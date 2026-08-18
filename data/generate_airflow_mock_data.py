#!/usr/bin/env python3
"""Generate mock Apache Airflow data that lines up with the SOTF mock dataset.

This script mocks **Airflow only**. It does not generate S3, ClickHouse, Jira or
TestRail data -- it *reads* the dataset produced by ``generate_sotf_mock_data.py``
and builds the pipeline layer that would have processed it:

    dags.json                  GET /api/v1/dags
    dag_details/<dag_id>.json  GET /api/v1/dags/{dag_id}/details
    tasks/<dag_id>.json        GET /api/v1/dags/{dag_id}/tasks
    dag_runs/<dag_id>.json     GET /api/v1/dags/{dag_id}/dagRuns
    task_instances/...         GET .../dagRuns/{run_id}/taskInstances
    xcom/...                   GET .../taskInstances/{task_id}/xcomEntries
    logs/...                   GET .../taskInstances/{task_id}/logs/{try_number}
    event_logs.json, import_errors.json, health.json, pools.json,
    variables.json, connections.json

The alignment is the point. Every DAG run covers a real time window from the
SOTF dataset, and its XCom payloads carry the actual trigger ids, TestRail run
ids, ZPT keys and row counts from that window. ``cross_reference_airflow.json``
maps each SOTF ride to the DAG runs that touched it, so "which pipeline loaded
this ride's logs?" has a checkable answer.

Usage:
    python3 generate_airflow_mock_data.py --clean
    python3 generate_airflow_mock_data.py --sotf-data ./out --out ./out_airflow
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Airflow-side constants
# ---------------------------------------------------------------------------

DAG_FOLDER = "/opt/airflow/dags/sotf"
WORKERS = ("airflow-worker-0", "airflow-worker-1", "airflow-worker-2", "airflow-worker-3")
DEFAULT_POOL = "default_pool"

DAG_RUN_STATES = ("queued", "running", "success", "failed")
TASK_STATES = ("success", "failed", "upstream_failed", "skipped", "running", "up_for_retry")


class _Defaulting(dict):
    """format_map helper so a missing log placeholder never raises."""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return "-"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    operator: str
    retries: int
    duration_range: tuple[int, int]
    can_fail: bool = False
    failure_message: str = ""
    log_body: tuple[str, ...] = ()


@dataclass(frozen=True)
class DagSpec:
    dag_id: str
    description: str
    schedule: str
    timetable_description: str
    interval_hours: int
    start_delay_minutes: int
    owners: tuple[str, ...]
    tags: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]

    @property
    def fileloc(self) -> str:
        return f"{DAG_FOLDER}/{self.dag_id}.py"


DAG_SPECS: tuple[DagSpec, ...] = (
    DagSpec(
        dag_id="sotf_artifact_ingest",
        description="Pull TEC artifact bundles from S3, unpack them and register artifact metadata.",
        schedule="0 */6 * * *",
        timetable_description="At 00:00, 06:00, 12:00 and 18:00",
        interval_hours=6,
        start_delay_minutes=4,
        owners=("sotf-data",),
        tags=("sotf", "ingest", "artifacts"),
        tasks=(
            TaskSpec(
                task_id="wait_for_artifact_manifests",
                operator="SqsSensor",
                retries=1,
                duration_range=(5, 240),
                can_fail=True,
                failure_message=(
                    "airflow.exceptions.AirflowSensorTimeout: Sensor has timed out; "
                    "run duration of 3600.0 seconds exceeds the specified timeout of 3600.0."
                ),
                log_body=(
                    "Poking SQS queue tec-artifacts for Artifacts messages",
                    "Received {artifact_count} artifact manifest message(s) for window {window}",
                ),
            ),
            TaskSpec(
                task_id="download_artifact_bundles",
                operator="PythonOperator",
                retries=2,
                duration_range=(20, 600),
                can_fail=True,
                failure_message=(
                    "botocore.exceptions.ReadTimeoutError: Read timeout on endpoint URL: "
                    '"https://s3.us-west-2.amazonaws.com/{bucket}/{prefix}/"'
                ),
                log_body=(
                    "Listing s3://{bucket}/{prefix}/ for objects modified in {window}",
                    "Downloading {artifact_count} bundle(s), {bytes_ingested} bytes total",
                    "First trigger in batch: {first_trigger}",
                ),
            ),
            TaskSpec(
                task_id="unpack_and_index_artifacts",
                operator="PythonOperator",
                retries=2,
                duration_range=(15, 420),
                can_fail=True,
                failure_message="zipfile.BadZipFile: File is not a zip file",
                log_body=(
                    "Unpacking {artifact_count} bundle(s) into the staging volume",
                    "Indexed snapshots/, video/ and logs/ members for {artifact_count} trigger(s)",
                ),
            ),
            TaskSpec(
                task_id="register_artifact_metadata",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 90),
                log_body=("Registered artifact metadata for {artifact_count} trigger(s)",),
            ),
            TaskSpec(
                task_id="verify_ingest_counts",
                operator="PythonOperator",
                retries=1,
                duration_range=(2, 30),
                can_fail=True,
                failure_message=(
                    "AssertionError: expected {artifact_count} bundles, indexed {indexed_count}"
                ),
                log_body=(
                    "Expected {artifact_count} bundle(s), indexed {indexed_count}",
                    "Pushing XCom key=ingest_summary",
                ),
            ),
        ),
    ),
    DagSpec(
        dag_id="sotf_ctm_log_ingest",
        description="Load CTM execution logs for the previous day into ClickHouse.",
        schedule="30 1 * * *",
        timetable_description="At 01:30",
        interval_hours=24,
        start_delay_minutes=90,
        owners=("sotf-data",),
        tags=("sotf", "ingest", "clickhouse"),
        tasks=(
            TaskSpec(
                task_id="extract_ctm_logs",
                operator="PythonOperator",
                retries=2,
                duration_range=(30, 300),
                can_fail=True,
                failure_message="FileNotFoundError: no CTM log export found for {window}",
                log_body=("Extracting CTM logs for {window}", "Read {rows_loaded} candidate row(s)"),
            ),
            TaskSpec(
                task_id="transform_log_rows",
                operator="PythonOperator",
                retries=2,
                duration_range=(20, 240),
                log_body=(
                    "Normalised {rows_loaded} row(s) across {distinct_triggers} trigger(s)",
                    "Severity split: {error_rows} ERROR, {warn_rows} WARN",
                ),
            ),
            TaskSpec(
                task_id="load_clickhouse",
                operator="PythonOperator",
                retries=3,
                duration_range=(25, 600),
                can_fail=True,
                failure_message=(
                    "requests.exceptions.ReadTimeout: HTTPConnectionPool(host='10.1.101.35', "
                    "port=8123): Read timed out. (read timeout=300)"
                ),
                log_body=(
                    "INSERT INTO {table} FORMAT JSONEachRow ({rows_loaded} rows)",
                    "ClickHouse acknowledged the insert",
                ),
            ),
            TaskSpec(
                task_id="validate_row_counts",
                operator="PythonOperator",
                retries=1,
                duration_range=(3, 45),
                can_fail=True,
                failure_message="AssertionError: row count drift for {window}",
                log_body=("Row count in {table} matches the extract: {rows_loaded}",),
            ),
        ),
    ),
    DagSpec(
        dag_id="sotf_testrail_sync",
        description="Sync TestRail runs and results for the previous day into the reporting warehouse.",
        schedule="0 2 * * *",
        timetable_description="At 02:00",
        interval_hours=24,
        start_delay_minutes=120,
        owners=("sotf-qa",),
        tags=("sotf", "sync", "testrail"),
        tasks=(
            TaskSpec(
                task_id="fetch_testrail_runs",
                operator="PythonOperator",
                retries=3,
                duration_range=(10, 180),
                can_fail=True,
                failure_message=(
                    "requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: "
                    "index.php?/api/v2/get_runs/{project_id}"
                ),
                log_body=(
                    "GET index.php?/api/v2/get_runs/{project_id}",
                    "Fetched {runs_synced} run(s) created in {window}",
                ),
            ),
            TaskSpec(
                task_id="fetch_testrail_results",
                operator="PythonOperator",
                retries=3,
                duration_range=(15, 300),
                can_fail=True,
                failure_message=(
                    "requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: "
                    "index.php?/api/v2/get_results/{first_test_id}"
                ),
                log_body=("Fetched {results_synced} result(s) for {runs_synced} run(s)",),
            ),
            TaskSpec(
                task_id="normalize_results",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 90),
                log_body=("Mapped status ids to labels for {results_synced} result(s)",),
            ),
            TaskSpec(
                task_id="load_warehouse",
                operator="PythonOperator",
                retries=2,
                duration_range=(10, 150),
                can_fail=True,
                failure_message="psycopg2.errors.DeadlockDetected: deadlock detected",
                log_body=("Upserted {results_synced} row(s) into warehouse.testrail_results",),
            ),
        ),
    ),
    DagSpec(
        dag_id="sotf_jira_defect_sync",
        description="Pull ZPT defects raised by SOTF automation and link them to executions.",
        schedule="15 2 * * *",
        timetable_description="At 02:15",
        interval_hours=24,
        start_delay_minutes=135,
        owners=("sotf-qa",),
        tags=("sotf", "sync", "jira"),
        tasks=(
            TaskSpec(
                task_id="fetch_zpt_issues",
                operator="PythonOperator",
                retries=3,
                duration_range=(8, 200),
                can_fail=True,
                failure_message=(
                    "requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: "
                    "/rest/api/2/search"
                ),
                log_body=(
                    'JQL: project = ZPT AND labels = "sotf" AND created >= "{window_start_short}"',
                    "Fetched {issues_synced} issue(s)",
                ),
            ),
            TaskSpec(
                task_id="link_issues_to_executions",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 60),
                can_fail=True,
                failure_message="KeyError: 'trigger_id' missing from issue {first_issue}",
                log_body=("Linked {issues_synced} issue(s) to {linked_rides} ride(s)",),
            ),
            TaskSpec(
                task_id="load_warehouse",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 90),
                log_body=("Upserted {issues_synced} row(s) into warehouse.zpt_defects",),
            ),
        ),
    ),
    DagSpec(
        dag_id="sotf_execution_metrics_daily",
        description="Build the daily SOTF execution fact table and publish pass-rate metrics.",
        schedule="0 3 * * *",
        timetable_description="At 03:00",
        interval_hours=24,
        start_delay_minutes=180,
        owners=("sotf-data",),
        tags=("sotf", "metrics", "reporting"),
        tasks=(
            TaskSpec(
                task_id="build_ride_facts",
                operator="PythonOperator",
                retries=2,
                duration_range=(20, 240),
                can_fail=True,
                failure_message="AirflowException: upstream ride facts incomplete for {window}",
                log_body=("Built {rides} ride fact row(s) for {window}",),
            ),
            TaskSpec(
                task_id="compute_pass_rate",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 60),
                log_body=("Pass rate {pass_rate} ({passed} passed / {failed} failed)",),
            ),
            TaskSpec(
                task_id="compute_pudo_breakdown",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 60),
                log_body=("Computed breakdown across {zone_count} PUDO zone(s)",),
            ),
            TaskSpec(
                task_id="compute_failure_mode_breakdown",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 60),
                log_body=("Top failure mode: {top_failure_mode}",),
            ),
            TaskSpec(
                task_id="publish_metrics",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 90),
                can_fail=True,
                failure_message="ConnectionError: metrics sink unreachable",
                log_body=("Published {rides} row(s) to warehouse.sotf_daily_metrics",),
            ),
        ),
    ),
    DagSpec(
        dag_id="sotf_scenario_health_report",
        description="Weekly scenario health rollup for the SOTF quality review.",
        schedule="0 7 * * 1",
        timetable_description="At 07:00, only on Monday",
        interval_hours=168,
        start_delay_minutes=420,
        owners=("sotf-qa",),
        tags=("sotf", "reporting", "weekly"),
        tasks=(
            TaskSpec(
                task_id="aggregate_week",
                operator="PythonOperator",
                retries=2,
                duration_range=(30, 300),
                log_body=("Aggregated {rides} ride(s) for week starting {window_start_short}",),
            ),
            TaskSpec(
                task_id="render_report",
                operator="PythonOperator",
                retries=1,
                duration_range=(10, 180),
                can_fail=True,
                failure_message="KeyError: 'pass_rate'",
                log_body=("Rendered weekly report with pass rate {pass_rate}",),
            ),
            TaskSpec(
                task_id="publish_report",
                operator="PythonOperator",
                retries=2,
                duration_range=(5, 60),
                log_body=("Published report to {report_uri}",),
            ),
        ),
    ),
    DagSpec(
        dag_id="sotf_artifact_retention",
        description="Expire TEC artifact bundles past the retention window.",
        schedule="45 4 * * *",
        timetable_description="At 04:45",
        interval_hours=24,
        start_delay_minutes=285,
        owners=("sotf-data",),
        tags=("sotf", "maintenance", "artifacts"),
        tasks=(
            TaskSpec(
                task_id="list_expired_artifacts",
                operator="PythonOperator",
                retries=2,
                duration_range=(10, 120),
                log_body=(
                    "Retention window is {retention_days} day(s)",
                    "Found {expired_objects} expired object(s) under s3://{bucket}/{prefix}/",
                ),
            ),
            TaskSpec(
                task_id="archive_expired_artifacts",
                operator="PythonOperator",
                retries=2,
                duration_range=(10, 240),
                can_fail=True,
                failure_message="botocore.exceptions.ClientError: An error occurred (SlowDown)",
                log_body=("Archived {expired_objects} object(s) to the cold tier",),
            ),
            TaskSpec(
                task_id="delete_expired_artifacts",
                operator="PythonOperator",
                retries=1,
                duration_range=(10, 180),
                can_fail=True,
                failure_message=(
                    "botocore.exceptions.ClientError: An error occurred (AccessDenied) when "
                    "calling the DeleteObjects operation"
                ),
                log_body=("Deleted {expired_objects} object(s), reclaimed {bytes_reclaimed} bytes",),
            ),
            TaskSpec(
                task_id="report_retention",
                operator="PythonOperator",
                retries=1,
                duration_range=(2, 20),
                log_body=("Retention run complete for {window}",),
            ),
        ),
    ),
)

AIRFLOW_VARIABLES = (
    {"key": "sotf_s3_bucket", "value": "{bucket}", "description": "TEC artifact bucket"},
    {"key": "sotf_s3_artifacts_prefix", "value": "{prefix}", "description": "TEC artifact prefix"},
    {"key": "sotf_clickhouse_table", "value": "{table}", "description": "CTM log table"},
    {"key": "sotf_testrail_project_id", "value": "{project_id}", "description": "TestRail project"},
    {"key": "sotf_jira_project_key", "value": "ZPT", "description": "Defect project key"},
    {"key": "sotf_artifact_retention_days", "value": "{retention_days}", "description": "Retention window"},
    # Airflow masks values whose key looks sensitive; the API returns them masked.
    {"key": "sotf_testrail_api_key", "value": "***", "description": "Masked by Airflow"},
    {"key": "sotf_jira_api_token", "value": "***", "description": "Masked by Airflow"},
)

AIRFLOW_CONNECTIONS = (
    {
        "connection_id": "sotf_s3",
        "conn_type": "aws",
        "description": "TEC artifact bucket",
        "host": None,
        "login": None,
        "schema": None,
        "port": None,
        "extra": '{"region_name": "us-west-2"}',
    },
    {
        "connection_id": "sotf_clickhouse",
        "conn_type": "http",
        "description": "CTM log store",
        "host": "10.1.101.35",
        "login": "default",
        "schema": "default",
        "port": 8123,
        "extra": None,
    },
    {
        "connection_id": "sotf_testrail",
        "conn_type": "http",
        "description": "TestRail API",
        "host": "testrail.example.com",
        "login": "sotf-automation@example.com",
        "schema": "https",
        "port": 443,
        "extra": None,
    },
    {
        "connection_id": "sotf_jira",
        "conn_type": "http",
        "description": "Jira API",
        "host": "jira.example.com",
        "login": "sotf-automation@example.com",
        "schema": "https",
        "port": 443,
        "extra": None,
    },
    {
        "connection_id": "sotf_warehouse",
        "conn_type": "postgres",
        "description": "Reporting warehouse",
        "host": "warehouse.example.internal",
        "login": "airflow",
        "schema": "sotf",
        "port": 5432,
        "extra": None,
    },
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def log_ts(dt: datetime) -> str:
    """Airflow task-log timestamp format."""
    utc = dt.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}+0000"


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def run_token(logical_date: datetime) -> str:
    """Filesystem-safe stand-in for dag_run_id, which contains ':' and '+'."""
    return logical_date.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def floor_to(dt: datetime, hours: int) -> datetime:
    """Floor to a schedule boundary: 6h slots, midnight, or Monday midnight."""
    base = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if hours == 168:
        midnight = base.replace(hour=0)
        return midnight - timedelta(days=midnight.weekday())
    if hours >= 24:
        return base.replace(hour=0)
    return base.replace(hour=base.hour - base.hour % hours)


def iter_intervals(
    window_start: datetime, now: datetime, hours: int
) -> list[tuple[datetime, datetime]]:
    """Data intervals that have fully closed by `now`."""
    step = timedelta(hours=hours)
    cursor = floor_to(window_start, hours)
    intervals: list[tuple[datetime, datetime]] = []
    while cursor + step <= now:
        intervals.append((cursor, cursor + step))
        cursor += step
    return intervals


# ---------------------------------------------------------------------------
# SOTF dataset input
# ---------------------------------------------------------------------------


@dataclass
class SotfData:
    root: Path
    now: datetime
    days: int
    bucket: str
    prefix: str
    table: str
    project_id: int
    rides: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    testrail_runs: list[dict[str, Any]]
    log_rows_by_date: dict[str, int]

    def rides_ended_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [r for r in self.rides if r["_ended"] and start <= r["_ended"] < end]

    def rides_started_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [r for r in self.rides if r["_started"] and start <= r["_started"] < end]

    def issues_created_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [i for i in self.issues if i["_created"] and start <= i["_created"] < end]

    def testrail_runs_between(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [r for r in self.testrail_runs if start <= r["_created"] < end]

    def log_rows_between(self, start: datetime, end: datetime) -> int:
        total = 0
        cursor = start
        while cursor < end:
            total += self.log_rows_by_date.get(cursor.strftime("%Y-%m-%d"), 0)
            cursor += timedelta(days=1)
        return total


def load_sotf_dataset(root: Path) -> SotfData:
    required = root / "cross_reference.json"
    summary_file = root / "summary.json"
    missing = [f.name for f in (required, summary_file) if not f.is_file()]
    if missing:
        raise SystemExit(
            f"SOTF dataset at {root} is missing {', '.join(missing)}\n"
            "Run generate_sotf_mock_data.py first, or pass --sotf-data."
        )

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    params = summary.get("parameters", {})

    rides = json.loads(required.read_text(encoding="utf-8"))
    for ride in rides:
        ride["_started"] = parse_dt(ride.get("started_at"))
        ride["_ended"] = parse_dt(ride.get("ended_at"))

    issues: list[dict[str, Any]] = []
    search_file = root / "jira/search_issues.json"
    if search_file.is_file():
        for issue in json.loads(search_file.read_text(encoding="utf-8")).get("issues", []):
            issue["_created"] = parse_dt(issue.get("fields", {}).get("created"))
            issues.append(issue)

    testrail_runs: list[dict[str, Any]] = []
    runs_file = root / "testrail/runs.json"
    if runs_file.is_file():
        for run in json.loads(runs_file.read_text(encoding="utf-8")):
            testrail_runs.append(
                {**run, "_created": datetime.fromtimestamp(run["created_on"], tz=timezone.utc)}
            )

    log_rows_by_date: dict[str, int] = {}
    logs_file = root / "clickhouse/logs.jsonl"
    if logs_file.is_file():
        with logs_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                # rows are {"timestamp": "YYYY-MM-DD HH:MM:SS", ...}
                day = json.loads(line)["timestamp"][:10]
                log_rows_by_date[day] = log_rows_by_date.get(day, 0) + 1
    else:
        for ride in rides:
            if ride["_started"]:
                day = ride["_started"].strftime("%Y-%m-%d")
                log_rows_by_date[day] = log_rows_by_date.get(day, 0) + int(
                    ride.get("clickhouse_log_rows", 0)
                )

    now = parse_dt(params.get("now")) or datetime.now(timezone.utc)
    return SotfData(
        root=root,
        now=now,
        days=int(params.get("days", 30)),
        bucket=params.get("bucket", "zoox-test-artifacts"),
        prefix=params.get("s3_prefix", "tec/dev/artifacts"),
        table=params.get("clickhouse", "default.logs"),
        project_id=int(params.get("testrail_project_id", 42)),
        rides=rides,
        issues=issues,
        testrail_runs=testrail_runs,
        log_rows_by_date=log_rows_by_date,
    )


# ---------------------------------------------------------------------------
# XCom payloads: this is where Airflow is tied to the SOTF data
# ---------------------------------------------------------------------------

PayloadFn = Callable[["Window"], dict[str, Any]]


@dataclass
class Window:
    """One data interval, plus everything a payload builder needs to describe it."""

    sotf: SotfData
    args: argparse.Namespace
    start: datetime
    end: datetime


def payload_artifact_ingest(w: Window) -> dict[str, Any]:
    rides = [r for r in w.sotf.rides_ended_between(w.start, w.end) if r.get("s3", {}).get("key")]
    trigger_ids = [r["trigger_id"] for r in rides]
    return {
        "window_start": iso(w.start),
        "window_end": iso(w.end),
        "bucket": w.sotf.bucket,
        "prefix": w.sotf.prefix,
        "artifact_count": len(rides),
        "indexed_count": len(rides),
        "bytes_ingested": sum(48_000 + len(r["trigger_id"]) * 40 for r in rides),
        "trigger_ids": trigger_ids,
        "keys_sample": [r["s3"]["key"] for r in rides[:5]],
    }


def payload_ctm_log_ingest(w: Window) -> dict[str, Any]:
    rides = w.sotf.rides_started_between(w.start, w.end)
    failed = [r for r in rides if r["ride_status"] == "Failed"]
    return {
        "window_start": iso(w.start),
        "window_end": iso(w.end),
        "table": w.sotf.table,
        "rows_loaded": w.sotf.log_rows_between(w.start, w.end),
        "distinct_triggers": len(rides),
        # Severity split is approximated from the failure count rather than re-read
        # row by row; the SOTF generator emits a handful of WARN/ERROR lines per failure.
        "error_rows": 3 * len(failed),
        "warn_rows": 2 * len(failed),
        "trigger_ids": [r["trigger_id"] for r in rides],
    }


def payload_testrail_sync(w: Window) -> dict[str, Any]:
    runs = w.sotf.testrail_runs_between(w.start, w.end)
    results = [
        r
        for r in w.sotf.rides_ended_between(w.start, w.end)
        if r.get("testrail", {}).get("result_id")
    ]
    test_ids = [r["testrail"]["test_id"] for r in results]
    return {
        "window_start": iso(w.start),
        "window_end": iso(w.end),
        "project_id": w.sotf.project_id,
        "runs_synced": len(runs),
        "results_synced": len(results),
        "run_ids": [r["id"] for r in runs],
        "first_test_id": test_ids[0] if test_ids else 0,
        "result_ids": [r["testrail"]["result_id"] for r in results],
        "trigger_ids": [r["trigger_id"] for r in results],
    }


def payload_jira_defect_sync(w: Window) -> dict[str, Any]:
    created = w.sotf.issues_created_between(w.start, w.end)
    # A ZPT issue is created once but keeps collecting occurrences, so the rides that
    # matter for this window are the ones that ended in it, not the issue's full history.
    linked = [r for r in w.sotf.rides_ended_between(w.start, w.end) if r.get("jira_key")]
    keys = sorted({r["jira_key"] for r in linked} | {i["key"] for i in created})
    return {
        "window_start": iso(w.start),
        "window_end": iso(w.end),
        "project_key": "ZPT",
        "issues_created": len(created),
        "issues_synced": len(keys),
        "issue_keys": keys,
        "first_issue": keys[0] if keys else "-",
        "linked_rides": len(linked),
        "trigger_ids": [r["trigger_id"] for r in linked],
    }


def _breakdown(rides: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ride in rides:
        value = ride.get(key)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def payload_execution_metrics(w: Window) -> dict[str, Any]:
    rides = [
        r for r in w.sotf.rides_ended_between(w.start, w.end) if r["ride_status"] in ("Passed", "Failed")
    ]
    passed = [r for r in rides if r["ride_status"] == "Passed"]
    failed = [r for r in rides if r["ride_status"] == "Failed"]
    by_zone = _breakdown(rides, "pudo_zone")
    by_mode = _breakdown(failed, "failure_mode")
    return {
        "window_start": iso(w.start),
        "window_end": iso(w.end),
        "rides": len(rides),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / len(rides), 4) if rides else None,
        "by_environment": _breakdown(rides, "environment"),
        "by_ride_service": _breakdown(rides, "ride_service"),
        "by_pudo_zone": by_zone,
        "by_failure_mode": by_mode,
        "zone_count": len(by_zone),
        "top_failure_mode": next(iter(by_mode), "-"),
        "trigger_ids": [r["trigger_id"] for r in rides],
    }


def payload_scenario_health(w: Window) -> dict[str, Any]:
    metrics = payload_execution_metrics(w)
    by_scenario = _breakdown(
        [r for r in w.sotf.rides_ended_between(w.start, w.end) if r["ride_status"] == "Failed"],
        "scenario",
    )
    metrics.update(
        {
            "week_start": w.start.strftime("%Y-%m-%d"),
            "top_failure_modes": list(metrics["by_failure_mode"].items())[:3],
            "failures_by_scenario": by_scenario,
            "report_uri": f"s3://{w.sotf.bucket}/reports/scenario_health/{w.start.strftime('%Y%m%d')}.html",
        }
    )
    return metrics


def payload_artifact_retention(w: Window) -> dict[str, Any]:
    cutoff = w.end - timedelta(days=w.args.retention_days)
    expired = [
        r
        for r in w.sotf.rides
        if r.get("s3", {}).get("key") and r["_ended"] and r["_ended"] < cutoff
    ]
    return {
        "window_start": iso(w.start),
        "window_end": iso(w.end),
        "bucket": w.sotf.bucket,
        "prefix": w.sotf.prefix,
        "retention_days": w.args.retention_days,
        "cutoff": iso(cutoff),
        "expired_objects": len(expired),
        "bytes_reclaimed": sum(48_000 for _ in expired),
        "keys_sample": [r["s3"]["key"] for r in expired[:5]],
    }


PAYLOAD_BUILDERS: dict[str, PayloadFn] = {
    "sotf_artifact_ingest": payload_artifact_ingest,
    "sotf_ctm_log_ingest": payload_ctm_log_ingest,
    "sotf_testrail_sync": payload_testrail_sync,
    "sotf_jira_defect_sync": payload_jira_defect_sync,
    "sotf_execution_metrics_daily": payload_execution_metrics,
    "sotf_scenario_health_report": payload_scenario_health,
    "sotf_artifact_retention": payload_artifact_retention,
}

XCOM_KEYS = {
    "sotf_artifact_ingest": ("verify_ingest_counts", "ingest_summary"),
    "sotf_ctm_log_ingest": ("validate_row_counts", "load_summary"),
    "sotf_testrail_sync": ("load_warehouse", "sync_summary"),
    "sotf_jira_defect_sync": ("load_warehouse", "sync_summary"),
    "sotf_execution_metrics_daily": ("publish_metrics", "metrics_summary"),
    "sotf_scenario_health_report": ("publish_report", "report_summary"),
    "sotf_artifact_retention": ("report_retention", "retention_summary"),
}


# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------


RETRY_DELAY = timedelta(minutes=5)


@dataclass
class TaskRun:
    spec: TaskSpec
    state: str | None
    try_number: int
    start: datetime | None
    end: datetime | None

    def attempt_window(self, attempt: int) -> tuple[datetime, datetime]:
        """Clock for one attempt. Earlier attempts sit one retry delay apart."""
        final_start = self.start or datetime.now(timezone.utc)
        final_end = self.end or final_start + timedelta(seconds=30)
        if attempt >= self.try_number:
            return final_start, final_end
        end = final_start - RETRY_DELAY * (self.try_number - attempt)
        return end - timedelta(seconds=30), end

    @property
    def duration(self) -> float | None:
        if not (self.start and self.end):
            return None
        return round((self.end - self.start).total_seconds(), 3)


@dataclass
class DagRun:
    spec: DagSpec
    logical_date: datetime
    data_interval_start: datetime
    data_interval_end: datetime
    start: datetime | None
    end: datetime | None
    state: str
    payload: dict[str, Any]
    tasks: list[TaskRun] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return f"scheduled__{iso(self.logical_date)}"

    @property
    def token(self) -> str:
        return run_token(self.logical_date)


def build_dag_run(
    spec: DagSpec, w: Window, rng: random.Random, args: argparse.Namespace, is_last: bool
) -> DagRun:
    payload = PAYLOAD_BUILDERS[spec.dag_id](w)
    scheduled_at = w.end + timedelta(minutes=spec.start_delay_minutes)
    run_start = scheduled_at + timedelta(seconds=rng.randint(1, 45))

    # An in-flight run at the tail of the window keeps the dataset realistic.
    running = is_last and rng.random() < 0.3

    fail_candidates = [t for t in spec.tasks if t.can_fail]
    hard_fail = bool(fail_candidates) and not running and rng.random() < args.failure_rate
    retry_then_pass = (
        bool(fail_candidates)
        and not running
        and not hard_fail
        and rng.random() < args.retry_rate
    )
    marked = rng.choice(fail_candidates).task_id if (hard_fail or retry_then_pass) else None

    tasks: list[TaskRun] = []
    cursor = run_start
    state = "success"
    stopped = False
    running_index = rng.randrange(len(spec.tasks)) if running else -1

    for index, task_spec in enumerate(spec.tasks):
        if stopped:
            tasks.append(TaskRun(task_spec, "upstream_failed", 0, None, None))
            continue
        if running and index > running_index:
            tasks.append(TaskRun(task_spec, None, 0, None, None))
            continue

        start = cursor + timedelta(seconds=rng.randint(1, 12))
        seconds = rng.randint(*task_spec.duration_range)

        if running and index == running_index:
            # Downstream tasks stay unscheduled (state null) via the branch above.
            tasks.append(TaskRun(task_spec, "running", 1, start, None))
            cursor = start + timedelta(seconds=seconds)
            state = "running"
            continue

        if task_spec.task_id == marked and hard_fail:
            attempts = task_spec.retries + 1
            first_start = start + RETRY_DELAY * task_spec.retries
            failed_at = first_start + timedelta(seconds=seconds)
            tasks.append(TaskRun(task_spec, "failed", attempts, first_start, failed_at))
            cursor = failed_at
            state = "failed"
            stopped = True
            continue

        if task_spec.task_id == marked and retry_then_pass:
            retry_start = start + RETRY_DELAY
            end = retry_start + timedelta(seconds=seconds)
            tasks.append(TaskRun(task_spec, "success", 2, retry_start, end))
            cursor = end
            continue

        end = start + timedelta(seconds=seconds)
        tasks.append(TaskRun(task_spec, "success", 1, start, end))
        cursor = end

    return DagRun(
        spec=spec,
        logical_date=w.start,
        data_interval_start=w.start,
        data_interval_end=w.end,
        start=run_start,
        end=None if state == "running" else cursor,
        state=state,
        payload=payload,
        tasks=tasks,
    )


def build_runs(sotf: SotfData, args: argparse.Namespace, rng: random.Random) -> list[DagRun]:
    window_start = sotf.now - timedelta(days=sotf.days)
    runs: list[DagRun] = []
    for spec in DAG_SPECS:
        intervals = iter_intervals(window_start, sotf.now, spec.interval_hours)
        for position, (start, end) in enumerate(intervals):
            window = Window(sotf=sotf, args=args, start=start, end=end)
            runs.append(
                build_dag_run(spec, window, rng, args, is_last=position == len(intervals) - 1)
            )
    return runs


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def emit_dags(out: Path, runs: list[DagRun], sotf: SotfData) -> None:
    last_run_by_dag: dict[str, DagRun] = {}
    for run in runs:
        current = last_run_by_dag.get(run.spec.dag_id)
        if current is None or run.logical_date > current.logical_date:
            last_run_by_dag[run.spec.dag_id] = run

    dags = []
    for spec in DAG_SPECS:
        last = last_run_by_dag.get(spec.dag_id)
        next_start = (
            last.data_interval_end
            if last
            else floor_to(sotf.now, spec.interval_hours)
        )
        next_end = next_start + timedelta(hours=spec.interval_hours)
        dags.append(
            {
                "dag_id": spec.dag_id,
                "root_dag_id": None,
                "is_paused": False,
                "is_active": True,
                "is_subdag": False,
                "last_parsed_time": iso(sotf.now - timedelta(minutes=3)),
                "last_pickled": None,
                "last_expired": None,
                "scheduler_lock": None,
                "pickle_id": None,
                "default_view": "grid",
                "fileloc": spec.fileloc,
                "file_token": spec.dag_id,
                "owners": list(spec.owners),
                "description": spec.description,
                "schedule_interval": {"__type": "CronExpression", "value": spec.schedule},
                "timetable_description": spec.timetable_description,
                "tags": [{"name": tag} for tag in spec.tags],
                "max_active_tasks": 16,
                "max_active_runs": 1,
                "has_task_concurrency_limits": False,
                "has_import_errors": False,
                "next_dagrun": iso(next_start),
                "next_dagrun_data_interval_start": iso(next_start),
                "next_dagrun_data_interval_end": iso(next_end),
                "next_dagrun_create_after": iso(
                    next_end + timedelta(minutes=spec.start_delay_minutes)
                ),
            }
        )

    write_json(out / "dags.json", {"dags": dags, "total_entries": len(dags)})

    for spec, payload in zip(DAG_SPECS, dags):
        details = dict(payload)
        details.update(
            {
                "catchup": False,
                "concurrency": 16,
                "dag_run_timeout": {"__type": "TimeDelta", "days": 0, "seconds": 14400, "microseconds": 0},
                "doc_md": (
                    f"# {spec.dag_id}\n\n{spec.description}\n\n"
                    f"Owner: {', '.join(spec.owners)}. Schedule: `{spec.schedule}`.\n"
                ),
                "end_date": None,
                "is_paused_upon_creation": False,
                "orientation": "LR",
                "params": {},
                "render_template_as_native_obj": False,
                "start_date": iso(sotf.now - timedelta(days=sotf.days + 30)),
                "timezone": "UTC",
            }
        )
        write_json(out / f"dag_details/{spec.dag_id}.json", details)


def emit_tasks(out: Path, sotf: SotfData) -> None:
    for spec in DAG_SPECS:
        tasks = []
        for index, task_spec in enumerate(spec.tasks):
            downstream = (
                [spec.tasks[index + 1].task_id] if index + 1 < len(spec.tasks) else []
            )
            tasks.append(
                {
                    "task_id": task_spec.task_id,
                    "owner": spec.owners[0],
                    "start_date": iso(sotf.now - timedelta(days=sotf.days + 30)),
                    "end_date": None,
                    "trigger_rule": "all_success",
                    "depends_on_past": False,
                    "wait_for_downstream": False,
                    "retries": task_spec.retries,
                    "queue": "default",
                    "pool": DEFAULT_POOL,
                    "pool_slots": 1,
                    "execution_timeout": {
                        "__type": "TimeDelta",
                        "days": 0,
                        "seconds": 3600,
                        "microseconds": 0,
                    },
                    "retry_delay": {"__type": "TimeDelta", "days": 0, "seconds": 300, "microseconds": 0},
                    "retry_exponential_backoff": False,
                    "priority_weight": len(spec.tasks) - index,
                    "operator_name": task_spec.operator,
                    "class_ref": {
                        "module_path": "airflow.operators.python"
                        if task_spec.operator == "PythonOperator"
                        else "airflow.providers.amazon.aws.sensors.sqs",
                        "class_name": task_spec.operator,
                    },
                    "downstream_task_ids": downstream,
                    "ui_color": "#ffefeb",
                    "ui_fgcolor": "#000",
                    "template_fields": ["templates_dict", "op_args", "op_kwargs"],
                    "sub_dag": None,
                    "is_mapped": False,
                }
            )
        write_json(out / f"tasks/{spec.dag_id}.json", {"tasks": tasks, "total_entries": len(tasks)})


def emit_dag_runs(out: Path, runs: list[DagRun]) -> None:
    by_dag: dict[str, list[DagRun]] = {}
    for run in runs:
        by_dag.setdefault(run.spec.dag_id, []).append(run)

    for dag_id, dag_runs in by_dag.items():
        payload = [
            {
                "dag_run_id": run.run_id,
                "dag_id": dag_id,
                "logical_date": iso(run.logical_date),
                "execution_date": iso(run.logical_date),
                "start_date": iso(run.start) if run.start else None,
                "end_date": iso(run.end) if run.end else None,
                "data_interval_start": iso(run.data_interval_start),
                "data_interval_end": iso(run.data_interval_end),
                "last_scheduling_decision": iso(run.end or run.start or run.logical_date),
                "run_type": "scheduled",
                "state": run.state,
                "external_trigger": False,
                "conf": {},
                "note": None,
            }
            for run in dag_runs
        ]
        write_json(
            out / f"dag_runs/{dag_id}.json",
            {"dag_runs": payload, "total_entries": len(payload)},
        )


def emit_task_instances(out: Path, runs: list[DagRun], rng: random.Random) -> None:
    for run in runs:
        instances = []
        for task in run.tasks:
            started = task.start
            instances.append(
                {
                    "task_id": task.spec.task_id,
                    "dag_id": run.spec.dag_id,
                    "dag_run_id": run.run_id,
                    "execution_date": iso(run.logical_date),
                    "start_date": iso(started) if started else None,
                    "end_date": iso(task.end) if task.end else None,
                    "duration": task.duration,
                    "state": task.state,
                    "try_number": task.try_number,
                    "map_index": -1,
                    "max_tries": task.spec.retries,
                    "hostname": rng.choice(WORKERS) if started else "",
                    "unixname": "airflow",
                    "pool": DEFAULT_POOL,
                    "pool_slots": 1,
                    "queue": "default",
                    "priority_weight": 1,
                    "operator": task.spec.operator,
                    "queued_when": iso(started - timedelta(seconds=3)) if started else None,
                    "pid": rng.randint(1000, 60000) if started else None,
                    "executor_config": "{}",
                    "sla_miss": None,
                    "rendered_fields": {},
                    "trigger": None,
                    "triggerer_job": None,
                    "note": None,
                }
            )
        write_json(
            out / f"task_instances/{run.spec.dag_id}/{run.token}.json",
            {"task_instances": instances, "total_entries": len(instances)},
        )


def emit_xcom(out: Path, runs: list[DagRun]) -> None:
    for run in runs:
        task_id, key = XCOM_KEYS[run.spec.dag_id]
        emitting = next((t for t in run.tasks if t.spec.task_id == task_id), None)
        entries = []
        if emitting and emitting.state == "success" and emitting.end:
            entries.append(
                {
                    "key": key,
                    "value": run.payload,
                    "timestamp": iso(emitting.end),
                    "execution_date": iso(run.logical_date),
                    "map_index": -1,
                    "task_id": task_id,
                    "dag_id": run.spec.dag_id,
                }
            )
        write_json(
            out / f"xcom/{run.spec.dag_id}/{run.token}.json",
            {"xcom_entries": entries, "total_entries": len(entries)},
        )


def log_context(run: DagRun) -> dict[str, Any]:
    context: dict[str, Any] = {
        "window": f"{run.data_interval_start.strftime('%Y-%m-%d %H:%M')} to "
        f"{run.data_interval_end.strftime('%Y-%m-%d %H:%M')}",
        "window_start_short": run.data_interval_start.strftime("%Y-%m-%d"),
        "run_id": run.run_id,
    }
    for key, value in run.payload.items():
        if isinstance(value, (str, int, float)) or value is None:
            context[key] = value
    triggers = run.payload.get("trigger_ids") or []
    context["first_trigger"] = triggers[0] if triggers else "-"
    return context


def render_log(run: DagRun, task: TaskRun, attempt: int, failed: bool, context: dict[str, Any]) -> str:
    spec = task.spec
    start, end = task.attempt_window(attempt)

    lines = [
        f"[{log_ts(start)}] {{taskinstance.py:1157}} INFO - Dependencies all met for "
        f"dep_context=non-requeueable deps ti=<TaskInstance: {run.spec.dag_id}.{spec.task_id} "
        f"{run.run_id} [queued]>",
        f"[{log_ts(start)}] {{taskinstance.py:1359}} INFO - Starting attempt {attempt} of "
        f"{spec.retries + 1}",
        f"[{log_ts(start + timedelta(milliseconds=180))}] {{standard_task_runner.py:60}} INFO - "
        f"Started process to run task",
        f"[{log_ts(start + timedelta(milliseconds=320))}] {{taskinstance.py:1660}} INFO - Exporting "
        f"env vars: AIRFLOW_CTX_DAG_ID={run.spec.dag_id} AIRFLOW_CTX_TASK_ID={spec.task_id} "
        f"AIRFLOW_CTX_EXECUTION_DATE={iso(run.logical_date)} AIRFLOW_CTX_DAG_RUN_ID={run.run_id}",
    ]

    body_at = start + timedelta(seconds=1)
    for template in spec.log_body:
        lines.append(
            f"[{log_ts(body_at)}] {{logging_mixin.py:151}} INFO - "
            + template.format_map(_Defaulting(context))
        )
        body_at += timedelta(seconds=2)

    if failed:
        message = spec.failure_message.format_map(_Defaulting(context))
        lines += [
            f"[{log_ts(end)}] {{taskinstance.py:1824}} ERROR - Task failed with exception",
            "Traceback (most recent call last):",
            f'  File "{run.spec.fileloc}", line 84, in {spec.task_id}',
            "    result = run_step(context)",
            '  File "/opt/airflow/plugins/sotf/steps.py", line 212, in run_step',
            "    return handler(**kwargs)",
            message,
            f"[{log_ts(end + timedelta(milliseconds=90))}] {{taskinstance.py:1400}} INFO - Marking "
            f"task as {'UP_FOR_RETRY' if attempt <= spec.retries else 'FAILED'}. "
            f"dag_id={run.spec.dag_id}, task_id={spec.task_id}, "
            f"execution_date={run.logical_date.strftime('%Y%m%dT%H%M%S')}",
            f"[{log_ts(end + timedelta(milliseconds=200))}] {{local_task_job_runner.py:225}} INFO - "
            f"Task exited with return code 1",
        ]
    else:
        xcom_task, xcom_key = XCOM_KEYS[run.spec.dag_id]
        if spec.task_id == xcom_task:
            lines.append(
                f"[{log_ts(end - timedelta(milliseconds=400))}] {{xcom.py:664}} INFO - Pushed XCom "
                f"key={xcom_key}"
            )
        lines += [
            f"[{log_ts(end)}] {{taskinstance.py:1400}} INFO - Marking task as SUCCESS. "
            f"dag_id={run.spec.dag_id}, task_id={spec.task_id}, "
            f"execution_date={run.logical_date.strftime('%Y%m%dT%H%M%S')}",
            f"[{log_ts(end + timedelta(milliseconds=140))}] {{local_task_job_runner.py:225}} INFO - "
            f"Task exited with return code 0",
        ]
    return "\n".join(lines) + "\n"


def emit_logs(out: Path, runs: list[DagRun], args: argparse.Namespace) -> int:
    written = 0
    sampled_dags: set[str] = set()

    for run in runs:
        interesting = run.state == "failed" or any(t.try_number > 1 for t in run.tasks)
        sample = run.spec.dag_id not in sampled_dags
        if not (args.all_logs or interesting or sample):
            continue
        sampled_dags.add(run.spec.dag_id)

        context = log_context(run)
        for task in run.tasks:
            if task.state in (None, "upstream_failed", "skipped"):
                continue
            directory = out / f"logs/{run.spec.dag_id}/{run.token}/{task.spec.task_id}"
            directory.mkdir(parents=True, exist_ok=True)
            if task.state == "failed":
                for attempt in range(1, task.try_number + 1):
                    (directory / f"attempt_{attempt}.log").write_text(
                        render_log(run, task, attempt, failed=True, context=context),
                        encoding="utf-8",
                    )
                    written += 1
            elif task.try_number > 1:
                (directory / "attempt_1.log").write_text(
                    render_log(run, task, 1, failed=True, context=context), encoding="utf-8"
                )
                (directory / f"attempt_{task.try_number}.log").write_text(
                    render_log(run, task, task.try_number, failed=False, context=context),
                    encoding="utf-8",
                )
                written += 2
            elif task.state == "success":
                (directory / "attempt_1.log").write_text(
                    render_log(run, task, 1, failed=False, context=context), encoding="utf-8"
                )
                written += 1
    return written


def emit_event_logs(out: Path, runs: list[DagRun]) -> None:
    events = []
    event_id = 900000
    for run in sorted(runs, key=lambda r: r.logical_date):
        for task in run.tasks:
            if task.state is None or not task.start:
                continue
            event_id += 1
            events.append(
                {
                    "event_log_id": event_id,
                    "when": iso(task.start),
                    "dag_id": run.spec.dag_id,
                    "task_id": task.spec.task_id,
                    "run_id": run.run_id,
                    "map_index": -1,
                    "try_number": task.try_number,
                    "event": "running",
                    "execution_date": iso(run.logical_date),
                    "owner": run.spec.owners[0],
                    "extra": None,
                }
            )
            if task.state in ("success", "failed"):
                event_id += 1
                events.append(
                    {
                        "event_log_id": event_id,
                        "when": iso(task.end or task.start),
                        "dag_id": run.spec.dag_id,
                        "task_id": task.spec.task_id,
                        "run_id": run.run_id,
                        "map_index": -1,
                        "try_number": task.try_number,
                        "event": task.state,
                        "execution_date": iso(run.logical_date),
                        "owner": run.spec.owners[0],
                        "extra": None,
                    }
                )
    write_json(out / "event_logs.json", {"event_logs": events, "total_entries": len(events)})


def emit_instance_metadata(out: Path, sotf: SotfData, runs: list[DagRun], args: argparse.Namespace) -> None:
    write_json(out / "import_errors.json", {"import_errors": [], "total_entries": 0})
    write_json(
        out / "health.json",
        {
            "metadatabase": {"status": "healthy"},
            "scheduler": {
                "status": "healthy",
                "latest_scheduler_heartbeat": iso(sotf.now - timedelta(seconds=4)),
            },
            "triggerer": {
                "status": "healthy",
                "latest_triggerer_heartbeat": iso(sotf.now - timedelta(seconds=6)),
            },
            "dag_processor": {
                "status": "healthy",
                "latest_dag_processor_heartbeat": iso(sotf.now - timedelta(seconds=11)),
            },
        },
    )

    running_tasks = sum(
        1 for run in runs if run.state == "running" for t in run.tasks if t.state == "running"
    )
    write_json(
        out / "pools.json",
        {
            "pools": [
                {
                    "name": DEFAULT_POOL,
                    "slots": 128,
                    "occupied_slots": running_tasks,
                    "running_slots": running_tasks,
                    "queued_slots": 0,
                    "scheduled_slots": 0,
                    "open_slots": 128 - running_tasks,
                    "description": "Default pool",
                    "include_deferred": False,
                }
            ],
            "total_entries": 1,
        },
    )

    substitutions = {
        "bucket": sotf.bucket,
        "prefix": sotf.prefix,
        "table": sotf.table,
        "project_id": sotf.project_id,
        "retention_days": args.retention_days,
    }
    write_json(
        out / "variables.json",
        {
            "variables": [
                {
                    "key": variable["key"],
                    "value": str(variable["value"]).format(**substitutions),
                    "description": variable["description"],
                }
                for variable in AIRFLOW_VARIABLES
            ],
            "total_entries": len(AIRFLOW_VARIABLES),
        },
    )
    write_json(
        out / "connections.json",
        {
            # The Airflow API never returns connection passwords; neither does this fixture.
            "connections": [dict(connection) for connection in AIRFLOW_CONNECTIONS],
            "total_entries": len(AIRFLOW_CONNECTIONS),
        },
    )


def emit_runs_index(out: Path, runs: list[DagRun]) -> None:
    write_json(
        out / "runs_index.json",
        {
            "note": (
                "dag_run_id contains ':' and '+', so fixture files are named by the compact "
                "logical date instead. Use this index to go from a dag_run_id to its files."
            ),
            "runs": [
                {
                    "dag_id": run.spec.dag_id,
                    "dag_run_id": run.run_id,
                    "logical_date": iso(run.logical_date),
                    "state": run.state,
                    "token": run.token,
                    "task_instances_file": f"task_instances/{run.spec.dag_id}/{run.token}.json",
                    "xcom_file": f"xcom/{run.spec.dag_id}/{run.token}.json",
                    "log_dir": f"logs/{run.spec.dag_id}/{run.token}",
                }
                for run in sorted(runs, key=lambda r: (r.spec.dag_id, r.logical_date))
            ],
        },
    )


def emit_cross_reference(out: Path, sotf: SotfData, runs: list[DagRun]) -> None:
    """Map every SOTF ride to the DAG runs that touched it."""
    by_dag_and_trigger: dict[str, dict[str, str]] = {spec.dag_id: {} for spec in DAG_SPECS}
    for run in runs:
        table = by_dag_and_trigger[run.spec.dag_id]
        for trigger_id in run.payload.get("trigger_ids", []) or []:
            table.setdefault(trigger_id, run.run_id)

    entries = []
    for ride in sotf.rides:
        trigger_id = ride["trigger_id"]
        entries.append(
            {
                "trigger_id": trigger_id,
                "ride_status": ride["ride_status"],
                "started_at": ride.get("started_at"),
                "ended_at": ride.get("ended_at"),
                "jira_key": ride.get("jira_key"),
                "s3_key": ride.get("s3", {}).get("key"),
                "testrail_result_id": ride.get("testrail", {}).get("result_id"),
                "airflow": {
                    dag_id: table.get(trigger_id) for dag_id, table in by_dag_and_trigger.items()
                },
            }
        )

    write_json(
        out / "cross_reference_airflow.json",
        {
            "note": (
                "Which DAG run covered each SOTF ride. Null means the ride had nothing for that "
                "pipeline to do (a Queued ride has no artifacts and no TestRail result). A run "
                "listed here may have failed -- check its state in dag_runs/<dag_id>.json before "
                "concluding the ride was actually processed."
            ),
            "rides": entries,
        },
    )


API_INDEX = {
    "base_url": "https://airflow.example.com",
    "api_version": "v1 (Airflow 2.x stable REST API)",
    "endpoints": {
        "GET /api/v1/dags": "dags.json",
        "GET /api/v1/dags/{dag_id}/details": "dag_details/{dag_id}.json",
        "GET /api/v1/dags/{dag_id}/tasks": "tasks/{dag_id}.json",
        "GET /api/v1/dags/{dag_id}/dagRuns": "dag_runs/{dag_id}.json",
        "GET /api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances": "task_instances/{dag_id}/{token}.json",
        "GET /api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries": "xcom/{dag_id}/{token}.json",
        "GET /api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}": "logs/{dag_id}/{token}/{task_id}/attempt_{try_number}.log",
        "GET /api/v1/eventLogs": "event_logs.json",
        "GET /api/v1/importErrors": "import_errors.json",
        "GET /api/v1/health": "health.json",
        "GET /api/v1/pools": "pools.json",
        "GET /api/v1/variables": "variables.json",
        "GET /api/v1/connections": "connections.json",
    },
    "notes": [
        "token = logical_date as %Y%m%dT%H%M; resolve a dag_run_id through runs_index.json.",
        "The real list-xcomEntries endpoint omits `value`; it is included here to be useful.",
        "Read-only fixtures. No trigger-DAG, clear-task or set-state payloads are emitted.",
        "Variables whose key looks sensitive are masked, and connections carry no password.",
    ],
}


# ---------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------


def run_self_check(out: Path, sotf: SotfData, runs: list[DagRun]) -> list[str]:
    problems: list[str] = []

    index = json.loads((out / "runs_index.json").read_text(encoding="utf-8"))["runs"]
    known_run_ids = {entry["dag_run_id"] for entry in index}

    for entry in index:
        for relative in (entry["task_instances_file"], entry["xcom_file"]):
            if not (out / relative).is_file():
                problems.append(f"{entry['dag_run_id']}: missing {relative}")

    for run in runs:
        if run.run_id not in known_run_ids:
            problems.append(f"{run.run_id}: absent from runs_index.json")
        if run.state == "failed" and not any(t.state == "failed" for t in run.tasks):
            problems.append(f"{run.run_id}: marked failed but no task instance failed")
        if run.state == "success" and any(t.state == "failed" for t in run.tasks):
            problems.append(f"{run.run_id}: marked success but a task instance failed")
        if run.state == "failed":
            log_dir = out / f"logs/{run.spec.dag_id}/{run.token}"
            if not log_dir.is_dir():
                problems.append(f"{run.run_id}: failed run has no logs")

    # Every artifact-bearing ride must be claimed by exactly one artifact ingest run.
    ingest_claims: dict[str, list[str]] = {}
    for run in runs:
        if run.spec.dag_id != "sotf_artifact_ingest":
            continue
        for trigger_id in run.payload.get("trigger_ids", []):
            ingest_claims.setdefault(trigger_id, []).append(run.run_id)

    # Only rides inside the window the ingest DAG actually covered can be claimed.
    ingest_runs = [run for run in runs if run.spec.dag_id == "sotf_artifact_ingest"]
    if ingest_runs:
        coverage_start = min(run.data_interval_start for run in ingest_runs)
        coverage_end = max(run.data_interval_end for run in ingest_runs)
        for ride in sotf.rides:
            if not ride.get("s3", {}).get("key") or not ride["_ended"]:
                continue
            if not (coverage_start <= ride["_ended"] < coverage_end):
                continue
            claims = ingest_claims.get(ride["trigger_id"], [])
            if len(claims) != 1:
                problems.append(
                    f"{ride['trigger_id']}: claimed by {len(claims)} artifact ingest "
                    "run(s), expected 1"
                )

    secrets = {"password", "token", "secret", "api_key", "apikey"}
    variables = json.loads((out / "variables.json").read_text(encoding="utf-8"))["variables"]
    for variable in variables:
        looks_sensitive = any(marker in variable["key"].lower() for marker in secrets)
        if looks_sensitive and variable["value"] != "***":
            problems.append(f"variable {variable['key']} is not masked")

    connections = json.loads((out / "connections.json").read_text(encoding="utf-8"))["connections"]
    for connection in connections:
        if "password" in connection:
            problems.append(f"connection {connection['connection_id']} exposes a password field")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mock Airflow data aligned with an existing SOTF mock dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sotf-data", type=Path, default=HERE / "out", help="output of generate_sotf_mock_data.py")
    parser.add_argument("--out", type=Path, default=HERE / "out_airflow")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    parser.add_argument("--failure-rate", type=float, default=0.07, help="share of runs that fail outright")
    parser.add_argument("--retry-rate", type=float, default=0.12, help="share of runs with a retried task")
    parser.add_argument("--retention-days", type=int, default=14, help="artifact retention window")
    parser.add_argument("--all-logs", action="store_true", help="write logs for every task instance")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sotf = load_sotf_dataset(args.sotf_data)
    rng = random.Random(args.seed)

    out: Path = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    runs = build_runs(sotf, args, rng)

    emit_dags(out, runs, sotf)
    emit_tasks(out, sotf)
    emit_dag_runs(out, runs)
    emit_task_instances(out, runs, rng)
    emit_xcom(out, runs)
    log_files = emit_logs(out, runs, args)
    emit_event_logs(out, runs)
    emit_instance_metadata(out, sotf, runs, args)
    emit_runs_index(out, runs)
    emit_cross_reference(out, sotf, runs)
    write_json(out / "api_index.json", API_INDEX)

    task_instances = sum(len(run.tasks) for run in runs)
    counts = {
        "dags": len(DAG_SPECS),
        "dag_runs": len(runs),
        "dag_runs_by_state": {
            state: sum(1 for run in runs if run.state == state) for state in DAG_RUN_STATES
        },
        "dag_runs_by_dag": {
            spec.dag_id: sum(1 for run in runs if run.spec.dag_id == spec.dag_id)
            for spec in DAG_SPECS
        },
        "task_instances": task_instances,
        "task_instances_by_state": {
            **{
                state: sum(1 for run in runs for t in run.tasks if t.state == state)
                for state in TASK_STATES
            },
            "none": sum(1 for run in runs for t in run.tasks if t.state is None),
        },
        "retried_tasks": sum(1 for run in runs for t in run.tasks if t.try_number > 1),
        "xcom_entries": sum(
            1
            for run in runs
            for t in run.tasks
            if t.spec.task_id == XCOM_KEYS[run.spec.dag_id][0] and t.state == "success"
        ),
        "log_files": log_files,
        "rides_referenced": len(
            {
                trigger_id
                for run in runs
                for trigger_id in (run.payload.get("trigger_ids") or [])
            }
        ),
    }
    write_json(
        out / "summary.json",
        {
            "generated_at": iso(datetime.now(timezone.utc)),
            "sotf_dataset": str(args.sotf_data),
            "parameters": {
                "seed": args.seed,
                "now": iso(sotf.now),
                "days": sotf.days,
                "failure_rate": args.failure_rate,
                "retry_rate": args.retry_rate,
                "retention_days": args.retention_days,
                "all_logs": args.all_logs,
            },
            "counts": counts,
        },
    )

    print(f"Wrote Airflow mock data to {out}")
    print(
        "  {dags} DAGs, {dag_runs} runs, {task_instances} task instances, "
        "{xcom_entries} XCom payloads, {log_files} log files, "
        "{rides_referenced} SOTF rides referenced".format(**counts)
    )

    if args.skip_self_check:
        return 0

    problems = run_self_check(out, sotf, runs)
    if problems:
        print(f"  self-check FAILED with {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    - {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return 1

    print("  self-check passed: every artifact-bearing ride is claimed by exactly one ingest run")
    print(f"  join table: {out / 'cross_reference_airflow.json'}")
    print(f"  run index:  {out / 'runs_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
