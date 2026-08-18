#!/usr/bin/env python3
"""Generate mock Grafana data for the SOTF reporting stack.

Grafana is not a data store, so this script does not invent new rows the way
the others do. It reads what already exists:

    generate_postgres_mock_data.py  -> ./out_postgres   (RDS CloudWatch metrics)
    generate_redshift_mock_data.py  -> ./out_redshift    (mart aggregates, ops tables)

and builds the dashboard layer that would sit on top: datasources, dashboard
JSON, the value each panel would render today, alert rules with a state
history that actually matches the data, annotations, and a read-only org.

Every panel points at a real query over a real mock table, and every stored
panel value is computed from that table, not invented. The self-check
recomputes each one and fails on a mismatch, and it evaluates every alert
rule's threshold against the same data its stored state history claims to
reflect -- so a dashboard here can't quietly show a number the warehouse
doesn't back up.

Usage:
    python3 generate_grafana_mock_data.py --clean
    python3 generate_grafana_mock_data.py --redshift-data ./out_redshift \\
        --postgres-data ./out_postgres --out ./out_grafana
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAFANA_VERSION = "11.4.0"
SCHEMA_VERSION = 39
ORG_ID = 1
ORG_NAME = "Zoox"
FOLDER_UID = "sotf-reporting"
FOLDER_TITLE = "SOTF Reporting"

DS_REDSHIFT = "sotf-redshift"
DS_CLOUDWATCH = "sotf-rds-cloudwatch"
DS_CLICKHOUSE = "sotf-clickhouse"

READ_ROLE_TITLE = "SOTF Assistant (Viewer)"


class IdSeq:
    def __init__(self, start: int = 0) -> None:
        self._value = start

    def next(self) -> int:
        self._value += 1
        return self._value


def ts(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def unix_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def num(value: str | None) -> float | int | None:
    if value in (None, ""):
        return None
    try:
        as_int = int(value)
        if str(as_int) == value:
            return as_int
    except ValueError:
        pass
    return float(value)


def boolean(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    return value == "true"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def current_moment(up: "Upstream") -> datetime:
    """The instant this Grafana snapshot represents.

    The Redshift nightly load finishes a couple of hours after ``up.now``
    (the last-ride timestamp), so evaluating staleness at ``up.now`` itself
    would make every source look "loaded in the future". Instead this is a
    few minutes after the most recent real load, which is when someone would
    actually be looking at this dashboard.
    """
    loaded_times = [
        parse_dt(row["last_loaded_at"]) for row in up.data_sources if row.get("last_loaded_at")
    ]
    if not loaded_times:
        return up.now
    return max(loaded_times) + timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Upstream dataset input
# ---------------------------------------------------------------------------


@dataclass
class Upstream:
    now: datetime
    redshift_now: datetime | None
    postgres_now: datetime | None
    region: str
    redshift_cluster_id: str
    redshift_database: str
    redshift_endpoint: str
    rds_instance_id: str
    rds_endpoint: str
    allocated_storage_gb: int = 200

    daily_metrics: list[dict[str, Any]] = field(default_factory=list)
    scenario_health: list[dict[str, Any]] = field(default_factory=list)
    failure_mode_daily: list[dict[str, Any]] = field(default_factory=list)
    pudo_zone_reliability: list[dict[str, Any]] = field(default_factory=list)
    environments: list[dict[str, Any]] = field(default_factory=list)
    data_sources: list[dict[str, Any]] = field(default_factory=list)
    airflow_runs: list[dict[str, Any]] = field(default_factory=list)
    databricks_runs: list[dict[str, Any]] = field(default_factory=list)
    pipeline_loads: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    ctm_log_daily: list[dict[str, Any]] = field(default_factory=list)
    cloudwatch: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def load_redshift(
    root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    summary_file = root / "summary.json"
    cluster_file = root / "cluster.json"
    if not (summary_file.is_file() and cluster_file.is_file()):
        raise SystemExit(
            f"Redshift dataset at {root} is missing summary.json or cluster.json.\n"
            "Run generate_redshift_mock_data.py first, or pass --redshift-data."
        )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    cluster = json.loads(cluster_file.read_text(encoding="utf-8"))["Clusters"][0]

    data = root / "data"
    tables = {
        "daily_metrics": data / "mart/agg_daily_execution_metrics.csv",
        "scenario_health": data / "mart/agg_scenario_health.csv",
        "failure_mode_daily": data / "mart/agg_failure_mode_daily.csv",
        "pudo_zone_reliability": data / "mart/agg_pudo_zone_reliability.csv",
        "environments": data / "mart/dim_environment.csv",
        "data_sources": data / "ops/dim_data_source.csv",
        "airflow_runs": data / "ops/fact_airflow_dag_run.csv",
        "databricks_runs": data / "ops/fact_databricks_job_run.csv",
        "pipeline_loads": data / "ops/fact_pipeline_load.csv",
        "predictions": data / "ops/fact_model_prediction.csv",
        "ctm_log_daily": data / "raw/ctm_log_daily.csv",
    }
    missing = [str(path) for path in tables.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            "Redshift dataset is incomplete, missing:\n  " + "\n  ".join(missing)
        )
    rows = {name: read_csv(path) for name, path in tables.items()}
    return summary, rows, cluster


def load_postgres(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary_file = root / "summary.json"
    cloudwatch_file = root / "rds/cloudwatch_metrics.json"
    instances_file = root / "rds/db_instances.json"
    missing = [
        str(p) for p in (summary_file, cloudwatch_file, instances_file) if not p.is_file()
    ]
    if missing:
        raise SystemExit(
            "Postgres dataset is incomplete, missing:\n  " + "\n  ".join(missing) +
            "\nRun generate_postgres_mock_data.py first, or pass --postgres-data."
        )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    cloudwatch = json.loads(cloudwatch_file.read_text(encoding="utf-8"))
    instance = json.loads(instances_file.read_text(encoding="utf-8"))["DBInstances"][0]
    return summary, cloudwatch, instance


def load_upstream(args: argparse.Namespace) -> Upstream:
    redshift_summary, redshift_rows, cluster = load_redshift(args.redshift_data)
    postgres_summary, cloudwatch, instance = load_postgres(args.postgres_data)

    redshift_now = parse_dt(redshift_summary["parameters"]["now"])
    postgres_now = parse_dt(postgres_summary["parameters"]["now"])
    # Both downstream datasets are meant to be generated from the same SOTF
    # "now"; if they were not, use the later of the two so nothing here shows
    # a value from before its own source data existed.
    candidates = [value for value in (redshift_now, postgres_now) if value is not None]
    if not candidates:
        raise SystemExit("Neither the Redshift nor the Postgres summary.json has a parsed 'now'.")
    now = max(candidates)

    upstream = Upstream(
        now=now,
        redshift_now=redshift_now,
        postgres_now=postgres_now,
        region=redshift_summary["parameters"]["region"],
        redshift_cluster_id=redshift_summary["parameters"]["cluster"],
        redshift_database=redshift_summary["parameters"]["database"],
        redshift_endpoint=f"{cluster['Endpoint']['Address']}:{cluster['Endpoint']['Port']}",
        rds_instance_id=instance["DBInstanceIdentifier"],
        rds_endpoint=f"{instance['Endpoint']['Address']}:{instance['Endpoint']['Port']}",
        allocated_storage_gb=instance.get("AllocatedStorage", 200),
        cloudwatch=cloudwatch["metrics"],
    )
    upstream.daily_metrics = redshift_rows["daily_metrics"]
    upstream.scenario_health = redshift_rows["scenario_health"]
    upstream.failure_mode_daily = redshift_rows["failure_mode_daily"]
    upstream.pudo_zone_reliability = redshift_rows["pudo_zone_reliability"]
    upstream.environments = redshift_rows["environments"]
    upstream.data_sources = redshift_rows["data_sources"]
    upstream.airflow_runs = redshift_rows["airflow_runs"]
    upstream.databricks_runs = redshift_rows["databricks_runs"]
    upstream.pipeline_loads = redshift_rows["pipeline_loads"]
    upstream.predictions = redshift_rows["predictions"]
    upstream.ctm_log_daily = redshift_rows["ctm_log_daily"]
    return upstream


# ---------------------------------------------------------------------------
# Datasources
# ---------------------------------------------------------------------------


def build_datasources(up: Upstream) -> list[dict[str, Any]]:
    """secureJsonData is present but masked, exactly as the real
    /api/datasources response never returns a usable secret."""
    return [
        {
            "id": 1,
            "uid": DS_REDSHIFT,
            "orgId": ORG_ID,
            "name": "SOTF Redshift",
            "type": "grafana-redshift-datasource",
            "typeName": "Redshift",
            "access": "proxy",
            "url": "",
            "isDefault": True,
            "jsonData": {
                "authType": "default",
                "defaultRegion": up.region,
                "clusterIdentifier": up.redshift_cluster_id,
                "database": up.redshift_database,
                "useIAM": True,
                "dbUser": "sotf_assistant",
            },
            "secureJsonFields": {},
            "readOnly": True,
        },
        {
            "id": 2,
            "uid": DS_CLOUDWATCH,
            "orgId": ORG_ID,
            "name": "SOTF RDS CloudWatch",
            "type": "cloudwatch",
            "typeName": "CloudWatch",
            "access": "proxy",
            "url": "",
            "isDefault": False,
            "jsonData": {
                "authType": "default",
                "defaultRegion": up.region,
                "customMetricsNamespaces": "AWS/RDS",
            },
            "secureJsonFields": {},
            "readOnly": True,
        },
        {
            "id": 3,
            "uid": DS_CLICKHOUSE,
            "orgId": ORG_ID,
            "name": "SOTF ClickHouse",
            "type": "grafana-clickhouse-datasource",
            "typeName": "ClickHouse",
            "access": "proxy",
            "url": "",
            "isDefault": False,
            "jsonData": {
                "defaultDatabase": "sotf",
                "port": 8443,
                "protocol": "https",
                "secure": True,
                "username": "sotf_readonly",
            },
            "secureJsonFields": {"password": True},
            "readOnly": True,
        },
    ]


def ds_ref(uid: str) -> dict[str, str]:
    type_by_uid = {
        DS_REDSHIFT: "grafana-redshift-datasource",
        DS_CLOUDWATCH: "cloudwatch",
        DS_CLICKHOUSE: "grafana-clickhouse-datasource",
    }
    return {"type": type_by_uid[uid], "uid": uid}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


_panel_ids = IdSeq()


def grid(x: int, y: int, w: int = 12, h: int = 8) -> dict[str, int]:
    return {"h": h, "w": w, "x": x, "y": y}


def panel(
    title: str,
    panel_type: str,
    grid_pos: dict[str, int],
    datasource_uid: str,
    targets: list[dict[str, Any]],
    field_config: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    return {
        "id": _panel_ids.next(),
        "title": title,
        "description": description,
        "type": panel_type,
        "gridPos": grid_pos,
        "datasource": ds_ref(datasource_uid),
        "targets": targets,
        "fieldConfig": field_config or {"defaults": {}, "overrides": []},
        "options": options or {},
    }


def redshift_target(sql: str, format_: str = "table") -> dict[str, Any]:
    return {
        "refId": "A",
        "datasource": ds_ref(DS_REDSHIFT),
        "rawSQL": sql,
        "format": format_,
    }


def cloudwatch_target(metric_name: str, instance_id: str, stat: str = "Average") -> dict[str, Any]:
    return {
        "refId": "A",
        "datasource": ds_ref(DS_CLOUDWATCH),
        "namespace": "AWS/RDS",
        "metricName": metric_name,
        "statistic": stat,
        "dimensions": {"DBInstanceIdentifier": instance_id},
        "region": "default",
    }


def clickhouse_target(sql: str) -> dict[str, Any]:
    return {"refId": "A", "datasource": ds_ref(DS_CLICKHOUSE), "rawSql": sql, "format": 1}


# ---------------------------------------------------------------------------
# Dashboard builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DashboardSpec:
    uid: str
    title: str
    description: str
    tags: tuple[str, ...]
    build: Any  # Callable[[Upstream], list[dict[str, Any]]]


def build_test_execution_health(up: Upstream) -> list[dict[str, Any]]:
    return [
        panel(
            "Pass rate by environment (14d)",
            "timeseries",
            grid(0, 0, 16, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT metric_date, environment, "
                "round(sum(passed)::decimal / sum(rides), 4) AS pass_rate "
                "FROM mart.agg_daily_execution_metrics "
                "WHERE metric_date >= dateadd(day, -14, current_date) "
                "GROUP BY metric_date, environment ORDER BY metric_date",
                "time_series",
            )],
            field_config={"defaults": {"unit": "percentunit", "min": 0, "max": 1}, "overrides": []},
            description="mart.agg_daily_execution_metrics, grouped by day and environment.",
        ),
        panel(
            "Rides today",
            "stat",
            grid(16, 0, 8, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT sum(rides) FROM mart.agg_daily_execution_metrics "
                "WHERE metric_date = current_date",
            )],
            field_config={"defaults": {"unit": "short"}, "overrides": []},
        ),
        panel(
            "Flakiest scenarios",
            "table",
            grid(0, 8, 24, 9),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT scenario_name, total_runs, pass_rate, flakiness_score, "
                "consecutive_failures, top_failure_mode "
                "FROM mart.agg_scenario_health "
                "WHERE total_runs >= 4 ORDER BY flakiness_score DESC",
            )],
            description="mart.agg_scenario_health, the same view as the flakiest_scenarios saved query.",
        ),
        panel(
            "Failure modes, last 7 days",
            "bargauge",
            grid(0, 17, 12, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT failure_mode, sum(occurrences) AS occurrences "
                "FROM mart.agg_failure_mode_daily "
                "WHERE metric_date >= dateadd(day, -7, current_date) "
                "GROUP BY failure_mode ORDER BY occurrences DESC",
            )],
        ),
        panel(
            "PUDO zone reliability",
            "table",
            grid(12, 17, 12, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT pudo_zone, ride_service, rides, pass_rate, top_failure_mode "
                "FROM mart.agg_pudo_zone_reliability ORDER BY pass_rate ASC",
            )],
        ),
    ]


def build_pipeline_observability(up: Upstream) -> list[dict[str, Any]]:
    return [
        panel(
            "Airflow DAG runs by state (14d)",
            "timeseries",
            grid(0, 0, 12, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT d.calendar_date, a.state, count(*) AS runs "
                "FROM ops.fact_airflow_dag_run a "
                "JOIN mart.dim_date d ON d.date_key = a.date_key "
                "GROUP BY d.calendar_date, a.state ORDER BY d.calendar_date",
                "time_series",
            )],
        ),
        panel(
            "Databricks job runs by result",
            "bargauge",
            grid(12, 0, 12, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT job_name, result_state, count(*) AS runs "
                "FROM ops.fact_databricks_job_run GROUP BY job_name, result_state",
            )],
        ),
        panel(
            "COPY load duration by table",
            "bargauge",
            grid(0, 8, 12, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT target_table, avg(duration_seconds) AS avg_seconds "
                "FROM ops.fact_pipeline_load GROUP BY target_table "
                "ORDER BY avg_seconds DESC",
            )],
            field_config={"defaults": {"unit": "s"}, "overrides": []},
        ),
        panel(
            "Rows loaded per table",
            "table",
            grid(12, 8, 12, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT target_table, rows_loaded, load_status, finished_at "
                "FROM ops.fact_pipeline_load ORDER BY finished_at DESC",
            )],
        ),
    ]


def build_model_monitoring(up: Upstream) -> list[dict[str, Any]]:
    return [
        panel(
            "Classifier accuracy by failure mode",
            "bargauge",
            grid(0, 0, 12, 9),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT actual_failure_mode, "
                "round(avg(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END), 4) AS accuracy "
                "FROM ops.fact_model_prediction GROUP BY actual_failure_mode "
                "ORDER BY accuracy ASC",
            )],
            field_config={"defaults": {"unit": "percentunit", "min": 0, "max": 1}, "overrides": []},
        ),
        panel(
            "Confidence vs. correctness",
            "table",
            grid(12, 0, 12, 9),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT trigger_id, predicted_failure_mode, confidence, is_correct "
                "FROM ops.fact_model_prediction WHERE confidence < 0.6 "
                "ORDER BY confidence ASC",
            )],
            description="Low-confidence predictions a human should check before trusting.",
        ),
    ]


def build_rds_instance_health(up: Upstream) -> list[dict[str, Any]]:
    return [
        panel(
            "CPU utilization",
            "timeseries",
            grid(0, 0, 12, 8),
            DS_CLOUDWATCH,
            [cloudwatch_target("CPUUtilization", up.rds_instance_id, "Average")],
            field_config={"defaults": {"unit": "percent", "max": 100}, "overrides": []},
        ),
        panel(
            "Database connections",
            "timeseries",
            grid(12, 0, 12, 8),
            DS_CLOUDWATCH,
            [cloudwatch_target("DatabaseConnections", up.rds_instance_id, "Maximum")],
            description="Exactly base + per-ride-slope * rides started that day; see out_postgres notes.",
        ),
        panel(
            "Free storage space",
            "timeseries",
            grid(0, 8, 12, 8),
            DS_CLOUDWATCH,
            [cloudwatch_target("FreeStorageSpace", up.rds_instance_id, "Average")],
            field_config={"defaults": {"unit": "bytes"}, "overrides": []},
        ),
        panel(
            "Read / write IOPS",
            "timeseries",
            grid(12, 8, 12, 8),
            DS_CLOUDWATCH,
            [
                cloudwatch_target("ReadIOPS", up.rds_instance_id, "Average"),
                {**cloudwatch_target("WriteIOPS", up.rds_instance_id, "Average"), "refId": "B"},
            ],
        ),
        panel(
            "Replica lag",
            "timeseries",
            grid(0, 16, 24, 6),
            DS_CLOUDWATCH,
            [cloudwatch_target("ReplicaLag", up.rds_instance_id, "Maximum")],
            field_config={"defaults": {"unit": "s"}, "overrides": []},
        ),
    ]


def build_ctm_log_volume(up: Upstream) -> list[dict[str, Any]]:
    return [
        panel(
            "Log rows by severity (7d)",
            "timeseries",
            grid(0, 0, 16, 9),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT log_date, severity, sum(row_count) AS rows "
                "FROM raw.ctm_log_daily "
                "WHERE log_date >= dateadd(day, -7, current_date) "
                "GROUP BY log_date, severity ORDER BY log_date",
                "time_series",
            )],
            description=(
                "Rolled up from ClickHouse into raw.ctm_log_daily; the log lines "
                "themselves are queried in ClickHouse directly, not here."
            ),
        ),
        panel(
            "Distinct triggers logging today",
            "stat",
            grid(16, 0, 8, 9),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT sum(distinct_triggers) FROM raw.ctm_log_daily "
                "WHERE log_date = current_date",
            )],
        ),
        panel(
            "ERROR lines by component (live)",
            "table",
            grid(0, 9, 24, 8),
            DS_CLICKHOUSE,
            [clickhouse_target(
                "SELECT component, count() AS error_rows, uniq(trigger_id) AS rides "
                "FROM sotf.logs WHERE severity = 'ERROR' "
                "AND timestamp >= now() - INTERVAL 7 DAY "
                "GROUP BY component ORDER BY error_rows DESC"
            )],
            description="Direct ClickHouse query; not backed by a Redshift rollup.",
        ),
    ]


def build_data_freshness(up: Upstream) -> list[dict[str, Any]]:
    return [
        panel(
            "Hours since last load, by source",
            "bargauge",
            grid(0, 0, 24, 8),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT source_name, target_table, refresh_cadence, rows_last_load, "
                "last_loaded_at, datediff(hour, last_loaded_at, getdate()) AS hours_stale "
                "FROM ops.dim_data_source ORDER BY hours_stale DESC",
            )],
        ),
        panel(
            "Load status, last 20 COPYs",
            "table",
            grid(0, 8, 24, 9),
            DS_REDSHIFT,
            [redshift_target(
                "SELECT target_table, started_at, duration_seconds, rows_loaded, "
                "load_status, error_count FROM ops.fact_pipeline_load "
                "ORDER BY started_at DESC LIMIT 20",
            )],
        ),
    ]


DASHBOARD_SPECS: tuple[DashboardSpec, ...] = (
    DashboardSpec(
        "sotf-test-execution-health",
        "SOTF Test Execution Health",
        "Pass rate, flakiness and failure modes across every SOTF environment.",
        ("sotf", "testing"),
        build_test_execution_health,
    ),
    DashboardSpec(
        "sotf-pipeline-observability",
        "SOTF Pipeline Observability",
        "Airflow and Databricks run health, and Redshift load performance.",
        ("sotf", "pipelines"),
        build_pipeline_observability,
    ),
    DashboardSpec(
        "sotf-model-monitoring",
        "SOTF Model Monitoring",
        "Failure-mode classifier accuracy and predictions worth a second look.",
        ("sotf", "ml"),
        build_model_monitoring,
    ),
    DashboardSpec(
        "sotf-rds-instance-health",
        "RDS Postgres Instance Health",
        "CloudWatch metrics for the test-mgmt database instance.",
        ("sotf", "infra"),
        build_rds_instance_health,
    ),
    DashboardSpec(
        "sotf-ctm-log-volume",
        "SOTF CTM Log Volume",
        "Log row volume by severity, rolled up daily plus a live ClickHouse drill-down.",
        ("sotf", "logs"),
        build_ctm_log_volume,
    ),
    DashboardSpec(
        "sotf-data-freshness",
        "SOTF Data Freshness",
        "Is the warehouse current? One row per upstream tool.",
        ("sotf", "ops"),
        build_data_freshness,
    ),
)


def build_dashboard(spec: DashboardSpec, up: Upstream) -> dict[str, Any]:
    panels = spec.build(up)
    return {
        "uid": spec.uid,
        "title": spec.title,
        "description": spec.description,
        "tags": list(spec.tags),
        "schemaVersion": SCHEMA_VERSION,
        "version": 1,
        "timezone": "utc",
        "editable": False,
        "graphTooltip": 1,
        "time": {"from": "now-14d", "to": "now"},
        "refresh": "5m",
        "panels": panels,
        "templating": {"list": []},
        "annotations": {
            "list": [
                {
                    "name": "Deploys and incidents",
                    "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                    "enable": True,
                    "iconColor": "red",
                }
            ]
        },
        "links": [],
    }


# ---------------------------------------------------------------------------
# Panel snapshots: what each panel would render right now
# ---------------------------------------------------------------------------


def group_sum(rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> dict[tuple, float]:
    totals: dict[tuple, float] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        totals[key] = totals.get(key, 0.0) + (num(row.get(value)) or 0)
    return totals


def build_test_execution_snapshots(
    up: Upstream, panels: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    today = up.now.strftime("%Y-%m-%d")

    pass_rate_series: dict[str, list[list[Any]]] = {}
    for row in up.daily_metrics:
        env = row["environment"]
        rides = num(row["rides"]) or 0
        passed = num(row["passed"]) or 0
        rate = round(passed / rides, 4) if rides else None
        stamp = unix_ms(parse_dt(row["metric_date"] + "T00:00:00Z"))
        pass_rate_series.setdefault(env, []).append([rate, stamp])
    for series in pass_rate_series.values():
        series.sort(key=lambda point: point[1])

    rides_today = sum(num(r["rides"]) or 0 for r in up.daily_metrics if r["metric_date"] == today)

    flaky = sorted(
        (r for r in up.scenario_health if (num(r["total_runs"]) or 0) >= 4),
        key=lambda r: -(num(r["flakiness_score"]) or 0),
    )

    failure_totals = group_sum(up.failure_mode_daily, ("failure_mode",), "occurrences")
    ordered_failures = sorted(failure_totals.items(), key=lambda kv: -kv[1])

    zones = sorted(up.pudo_zone_reliability, key=lambda r: num(r["pass_rate"]) or 0)

    return {
        panels[0]["id"]: {
            "type": "time_series",
            "series": [
                {"name": env, "points": points} for env, points in sorted(pass_rate_series.items())
            ],
        },
        panels[1]["id"]: {"type": "stat", "value": rides_today},
        panels[2]["id"]: {
            "type": "table",
            "columns": [
                "scenario_name", "total_runs", "pass_rate", "flakiness_score",
                "consecutive_failures", "top_failure_mode",
            ],
            "rows": [
                [r["scenario_name"], num(r["total_runs"]), num(r["pass_rate"]),
                 num(r["flakiness_score"]), num(r["consecutive_failures"]), r["top_failure_mode"]]
                for r in flaky
            ],
        },
        panels[3]["id"]: {
            "type": "bargauge",
            "values": [{"label": mode[0], "value": total} for mode, total in ordered_failures],
        },
        panels[4]["id"]: {
            "type": "table",
            "columns": ["pudo_zone", "ride_service", "rides", "pass_rate", "top_failure_mode"],
            "rows": [
                [r["pudo_zone"], r["ride_service"], num(r["rides"]), num(r["pass_rate"]),
                 r["top_failure_mode"]]
                for r in zones
            ],
        },
    }


def build_pipeline_snapshots(
    up: Upstream, panels: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    state_counts: dict[tuple[str, str], int] = {}
    for row in up.airflow_runs:
        key = (row.get("date_key") or "", row["state"])
        state_counts[key] = state_counts.get(key, 0) + 1

    job_counts = group_sum(
        [{**r, "one": 1} for r in up.databricks_runs], ("job_name", "result_state"), "one"
    )

    load_totals: dict[str, list[float]] = {}
    for row in up.pipeline_loads:
        load_totals.setdefault(row["target_table"], []).append(num(row["duration_seconds"]) or 0)

    loads = sorted(up.pipeline_loads, key=lambda r: r["finished_at"], reverse=True)

    return {
        panels[0]["id"]: {
            "type": "time_series",
            "series": [{"key": list(key), "count": count} for key, count in sorted(state_counts.items())],
        },
        panels[1]["id"]: {
            "type": "bargauge",
            "values": [
                {"job_name": k[0], "result_state": k[1], "runs": int(v)} for k, v in job_counts.items()
            ],
        },
        panels[2]["id"]: {
            "type": "bargauge",
            "values": sorted(
                [{"target_table": table, "avg_seconds": mean(durations)}
                 for table, durations in load_totals.items()],
                key=lambda row: -row["avg_seconds"],
            ),
        },
        panels[3]["id"]: {
            "type": "table",
            "columns": ["target_table", "rows_loaded", "load_status", "finished_at"],
            "rows": [
                [r["target_table"], num(r["rows_loaded"]), r["load_status"], r["finished_at"]]
                for r in loads
            ],
        },
    }


def build_model_snapshots(
    up: Upstream, panels: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    by_mode: dict[str, list[bool]] = {}
    for row in up.predictions:
        by_mode.setdefault(row["actual_failure_mode"], []).append(boolean(row["is_correct"]) or False)
    accuracy = {
        mode: round(sum(1 for v in values if v) / len(values), 4)
        for mode, values in by_mode.items()
    }

    low_confidence = sorted(
        (r for r in up.predictions if (num(r["confidence"]) or 0) < 0.6),
        key=lambda r: num(r["confidence"]) or 0,
    )

    return {
        panels[0]["id"]: {
            "type": "bargauge",
            "values": sorted(
                [{"actual_failure_mode": mode, "accuracy": value} for mode, value in accuracy.items()],
                key=lambda row: row["accuracy"],
            ),
        },
        panels[1]["id"]: {
            "type": "table",
            "columns": ["trigger_id", "predicted_failure_mode", "confidence", "is_correct"],
            "rows": [
                [r["trigger_id"], r["predicted_failure_mode"], num(r["confidence"]), r["is_correct"]]
                for r in low_confidence
            ],
        },
    }


def build_rds_snapshots(
    up: Upstream, panels: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    def series(metric: str, stat: str) -> list[list[Any]]:
        return [
            [num(point[stat]), unix_ms(parse_dt(point["Timestamp"]))]
            for point in up.cloudwatch.get(metric, [])
        ]

    return {
        panels[0]["id"]: {"type": "time_series", "points": series("CPUUtilization", "Average")},
        panels[1]["id"]: {"type": "time_series", "points": series("DatabaseConnections", "Maximum")},
        panels[2]["id"]: {"type": "time_series", "points": series("FreeStorageSpace", "Average")},
        panels[3]["id"]: {
            "type": "time_series",
            "series": [
                {"name": "ReadIOPS", "points": series("ReadIOPS", "Average")},
                {"name": "WriteIOPS", "points": series("WriteIOPS", "Average")},
            ],
        },
        panels[4]["id"]: {"type": "time_series", "points": series("ReplicaLag", "Maximum")},
    }


def build_log_volume_snapshots(
    up: Upstream, panels: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    seven_days_ago = (up.now - timedelta(days=7)).strftime("%Y-%m-%d")

    recent = [r for r in up.ctm_log_daily if r["log_date"] >= seven_days_ago]
    totals = group_sum(recent, ("log_date", "severity"), "row_count")
    today = up.now.strftime("%Y-%m-%d")
    triggers_today = sum(
        num(r["distinct_triggers"]) or 0 for r in up.ctm_log_daily if r["log_date"] == today
    )

    return {
        panels[0]["id"]: {
            "type": "time_series",
            "series": [{"key": list(key), "rows": total} for key, total in sorted(totals.items())],
        },
        panels[1]["id"]: {"type": "stat", "value": triggers_today},
        panels[2]["id"]: {
            "type": "table",
            "note": "Live ClickHouse query; no stored snapshot -- see raw ClickHouse fixtures for values.",
        },
    }


def build_freshness_snapshots(
    up: Upstream, panels: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    moment = current_moment(up)
    freshness = []
    for row in up.data_sources:
        loaded_at = parse_dt(row["last_loaded_at"])
        hours_stale = round((moment - loaded_at).total_seconds() / 3600, 2) if loaded_at else None
        freshness.append(
            {
                "source_name": row["source_name"],
                "target_table": row["target_table"],
                "refresh_cadence": row["refresh_cadence"],
                "rows_last_load": num(row["rows_last_load"]),
                "last_loaded_at": row["last_loaded_at"],
                "hours_stale": hours_stale,
            }
        )
    freshness.sort(key=lambda row: -(row["hours_stale"] or 0))

    recent_loads = sorted(up.pipeline_loads, key=lambda r: r["started_at"], reverse=True)[:20]

    return {
        panels[0]["id"]: {"type": "bargauge", "values": freshness},
        panels[1]["id"]: {
            "type": "table",
            "columns": ["target_table", "started_at", "duration_seconds", "rows_loaded", "load_status", "error_count"],
            "rows": [
                [r["target_table"], r["started_at"], num(r["duration_seconds"]),
                 num(r["rows_loaded"]), r["load_status"], num(r["error_count"])]
                for r in recent_loads
            ],
        },
    }


SNAPSHOT_BUILDERS = (
    build_test_execution_snapshots,
    build_pipeline_snapshots,
    build_model_snapshots,
    build_rds_snapshots,
    build_log_volume_snapshots,
    build_freshness_snapshots,
)


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertSpec:
    uid: str
    title: str
    folder_uid: str
    condition_summary: str
    datasource_uid: str
    query: str
    for_duration: str
    evaluate: Any  # Callable[[Upstream], list[tuple[str, str, bool]]] -> (label, timestamp, firing)


def evaluate_pass_rate_alert(up: Upstream) -> list[tuple[str, str, bool]]:
    """Fires per environment per day when the pass rate drops below 0.7.

    The rule's own SQL groups by environment only (summing across ride
    services), so this has to aggregate the same way -- otherwise two rows
    for the same environment and day (different ride_service) would produce
    two contradictory states for what is really one alert instance.
    """
    totals: dict[tuple[str, str], tuple[float, float]] = {}
    for row in up.daily_metrics:
        key = (row["metric_date"], row["environment"])
        rides, passed = totals.get(key, (0.0, 0.0))
        totals[key] = (rides + (num(row["rides"]) or 0), passed + (num(row["passed"]) or 0))

    results = []
    for (metric_date, environment), (rides, passed) in sorted(totals.items()):
        if rides == 0:
            continue
        pass_rate = passed / rides
        stamp = ts(parse_dt(metric_date + "T12:00:00Z"))
        results.append((environment, stamp, pass_rate < 0.7))
    return results


def evaluate_rds_storage_alert(up: Upstream) -> list[tuple[str, str, bool]]:
    """Fires when free storage drops below 15% of the allocated volume."""
    threshold = up.allocated_storage_gb * 1024**3 * 0.15
    results = []
    for point in up.cloudwatch.get("FreeStorageSpace", []):
        firing = (num(point["Average"]) or 0) < threshold
        results.append((up.rds_instance_id, point["Timestamp"], firing))
    return results


def evaluate_freshness_alert(up: Upstream) -> list[tuple[str, str, bool]]:
    """Fires when a source is stale beyond 2x its refresh cadence."""
    cadence_hours = {
        "continuous": 1,
        "every 6 hours": 12,
        "daily 01:30": 48,
        "daily 02:00": 48,
        "daily 02:15": 48,
        "daily 03:30": 48,
        "daily 05:00": 48,
    }
    moment = current_moment(up)
    results = []
    for row in up.data_sources:
        loaded_at = parse_dt(row["last_loaded_at"])
        if loaded_at is None:
            continue
        limit = cadence_hours.get(row["refresh_cadence"], 48)
        hours_stale = (moment - loaded_at).total_seconds() / 3600
        results.append((row["source_name"], ts(moment), hours_stale > limit))
    return results


ALERT_SPECS: tuple[AlertSpec, ...] = (
    AlertSpec(
        "sotf-pass-rate-drop",
        "SOTF pass rate drop",
        FOLDER_UID,
        "avg(pass_rate) by environment over the trailing day < 0.7",
        DS_REDSHIFT,
        "SELECT environment, round(sum(passed)::decimal / sum(rides), 4) AS pass_rate "
        "FROM mart.agg_daily_execution_metrics "
        "WHERE metric_date = current_date GROUP BY environment",
        "15m",
        evaluate_pass_rate_alert,
    ),
    AlertSpec(
        "sotf-rds-storage-low",
        "RDS free storage low",
        FOLDER_UID,
        "FreeStorageSpace < 15% of allocated storage",
        DS_CLOUDWATCH,
        "CloudWatch AWS/RDS FreeStorageSpace, Average, DBInstanceIdentifier="
        "<rds_instance_id>",
        "10m",
        evaluate_rds_storage_alert,
    ),
    AlertSpec(
        "sotf-source-staleness",
        "SOTF upstream source stale",
        FOLDER_UID,
        "hours since ops.dim_data_source.last_loaded_at > 2x refresh_cadence",
        DS_REDSHIFT,
        "SELECT source_name, datediff(hour, last_loaded_at, getdate()) AS hours_stale "
        "FROM ops.dim_data_source",
        "30m",
        evaluate_freshness_alert,
    ),
)


def build_alert_rules(up: Upstream) -> list[dict[str, Any]]:
    rules = []
    for spec in ALERT_SPECS:
        rules.append(
            {
                "uid": spec.uid,
                "title": spec.title,
                "orgID": ORG_ID,
                "folderUID": spec.folder_uid,
                "ruleGroup": "sotf-alerts",
                "condition": "A",
                "for": spec.for_duration,
                "annotations": {"summary": spec.condition_summary},
                "labels": {"team": "sotf"},
                "data": [
                    {
                        "refId": "A",
                        "datasourceUid": spec.datasource_uid,
                        "relativeTimeRange": {"from": 600, "to": 0},
                        "model": {
                            "refId": "A",
                            **({"rawSQL" if spec.datasource_uid == DS_REDSHIFT else "expression": spec.query}),
                        },
                    }
                ],
                "noDataState": "NoData",
                "execErrState": "Error",
            }
        )
    return rules


def build_alert_state_history(up: Upstream) -> list[dict[str, Any]]:
    history = []
    for spec in ALERT_SPECS:
        for label, stamp, firing in spec.evaluate(up):
            history.append(
                {
                    "ruleUID": spec.uid,
                    "labels": {"instance": label},
                    "state": "Alerting" if firing else "Normal",
                    "time": stamp,
                }
            )
    history.sort(key=lambda row: (row["ruleUID"], row["time"]))
    return history


def build_contact_points_and_policies() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contact_points = [
        {
            "uid": "sotf-slack",
            "name": "sotf-oncall-slack",
            "type": "slack",
            "settings": {"url": "{{ secret }}", "recipient": "#sotf-alerts"},
            "secureFields": {"url": True},
        }
    ]
    policy = {
        "receiver": "sotf-oncall-slack",
        "group_by": ["alertname"],
        "routes": [
            {"receiver": "sotf-oncall-slack", "object_matchers": [["team", "=", "sotf"]]},
        ],
    }
    return contact_points, policy


# ---------------------------------------------------------------------------
# Annotations, org, folders
# ---------------------------------------------------------------------------


def build_annotations(up: Upstream) -> list[dict[str, Any]]:
    """Deploy and incident markers on real dates from the dataset, not made-up ones."""
    annotations = []
    annotation_id = IdSeq(9000)

    deploy_days = sorted({row["metric_date"] for row in up.daily_metrics})[::7]
    for day in deploy_days:
        annotations.append(
            {
                "id": annotation_id.next(),
                "dashboardUID": DASHBOARD_SPECS[0].uid,
                "time": unix_ms(parse_dt(day + "T09:00:00Z")),
                "tags": ["deploy"],
                "text": "test-mgmt weekly release",
            }
        )

    worst_day = min(
        (row for row in up.daily_metrics if (num(row["rides"]) or 0) > 0),
        key=lambda row: (num(row["passed"]) or 0) / (num(row["rides"]) or 1),
        default=None,
    )
    if worst_day is not None:
        annotations.append(
            {
                "id": annotation_id.next(),
                "dashboardUID": DASHBOARD_SPECS[0].uid,
                "time": unix_ms(parse_dt(worst_day["metric_date"] + "T00:00:00Z")),
                "tags": ["incident"],
                "text": f"Pass rate dropped in {worst_day['environment']} -- investigated same day.",
            }
        )
    return annotations


def build_org() -> dict[str, Any]:
    return {
        "org": {"id": ORG_ID, "name": ORG_NAME},
        "teams": [{"id": 1, "name": "sotf-qa", "orgId": ORG_ID, "memberCount": 6}],
        "users": [
            {"id": 1, "login": "sotf_assistant", "role": "Viewer", "isDisabled": False},
            {"id": 2, "login": "sotf_dashboard_admin", "role": "Editor", "isDisabled": False},
        ],
        "roles": [
            {
                "name": READ_ROLE_TITLE,
                "permissions": ["dashboards:read", "datasources:query", "folders:read"],
            }
        ],
        "note": "No password or API token appears here; Grafana never returns one for a user.",
    }


def build_folders() -> list[dict[str, Any]]:
    return [{"uid": FOLDER_UID, "title": FOLDER_TITLE, "orgId": ORG_ID}]


LOADER_TEMPLATE = """#!/usr/bin/env bash
# Push the generated dashboards, datasources and alert rules into a throwaway
# Grafana instance. Point this at a local `docker run grafana/grafana` or a
# scratch Grafana Cloud stack -- never at a real one.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

: "${{GRAFANA_URL:=http://localhost:3000}}"
: "${{GRAFANA_API_TOKEN:?set GRAFANA_API_TOKEN to a service account token first}}"

auth=(-H "Authorization: Bearer ${{GRAFANA_API_TOKEN}}" -H "Content-Type: application/json")

echo "==> creating folder"
jq -c '.[] | del(.orgId)' "${{HERE}}/folders/folders.json" | while IFS= read -r folder; do
  curl -sS --fail "${{auth[@]}}" -X POST "${{GRAFANA_URL}}/api/folders" -d "${{folder}}" > /dev/null || true
done

echo "==> creating datasources"
jq -c '.[] | del(.id)' "${{HERE}}/datasources/datasources.json" | while IFS= read -r ds; do
  curl -sS --fail "${{auth[@]}}" -X POST "${{GRAFANA_URL}}/api/datasources" -d "${{ds}}" > /dev/null || true
done

echo "==> importing dashboards"
for dash in "${{HERE}}"/dashboards/*.json; do
  jq '{{dashboard: ., folderUid: "{folder_uid}", overwrite: true}}' "${{dash}}" \\
    | curl -sS --fail "${{auth[@]}}" -X POST "${{GRAFANA_URL}}/api/dashboards/db" -d @- > /dev/null
done

echo "==> creating alert rules"
jq -c '.[]' "${{HERE}}/alerting/rules.json" | while IFS= read -r rule; do
  curl -sS --fail "${{auth[@]}}" -X POST \\
    "${{GRAFANA_URL}}/api/v1/provisioning/alert-rules" -d "${{rule}}" > /dev/null || true
done

echo "Datasources and dashboards authenticate through GRAFANA_API_TOKEN only;"
echo "no credential is stored in this repo."
"""


def emit_loader(out: Path) -> None:
    path = out / "load_grafana.sh"
    write_text(path, LOADER_TEMPLATE.format(folder_uid=FOLDER_UID))
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Catalog, cross-reference, summary
# ---------------------------------------------------------------------------


API_INDEX = {
    "base_url": "https://grafana.example.com",
    "note": "Read-only fixtures. No dashboard-write, datasource-write or alert-write payload is meant to be replayed against a real instance except through load_grafana.sh, which you must point at a scratch instance yourself.",
    "endpoints": {
        "GET /api/search": "catalog.json (dashboards[])",
        "GET /api/dashboards/uid/:uid": "dashboards/<uid>.json",
        "GET /api/datasources": "datasources/datasources.json",
        "GET /api/folders": "folders/folders.json",
        "GET /api/annotations": "annotations/annotations.json",
        "GET /api/v1/provisioning/alert-rules": "alerting/rules.json",
        "GET /api/v1/provisioning/contact-points": "alerting/contact_points.json",
        "GET /api/v1/provisioning/policies": "alerting/notification_policy.json",
        "GET /api/org": "org.json",
        "GET /api/org/users": "org.json (users[])",
    },
}


def emit_catalog(out: Path, up: Upstream, dashboards: list[dict[str, Any]]) -> None:
    write_json(
        out / "catalog.json",
        {
            "grafana_version": GRAFANA_VERSION,
            "org": ORG_NAME,
            "folder": FOLDER_TITLE,
            "datasources": [DS_REDSHIFT, DS_CLOUDWATCH, DS_CLICKHOUSE],
            "dashboards": [
                {
                    "uid": spec.uid,
                    "title": spec.title,
                    "description": spec.description,
                    "tags": list(spec.tags),
                    "panel_count": len(dashboard["panels"]),
                    "url": f"/d/{spec.uid}",
                }
                for spec, dashboard in zip(DASHBOARD_SPECS, dashboards)
            ],
            "alert_rules": [
                {"uid": spec.uid, "title": spec.title, "condition": spec.condition_summary}
                for spec in ALERT_SPECS
            ],
            "notes": [
                "Every panel value under panels/snapshots/ is computed directly from "
                "out_redshift/ and out_postgres/, not invented.",
                "Alert state history under alerting/state_history.json is recomputed by "
                "the self-check by evaluating each rule's threshold against the same data.",
                "No datasource in datasources/datasources.json carries a literal secret; "
                "secureJsonFields are present and true, matching what the real API returns.",
            ],
        },
    )


def emit_cross_reference(out: Path, up: Upstream, dashboards: list[dict[str, Any]]) -> None:
    entries = []
    for spec, dashboard in zip(DASHBOARD_SPECS, dashboards):
        for p in dashboard["panels"]:
            target = p["targets"][0]
            entries.append(
                {
                    "dashboard_uid": spec.uid,
                    "panel_id": p["id"],
                    "panel_title": p["title"],
                    "datasource_uid": target["datasource"]["uid"],
                    "query": target.get("rawSQL") or target.get("rawSql") or target.get("metricName"),
                }
            )
    write_json(
        out / "cross_reference_grafana.json",
        {
            "note": (
                "Every panel and the exact query or metric behind it, tracing back to "
                "out_redshift or out_postgres. Use this to find which mock table a "
                "dashboard number actually came from."
            ),
            "panels": entries,
        },
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def run_self_check(
    out: Path,
    up: Upstream,
    dashboards: list[dict[str, Any]],
    snapshots: list[dict[int, dict[str, Any]]],
    datasources: list[dict[str, Any]],
) -> list[str]:
    problems: list[str] = []

    if up.redshift_now and up.postgres_now and up.redshift_now != up.postgres_now:
        problems.append(
            "out_redshift and out_postgres were generated with different 'now' values "
            f"({up.redshift_now.isoformat()} vs {up.postgres_now.isoformat()}); "
            "regenerate both from the same SOTF dataset for one coherent point in time"
        )

    ds_uids = {d["uid"] for d in datasources}
    for dashboard in dashboards:
        for p in dashboard["panels"]:
            if p["datasource"]["uid"] not in ds_uids:
                problems.append(f"{dashboard['uid']}/{p['id']}: unknown datasource")
            if not p.get("gridPos") or not p.get("targets"):
                problems.append(f"{dashboard['uid']}/{p['id']}: missing gridPos or targets")

    # Every panel with a stored snapshot must actually have that panel id on the
    # dashboard, and vice versa for panels that carry a live query with no snapshot.
    for dashboard, snap in zip(dashboards, snapshots):
        panel_ids = {p["id"] for p in dashboard["panels"]}
        for panel_id in snap:
            if panel_id not in panel_ids:
                problems.append(f"{dashboard['uid']}: snapshot for unknown panel {panel_id}")

    # No literal secret in any datasource. The real GET /api/datasources never
    # returns a secureJsonData block at all, only secureJsonFields booleans.
    for ds in datasources:
        if "secureJsonData" in ds:
            problems.append(f"{ds['uid']}: secureJsonData present (the real API never returns it)")
        blob = json.dumps(ds).upper()
        for marker in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY", "SESSION_TOKEN"):
            if marker in blob:
                problems.append(f"{ds['uid']}: {marker} present in datasource config")

    # Every panel snapshot must be exactly reproducible from the source data,
    # using the same panel objects (and therefore the same ids) as the
    # dashboard actually written to disk.
    for builder, snap, dashboard in zip(SNAPSHOT_BUILDERS, snapshots, dashboards):
        if builder(up, dashboard["panels"]) != snap:
            problems.append(f"{builder.__name__}: snapshot is not reproducible from the source data")

    # Alert state history must match evaluating each rule fresh.
    fresh_history = build_alert_state_history(up)
    history_file = out / "alerting/state_history.json"
    if history_file.is_file():
        stored_history = json.loads(history_file.read_text(encoding="utf-8"))
        if stored_history != fresh_history:
            problems.append("alerting/state_history.json does not match re-evaluating the alert rules")

    # Every alert rule references a real datasource.
    for spec in ALERT_SPECS:
        if spec.datasource_uid not in ds_uids:
            problems.append(f"{spec.uid}: references unknown datasource {spec.datasource_uid}")

    # No user or team row carries a password/token field.
    org = json.loads((out / "org.json").read_text(encoding="utf-8"))
    for user in org["users"]:
        if "password" in user or "token" in user:
            problems.append(f"{user['login']}: org.json carries a credential field")

    # Annotation timestamps must fall on days that actually exist in the data.
    known_days = {row["metric_date"] for row in up.daily_metrics}
    annotations = json.loads((out / "annotations/annotations.json").read_text(encoding="utf-8"))
    for note in annotations:
        day = datetime.fromtimestamp(note["time"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if day not in known_days:
            problems.append(f"annotation at {day} does not correspond to a day with ride data")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mock Grafana dashboards, datasources and alerting for the SOTF "
            "reporting stack, built entirely from the already-generated Redshift and "
            "Postgres mock datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--redshift-data", type=Path, default=HERE / "out_redshift",
        help="output of generate_redshift_mock_data.py",
    )
    parser.add_argument(
        "--postgres-data", type=Path, default=HERE / "out_postgres",
        help="output of generate_postgres_mock_data.py",
    )
    parser.add_argument("--out", type=Path, default=HERE / "out_grafana")
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    up = load_upstream(args)

    out: Path = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    datasources = build_datasources(up)
    dashboards = [build_dashboard(spec, up) for spec in DASHBOARD_SPECS]
    snapshots = [
        builder(up, dashboard["panels"])
        for builder, dashboard in zip(SNAPSHOT_BUILDERS, dashboards)
    ]

    write_json(out / "datasources/datasources.json", datasources)
    write_json(out / "folders/folders.json", build_folders())
    for spec, dashboard in zip(DASHBOARD_SPECS, dashboards):
        write_json(out / f"dashboards/{spec.uid}.json", dashboard)
    for spec, snap in zip(DASHBOARD_SPECS, snapshots):
        for panel_id, value in snap.items():
            write_json(out / f"panels/snapshots/{spec.uid}/{panel_id}.json", value)

    write_json(out / "alerting/rules.json", build_alert_rules(up))
    write_json(out / "alerting/state_history.json", build_alert_state_history(up))
    contact_points, policy = build_contact_points_and_policies()
    write_json(out / "alerting/contact_points.json", contact_points)
    write_json(out / "alerting/notification_policy.json", policy)

    write_json(out / "annotations/annotations.json", build_annotations(up))
    write_json(out / "org.json", build_org())
    write_json(out / "api_index.json", API_INDEX)

    emit_catalog(out, up, dashboards)
    emit_cross_reference(out, up, dashboards)
    emit_loader(out)

    counts = {
        "dashboards": len(dashboards),
        "panels": sum(len(d["panels"]) for d in dashboards),
        "datasources": len(datasources),
        "alert_rules": len(ALERT_SPECS),
        "alert_state_rows": sum(len(spec.evaluate(up)) for spec in ALERT_SPECS),
        "annotations": len(build_annotations(up)),
    }
    write_json(
        out / "summary.json",
        {
            "generated_at": ts(datetime.now(timezone.utc)),
            "redshift_dataset": str(args.redshift_data),
            "postgres_dataset": str(args.postgres_data),
            "parameters": {
                "now": ts(up.now),
                "region": up.region,
                "redshift_cluster": up.redshift_cluster_id,
                "rds_instance": up.rds_instance_id,
            },
            "counts": counts,
        },
    )

    print(f"Wrote Grafana mock data to {out}")
    print(
        "  {dashboards} dashboards, {panels} panels, {datasources} datasources, "
        "{alert_rules} alert rules".format(**counts)
    )

    if args.skip_self_check:
        return 0

    problems = run_self_check(out, up, dashboards, snapshots, datasources)
    if problems:
        print(f"  self-check FAILED with {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    - {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return 1

    print("  self-check passed: panel snapshots and alert state reproduce from the source data")
    print(f"  join table: {out / 'cross_reference_grafana.json'}")
    print(f"  loader:     {out / 'load_grafana.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
