#!/usr/bin/env python3
"""Generate mock Amazon Redshift data for the SOTF reporting warehouse.

This script mocks **Redshift only**. It regenerates nothing -- it *reads* the
datasets the other generators produced and models the warehouse that would sit
downstream of all of them:

    generate_sotf_mock_data.py       -> ./out             (required)
    generate_airflow_mock_data.py    -> ./out_airflow     (optional)
    generate_databricks_mock_data.py -> ./out_databricks  (optional)
    generate_postgres_mock_data.py   -> ./out_postgres    (optional)

Four schemas:

    raw            landing tables, one per upstream system, loaded by COPY
    mart           conformed dimensions, facts and daily aggregates
    ops            pipeline observability: Airflow runs, Databricks runs, model
                   predictions, COPY history, and one row per upstream tool
    spectrum_sotf  external schema over the S3 artifact bucket

Every upstream tool in the SOTF chatbot plan appears somewhere: RDS Postgres,
S3, ClickHouse, TestRail and Jira land in ``raw``; Airflow and Databricks land in
``ops``; and ``ops.dim_data_source`` names all of them with their freshness.

The alignment is the point. ``mart.agg_daily_execution_metrics`` is computed the
same way ``sotf.gold.daily_execution_metrics`` is in the Databricks dataset, and
the self-check fails if the two disagree -- so "what was the pass rate on the
14th" has one answer no matter which system you ask.

Usage:
    python3 generate_redshift_mock_data.py --clean
    python3 generate_redshift_mock_data.py --sotf-data ./out --out ./out_redshift
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Cluster constants
# ---------------------------------------------------------------------------

CLUSTER_ID = "sotf-analytics"
DATABASE = "sotf_analytics"
REGION = "us-west-2"
ACCOUNT_ID = "123456789012"
IAM_ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/RedshiftSotfLoader"
STAGE_BUCKET = "zoox-redshift-stage"
STAGE_PREFIX = "sotf"
GLUE_DATABASE = "sotf_lake"

SCHEMAS = ("raw", "mart", "ops")
EXTERNAL_SCHEMA = "spectrum_sotf"

READ_GROUP = "sotf_readonly"
ETL_GROUP = "sotf_etl"

# Repeated here only so the failure trend table can carry it; the SOTF generator
# owns the taxonomy.
FAILURE_COMPONENT = {
    "pickup_geofence_mismatch": "geofence_monitor",
    "milestone_timeout": "mission_monitor",
    "vehicle_unassigned": "dispatch",
    "rider_app_crash_on_confirm": "rider_app",
    "webapi_timeout": "webapi_client",
    "device_disconnect": "device_agent",
    "sim_pose_stall": "sim_bridge",
    "door_unlock_not_acked": "vehicle_gateway",
    "dropoff_geofence_blocked": "geofence_monitor",
}

ENV_SLUGS = {
    "Dev": "dev",
    "Staging": "staging",
    "SDET": "sdet",
    "Prod 0": "prod",
    "Prod 1": "prod1",
}

# Redshift has no array type, so list-valued columns are collapsed to a
# delimited string. '|' never appears in the source values.
LIST_SEPARATOR = "|"

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


# ---------------------------------------------------------------------------
# Table catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RCol:
    name: str
    type: str
    encode: str = "az64"
    nullable: bool = True
    comment: str = ""


@dataclass(frozen=True)
class RTable:
    schema: str
    name: str
    comment: str
    columns: tuple[RCol, ...]
    source: str
    diststyle: str = "even"  # even | key | all | auto
    distkey: str | None = None
    sortkey: tuple[str, ...] = ()
    primary_key: tuple[str, ...] = ()
    # (column, "schema.table", referenced_column)
    foreign_keys: tuple[tuple[str, str, str], ...] = ()

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def relative_path(self) -> str:
        return f"data/{self.schema}/{self.name}.csv"

    @property
    def s3_uri(self) -> str:
        return f"s3://{STAGE_BUCKET}/{STAGE_PREFIX}/{self.schema}/{self.name}.csv"

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def c(name: str, type_name: str, **kwargs: Any) -> RCol:
    return RCol(name, type_name, **kwargs)


TS = "TIMESTAMP"


def v(size: int) -> str:
    return f"VARCHAR({size})"


TABLES: tuple[RTable, ...] = (
    # ---- raw: one landing table per upstream system -----------------------
    RTable(
        schema="raw",
        name="rides",
        comment="Rides replicated out of the test-mgmt RDS Postgres instance by DMS.",
        source="Amazon RDS for PostgreSQL",
        columns=(
            c("trigger_id", v(36), encode="raw", nullable=False),
            c("execution_id", v(36), encode="lzo", nullable=False),
            c("scenario_name", v(255), encode="bytedict"),
            c("scenario_version", "DECIMAL(3,1)"),
            c("ride_status", v(20), encode="bytedict"),
            c("environment", v(20), encode="bytedict"),
            c("ride_service", v(64), encode="bytedict"),
            c("pudo_zone", v(64), encode="bytedict"),
            c("pickup_poi", v(255), encode="zstd"),
            c("dropoff_poi", v(255), encode="zstd"),
            c("host_machine", v(64), encode="bytedict"),
            c("device_id", v(64), encode="bytedict"),
            c("vehicle_id", v(64), encode="bytedict"),
            c("started_at", TS),
            c("ended_at", TS),
            c("failure_mode", v(64), encode="bytedict"),
            c("failed_step_id", "INTEGER"),
            c("testrail_run_id", "BIGINT"),
            c("testrail_test_id", "BIGINT"),
            c("testrail_result_id", "BIGINT"),
            c("jira_key", v(20), encode="lzo"),
            c("s3_key", v(512), encode="zstd"),
            c("log_row_count", "INTEGER"),
            c("loaded_at", TS),
        ),
        diststyle="key",
        distkey="trigger_id",
        sortkey=("started_at",),
        primary_key=("trigger_id",),
    ),
    RTable(
        schema="raw",
        name="executions",
        comment="test_executions rows replicated out of RDS Postgres.",
        source="Amazon RDS for PostgreSQL",
        columns=(
            c("execution_id", v(36), encode="raw", nullable=False),
            c("scenario_id", v(36), encode="lzo"),
            c("suite_id", v(36), encode="lzo"),
            c("scenario_name", v(255), encode="bytedict"),
            c("scenario_version", "DECIMAL(3,1)"),
            c("status", v(20), encode="bytedict"),
            c("environment", v(20), encode="bytedict"),
            c("ride_service", v(64), encode="bytedict"),
            c("pudo_zone", v(64), encode="bytedict"),
            c("no_of_triggers", "SMALLINT"),
            c("testrail_run_id", "BIGINT"),
            c("testrail_case_id", "BIGINT"),
            c("created_by", v(150), encode="bytedict"),
            c("created_at", TS),
            c("scheduled_at", TS),
            c("loaded_at", TS),
        ),
        diststyle="even",
        sortkey=("created_at",),
        primary_key=("execution_id",),
    ),
    RTable(
        schema="raw",
        name="scenarios",
        comment="test_scenarios rows replicated out of RDS Postgres.",
        source="Amazon RDS for PostgreSQL",
        columns=(
            c("scenario_id", v(36), encode="raw", nullable=False),
            c("scenario_name", v(255), encode="zstd"),
            c("version", "DECIMAL(3,1)"),
            c("status", v(20), encode="bytedict"),
            c("testrail_case_id", "BIGINT"),
            c("step_count", "SMALLINT"),
            c("labels", v(255), encode="zstd", comment="pipe delimited"),
            c("created_by", v(100), encode="bytedict"),
            c("updated_by", v(100), encode="bytedict"),
            c("created_at", TS),
            c("updated_at", TS),
            c("loaded_at", TS),
        ),
        diststyle="all",
        sortkey=("scenario_name",),
        primary_key=("scenario_id",),
    ),
    RTable(
        schema="raw",
        name="testrail_results",
        comment="TestRail results pulled by the sotf_testrail_sync Airflow DAG.",
        source="TestRail",
        columns=(
            c("result_id", "BIGINT", encode="raw", nullable=False),
            c("test_id", "BIGINT"),
            c("run_id", "BIGINT"),
            c("case_id", "BIGINT"),
            c("trigger_id", v(36), encode="lzo"),
            c("status_id", "SMALLINT"),
            c("status", v(20), encode="bytedict"),
            c("defects", v(20), encode="lzo", comment="ZPT key, when one was filed"),
            c("elapsed_seconds", "INTEGER"),
            c("recorded_at", TS),
            c("loaded_at", TS),
        ),
        diststyle="key",
        distkey="trigger_id",
        sortkey=("recorded_at",),
        primary_key=("result_id",),
    ),
    RTable(
        schema="raw",
        name="jira_defects",
        comment="ZPT issues pulled by the sotf_jira_defect_sync Airflow DAG.",
        source="Jira",
        columns=(
            c("issue_key", v(20), encode="raw", nullable=False),
            c("issue_id", v(20), encode="lzo"),
            c("summary", v(1024), encode="zstd"),
            c("issue_status", v(32), encode="bytedict"),
            c("failure_mode", v(64), encode="bytedict"),
            c("ride_service", v(64), encode="bytedict"),
            c("occurrences", "INTEGER"),
            c("linked_trigger_ids", v(65535), encode="zstd", comment="pipe delimited"),
            c("created_at", TS),
            c("updated_at", TS),
            c("loaded_at", TS),
        ),
        diststyle="all",
        sortkey=("created_at",),
        primary_key=("issue_key",),
    ),
    RTable(
        schema="raw",
        name="ctm_log_daily",
        comment=(
            "Daily counts of CTM log rows by severity and component. The log lines "
            "themselves stay in ClickHouse; only the rollup is warehoused."
        ),
        source="ClickHouse",
        columns=(
            c("log_date", "DATE", encode="raw", nullable=False),
            c("severity", v(10), encode="bytedict", nullable=False),
            c("component", v(64), encode="bytedict", nullable=False),
            c("environment", v(20), encode="bytedict"),
            c("row_count", "BIGINT"),
            c("distinct_triggers", "INTEGER"),
            c("loaded_at", TS),
        ),
        diststyle="even",
        sortkey=("log_date", "severity"),
        primary_key=("log_date", "severity", "component", "environment"),
    ),
    RTable(
        schema="raw",
        name="s3_artifacts",
        comment="Artifact bundle inventory from the TEC artifacts prefix.",
        source="Amazon S3",
        columns=(
            c("trigger_id", v(36), encode="raw", nullable=False),
            c("bucket", v(128), encode="bytedict"),
            c("s3_key", v(512), encode="zstd"),
            c("size_bytes", "BIGINT"),
            c("member_count", "SMALLINT"),
            c("last_modified", TS),
            c("loaded_at", TS),
        ),
        diststyle="key",
        distkey="trigger_id",
        sortkey=("last_modified",),
        primary_key=("trigger_id",),
    ),
    # ---- mart: dimensions -------------------------------------------------
    RTable(
        schema="mart",
        name="dim_date",
        comment="Calendar dimension covering every day the warehouse has facts for.",
        source="generated",
        columns=(
            c("date_key", "INTEGER", encode="raw", nullable=False),
            c("calendar_date", "DATE", nullable=False),
            c("day_of_week", "SMALLINT"),
            c("day_name", v(10), encode="bytedict"),
            c("week_of_year", "SMALLINT"),
            c("month_number", "SMALLINT"),
            c("month_name", v(10), encode="bytedict"),
            c("quarter", "SMALLINT"),
            c("year", "SMALLINT"),
            c("is_weekend", "BOOLEAN", encode="raw"),
        ),
        diststyle="all",
        sortkey=("date_key",),
        primary_key=("date_key",),
    ),
    RTable(
        schema="mart",
        name="dim_scenario",
        comment="One row per scenario authored in the SOTF app.",
        source="Amazon RDS for PostgreSQL",
        columns=(
            c("scenario_key", "INTEGER", encode="raw", nullable=False),
            c("scenario_id", v(36), encode="lzo", nullable=False),
            c("scenario_name", v(255), encode="zstd", nullable=False),
            c("current_version", "DECIMAL(3,1)"),
            c("scenario_status", v(20), encode="bytedict"),
            c("testrail_case_id", "BIGINT"),
            c("step_count", "SMALLINT"),
            c("primary_label", v(64), encode="bytedict"),
            c("labels", v(255), encode="zstd"),
            c("created_at", TS),
            c("updated_at", TS),
        ),
        diststyle="all",
        sortkey=("scenario_key",),
        primary_key=("scenario_key",),
    ),
    RTable(
        schema="mart",
        name="dim_environment",
        comment="Execution environments, with the slug used in service URLs and S3 prefixes.",
        source="test-mgmt metadata.json",
        columns=(
            c("environment_key", "INTEGER", encode="raw", nullable=False),
            c("environment", v(20), encode="bytedict", nullable=False),
            c("env_slug", v(20), encode="bytedict"),
            c("is_production", "BOOLEAN", encode="raw"),
        ),
        diststyle="all",
        sortkey=("environment_key",),
        primary_key=("environment_key",),
    ),
    RTable(
        schema="mart",
        name="dim_pudo_zone",
        comment="Pick-up / drop-off zones, grouped under their ride service.",
        source="test-mgmt metadata.json",
        columns=(
            c("pudo_zone_key", "INTEGER", encode="raw", nullable=False),
            c("pudo_zone", v(64), encode="bytedict", nullable=False),
            c("ride_service", v(64), encode="bytedict"),
            c("distinct_pickup_pois", "SMALLINT"),
            c("distinct_dropoff_pois", "SMALLINT"),
        ),
        diststyle="all",
        sortkey=("pudo_zone_key",),
        primary_key=("pudo_zone_key",),
    ),
    # ---- mart: facts ------------------------------------------------------
    RTable(
        schema="mart",
        name="fact_ride",
        comment=(
            "One row per SOTF ride. date_key is null for a ride that never started, "
            "which is correct: it has no calendar day."
        ),
        source="raw.rides",
        columns=(
            c("trigger_id", v(36), encode="raw", nullable=False),
            c("date_key", "INTEGER"),
            c("scenario_key", "INTEGER"),
            c("environment_key", "INTEGER"),
            c("pudo_zone_key", "INTEGER"),
            c("execution_id", v(36), encode="lzo"),
            c("ride_status", v(20), encode="bytedict"),
            c("is_completed", "BOOLEAN", encode="raw"),
            c("is_passed", "BOOLEAN", encode="raw"),
            c("is_failed", "BOOLEAN", encode="raw"),
            c("failure_mode", v(64), encode="bytedict"),
            c("failure_component", v(64), encode="bytedict"),
            c("failed_step_id", "INTEGER"),
            c("duration_seconds", "INTEGER"),
            c("log_row_count", "INTEGER"),
            c("has_artifact", "BOOLEAN", encode="raw"),
            c("has_defect", "BOOLEAN", encode="raw"),
            c("testrail_result_id", "BIGINT"),
            c("jira_key", v(20), encode="lzo"),
            c("started_at", TS),
            c("ended_at", TS),
        ),
        diststyle="key",
        distkey="trigger_id",
        sortkey=("date_key", "environment_key"),
        primary_key=("trigger_id",),
        foreign_keys=(
            ("date_key", "mart.dim_date", "date_key"),
            ("scenario_key", "mart.dim_scenario", "scenario_key"),
            ("environment_key", "mart.dim_environment", "environment_key"),
            ("pudo_zone_key", "mart.dim_pudo_zone", "pudo_zone_key"),
        ),
    ),
    RTable(
        schema="mart",
        name="fact_test_result",
        comment="One row per TestRail result, joined back to the ride that produced it.",
        source="raw.testrail_results",
        columns=(
            c("result_id", "BIGINT", encode="raw", nullable=False),
            c("date_key", "INTEGER"),
            c("trigger_id", v(36), encode="lzo"),
            c("scenario_key", "INTEGER"),
            c("run_id", "BIGINT"),
            c("test_id", "BIGINT"),
            c("case_id", "BIGINT"),
            c("status_id", "SMALLINT"),
            c("status", v(20), encode="bytedict"),
            c("defects", v(20), encode="lzo"),
            c("elapsed_seconds", "INTEGER"),
            c("recorded_at", TS),
        ),
        diststyle="key",
        distkey="trigger_id",
        sortkey=("date_key",),
        primary_key=("result_id",),
        foreign_keys=(("date_key", "mart.dim_date", "date_key"),),
    ),
    RTable(
        schema="mart",
        name="fact_defect",
        comment="One row per ZPT issue, with the occurrence count it accumulated.",
        source="raw.jira_defects",
        columns=(
            c("issue_key", v(20), encode="raw", nullable=False),
            c("date_key", "INTEGER"),
            c("failure_mode", v(64), encode="bytedict"),
            c("failure_component", v(64), encode="bytedict"),
            c("ride_service", v(64), encode="bytedict"),
            c("issue_status", v(32), encode="bytedict"),
            c("occurrences", "INTEGER"),
            c("linked_ride_count", "INTEGER"),
            c("summary", v(1024), encode="zstd"),
            c("created_at", TS),
            c("updated_at", TS),
        ),
        diststyle="all",
        sortkey=("date_key",),
        primary_key=("issue_key",),
        foreign_keys=(("date_key", "mart.dim_date", "date_key"),),
    ),
    # ---- mart: aggregates -------------------------------------------------
    RTable(
        schema="mart",
        name="agg_daily_execution_metrics",
        comment=(
            "Daily pass rate by environment and ride service. Computed exactly as "
            "sotf.gold.daily_execution_metrics is in the Databricks dataset."
        ),
        source="mart.fact_ride",
        columns=(
            c("metric_date", "DATE", encode="raw", nullable=False),
            c("date_key", "INTEGER"),
            c("environment", v(20), encode="bytedict", nullable=False),
            c("ride_service", v(64), encode="bytedict", nullable=False),
            c("rides", "BIGINT"),
            c("passed", "BIGINT"),
            c("failed", "BIGINT"),
            c("pass_rate", "DECIMAL(6,4)"),
            c("avg_duration_seconds", "DECIMAL(10,1)"),
            c("p90_duration_seconds", "DECIMAL(10,1)"),
            c("artifacts_captured", "BIGINT"),
            c("defects_filed", "BIGINT"),
        ),
        diststyle="even",
        sortkey=("metric_date", "environment"),
        primary_key=("metric_date", "environment", "ride_service"),
    ),
    RTable(
        schema="mart",
        name="agg_scenario_health",
        comment=(
            "Per-scenario reliability with a flip-rate flakiness score. Mirrors "
            "sotf.gold.scenario_health."
        ),
        source="mart.fact_ride",
        columns=(
            c("scenario_key", "INTEGER", encode="raw", nullable=False),
            c("scenario_name", v(255), encode="zstd"),
            c("total_runs", "BIGINT"),
            c("passed", "BIGINT"),
            c("failed", "BIGINT"),
            c("pass_rate", "DECIMAL(6,4)"),
            c("flakiness_score", "DECIMAL(6,4)", comment="Share of consecutive runs that flipped"),
            c("consecutive_failures", "SMALLINT"),
            c("distinct_failure_modes", "SMALLINT"),
            c("top_failure_mode", v(64), encode="bytedict"),
            c("environments_covered", v(128), encode="zstd"),
            c("last_run_at", TS),
            c("last_failure_at", TS),
        ),
        diststyle="all",
        sortkey=("scenario_key",),
        primary_key=("scenario_key",),
    ),
    RTable(
        schema="mart",
        name="agg_failure_mode_daily",
        comment="Daily occurrences per failure mode, with the defects they produced.",
        source="mart.fact_ride",
        columns=(
            c("metric_date", "DATE", encode="raw", nullable=False),
            c("failure_mode", v(64), encode="bytedict", nullable=False),
            c("failure_component", v(64), encode="bytedict"),
            c("occurrences", "BIGINT"),
            c("distinct_scenarios", "SMALLINT"),
            c("distinct_zones", "SMALLINT"),
            c("distinct_environments", "SMALLINT"),
            c("zpt_keys", v(1024), encode="zstd", comment="pipe delimited"),
            c("first_seen_at", TS),
            c("last_seen_at", TS),
        ),
        diststyle="even",
        sortkey=("metric_date", "failure_mode"),
        primary_key=("metric_date", "failure_mode"),
    ),
    RTable(
        schema="mart",
        name="agg_pudo_zone_reliability",
        comment="Reliability rollup by pick-up / drop-off zone.",
        source="mart.fact_ride",
        columns=(
            c("pudo_zone_key", "INTEGER", encode="raw", nullable=False),
            c("pudo_zone", v(64), encode="bytedict"),
            c("ride_service", v(64), encode="bytedict"),
            c("rides", "BIGINT"),
            c("passed", "BIGINT"),
            c("failed", "BIGINT"),
            c("pass_rate", "DECIMAL(6,4)"),
            c("distinct_scenarios", "SMALLINT"),
            c("distinct_pickup_pois", "SMALLINT"),
            c("top_failure_mode", v(64), encode="bytedict"),
            c("avg_duration_seconds", "DECIMAL(10,1)"),
        ),
        diststyle="all",
        sortkey=("pudo_zone_key",),
        primary_key=("pudo_zone_key",),
    ),
    # ---- ops: cross-tool pipeline observability ---------------------------
    RTable(
        schema="ops",
        name="dim_data_source",
        comment="One row per upstream system the warehouse reads, with its freshness.",
        source="generated",
        columns=(
            c("source_key", "INTEGER", encode="raw", nullable=False),
            c("source_name", v(64), encode="bytedict", nullable=False),
            c("source_system", v(64), encode="bytedict"),
            c("ingestion_method", v(64), encode="bytedict"),
            c("target_table", v(128), encode="zstd"),
            c("refresh_cadence", v(64), encode="bytedict"),
            c("is_authoritative", "BOOLEAN", encode="raw"),
            c("rows_last_load", "BIGINT"),
            c("last_loaded_at", TS),
        ),
        diststyle="all",
        sortkey=("source_key",),
        primary_key=("source_key",),
    ),
    RTable(
        schema="ops",
        name="fact_airflow_dag_run",
        comment="One row per Airflow DAG run, from the out_airflow dataset.",
        source="Apache Airflow",
        columns=(
            c("dag_run_id", v(80), encode="raw", nullable=False),
            c("dag_id", v(64), encode="bytedict", nullable=False),
            c("date_key", "INTEGER"),
            c("logical_date", TS),
            c("started_at", TS),
            c("ended_at", TS),
            c("duration_seconds", "INTEGER"),
            c("state", v(16), encode="bytedict"),
            c("run_type", v(16), encode="bytedict"),
            c("task_count", "SMALLINT"),
            c("failed_task_count", "SMALLINT"),
            c("retried_task_count", "SMALLINT"),
        ),
        diststyle="even",
        sortkey=("date_key", "dag_id"),
        primary_key=("dag_run_id", "dag_id"),
    ),
    RTable(
        schema="ops",
        name="fact_airflow_task_instance",
        comment="One row per Airflow task instance.",
        source="Apache Airflow",
        columns=(
            c("dag_run_id", v(80), encode="raw", nullable=False),
            c("dag_id", v(64), encode="bytedict", nullable=False),
            c("task_id", v(64), encode="bytedict", nullable=False),
            c("date_key", "INTEGER"),
            c("operator", v(48), encode="bytedict"),
            c("state", v(16), encode="bytedict"),
            c("try_number", "SMALLINT"),
            c("duration_seconds", "DECIMAL(10,1)"),
            c("started_at", TS),
            c("ended_at", TS),
            c("hostname", v(64), encode="bytedict"),
        ),
        diststyle="even",
        sortkey=("date_key", "dag_id"),
        primary_key=("dag_run_id", "dag_id", "task_id"),
    ),
    RTable(
        schema="ops",
        name="fact_databricks_job_run",
        comment="One row per Databricks job run, from the out_databricks dataset.",
        source="Databricks",
        columns=(
            c("run_id", "BIGINT", encode="raw", nullable=False),
            c("job_id", "BIGINT"),
            c("job_name", v(64), encode="bytedict"),
            c("date_key", "INTEGER"),
            c("started_at", TS),
            c("ended_at", TS),
            c("duration_seconds", "INTEGER"),
            c("result_state", v(16), encode="bytedict"),
            c("task_count", "SMALLINT"),
            c("failed_task_count", "SMALLINT"),
        ),
        diststyle="even",
        sortkey=("date_key", "job_name"),
        primary_key=("run_id",),
    ),
    RTable(
        schema="ops",
        name="fact_model_prediction",
        comment=(
            "Batch scoring output of the Databricks failure-mode classifier, "
            "unloaded into Redshift so BI can join it to the facts."
        ),
        source="Databricks",
        columns=(
            c("trigger_id", v(36), encode="raw", nullable=False),
            c("date_key", "INTEGER"),
            c("model_name", v(64), encode="bytedict"),
            c("model_version", "SMALLINT"),
            c("mlflow_run_id", v(32), encode="lzo"),
            c("actual_failure_mode", v(64), encode="bytedict"),
            c("predicted_failure_mode", v(64), encode="bytedict"),
            c("confidence", "DECIMAL(6,4)"),
            c("is_correct", "BOOLEAN", encode="raw"),
            c("predicted_at", TS),
        ),
        diststyle="key",
        distkey="trigger_id",
        sortkey=("date_key",),
        primary_key=("trigger_id",),
    ),
    RTable(
        schema="ops",
        name="fact_pipeline_load",
        comment="COPY history for every table in this warehouse, one row per load.",
        source="Amazon Redshift",
        columns=(
            c("load_id", "BIGINT", encode="raw", nullable=False),
            c("target_table", v(128), encode="bytedict", nullable=False),
            c("source_uri", v(512), encode="zstd"),
            c("source_system", v(64), encode="bytedict"),
            c("started_at", TS),
            c("finished_at", TS),
            c("duration_seconds", "DECIMAL(10,2)"),
            c("rows_loaded", "BIGINT"),
            c("files_scanned", "SMALLINT"),
            c("load_status", v(16), encode="bytedict"),
            c("error_count", "SMALLINT"),
        ),
        diststyle="even",
        sortkey=("started_at",),
        primary_key=("load_id",),
    ),
)

TABLE_BY_NAME = {table.full_name: table for table in TABLES}


# ---------------------------------------------------------------------------
# Saved queries: the row count each one returns is checked against the data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedQuery:
    name: str
    sql: str
    # Key into the built dataset used to work out how many rows it returns.
    result_rows_from: str


SAVED_QUERIES: tuple[SavedQuery, ...] = (
    SavedQuery(
        name="daily_pass_rate_by_environment",
        sql=(
            "SELECT metric_date, environment, sum(rides) AS rides, "
            "round(sum(passed)::decimal / nullif(sum(rides), 0), 4) AS pass_rate "
            "FROM mart.agg_daily_execution_metrics "
            "GROUP BY metric_date, environment ORDER BY metric_date DESC, environment"
        ),
        result_rows_from="daily_pass_rate_by_environment",
    ),
    SavedQuery(
        name="flakiest_scenarios",
        sql=(
            "SELECT scenario_name, total_runs, pass_rate, flakiness_score, top_failure_mode "
            "FROM mart.agg_scenario_health WHERE total_runs >= 4 AND flakiness_score >= 0.3 "
            "ORDER BY flakiness_score DESC, total_runs DESC"
        ),
        result_rows_from="flakiest_scenarios",
    ),
    SavedQuery(
        name="failure_modes_last_7_days",
        sql=(
            "SELECT failure_mode, failure_component, sum(occurrences) AS occurrences "
            "FROM mart.agg_failure_mode_daily "
            "WHERE metric_date >= dateadd(day, -7, current_date) "
            "GROUP BY failure_mode, failure_component ORDER BY occurrences DESC"
        ),
        result_rows_from="failure_modes_last_7_days",
    ),
    SavedQuery(
        name="pudo_zone_reliability",
        sql=(
            "SELECT pudo_zone, ride_service, rides, pass_rate, top_failure_mode "
            "FROM mart.agg_pudo_zone_reliability ORDER BY pass_rate ASC"
        ),
        result_rows_from="pudo_zone_reliability",
    ),
    SavedQuery(
        name="pipeline_health_by_day",
        sql=(
            "SELECT d.calendar_date, "
            "count(*) FILTER (WHERE a.state = 'failed') AS failed_dag_runs, "
            "count(*) FILTER (WHERE a.state = 'success') AS successful_dag_runs "
            "FROM ops.fact_airflow_dag_run a "
            "JOIN mart.dim_date d ON d.date_key = a.date_key "
            "GROUP BY d.calendar_date ORDER BY d.calendar_date DESC"
        ),
        result_rows_from="pipeline_health_by_day",
    ),
    SavedQuery(
        name="model_accuracy_by_failure_mode",
        sql=(
            "SELECT actual_failure_mode, count(*) AS scored, "
            "sum(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct "
            "FROM ops.fact_model_prediction GROUP BY actual_failure_mode ORDER BY scored DESC"
        ),
        result_rows_from="model_accuracy_by_failure_mode",
    ),
    SavedQuery(
        name="rides_missing_artifacts",
        sql=(
            "SELECT f.trigger_id, s.scenario_name, f.ride_status, f.ended_at "
            "FROM mart.fact_ride f JOIN mart.dim_scenario s USING (scenario_key) "
            "WHERE f.is_completed AND NOT f.has_artifact ORDER BY f.ended_at DESC"
        ),
        result_rows_from="rides_missing_artifacts",
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ts(value: datetime | None) -> str | None:
    """Redshift TIMESTAMP has no zone: store UTC wall time."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def date_key_of(value: datetime | date | None) -> int | None:
    if value is None:
        return None
    return int(value.strftime("%Y%m%d"))


def join_list(values: Iterable[Any]) -> str | None:
    text = LIST_SEPARATOR.join(str(value) for value in values if value is not None)
    return text or None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[index], 1)


def top_key(counter: dict[str, int]) -> str | None:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


class Sequence:
    def __init__(self, start: int) -> None:
        self._value = start

    def next(self) -> int:
        self._value += 1
        return self._value


# ---------------------------------------------------------------------------
# Upstream dataset input
# ---------------------------------------------------------------------------


@dataclass
class Upstream:
    now: datetime
    days: int
    bucket: str
    prefix: str
    airflow_root: Path
    counts: dict[str, Any]
    rides: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    scenarios: list[dict[str, Any]]
    defects: list[dict[str, Any]]
    log_daily: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    airflow_runs: list[dict[str, Any]]
    airflow_tasks: list[dict[str, Any]]
    airflow_counts: dict[str, Any]
    databricks_runs: list[dict[str, Any]]
    databricks_counts: dict[str, Any]
    predictions: list[dict[str, Any]]
    databricks_gold: dict[str, list[dict[str, Any]]]
    has_airflow: bool
    has_databricks: bool
    has_postgres: bool

    @property
    def completed(self) -> list[dict[str, Any]]:
        return [
            ride
            for ride in self.rides
            if ride["ride_status"] in ("Passed", "Failed") and ride["_started"]
        ]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.completed if r["ride_status"] == "Failed" and r["failure_mode"]]


def load_sotf(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_file = root / "summary.json"
    cross_file = root / "cross_reference.json"
    missing = [f.name for f in (summary_file, cross_file) if not f.is_file()]
    if missing:
        raise SystemExit(
            f"SOTF dataset at {root} is missing {', '.join(missing)}\n"
            "Run generate_sotf_mock_data.py first, or pass --sotf-data."
        )
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    rides = json.loads(cross_file.read_text(encoding="utf-8"))
    for ride in rides:
        ride["_started"] = parse_dt(ride.get("started_at"))
        ride["_ended"] = parse_dt(ride.get("ended_at"))
        if ride["_started"] and ride["_ended"]:
            ride["_duration"] = (ride["_ended"] - ride["_started"]).total_seconds()
        else:
            ride["_duration"] = None
    return summary, rides


def load_ctm_log_daily(root: Path) -> list[dict[str, Any]]:
    """Roll ClickHouse log rows up to (date, severity, component, environment)."""
    path = root / "clickhouse/logs.jsonl"
    if not path.is_file():
        return []

    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    triggers: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["timestamp"][:10],
                row["severity"],
                row["component"],
                row.get("environment") or "unknown",
            )
            counts[key] += 1
            triggers[key].add(row.get("trigger_id") or "")

    return [
        {
            "log_date": log_date,
            "severity": severity,
            "component": component,
            "environment": environment,
            "row_count": count,
            "distinct_triggers": len(triggers[(log_date, severity, component, environment)]),
        }
        for (log_date, severity, component, environment), count in sorted(counts.items())
    ]


def load_defects(root: Path, rides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read the ZPT issues, taking failure mode and ride service from the rides
    they are linked to rather than from Jira custom fields, so the warehouse
    cannot disagree with the facts about why an issue exists."""
    search = root / "jira/search_issues.json"
    if not search.is_file():
        return []

    ride_by_id = {ride["trigger_id"]: ride for ride in rides}
    defects = []
    for issue in json.loads(search.read_text(encoding="utf-8")).get("issues", []):
        linked = issue.get("linked_rides", [])
        origin = next((ride_by_id[t] for t in linked if t in ride_by_id), None)
        defects.append(
            {
                "issue_key": issue["key"],
                "issue_id": issue["id"],
                "summary": issue["fields"]["summary"],
                "issue_status": issue["fields"]["status"]["name"],
                "failure_mode": origin["failure_mode"] if origin else None,
                "ride_service": origin["ride_service"] if origin else None,
                "occurrences": issue.get("occurrences", 1),
                "linked_rides": linked,
                "created_at": parse_dt(issue["fields"].get("created")),
                "updated_at": parse_dt(issue["fields"].get("updated")),
            }
        )
    return defects


def load_airflow(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index_file = root / "runs_index.json"
    summary_file = root / "summary.json"
    if not (index_file.is_file() and summary_file.is_file()):
        return [], [], {}

    counts = json.loads(summary_file.read_text(encoding="utf-8")).get("counts", {})
    index = json.loads(index_file.read_text(encoding="utf-8"))["runs"]

    detail_by_run: dict[tuple[str, str], dict[str, Any]] = {}
    for dag_file in sorted((root / "dag_runs").glob("*.json")):
        for run in json.loads(dag_file.read_text(encoding="utf-8"))["dag_runs"]:
            detail_by_run[(run["dag_id"], run["dag_run_id"])] = run

    runs: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for entry in index:
        detail = detail_by_run.get((entry["dag_id"], entry["dag_run_id"]), {})
        instances_file = root / entry["task_instances_file"]
        instances = (
            json.loads(instances_file.read_text(encoding="utf-8"))["task_instances"]
            if instances_file.is_file()
            else []
        )
        runs.append({**entry, **detail, "_task_instances": instances})
        for instance in instances:
            tasks.append({**instance, "dag_run_id": entry["dag_run_id"]})
    return runs, tasks, counts


def load_databricks(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    runs_file = root / "jobs/runs.json"
    summary_file = root / "summary.json"
    if not (runs_file.is_file() and summary_file.is_file()):
        return [], {}, [], {}

    counts = json.loads(summary_file.read_text(encoding="utf-8")).get("counts", {})
    runs = json.loads(runs_file.read_text(encoding="utf-8"))["runs"]
    for run in runs:
        detail_file = root / f"jobs/runs/{run['run_id']}.json"
        if detail_file.is_file():
            run["tasks"] = json.loads(detail_file.read_text(encoding="utf-8")).get("tasks", [])

    def read_jsonl(relative: str) -> list[dict[str, Any]]:
        path = root / relative
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    gold = {
        "daily_execution_metrics": read_jsonl("tables/gold/daily_execution_metrics.jsonl"),
        "scenario_health": read_jsonl("tables/gold/scenario_health.jsonl"),
        "pudo_zone_reliability": read_jsonl("tables/gold/pudo_zone_reliability.jsonl"),
    }
    predictions = read_jsonl("tables/gold/ride_failure_predictions.jsonl")
    return runs, counts, predictions, gold


def load_upstream(args: argparse.Namespace) -> Upstream:
    summary, rides = load_sotf(args.sotf_data)
    params = summary.get("parameters", {})

    # dim_scenario is keyed off domain/scenarios.json, so a missing file would
    # silently produce a star schema whose facts point at nothing.
    domain = args.sotf_data / "domain"
    required = {
        "scenarios": domain / "scenarios.json",
        "executions": domain / "executions.json",
        "artifacts": args.sotf_data / "s3/objects.json",
    }
    absent = [str(path) for path in required.values() if not path.is_file()]
    if absent:
        raise SystemExit(
            "SOTF dataset is incomplete, missing:\n  " + "\n  ".join(absent) +
            "\nRe-run generate_sotf_mock_data.py."
        )
    scenarios = json.loads(required["scenarios"].read_text(encoding="utf-8"))
    executions = json.loads(required["executions"].read_text(encoding="utf-8"))
    artifacts = json.loads(required["artifacts"].read_text(encoding="utf-8"))

    airflow_runs, airflow_tasks, airflow_counts = (
        load_airflow(args.airflow_data) if args.airflow_data.is_dir() else ([], [], {})
    )
    databricks_runs, databricks_counts, predictions, gold = (
        load_databricks(args.databricks_data) if args.databricks_data.is_dir() else ([], {}, [], {})
    )

    return Upstream(
        now=parse_dt(params.get("now")) or datetime.now(timezone.utc),
        days=int(params.get("days", 30)),
        bucket=params.get("bucket", "zoox-test-artifacts"),
        prefix=params.get("s3_prefix", "tec/dev/artifacts"),
        airflow_root=args.airflow_data,
        counts=summary.get("counts", {}),
        rides=rides,
        executions=executions,
        scenarios=scenarios,
        defects=load_defects(args.sotf_data, rides),
        log_daily=load_ctm_log_daily(args.sotf_data),
        artifacts=artifacts,
        airflow_runs=airflow_runs,
        airflow_tasks=airflow_tasks,
        airflow_counts=airflow_counts,
        databricks_runs=databricks_runs,
        databricks_counts=databricks_counts,
        predictions=predictions,
        databricks_gold=gold,
        has_airflow=bool(airflow_runs),
        has_databricks=bool(databricks_runs),
        has_postgres=args.postgres_data.is_dir(),
    )


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------


def build_dim_date(up: Upstream) -> list[dict[str, Any]]:
    days: set[date] = set()
    for ride in up.rides:
        for value in (ride["_started"], ride["_ended"]):
            if value:
                days.add(value.date())
    for run in up.airflow_runs:
        started = parse_dt(run.get("start_date"))
        if started:
            days.add(started.date())
    for run in up.databricks_runs:
        if run.get("start_time"):
            days.add(datetime.fromtimestamp(run["start_time"] / 1000, tz=timezone.utc).date())
    if not days:
        days.add(up.now.date())

    cursor, last = min(days), max(days)
    rows = []
    while cursor <= last:
        rows.append(
            {
                "date_key": int(cursor.strftime("%Y%m%d")),
                "calendar_date": cursor.isoformat(),
                "day_of_week": cursor.weekday() + 1,
                "day_name": DAY_NAMES[cursor.weekday()],
                "week_of_year": int(cursor.strftime("%V")),
                "month_number": cursor.month,
                "month_name": MONTH_NAMES[cursor.month - 1],
                "quarter": (cursor.month - 1) // 3 + 1,
                "year": cursor.year,
                "is_weekend": cursor.weekday() >= 5,
            }
        )
        cursor += timedelta(days=1)
    return rows


def build_dim_scenario(up: Upstream) -> list[dict[str, Any]]:
    rows = []
    for index, scenario in enumerate(sorted(up.scenarios, key=lambda s: s["name"]), start=1):
        labels = (scenario.get("metadata") or {}).get("labels") or []
        testrail_id = (scenario.get("metadata") or {}).get("testrail_id")
        rows.append(
            {
                "scenario_key": index,
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "current_version": scenario["version"],
                "scenario_status": scenario["status"],
                "testrail_case_id": int(testrail_id) if testrail_id else None,
                "step_count": len(scenario.get("test_steps") or []),
                "primary_label": labels[0] if labels else None,
                "labels": join_list(labels),
                "created_at": ts(parse_dt(scenario.get("created_at"))),
                "updated_at": ts(parse_dt(scenario.get("updated_at"))),
            }
        )
    return rows


def build_dim_environment(up: Upstream) -> list[dict[str, Any]]:
    names = sorted({ride["environment"] for ride in up.rides if ride.get("environment")})
    return [
        {
            "environment_key": index,
            "environment": name,
            "env_slug": ENV_SLUGS.get(name, name.lower().replace(" ", "")),
            "is_production": name.startswith("Prod"),
        }
        for index, name in enumerate(names, start=1)
    ]


def build_dim_pudo_zone(up: Upstream) -> list[dict[str, Any]]:
    service_by_zone: dict[str, str] = {}
    pickups: dict[str, set[str]] = defaultdict(set)
    dropoffs: dict[str, set[str]] = defaultdict(set)
    for ride in up.rides:
        zone = ride.get("pudo_zone")
        if not zone:
            continue
        service_by_zone.setdefault(zone, ride.get("ride_service") or "unknown")
        if ride.get("pickup_poi"):
            pickups[zone].add(ride["pickup_poi"])
        if ride.get("dropoff_poi"):
            dropoffs[zone].add(ride["dropoff_poi"])

    return [
        {
            "pudo_zone_key": index,
            "pudo_zone": zone,
            "ride_service": service_by_zone[zone],
            "distinct_pickup_pois": len(pickups[zone]),
            "distinct_dropoff_pois": len(dropoffs[zone]),
        }
        for index, zone in enumerate(sorted(service_by_zone), start=1)
    ]


# ---------------------------------------------------------------------------
# Raw table builders
# ---------------------------------------------------------------------------


def build_raw_rides(up: Upstream, loaded_at: str) -> list[dict[str, Any]]:
    rows = []
    for ride in up.rides:
        testrail = ride.get("testrail") or {}
        rows.append(
            {
                "trigger_id": ride["trigger_id"],
                "execution_id": ride["execution_id"],
                "scenario_name": ride["scenario"],
                "scenario_version": ride["scenario_version"],
                "ride_status": ride["ride_status"],
                "environment": ride["environment"],
                "ride_service": ride["ride_service"],
                "pudo_zone": ride["pudo_zone"],
                "pickup_poi": ride["pickup_poi"],
                "dropoff_poi": ride["dropoff_poi"],
                "host_machine": ride["host_machine"],
                "device_id": ride["device_id"],
                "vehicle_id": ride["vehicle_id"],
                "started_at": ts(ride["_started"]),
                "ended_at": ts(ride["_ended"]),
                "failure_mode": ride["failure_mode"],
                "failed_step_id": ride["failed_step_id"],
                "testrail_run_id": testrail.get("run_id"),
                "testrail_test_id": testrail.get("test_id"),
                "testrail_result_id": testrail.get("result_id"),
                "jira_key": ride.get("jira_key"),
                "s3_key": (ride.get("s3") or {}).get("key"),
                "log_row_count": ride.get("clickhouse_log_rows", 0),
                "loaded_at": loaded_at,
            }
        )
    return rows


def build_raw_executions(up: Upstream, loaded_at: str) -> list[dict[str, Any]]:
    rows = []
    for execution in up.executions:
        payload = execution.get("payload") or {}
        testrail = execution.get("testrail") or {}
        rows.append(
            {
                "execution_id": execution["id"],
                "scenario_id": execution["scenario_id"],
                "suite_id": execution["suite_id"],
                "scenario_name": execution["scenario_name"],
                "scenario_version": execution["scenario_version"],
                "status": execution["status"],
                "environment": payload.get("environment"),
                "ride_service": payload.get("ride_service"),
                "pudo_zone": payload.get("pudo_zone"),
                "no_of_triggers": execution["no_of_triggers"],
                "testrail_run_id": testrail.get("run_id"),
                "testrail_case_id": testrail.get("case_id"),
                "created_by": execution["created_by"],
                "created_at": ts(parse_dt(execution["created_at"])),
                "scheduled_at": ts(parse_dt(execution["scheduled_at"])),
                "loaded_at": loaded_at,
            }
        )
    return rows


def build_raw_scenarios(
    up: Upstream, dim_scenario: list[dict[str, Any]], loaded_at: str
) -> list[dict[str, Any]]:
    by_id = {row["scenario_id"]: row for row in dim_scenario}
    rows = []
    for scenario in up.scenarios:
        dim = by_id.get(scenario["id"], {})
        rows.append(
            {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "version": scenario["version"],
                "status": scenario["status"],
                "testrail_case_id": dim.get("testrail_case_id"),
                "step_count": len(scenario.get("test_steps") or []),
                "labels": dim.get("labels"),
                "created_by": scenario["created_by"],
                "updated_by": scenario["updated_by"],
                "created_at": ts(parse_dt(scenario["created_at"])),
                "updated_at": ts(parse_dt(scenario["updated_at"])),
                "loaded_at": loaded_at,
            }
        )
    return rows


def build_raw_testrail_results(up: Upstream, loaded_at: str) -> list[dict[str, Any]]:
    rows = []
    for ride in up.rides:
        testrail = ride.get("testrail") or {}
        if not testrail.get("result_id"):
            continue
        passed = ride["ride_status"] == "Passed"
        duration = ride["_duration"]
        rows.append(
            {
                "result_id": testrail["result_id"],
                "test_id": testrail.get("test_id"),
                "run_id": testrail.get("run_id"),
                "case_id": testrail.get("case_id"),
                "trigger_id": ride["trigger_id"],
                "status_id": 1 if passed else 5,
                "status": "PASSED" if passed else "FAILED",
                "defects": ride.get("jira_key"),
                "elapsed_seconds": int(duration) if duration is not None else None,
                "recorded_at": ts(ride["_ended"] or ride["_started"]),
                "loaded_at": loaded_at,
            }
        )
    rows.sort(key=lambda row: row["result_id"])
    return rows


def build_raw_jira_defects(up: Upstream, loaded_at: str) -> list[dict[str, Any]]:
    return [
        {
            "issue_key": defect["issue_key"],
            "issue_id": defect["issue_id"],
            "summary": defect["summary"],
            "issue_status": defect["issue_status"],
            "failure_mode": defect["failure_mode"],
            "ride_service": defect["ride_service"],
            "occurrences": defect["occurrences"],
            "linked_trigger_ids": join_list(defect["linked_rides"]),
            "created_at": ts(defect["created_at"]),
            "updated_at": ts(defect["updated_at"]),
            "loaded_at": loaded_at,
        }
        for defect in up.defects
    ]


def build_raw_ctm_log_daily(up: Upstream, loaded_at: str) -> list[dict[str, Any]]:
    return [{**row, "loaded_at": loaded_at} for row in up.log_daily]


def build_raw_s3_artifacts(up: Upstream, loaded_at: str) -> list[dict[str, Any]]:
    return [
        {
            "trigger_id": obj["trigger_id"],
            "bucket": obj["bucket"],
            "s3_key": obj["key"],
            "size_bytes": obj["size"],
            "member_count": len(obj.get("contents") or []),
            "last_modified": ts(parse_dt(obj.get("last_modified"))),
            "loaded_at": loaded_at,
        }
        for obj in up.artifacts
    ]


# ---------------------------------------------------------------------------
# Fact and aggregate builders
# ---------------------------------------------------------------------------


def build_fact_ride(
    up: Upstream,
    scenario_keys: dict[str, int],
    environment_keys: dict[str, int],
    zone_keys: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for ride in up.rides:
        completed = ride["ride_status"] in ("Passed", "Failed") and ride["_started"] is not None
        duration = ride["_duration"]
        rows.append(
            {
                "trigger_id": ride["trigger_id"],
                "date_key": date_key_of(ride["_started"]),
                "scenario_key": scenario_keys.get(ride["scenario"]),
                "environment_key": environment_keys.get(ride["environment"]),
                "pudo_zone_key": zone_keys.get(ride["pudo_zone"]),
                "execution_id": ride["execution_id"],
                "ride_status": ride["ride_status"],
                "is_completed": completed,
                "is_passed": ride["ride_status"] == "Passed",
                "is_failed": ride["ride_status"] == "Failed",
                "failure_mode": ride["failure_mode"],
                "failure_component": (
                    FAILURE_COMPONENT.get(ride["failure_mode"], "unknown")
                    if ride["failure_mode"]
                    else None
                ),
                "failed_step_id": ride["failed_step_id"],
                "duration_seconds": int(duration) if duration is not None else None,
                "log_row_count": ride.get("clickhouse_log_rows", 0),
                "has_artifact": bool((ride.get("s3") or {}).get("key")),
                "has_defect": bool(ride.get("jira_key")),
                "testrail_result_id": (ride.get("testrail") or {}).get("result_id"),
                "jira_key": ride.get("jira_key"),
                "started_at": ts(ride["_started"]),
                "ended_at": ts(ride["_ended"]),
            }
        )
    return rows


def build_fact_test_result(
    raw_results: list[dict[str, Any]], ride_scenario_key: dict[str, int | None]
) -> list[dict[str, Any]]:
    rows = []
    for result in raw_results:
        recorded = datetime.strptime(result["recorded_at"], "%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "result_id": result["result_id"],
                "date_key": date_key_of(recorded),
                "trigger_id": result["trigger_id"],
                "scenario_key": ride_scenario_key.get(result["trigger_id"]),
                "run_id": result["run_id"],
                "test_id": result["test_id"],
                "case_id": result["case_id"],
                "status_id": result["status_id"],
                "status": result["status"],
                "defects": result["defects"],
                "elapsed_seconds": result["elapsed_seconds"],
                "recorded_at": result["recorded_at"],
            }
        )
    return rows


def build_fact_defect(up: Upstream) -> list[dict[str, Any]]:
    return [
        {
            "issue_key": defect["issue_key"],
            "date_key": date_key_of(defect["created_at"]),
            "failure_mode": defect["failure_mode"],
            "failure_component": FAILURE_COMPONENT.get(defect["failure_mode"] or "", "unknown"),
            "ride_service": defect["ride_service"],
            "issue_status": defect["issue_status"],
            "occurrences": defect["occurrences"],
            "linked_ride_count": len(defect["linked_rides"]),
            "summary": defect["summary"],
            "created_at": ts(defect["created_at"]),
            "updated_at": ts(defect["updated_at"]),
        }
        for defect in up.defects
    ]


def build_agg_daily_execution_metrics(up: Upstream) -> list[dict[str, Any]]:
    """Identical arithmetic to the Databricks gold table, deliberately."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for ride in up.completed:
        key = (
            ride["_started"].strftime("%Y-%m-%d"),
            ride["environment"],
            ride["ride_service"],
        )
        groups[key].append(ride)

    rows = []
    for (metric_date, environment, ride_service), members in sorted(groups.items()):
        durations = [r["_duration"] for r in members if r["_duration"] is not None]
        passed = sum(1 for r in members if r["ride_status"] == "Passed")
        rows.append(
            {
                "metric_date": metric_date,
                "date_key": int(metric_date.replace("-", "")),
                "environment": environment,
                "ride_service": ride_service,
                "rides": len(members),
                "passed": passed,
                "failed": len(members) - passed,
                "pass_rate": round(passed / len(members), 4),
                "avg_duration_seconds": mean(durations),
                "p90_duration_seconds": percentile(durations, 0.9),
                "artifacts_captured": sum(1 for r in members if (r.get("s3") or {}).get("key")),
                "defects_filed": len({r["jira_key"] for r in members if r.get("jira_key")}),
            }
        )
    return rows


def build_agg_scenario_health(
    up: Upstream, scenario_keys: dict[str, int]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ride in up.completed:
        groups[ride["scenario"]].append(ride)

    rows = []
    for scenario, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda r: r["_started"])
        statuses = [r["ride_status"] for r in ordered]
        passed = statuses.count("Passed")

        flips = sum(1 for a, b in zip(statuses, statuses[1:]) if a != b)
        flakiness = round(flips / (len(statuses) - 1), 4) if len(statuses) > 1 else 0.0

        streak = 0
        for status in reversed(statuses):
            if status != "Failed":
                break
            streak += 1

        modes: dict[str, int] = defaultdict(int)
        for ride in ordered:
            if ride["failure_mode"]:
                modes[ride["failure_mode"]] += 1
        failures = [r for r in ordered if r["ride_status"] == "Failed"]

        rows.append(
            {
                "scenario_key": scenario_keys.get(scenario),
                "scenario_name": scenario,
                "total_runs": len(ordered),
                "passed": passed,
                "failed": len(ordered) - passed,
                "pass_rate": round(passed / len(ordered), 4),
                "flakiness_score": flakiness,
                "consecutive_failures": streak,
                "distinct_failure_modes": len(modes),
                "top_failure_mode": top_key(dict(modes)),
                "environments_covered": join_list(sorted({r["environment"] for r in ordered})),
                "last_run_at": ts(ordered[-1]["_started"]),
                "last_failure_at": ts(failures[-1]["_started"]) if failures else None,
            }
        )
    return rows


def build_agg_failure_mode_daily(up: Upstream) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ride in up.failed:
        groups[(ride["_started"].strftime("%Y-%m-%d"), ride["failure_mode"])].append(ride)

    rows = []
    for (metric_date, failure_mode), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda r: r["_started"])
        rows.append(
            {
                "metric_date": metric_date,
                "failure_mode": failure_mode,
                "failure_component": FAILURE_COMPONENT.get(failure_mode, "unknown"),
                "occurrences": len(ordered),
                "distinct_scenarios": len({r["scenario"] for r in ordered}),
                "distinct_zones": len({r["pudo_zone"] for r in ordered if r["pudo_zone"]}),
                "distinct_environments": len({r["environment"] for r in ordered}),
                "zpt_keys": join_list(sorted({r["jira_key"] for r in ordered if r.get("jira_key")})),
                "first_seen_at": ts(ordered[0]["_started"]),
                "last_seen_at": ts(ordered[-1]["_started"]),
            }
        )
    return rows


def build_agg_pudo_zone_reliability(
    up: Upstream, zone_keys: dict[str, int], zone_service: dict[str, str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ride in up.completed:
        if ride["pudo_zone"]:
            groups[ride["pudo_zone"]].append(ride)

    rows = []
    for zone, members in sorted(groups.items()):
        passed = sum(1 for r in members if r["ride_status"] == "Passed")
        modes: dict[str, int] = defaultdict(int)
        for ride in members:
            if ride["failure_mode"]:
                modes[ride["failure_mode"]] += 1
        durations = [r["_duration"] for r in members if r["_duration"] is not None]
        rows.append(
            {
                "pudo_zone_key": zone_keys.get(zone),
                "pudo_zone": zone,
                "ride_service": zone_service.get(zone),
                "rides": len(members),
                "passed": passed,
                "failed": len(members) - passed,
                "pass_rate": round(passed / len(members), 4),
                "distinct_scenarios": len({r["scenario"] for r in members}),
                "distinct_pickup_pois": len({r["pickup_poi"] for r in members if r.get("pickup_poi")}),
                "top_failure_mode": top_key(dict(modes)),
                "avg_duration_seconds": mean(durations),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Ops table builders
# ---------------------------------------------------------------------------


def build_fact_airflow_dag_run(up: Upstream) -> list[dict[str, Any]]:
    rows = []
    for run in up.airflow_runs:
        started = parse_dt(run.get("start_date"))
        ended = parse_dt(run.get("end_date"))
        instances = run.get("_task_instances") or []
        rows.append(
            {
                "dag_run_id": run["dag_run_id"],
                "dag_id": run["dag_id"],
                "date_key": date_key_of(started),
                "logical_date": ts(parse_dt(run.get("logical_date"))),
                "started_at": ts(started),
                "ended_at": ts(ended),
                "duration_seconds": (
                    int((ended - started).total_seconds()) if started and ended else None
                ),
                "state": run.get("state"),
                "run_type": run.get("run_type", "scheduled"),
                "task_count": len(instances),
                "failed_task_count": sum(1 for t in instances if t.get("state") == "failed"),
                "retried_task_count": sum(1 for t in instances if (t.get("try_number") or 0) > 1),
            }
        )
    rows.sort(key=lambda row: (row["dag_id"], row["logical_date"] or ""))
    return rows


def build_fact_airflow_task_instance(up: Upstream) -> list[dict[str, Any]]:
    rows = []
    for task in up.airflow_tasks:
        started = parse_dt(task.get("start_date"))
        rows.append(
            {
                "dag_run_id": task["dag_run_id"],
                "dag_id": task["dag_id"],
                "task_id": task["task_id"],
                "date_key": date_key_of(started),
                "operator": task.get("operator"),
                "state": task.get("state"),
                "try_number": task.get("try_number"),
                "duration_seconds": task.get("duration"),
                "started_at": ts(started),
                "ended_at": ts(parse_dt(task.get("end_date"))),
                "hostname": task.get("hostname") or None,
            }
        )
    rows.sort(key=lambda row: (row["dag_id"], row["dag_run_id"], row["task_id"]))
    return rows


def build_fact_databricks_job_run(up: Upstream) -> list[dict[str, Any]]:
    rows = []
    for run in up.databricks_runs:
        started = (
            datetime.fromtimestamp(run["start_time"] / 1000, tz=timezone.utc)
            if run.get("start_time")
            else None
        )
        ended = (
            datetime.fromtimestamp(run["end_time"] / 1000, tz=timezone.utc)
            if run.get("end_time")
            else None
        )
        tasks = run.get("tasks") or []
        rows.append(
            {
                "run_id": run["run_id"],
                "job_id": run["job_id"],
                "job_name": run.get("run_name"),
                "date_key": date_key_of(started),
                "started_at": ts(started),
                "ended_at": ts(ended),
                "duration_seconds": (
                    int((ended - started).total_seconds()) if started and ended else None
                ),
                "result_state": (run.get("state") or {}).get("result_state"),
                "task_count": len(tasks),
                "failed_task_count": sum(
                    1 for task in tasks if (task.get("state") or {}).get("result_state") == "FAILED"
                ),
            }
        )
    rows.sort(key=lambda row: row["run_id"])
    return rows


def build_fact_model_prediction(up: Upstream) -> list[dict[str, Any]]:
    rows = []
    for prediction in up.predictions:
        predicted_at = parse_dt(prediction.get("predicted_at"))
        rows.append(
            {
                "trigger_id": prediction["trigger_id"],
                "date_key": date_key_of(predicted_at),
                "model_name": prediction["model_name"],
                "model_version": prediction["model_version"],
                "mlflow_run_id": prediction["mlflow_run_id"],
                "actual_failure_mode": prediction["actual_failure_mode"],
                "predicted_failure_mode": prediction["predicted_failure_mode"],
                "confidence": prediction["confidence"],
                "is_correct": prediction["is_correct"],
                "predicted_at": ts(predicted_at),
            }
        )
    rows.sort(key=lambda row: row["predicted_at"] or "")
    return rows


def build_dim_data_source(
    data: dict[str, list[dict[str, Any]]], loaded_at: str
) -> list[dict[str, Any]]:
    """One row per upstream tool, whether or not it landed rows this run."""
    entries = [
        (
            "test_mgmt_postgres",
            "Amazon RDS for PostgreSQL",
            "DMS change data capture",
            "raw.rides, raw.executions, raw.scenarios",
            "continuous",
            True,
            len(data["raw.rides"]) + len(data["raw.executions"]) + len(data["raw.scenarios"]),
        ),
        (
            "tec_artifacts_s3",
            "Amazon S3",
            "COPY from inventory manifest",
            "raw.s3_artifacts, spectrum_sotf.artifact_objects",
            "every 6 hours",
            True,
            len(data["raw.s3_artifacts"]),
        ),
        (
            "ctm_logs_clickhouse",
            "ClickHouse",
            "daily rollup export",
            "raw.ctm_log_daily",
            "daily 01:30",
            False,
            len(data["raw.ctm_log_daily"]),
        ),
        (
            "testrail",
            "TestRail",
            "REST API pull",
            "raw.testrail_results",
            "daily 02:00",
            True,
            len(data["raw.testrail_results"]),
        ),
        (
            "jira_zpt",
            "Jira",
            "REST API pull",
            "raw.jira_defects",
            "daily 02:15",
            True,
            len(data["raw.jira_defects"]),
        ),
        (
            "airflow",
            "Apache Airflow",
            "metadata database export",
            "ops.fact_airflow_dag_run, ops.fact_airflow_task_instance",
            "daily 03:30",
            False,
            len(data["ops.fact_airflow_dag_run"]) + len(data["ops.fact_airflow_task_instance"]),
        ),
        (
            "databricks",
            "Databricks",
            "UNLOAD from the lakehouse",
            "ops.fact_databricks_job_run, ops.fact_model_prediction",
            "daily 05:00",
            False,
            len(data["ops.fact_databricks_job_run"]) + len(data["ops.fact_model_prediction"]),
        ),
    ]
    return [
        {
            "source_key": index,
            "source_name": name,
            "source_system": system,
            "ingestion_method": method,
            "target_table": target,
            "refresh_cadence": cadence,
            "is_authoritative": authoritative,
            "rows_last_load": rows,
            "last_loaded_at": loaded_at,
        }
        for index, (name, system, method, target, cadence, authoritative, rows) in enumerate(
            entries, start=1
        )
    ]


def build_fact_pipeline_load(
    rng: random.Random,
    data: dict[str, list[dict[str, Any]]],
    load_start: datetime,
) -> list[dict[str, Any]]:
    """One COPY per table. rows_loaded must equal what the CSV ends up holding."""
    sequence = Sequence(880000)
    cursor = load_start
    rows = []
    for table in TABLES:
        # This table describes its own load too, and there is exactly one load
        # per table, so its final row count is known before the rows exist.
        loaded = len(TABLES) if table.name == "fact_pipeline_load" else len(data[table.full_name])
        # A load is a few seconds plus a slice proportional to the row count.
        duration = round(1.8 + loaded / 900.0 + rng.uniform(0.0, 1.2), 2)
        finished = cursor + timedelta(seconds=duration)
        rows.append(
            {
                "load_id": sequence.next(),
                "target_table": table.full_name,
                "source_uri": table.s3_uri,
                "source_system": table.source,
                "started_at": ts(cursor),
                "finished_at": ts(finished),
                "duration_seconds": duration,
                "rows_loaded": loaded,
                "files_scanned": 1,
                "load_status": "COMPLETED",
                "error_count": 0,
            }
        )
        cursor = finished + timedelta(seconds=rng.uniform(0.3, 2.0))
    return rows


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


def render_create_table(table: RTable) -> str:
    parts: list[str] = []
    width = max(len(c.name) for c in table.columns) + 2
    type_width = max(len(c.type) for c in table.columns) + 2

    for column in table.columns:
        piece = f"    {column.name.ljust(width)}{column.type.ljust(type_width)}"
        piece += f"ENCODE {column.encode}"
        if not column.nullable:
            piece += " NOT NULL"
        parts.append(piece)

    if table.primary_key:
        parts.append(f"    PRIMARY KEY ({', '.join(table.primary_key)})")
    for column, referenced, referenced_column in table.foreign_keys:
        parts.append(
            f"    FOREIGN KEY ({column}) REFERENCES {referenced} ({referenced_column})"
        )

    lines = [f"CREATE TABLE IF NOT EXISTS {table.full_name} (", ",\n".join(parts), ")"]
    if table.diststyle == "key" and table.distkey:
        lines.append(f"DISTSTYLE KEY\nDISTKEY ({table.distkey})")
    else:
        lines.append(f"DISTSTYLE {table.diststyle.upper()}")
    if table.sortkey:
        lines.append(f"COMPOUND SORTKEY ({', '.join(table.sortkey)})")
    body = "\n".join(lines) + ";"

    comment = table.comment.replace("'", "''")
    body += f"\nCOMMENT ON TABLE {table.full_name} IS '{comment}';"
    for column in table.columns:
        if column.comment:
            escaped = column.comment.replace("'", "''")
            body += f"\nCOMMENT ON COLUMN {table.full_name}.{column.name} IS '{escaped}';"
    return body


SCHEMA_HEADER = f"""-- SOTF reporting warehouse: schemas and groups.
--
-- Redshift accepts PRIMARY KEY and FOREIGN KEY declarations but does not enforce
-- them; they are here because the query planner uses them. The data is kept
-- consistent by the generator, and the self-check verifies it.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

COMMENT ON SCHEMA raw  IS 'Landing tables, one per upstream system, loaded by COPY.';
COMMENT ON SCHEMA mart IS 'Conformed dimensions, facts and daily aggregates.';
COMMENT ON SCHEMA ops  IS 'Pipeline observability across Airflow, Databricks and the loaders.';

CREATE GROUP {READ_GROUP};
CREATE GROUP {ETL_GROUP};
"""


def external_schema_sql(up: Upstream) -> str:
    return f"""-- Redshift Spectrum: the artifact bundles stay in S3 and are queried in place.
--
-- Unpacking every zip into the warehouse would duplicate terabytes for no
-- benefit, so only the object inventory is external. The zip interior
-- (snapshots/, video/, logs/, summary.json) is not queryable from here.

CREATE EXTERNAL SCHEMA IF NOT EXISTS {EXTERNAL_SCHEMA}
FROM DATA CATALOG
DATABASE '{GLUE_DATABASE}'
REGION '{REGION}'
IAM_ROLE '{IAM_ROLE}'
CREATE EXTERNAL DATABASE IF NOT EXISTS;

CREATE EXTERNAL TABLE {EXTERNAL_SCHEMA}.artifact_objects (
    trigger_id     VARCHAR(36),
    s3_key         VARCHAR(512),
    size_bytes     BIGINT,
    member_count   SMALLINT,
    last_modified  TIMESTAMP
)
PARTITIONED BY (ingest_date DATE)
STORED AS PARQUET
LOCATION 's3://{up.bucket}/{up.prefix}/_inventory/';
"""


MATERIALIZED_VIEWS_SQL = """-- Materialized views the BI layer reads. AUTO REFRESH keeps them within a few
-- minutes of the aggregates without a scheduled job.

CREATE MATERIALIZED VIEW mart.mv_env_pass_rate_daily
AUTO REFRESH YES
AS
SELECT metric_date,
       environment,
       sum(rides)  AS rides,
       sum(passed) AS passed,
       sum(failed) AS failed,
       round(sum(passed)::decimal(18, 4) / nullif(sum(rides), 0), 4) AS pass_rate
  FROM mart.agg_daily_execution_metrics
 GROUP BY metric_date, environment;

CREATE MATERIALIZED VIEW mart.mv_scenario_failures_last_7d
AUTO REFRESH YES
AS
SELECT s.scenario_name,
       f.failure_mode,
       f.failure_component,
       count(*)                        AS failures,
       count(DISTINCT f.jira_key)      AS defects,
       max(f.ended_at)                 AS last_failure_at
  FROM mart.fact_ride f
  JOIN mart.dim_scenario s ON s.scenario_key = f.scenario_key
  JOIN mart.dim_date d     ON d.date_key = f.date_key
 WHERE f.is_failed
   AND d.calendar_date >= dateadd(day, -7, current_date)
 GROUP BY s.scenario_name, f.failure_mode, f.failure_component;
"""


GRANTS_SQL = f"""-- Access control.
--
-- The assistant reads through {READ_GROUP}, which has SELECT and nothing else:
-- no INSERT, UPDATE, DELETE, TRUNCATE or CREATE anywhere. Both users
-- authenticate with IAM, so no password is stored, set or returned here.

CREATE USER sotf_assistant PASSWORD DISABLE IN GROUP {READ_GROUP};
CREATE USER sotf_loader    PASSWORD DISABLE IN GROUP {ETL_GROUP};

GRANT USAGE ON SCHEMA raw, mart, ops TO GROUP {READ_GROUP};
GRANT SELECT ON ALL TABLES IN SCHEMA raw, mart, ops TO GROUP {READ_GROUP};
ALTER DEFAULT PRIVILEGES IN SCHEMA raw, mart, ops
    GRANT SELECT ON TABLES TO GROUP {READ_GROUP};

GRANT USAGE ON SCHEMA {EXTERNAL_SCHEMA} TO GROUP {READ_GROUP};

GRANT ALL ON SCHEMA raw, mart, ops TO GROUP {ETL_GROUP};

-- What an assistant query is allowed to cost lives in the WLM queue rules, not
-- here; see system/wlm_config.json.
"""


SAMPLE_QUERIES_SQL = """-- Queries a read-only SOTF assistant would run against the warehouse.

-- 1. Pass rate per environment per day. This is the question the aggregates exist for.
SELECT metric_date, environment, sum(rides) AS rides,
       round(sum(passed)::decimal(18,4) / nullif(sum(rides), 0), 4) AS pass_rate
  FROM mart.agg_daily_execution_metrics
 WHERE metric_date >= dateadd(day, -14, current_date)
 GROUP BY metric_date, environment
 ORDER BY metric_date DESC, environment;

-- 2. Why did scenario X fail at PUDO Y last week: the whole join in one query.
SELECT f.trigger_id, s.scenario_name, z.pudo_zone, e.environment,
       f.failure_mode, f.failure_component, f.failed_step_id,
       f.jira_key, f.testrail_result_id, f.log_row_count, f.ended_at
  FROM mart.fact_ride f
  JOIN mart.dim_scenario    s ON s.scenario_key    = f.scenario_key
  JOIN mart.dim_pudo_zone   z ON z.pudo_zone_key   = f.pudo_zone_key
  JOIN mart.dim_environment e ON e.environment_key = f.environment_key
  JOIN mart.dim_date        d ON d.date_key        = f.date_key
 WHERE f.is_failed
   AND s.scenario_name = 'Rider unlocks vehicle at pickup'
   AND z.pudo_zone = 'LV-Strip-South'
   AND d.calendar_date >= dateadd(day, -7, current_date)
 ORDER BY f.ended_at DESC;

-- 3. Is this a known issue: the defect the failure mode usually produces.
SELECT d.issue_key, d.issue_status, d.occurrences, d.linked_ride_count, d.summary
  FROM mart.fact_defect d
 WHERE d.failure_mode = 'pickup_geofence_mismatch'
 ORDER BY d.occurrences DESC;

-- 4. Flaky versus consistently broken. A high flip rate with a middling pass
--    rate is flaky; a low flip rate with a low pass rate is just broken.
SELECT scenario_name, total_runs, pass_rate, flakiness_score,
       consecutive_failures, top_failure_mode
  FROM mart.agg_scenario_health
 WHERE total_runs >= 4
 ORDER BY flakiness_score DESC, pass_rate ASC;

-- 5. Did the pipeline actually run: Airflow and Databricks side by side.
SELECT d.calendar_date,
       sum(CASE WHEN a.state = 'failed'  THEN 1 ELSE 0 END) AS airflow_failed,
       sum(CASE WHEN a.state = 'success' THEN 1 ELSE 0 END) AS airflow_ok
  FROM ops.fact_airflow_dag_run a
  JOIN mart.dim_date d ON d.date_key = a.date_key
 GROUP BY d.calendar_date
 ORDER BY d.calendar_date DESC
 LIMIT 14;

SELECT job_name, result_state, count(*) AS runs,
       round(avg(duration_seconds), 1) AS avg_seconds
  FROM ops.fact_databricks_job_run
 GROUP BY job_name, result_state
 ORDER BY job_name, result_state;

-- 6. Can I trust the model on this ride? Its confidence next to the accuracy the
--    model actually achieves on that failure mode.
SELECT p.trigger_id, p.predicted_failure_mode, p.confidence,
       m.scored, m.correct, round(m.correct::decimal(18,4) / m.scored, 4) AS mode_accuracy
  FROM ops.fact_model_prediction p
  JOIN (SELECT actual_failure_mode,
               count(*) AS scored,
               sum(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct
          FROM ops.fact_model_prediction
         GROUP BY actual_failure_mode) m
    ON m.actual_failure_mode = p.predicted_failure_mode
 WHERE p.trigger_id = '00000000-0000-0000-0000-000000000000';

-- 7. Is the data current? One row per upstream tool.
SELECT source_name, source_system, target_table, refresh_cadence,
       rows_last_load, last_loaded_at,
       datediff(hour, last_loaded_at, getdate()) AS hours_stale
  FROM ops.dim_data_source
 ORDER BY hours_stale DESC;

-- 8. Which loads are getting slower.
SELECT target_table, count(*) AS loads,
       round(avg(duration_seconds), 2) AS avg_seconds,
       max(rows_loaded) AS max_rows
  FROM ops.fact_pipeline_load
 GROUP BY target_table
 ORDER BY avg_seconds DESC;

-- 9. Log volume by severity, from the ClickHouse rollup. The log lines
--    themselves are not in Redshift: query ClickHouse for those.
SELECT log_date, severity, sum(row_count) AS rows, sum(distinct_triggers) AS triggers
  FROM raw.ctm_log_daily
 WHERE log_date >= dateadd(day, -7, current_date)
 GROUP BY log_date, severity
 ORDER BY log_date DESC, severity;

-- 10. Artifact coverage: completed rides whose bundle never landed.
SELECT f.trigger_id, s.scenario_name, f.ride_status, f.ended_at
  FROM mart.fact_ride f
  JOIN mart.dim_scenario s ON s.scenario_key = f.scenario_key
  LEFT JOIN raw.s3_artifacts a ON a.trigger_id = f.trigger_id
 WHERE f.is_completed AND a.trigger_id IS NULL
 ORDER BY f.ended_at DESC;
"""


def emit_ddl(out: Path, up: Upstream) -> None:
    write_text(out / "ddl/001_schemas_and_groups.sql", SCHEMA_HEADER)

    for schema, filename in (
        ("raw", "002_raw.sql"),
        ("mart", "003_mart.sql"),
        ("ops", "004_ops.sql"),
    ):
        body = [f"-- SOTF reporting warehouse: {schema} tables.", ""]
        for table in TABLES:
            if table.schema != schema:
                continue
            body.append(render_create_table(table))
            body.append("")
        write_text(out / f"ddl/{filename}", "\n".join(body))

    write_text(out / "ddl/005_external_spectrum.sql", external_schema_sql(up))
    write_text(out / "ddl/006_materialized_views.sql", MATERIALIZED_VIEWS_SQL)
    write_text(out / "ddl/007_grants.sql", GRANTS_SQL)


def emit_copy_commands(out: Path) -> None:
    lines = [
        "-- COPY commands, in dependency order.",
        "--",
        "-- Authentication is the cluster's IAM role. No ACCESS_KEY_ID or",
        "-- SECRET_ACCESS_KEY appears here, and the self-check fails if one ever does.",
        "-- In CSV format Redshift reads an unquoted empty field as NULL, which is how",
        "-- the generator writes nulls; no NULL AS clause is needed.",
        "",
    ]
    for table in TABLES:
        lines.append(
            f"COPY {table.full_name} ({', '.join(table.column_names)})\n"
            f"FROM '{table.s3_uri}'\n"
            f"IAM_ROLE '{IAM_ROLE}'\n"
            "FORMAT AS CSV\n"
            "IGNOREHEADER 1\n"
            f"REGION '{REGION}';\n"
        )
    lines.append("ANALYZE;")
    lines.append("")
    write_text(out / "copy/copy_commands.sql", "\n".join(lines))

    unload = [
        "-- UNLOAD examples for handing a result set to another tool.",
        "-- Same rule: the IAM role authenticates, never a key pair.",
        "",
        f"UNLOAD ('SELECT * FROM mart.agg_daily_execution_metrics')\n"
        f"TO 's3://{STAGE_BUCKET}/{STAGE_PREFIX}/exports/daily_execution_metrics_'\n"
        f"IAM_ROLE '{IAM_ROLE}'\n"
        "FORMAT AS PARQUET\n"
        "ALLOWOVERWRITE;\n",
        f"UNLOAD ('SELECT * FROM mart.agg_scenario_health ORDER BY flakiness_score DESC')\n"
        f"TO 's3://{STAGE_BUCKET}/{STAGE_PREFIX}/exports/scenario_health_'\n"
        f"IAM_ROLE '{IAM_ROLE}'\n"
        "FORMAT AS CSV\n"
        "HEADER\n"
        "ALLOWOVERWRITE;\n",
    ]
    write_text(out / "unload/unload_commands.sql", "\n".join(unload))


# ---------------------------------------------------------------------------
# Data emitters
# ---------------------------------------------------------------------------


def csv_cell(value: Any) -> str:
    """Redshift CSV: an unquoted empty field is NULL."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def emit_csv(out: Path, table: RTable, rows: list[dict[str, Any]]) -> None:
    path = out / table.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(table.column_names)
        for row in rows:
            writer.writerow([csv_cell(row.get(name)) for name in table.column_names])


def emit_system_tables(
    out: Path,
    rng: random.Random,
    up: Upstream,
    data: dict[str, list[dict[str, Any]]],
    loads: list[dict[str, Any]],
    query_result_rows: dict[str, int],
) -> None:
    # svv_table_info: tbl_rows must be the number of rows actually written.
    table_info = []
    for table in TABLES:
        rows = data[table.full_name]
        width = sum(
            64 if column.type.startswith("VARCHAR") else 8 for column in table.columns
        )
        size_mb = max(1, round(len(rows) * width / 1024 / 1024) + len(table.columns))
        table_info.append(
            {
                "schema": table.schema,
                "table": table.name,
                "diststyle": (
                    f"KEY({table.distkey})" if table.diststyle == "key" else table.diststyle.upper()
                ),
                "sortkey1": table.sortkey[0] if table.sortkey else None,
                "sortkey_num": len(table.sortkey),
                "tbl_rows": len(rows),
                "size": size_mb,
                "pct_used": round(size_mb / 4096 * 100, 4),
                "unsorted": 0.0,
                "stats_off": 0.0,
                "skew_rows": round(rng.uniform(1.0, 1.4), 2) if table.diststyle == "key" else None,
                "encoded": "Y",
            }
        )
    write_json(out / "system/svv_table_info.json", table_info)

    # stl_load_commits: one row per COPY, lines_scanned includes the header.
    commits = []
    for load in loads:
        table = TABLE_BY_NAME[load["target_table"]]
        commits.append(
            {
                "query": load["load_id"],
                "slice": 0,
                "name": load["source_uri"],
                "lines_scanned": load["rows_loaded"] + 1,
                "rows_inserted": load["rows_loaded"],
                "errors": load["error_count"],
                "curtime": load["finished_at"],
                "status": 1,
                "table": table.full_name,
            }
        )
    write_json(out / "system/stl_load_commits.json", commits)

    # sys_query_history: returned_rows is what the query would actually return.
    history = []
    query_id = Sequence(4210000)
    base = up.now - timedelta(hours=len(SAVED_QUERIES) * 2)
    for index, query in enumerate(SAVED_QUERIES):
        returned = query_result_rows.get(query.result_rows_from, 0)
        started = base + timedelta(hours=index * 2, minutes=rng.randint(0, 50))
        # A result cache hit does no work, so it has to look like it did none.
        cached = index % 4 == 3
        elapsed_ms = (
            rng.randint(6, 30)
            if cached
            else int(180 + returned * rng.uniform(1.5, 9.0) + rng.uniform(40, 600))
        )
        history.append(
            {
                "query_id": query_id.next(),
                "query_label": query.name,
                "user_id": 108,
                "database_name": DATABASE,
                "query_type": "SELECT",
                "status": "success",
                "start_time": ts(started),
                "end_time": ts(started + timedelta(milliseconds=elapsed_ms)),
                "elapsed_time": elapsed_ms * 1000,
                "queue_time": 0 if cached else rng.randint(0, 90000),
                "execution_time": 0 if cached else elapsed_ms * 900,
                "returned_rows": returned,
                "returned_bytes": returned * rng.randint(60, 400),
                "query_text": query.sql,
                "result_cache_hit": cached,
            }
        )
    write_json(
        out / "system/sys_query_history.json",
        {
            "note": (
                "returned_rows is the row count these queries actually produce against "
                "the data in data/, not an invented number."
            ),
            "queries": history,
        },
    )

    biggest_fact = max(TABLES, key=lambda t: len(data[t.full_name]))
    write_json(
        out / "system/stl_alert_event_log.json",
        [
            {
                "query": history[0]["query_id"] if history else 4210001,
                "event": "Missing query planner statistics",
                "solution": f"Run the ANALYZE command on {biggest_fact.full_name}",
                "event_time": ts(up.now - timedelta(hours=19)),
            },
            {
                "query": history[1]["query_id"] if len(history) > 1 else 4210002,
                "event": "Nested Loop Join in the query plan",
                "solution": "Review the join predicates to avoid Cartesian products",
                "event_time": ts(up.now - timedelta(hours=11)),
            },
            {
                "query": history[2]["query_id"] if len(history) > 2 else 4210003,
                "event": "Scanned a large number of deleted rows",
                "solution": "Run the VACUUM command to reclaim deleted rows",
                "event_time": ts(up.now - timedelta(hours=5)),
            },
        ],
    )

    write_json(
        out / "system/svv_external_tables.json",
        [
            {
                "schemaname": EXTERNAL_SCHEMA,
                "tablename": "artifact_objects",
                "location": f"s3://{up.bucket}/{up.prefix}/_inventory/",
                "input_format": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "serialization_lib": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                "parameters": "{'classification'='parquet'}",
            }
        ],
    )

    write_json(
        out / "system/wlm_config.json",
        {
            "workload_management": "auto",
            "queues": [
                {
                    "name": "etl",
                    "user_group": ETL_GROUP,
                    "query_priority": "high",
                    "concurrency_scaling": "off",
                    "rules": [
                        {"rule_name": "etl_long_running", "predicate": "query_execution_time > 3600", "action": "abort"}
                    ],
                },
                {
                    "name": "assistant",
                    "user_group": READ_GROUP,
                    "query_priority": "normal",
                    "concurrency_scaling": "auto",
                    "rules": [
                        {"rule_name": "assistant_scan_cap", "predicate": "scan_row_count > 50000000", "action": "abort"},
                        {"rule_name": "assistant_timeout", "predicate": "query_execution_time > 120", "action": "abort"},
                    ],
                },
                {
                    "name": "default",
                    "user_group": None,
                    "query_priority": "normal",
                    "concurrency_scaling": "off",
                    "rules": [],
                },
            ],
            "note": (
                "The assistant queue aborts anything scanning more than 50M rows or "
                "running longer than 120s, which is the latency guard the chatbot plan "
                "calls for."
            ),
        },
    )


def emit_cluster(out: Path, up: Upstream) -> None:
    write_json(
        out / "cluster.json",
        {
            "Clusters": [
                {
                    "ClusterIdentifier": CLUSTER_ID,
                    "NodeType": "ra3.4xlarge",
                    "ClusterStatus": "available",
                    "ClusterAvailabilityStatus": "Available",
                    "DBName": DATABASE,
                    "Endpoint": {
                        "Address": f"{CLUSTER_ID}.abc123xyz789.{REGION}.redshift.amazonaws.com",
                        "Port": 5439,
                    },
                    "ClusterCreateTime": iso(up.now - timedelta(days=612)),
                    "MasterUsername": "sotf_admin",
                    "NumberOfNodes": 4,
                    "VpcId": "vpc-04ad2e91c7b3f6a10",
                    "PubliclyAccessible": False,
                    "Encrypted": True,
                    "KmsKeyId": (
                        f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/"
                        "3c7a19be-52d4-4a08-9d61-0b8e5f2c47aa"
                    ),
                    "EnhancedVpcRouting": True,
                    "AvailabilityZone": f"{REGION}b",
                    "PreferredMaintenanceWindow": "sat:12:30-sat:13:00",
                    "AutomatedSnapshotRetentionPeriod": 7,
                    "ManualSnapshotRetentionPeriod": 30,
                    "IamRoles": [{"IamRoleArn": IAM_ROLE, "ApplyStatus": "in-sync"}],
                    "DefaultIamRoleArn": IAM_ROLE,
                    "Tags": [
                        {"Key": "service", "Value": "sotf-analytics"},
                        {"Key": "owner", "Value": "sotf-data"},
                    ],
                }
            ],
            "note": (
                "describe-clusters never returns a password. Both database users "
                "authenticate through IAM (PASSWORD DISABLE in ddl/007_grants.sql)."
            ),
        },
    )


def emit_catalog(out: Path, up: Upstream, data: dict[str, list[dict[str, Any]]]) -> None:
    write_json(
        out / "catalog.json",
        {
            "cluster": CLUSTER_ID,
            "database": DATABASE,
            "region": REGION,
            "schemas": list(SCHEMAS) + [EXTERNAL_SCHEMA],
            "stage": f"s3://{STAGE_BUCKET}/{STAGE_PREFIX}/",
            "note": (
                "One CSV per table under data/<schema>/. Nulls are unquoted empty "
                "fields, which is how Redshift COPY reads NULL in CSV format. Load "
                "order is the order below; Redshift does not enforce the declared "
                "foreign keys, but the data satisfies them."
            ),
            "upstream": {
                "sotf": True,
                "airflow": up.has_airflow,
                "databricks": up.has_databricks,
                "postgres": up.has_postgres,
            },
            "tables": [
                {
                    "name": table.full_name,
                    "comment": table.comment,
                    "source": table.source,
                    "rows": len(data[table.full_name]),
                    "csv": table.relative_path,
                    "s3_uri": table.s3_uri,
                    "diststyle": table.diststyle,
                    "distkey": table.distkey,
                    "sortkey": list(table.sortkey),
                    "primary_key": list(table.primary_key),
                    "columns": [
                        {
                            "name": column.name,
                            "type": column.type,
                            "encode": column.encode,
                            "nullable": column.nullable,
                            "comment": column.comment,
                        }
                        for column in table.columns
                    ],
                }
                for table in TABLES
            ],
            "external_tables": [
                {
                    "name": f"{EXTERNAL_SCHEMA}.artifact_objects",
                    "location": f"s3://{up.bucket}/{up.prefix}/_inventory/",
                    "note": "Spectrum over the S3 inventory; the zip contents stay in S3.",
                }
            ],
            "saved_queries": [
                {"name": query.name, "sql": query.sql} for query in SAVED_QUERIES
            ],
            "notes": [
                "Read-only fixtures. COPY and UNLOAD authenticate with an IAM role; no "
                "key pair appears anywhere.",
                "mart.agg_daily_execution_metrics is computed the same way "
                "sotf.gold.daily_execution_metrics is in the Databricks dataset, and the "
                "self-check fails if they disagree.",
                "Redshift has no array type: list columns are pipe delimited.",
            ],
        },
    )


def emit_cross_reference(
    out: Path, up: Upstream, data: dict[str, list[dict[str, Any]]]
) -> None:
    predictions = {row["trigger_id"]: row for row in data["ops.fact_model_prediction"]}
    artifacts = {row["trigger_id"] for row in data["raw.s3_artifacts"]}
    results = {row["trigger_id"]: row["result_id"] for row in data["mart.fact_test_result"]}

    # Which Airflow DAG run covered a ride is already recorded in the Airflow
    # dataset; here we only record whether the warehouse saw the run at all.
    dag_run_states = {row["dag_run_id"]: row["state"] for row in data["ops.fact_airflow_dag_run"]}
    airflow_reference = {
        entry["trigger_id"]: entry.get("airflow", {})
        for entry in load_airflow_cross_reference(up)
    }

    rides = []
    for ride in data["mart.fact_ride"]:
        trigger_id = ride["trigger_id"]
        covering = airflow_reference.get(trigger_id, {})
        rides.append(
            {
                "trigger_id": trigger_id,
                "ride_status": ride["ride_status"],
                "date_key": ride["date_key"],
                "scenario_key": ride["scenario_key"],
                "environment_key": ride["environment_key"],
                "pudo_zone_key": ride["pudo_zone_key"],
                "in_fact_ride": True,
                "testrail_result_id": results.get(trigger_id),
                "jira_key": ride["jira_key"],
                "has_artifact_row": trigger_id in artifacts,
                "prediction": (
                    {
                        "predicted_failure_mode": predictions[trigger_id]["predicted_failure_mode"],
                        "actual_failure_mode": predictions[trigger_id]["actual_failure_mode"],
                        "confidence": predictions[trigger_id]["confidence"],
                        "is_correct": predictions[trigger_id]["is_correct"],
                    }
                    if trigger_id in predictions
                    else None
                ),
                "airflow_dag_runs": {
                    dag_id: {"dag_run_id": run_id, "state": dag_run_states.get(run_id)}
                    for dag_id, run_id in covering.items()
                    if run_id
                },
            }
        )

    write_json(
        out / "cross_reference_redshift.json",
        {
            "note": (
                "Where each SOTF ride lands in the warehouse, and which Airflow DAG runs "
                "claimed it upstream. A named DAG run may have failed: its state is "
                "included so a ride that looks processed but was not is visible. Only "
                "failed rides carry a prediction, and a prediction may be wrong on "
                "purpose -- is_correct records whether the model got it right."
            ),
            "row_counts": {name: len(rows) for name, rows in data.items()},
            "rides": rides,
        },
    )


def load_airflow_cross_reference(up: Upstream) -> list[dict[str, Any]]:
    """Read the Airflow join table if the Airflow dataset was generated."""
    if not up.has_airflow:
        return []
    path = up.airflow_root / "cross_reference_airflow.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("rides", [])


LOADER_TEMPLATE = """#!/usr/bin/env bash
# Stage the generated CSVs and load them into a throwaway Redshift cluster.
# Point this at a scratch cluster or a Redshift Serverless workgroup -- never at
# a real warehouse.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

: "${{STAGE_BUCKET:={stage_bucket}}}"
: "${{STAGE_PREFIX:={stage_prefix}}}"
: "${{CLUSTER_ID:={cluster_id}}}"
: "${{DATABASE:={database}}}"
: "${{DB_USER:=sotf_loader}}"

echo "==> staging CSVs to s3://${{STAGE_BUCKET}}/${{STAGE_PREFIX}}/"
aws s3 sync "${{HERE}}/data/" "s3://${{STAGE_BUCKET}}/${{STAGE_PREFIX}}/" --no-progress

run_sql() {{
  aws redshift-data execute-statement \\
    --cluster-identifier "${{CLUSTER_ID}}" \\
    --database "${{DATABASE}}" \\
    --db-user "${{DB_USER}}" \\
    --sql "file://$1" \\
    --no-cli-pager > /dev/null
}}

echo "==> applying DDL"
for file in "${{HERE}}"/ddl/*.sql; do
  echo "    ${{file##*/}}"
  run_sql "${{file}}"
done

echo "==> running COPY"
run_sql "${{HERE}}/copy/copy_commands.sql"

echo "COPY statements authenticate with the cluster's IAM role; the role must be"
echo "able to read s3://${{STAGE_BUCKET}}/${{STAGE_PREFIX}}/."
"""


def emit_loader(out: Path) -> None:
    path = out / "load_redshift.sh"
    write_text(
        path,
        LOADER_TEMPLATE.format(
            stage_bucket=STAGE_BUCKET,
            stage_prefix=STAGE_PREFIX,
            cluster_id=CLUSTER_ID,
            database=DATABASE,
        ),
    )
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------


def run_self_check(
    out: Path,
    up: Upstream,
    data: dict[str, list[dict[str, Any]]],
    loads: list[dict[str, Any]],
    query_result_rows: dict[str, int],
) -> list[str]:
    problems: list[str] = []

    # Redshift restricts which encodings a column type may use, and rejects the
    # CREATE TABLE outright when the pairing is illegal.
    for table in TABLES:
        names = set(table.column_names)
        for column in table.columns:
            if column.type == "BOOLEAN" and column.encode not in ("raw", "runlength"):
                problems.append(
                    f"{table.full_name}.{column.name}: BOOLEAN cannot use {column.encode}"
                )
            if column.type.startswith("VARCHAR") and column.encode == "az64":
                problems.append(f"{table.full_name}.{column.name}: VARCHAR cannot use az64")
        for key in (table.distkey, *table.sortkey, *table.primary_key):
            if key is not None and key not in names:
                problems.append(f"{table.full_name}: key column {key} does not exist")

    # CSV files match the built rows.
    for table in TABLES:
        path = out / table.relative_path
        if not path.is_file():
            problems.append(f"missing {table.relative_path}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            written = sum(1 for _ in reader)
        if header != table.column_names:
            problems.append(f"{table.relative_path} header does not match the table definition")
        if written != len(data[table.full_name]):
            problems.append(
                f"{table.relative_path} has {written} rows, expected {len(data[table.full_name])}"
            )

    # Nulls must be unambiguous: an empty string would load as NULL anyway.
    for table in TABLES:
        for row in data[table.full_name]:
            if any(row.get(name) == "" for name in table.column_names):
                problems.append(
                    f"{table.full_name} has an empty string, which COPY would read as NULL"
                )
                break

    # Declared primary keys are unique.
    for table in TABLES:
        if not table.primary_key:
            continue
        keys = [
            tuple(row.get(column) for column in table.primary_key)
            for row in data[table.full_name]
        ]
        if len(set(keys)) != len(keys):
            problems.append(f"{table.full_name} has duplicate primary keys")

    # Declared foreign keys resolve, even though Redshift would not enforce them.
    for table in TABLES:
        for column, referenced, referenced_column in table.foreign_keys:
            targets = {row[referenced_column] for row in data[referenced]}
            missing = {
                row[column]
                for row in data[table.full_name]
                if row.get(column) is not None and row[column] not in targets
            }
            if missing:
                problems.append(
                    f"{table.full_name}.{column}: {len(missing)} value(s) absent from {referenced}"
                )

    # The fact table covers exactly the SOTF rides.
    expected_rides = up.counts.get("rides")
    if expected_rides is not None and len(data["mart.fact_ride"]) != expected_rides:
        problems.append(
            f"mart.fact_ride has {len(data['mart.fact_ride'])} rows, "
            f"SOTF summary says {expected_rides}"
        )

    # The daily aggregate covers exactly the completed rides.
    aggregated = sum(row["rides"] for row in data["mart.agg_daily_execution_metrics"])
    if aggregated != len(up.completed):
        problems.append(
            f"agg_daily_execution_metrics covers {aggregated} rides, expected {len(up.completed)}"
        )
    fact_completed = sum(1 for row in data["mart.fact_ride"] if row["is_completed"])
    if aggregated != fact_completed:
        problems.append(
            f"agg_daily_execution_metrics covers {aggregated} rides but fact_ride has "
            f"{fact_completed} completed"
        )

    # Redshift and Databricks must agree on the numbers they both publish.
    if up.has_databricks and up.databricks_gold.get("daily_execution_metrics"):
        gold = {
            (row["metric_date"], row["environment"], row["ride_service"]): row
            for row in up.databricks_gold["daily_execution_metrics"]
        }
        here = {
            (row["metric_date"], row["environment"], row["ride_service"]): row
            for row in data["mart.agg_daily_execution_metrics"]
        }
        if set(gold) != set(here):
            problems.append(
                "agg_daily_execution_metrics and sotf.gold.daily_execution_metrics cover "
                "different day/environment/service combinations"
            )
        for key in set(gold) & set(here):
            for field in ("rides", "passed", "failed", "pass_rate"):
                if gold[key][field] != here[key][field]:
                    problems.append(
                        f"{key}: {field} is {here[key][field]} in Redshift but "
                        f"{gold[key][field]} in Databricks"
                    )
                    break

    if up.has_databricks and up.databricks_gold.get("scenario_health"):
        gold_health = {row["scenario_name"]: row for row in up.databricks_gold["scenario_health"]}
        for row in data["mart.agg_scenario_health"]:
            other = gold_health.get(row["scenario_name"])
            if other is None:
                problems.append(f"{row['scenario_name']} is missing from sotf.gold.scenario_health")
                continue
            if (row["total_runs"], row["pass_rate"], row["flakiness_score"]) != (
                other["total_runs"],
                other["pass_rate"],
                other["flakiness_score"],
            ):
                problems.append(
                    f"{row['scenario_name']}: scenario health disagrees with Databricks"
                )
                break

    # Every ZPT key on a ride must have a defect row, and every defect must have
    # come from a ride that failed.
    defect_keys = {row["issue_key"] for row in data["mart.fact_defect"]}
    ride_keys = {row["jira_key"] for row in data["mart.fact_ride"] if row["jira_key"]}
    if ride_keys - defect_keys:
        problems.append(
            f"{len(ride_keys - defect_keys)} ZPT key(s) on rides have no row in mart.fact_defect"
        )
    if defect_keys - ride_keys:
        problems.append(
            f"{len(defect_keys - ride_keys)} defect(s) are linked to no ride in mart.fact_ride"
        )
    for row in data["mart.fact_defect"]:
        if row["failure_mode"] is None:
            problems.append(f"{row['issue_key']}: defect has no failure mode")
            break

    # Predictions must be real failed rides, and only failed rides.
    failed_ids = {ride["trigger_id"] for ride in up.failed}
    predicted_ids = {row["trigger_id"] for row in data["ops.fact_model_prediction"]}
    if predicted_ids - failed_ids:
        problems.append(
            f"{len(predicted_ids - failed_ids)} prediction(s) reference a ride that did not fail"
        )

    # Pipeline run counts must match the upstream datasets.
    if up.has_airflow:
        expected = up.airflow_counts.get("dag_runs")
        if expected is not None and len(data["ops.fact_airflow_dag_run"]) != expected:
            problems.append(
                f"ops.fact_airflow_dag_run has {len(data['ops.fact_airflow_dag_run'])} rows, "
                f"the Airflow dataset says {expected}"
            )
        expected_tasks = up.airflow_counts.get("task_instances")
        if expected_tasks is not None and len(data["ops.fact_airflow_task_instance"]) != expected_tasks:
            problems.append(
                f"ops.fact_airflow_task_instance has "
                f"{len(data['ops.fact_airflow_task_instance'])} rows, the Airflow dataset "
                f"says {expected_tasks}"
            )
    if up.has_databricks:
        expected = up.databricks_counts.get("job_runs")
        if expected is not None and len(data["ops.fact_databricks_job_run"]) != expected:
            problems.append(
                f"ops.fact_databricks_job_run has {len(data['ops.fact_databricks_job_run'])} "
                f"rows, the Databricks dataset says {expected}"
            )

    # A run marked failed must have a failed task, and vice versa.
    for row in data["ops.fact_airflow_dag_run"]:
        if row["state"] == "failed" and row["failed_task_count"] == 0:
            problems.append(f"{row['dag_run_id']}: marked failed but no task instance failed")
            break
        if row["state"] == "success" and row["failed_task_count"] > 0:
            problems.append(f"{row['dag_run_id']}: marked success but a task instance failed")
            break

    # Load history and svv_table_info must report the rows that exist.
    for load in loads:
        actual = len(data[load["target_table"]])
        if load["rows_loaded"] != actual:
            problems.append(
                f"fact_pipeline_load says {load['rows_loaded']} rows for "
                f"{load['target_table']}, which holds {actual}"
            )
            break

    info_file = out / "system/svv_table_info.json"
    if info_file.is_file():
        for entry in json.loads(info_file.read_text(encoding="utf-8")):
            full_name = f"{entry['schema']}.{entry['table']}"
            if entry["tbl_rows"] != len(data[full_name]):
                problems.append(
                    f"svv_table_info reports {entry['tbl_rows']} rows for {full_name}, "
                    f"which holds {len(data[full_name])}"
                )
                break

    history_file = out / "system/sys_query_history.json"
    if history_file.is_file():
        by_label = {
            entry["query_label"]: entry
            for entry in json.loads(history_file.read_text(encoding="utf-8"))["queries"]
        }
        for query in SAVED_QUERIES:
            entry = by_label.get(query.name)
            expected = query_result_rows.get(query.result_rows_from, 0)
            if entry is None:
                problems.append(f"{query.name} is missing from sys_query_history")
            elif entry["returned_rows"] != expected:
                problems.append(
                    f"{query.name} reports {entry['returned_rows']} rows but returns {expected}"
                )

    # Every table named in a saved query must exist.
    known = {table.full_name for table in TABLES} | {f"{EXTERNAL_SCHEMA}.artifact_objects"}
    for query in SAVED_QUERIES:
        for schema in SCHEMAS:
            for token in query.sql.split():
                cleaned = token.strip("(),;").lower()
                if cleaned.startswith(f"{schema}.") and cleaned not in known:
                    problems.append(f"{query.name} references unknown table {cleaned}")

    # Credentials never appear in the SQL.
    for path in sorted(out.glob("**/*.sql")):
        text = path.read_text(encoding="utf-8").upper()
        for marker in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY", "SESSION_TOKEN", "PASSWORD '"):
            if marker in text:
                problems.append(f"{path.name} contains {marker.strip()}")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mock Amazon Redshift data for the SOTF reporting warehouse, "
            "aligned with the SOTF, Airflow and Databricks mock datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sotf-data", type=Path, default=HERE / "out", help="output of generate_sotf_mock_data.py"
    )
    parser.add_argument(
        "--airflow-data",
        type=Path,
        default=HERE / "out_airflow",
        help="output of generate_airflow_mock_data.py; skipped if absent",
    )
    parser.add_argument(
        "--databricks-data",
        type=Path,
        default=HERE / "out_databricks",
        help="output of generate_databricks_mock_data.py; skipped if absent",
    )
    parser.add_argument(
        "--postgres-data",
        type=Path,
        default=HERE / "out_postgres",
        help="output of generate_postgres_mock_data.py; only used to record lineage",
    )
    parser.add_argument("--out", type=Path, default=HERE / "out_redshift")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    up = load_upstream(args)
    rng = random.Random(args.seed)

    out: Path = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # The nightly load finishes a couple of hours after the last ride.
    load_start = (up.now + timedelta(hours=2)).replace(minute=15, second=0, microsecond=0)
    loaded_at = ts(load_start)

    dim_scenario = build_dim_scenario(up)
    dim_environment = build_dim_environment(up)
    dim_pudo_zone = build_dim_pudo_zone(up)
    dim_date = build_dim_date(up)

    scenario_keys = {row["scenario_name"]: row["scenario_key"] for row in dim_scenario}
    environment_keys = {row["environment"]: row["environment_key"] for row in dim_environment}
    zone_keys = {row["pudo_zone"]: row["pudo_zone_key"] for row in dim_pudo_zone}
    zone_service = {row["pudo_zone"]: row["ride_service"] for row in dim_pudo_zone}

    raw_testrail = build_raw_testrail_results(up, loaded_at)
    fact_ride = build_fact_ride(up, scenario_keys, environment_keys, zone_keys)
    ride_scenario_key = {row["trigger_id"]: row["scenario_key"] for row in fact_ride}

    data: dict[str, list[dict[str, Any]]] = {
        "raw.rides": build_raw_rides(up, loaded_at),
        "raw.executions": build_raw_executions(up, loaded_at),
        "raw.scenarios": build_raw_scenarios(up, dim_scenario, loaded_at),
        "raw.testrail_results": raw_testrail,
        "raw.jira_defects": build_raw_jira_defects(up, loaded_at),
        "raw.ctm_log_daily": build_raw_ctm_log_daily(up, loaded_at),
        "raw.s3_artifacts": build_raw_s3_artifacts(up, loaded_at),
        "mart.dim_date": dim_date,
        "mart.dim_scenario": dim_scenario,
        "mart.dim_environment": dim_environment,
        "mart.dim_pudo_zone": dim_pudo_zone,
        "mart.fact_ride": fact_ride,
        "mart.fact_test_result": build_fact_test_result(raw_testrail, ride_scenario_key),
        "mart.fact_defect": build_fact_defect(up),
        "mart.agg_daily_execution_metrics": build_agg_daily_execution_metrics(up),
        "mart.agg_scenario_health": build_agg_scenario_health(up, scenario_keys),
        "mart.agg_failure_mode_daily": build_agg_failure_mode_daily(up),
        "mart.agg_pudo_zone_reliability": build_agg_pudo_zone_reliability(
            up, zone_keys, zone_service
        ),
        "ops.dim_data_source": [],
        "ops.fact_airflow_dag_run": build_fact_airflow_dag_run(up),
        "ops.fact_airflow_task_instance": build_fact_airflow_task_instance(up),
        "ops.fact_databricks_job_run": build_fact_databricks_job_run(up),
        "ops.fact_model_prediction": build_fact_model_prediction(up),
        "ops.fact_pipeline_load": [],
    }
    data["ops.dim_data_source"] = build_dim_data_source(data, loaded_at)

    loads = build_fact_pipeline_load(rng, data, load_start)
    data["ops.fact_pipeline_load"] = loads

    query_result_rows = compute_query_result_rows(up, data)

    for table in TABLES:
        emit_csv(out, table, data[table.full_name])

    emit_ddl(out, up)
    emit_copy_commands(out)
    emit_system_tables(out, rng, up, data, loads, query_result_rows)
    emit_cluster(out, up)
    emit_catalog(out, up, data)
    emit_cross_reference(out, up, data)
    write_text(out / "sample_queries.sql", SAMPLE_QUERIES_SQL)
    emit_loader(out)

    counts = {
        "schemas": len(SCHEMAS) + 1,
        "tables": len(TABLES),
        "rows": {table.full_name: len(data[table.full_name]) for table in TABLES},
        "total_rows": sum(len(rows) for rows in data.values()),
        "saved_queries": len(SAVED_QUERIES),
        "upstream_present": {
            "sotf": True,
            "airflow": up.has_airflow,
            "databricks": up.has_databricks,
            "postgres": up.has_postgres,
        },
    }
    write_json(
        out / "summary.json",
        {
            "generated_at": iso(datetime.now(timezone.utc)),
            "sotf_dataset": str(args.sotf_data),
            "airflow_dataset": str(args.airflow_data) if up.has_airflow else None,
            "databricks_dataset": str(args.databricks_data) if up.has_databricks else None,
            "parameters": {
                "seed": args.seed,
                "now": iso(up.now),
                "days": up.days,
                "cluster": CLUSTER_ID,
                "database": DATABASE,
                "region": REGION,
            },
            "counts": counts,
        },
    )

    print(f"Wrote Redshift mock data to {out}")
    print(
        "  {tables} tables across {schemas} schemas, {total_rows} rows".format(**counts)
    )
    print(
        "  upstream: sotf=yes airflow={airflow} databricks={databricks} "
        "postgres={postgres}".format(
            airflow="yes" if up.has_airflow else "no",
            databricks="yes" if up.has_databricks else "no",
            postgres="yes" if up.has_postgres else "no",
        )
    )
    if not up.has_airflow:
        print("  note: ops.fact_airflow_* are empty; run generate_airflow_mock_data.py first")
    if not up.has_databricks:
        print("  note: ops.fact_databricks_* are empty; run generate_databricks_mock_data.py first")

    if args.skip_self_check:
        return 0

    problems = run_self_check(out, up, data, loads, query_result_rows)
    if problems:
        print(f"  self-check FAILED with {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    - {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return 1

    print("  self-check passed: keys, aggregates, load history and query history agree")
    print(f"  join table: {out / 'cross_reference_redshift.json'}")
    print(f"  loader:     {out / 'load_redshift.sh'}")
    return 0


def compute_query_result_rows(
    up: Upstream, data: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    """How many rows each saved query returns, computed from the data itself."""
    daily = data["mart.agg_daily_execution_metrics"]
    seven_days_ago = (up.now - timedelta(days=7)).strftime("%Y-%m-%d")

    return {
        "daily_pass_rate_by_environment": len(
            {(row["metric_date"], row["environment"]) for row in daily}
        ),
        "flakiest_scenarios": sum(
            1
            for row in data["mart.agg_scenario_health"]
            if row["total_runs"] >= 4 and row["flakiness_score"] >= 0.3
        ),
        "failure_modes_last_7_days": len(
            {
                (row["failure_mode"], row["failure_component"])
                for row in data["mart.agg_failure_mode_daily"]
                if row["metric_date"] >= seven_days_ago
            }
        ),
        "pudo_zone_reliability": len(data["mart.agg_pudo_zone_reliability"]),
        "pipeline_health_by_day": len(
            {row["date_key"] for row in data["ops.fact_airflow_dag_run"] if row["date_key"]}
        ),
        "model_accuracy_by_failure_mode": len(
            {row["actual_failure_mode"] for row in data["ops.fact_model_prediction"]}
        ),
        "rides_missing_artifacts": sum(
            1
            for row in data["mart.fact_ride"]
            if row["is_completed"] and not row["has_artifact"]
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
