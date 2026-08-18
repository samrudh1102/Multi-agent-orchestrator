#!/usr/bin/env python3
"""Generate mock Databricks data that lines up with the SOTF mock dataset.

This script mocks **Databricks only**. It does not generate S3, ClickHouse, Jira,
TestRail or Airflow data -- it *reads* the dataset produced by
``generate_sotf_mock_data.py`` and builds the lakehouse layer on top of it:

    unity_catalog/   catalogs, schemas, table metadata, lineage
    tables/          the actual gold + ML table contents, as Delta-style JSONL
    jobs/            Jobs API 2.1 job definitions, runs and run output
    clusters/        cluster list and events
    sql/             SQL warehouse, query history, saved queries
    mlflow/          experiments, runs, registered models, model versions
    workspace/       notebook listing and notebook source

The alignment is the point. The gold tables are aggregated from the real rides,
the feature table and the failure-mode classifier's predictions are keyed by real
trigger ids, and the accuracy metric logged in MLflow is the accuracy those
predictions actually have. ``cross_reference_databricks.json`` maps each SOTF ride
to the rows and the job run that produced them.

Usage:
    python3 generate_databricks_mock_data.py --clean
    python3 generate_databricks_mock_data.py --sotf-data ./out --out ./out_databricks
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Workspace constants
# ---------------------------------------------------------------------------

WORKSPACE_URL = "https://sotf.cloud.databricks.com"
METASTORE_ID = "6a1f4e2c-9d38-4a17-8b02-5c7d1e93af40"
CATALOG = "sotf"
SCHEMAS = ("bronze", "silver", "gold", "ml")
NOTEBOOK_ROOT = "/Shared/sotf"
STORAGE_ROOT = "s3://zoox-lakehouse/sotf"

OWNER = "sotf-data@example.com"
USERS = (
    "sotf-data@example.com",
    "sotf-qa@example.com",
    "priya.raman@example.com",
    "daniel.okafor@example.com",
    "mei.tanaka@example.com",
)

SPARK_VERSION = "14.3.x-scala2.12"
NODE_TYPE = "m5d.2xlarge"
DRIVER_NODE_TYPE = "m5d.4xlarge"

MODEL_NAME = "sotf_failure_mode_classifier"
FLAKINESS_MODEL_NAME = "sotf_scenario_flakiness"

CLASSIFIER_RUN_COUNT = 8
FLAKINESS_RUN_COUNT = 5
# Version 7 is the champion; version 8 is the newer candidate still in Staging.
CHAMPION_VERSION = 7
QUERY_HISTORY_ENTRIES = 48

# The SOTF generator owns this taxonomy; the component is repeated here only so
# the trend table can carry it. Unknown keys degrade to "unknown".
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

# Which failure modes a classifier would plausibly confuse: same mission phase or
# same subsystem. Used to make the wrong predictions look like model error rather
# than noise.
CONFUSABLE = {
    "pickup_geofence_mismatch": ("door_unlock_not_acked", "dropoff_geofence_blocked"),
    "milestone_timeout": ("sim_pose_stall", "vehicle_unassigned"),
    "vehicle_unassigned": ("milestone_timeout", "webapi_timeout"),
    "rider_app_crash_on_confirm": ("device_disconnect", "webapi_timeout"),
    "webapi_timeout": ("rider_app_crash_on_confirm", "vehicle_unassigned"),
    "device_disconnect": ("rider_app_crash_on_confirm", "sim_pose_stall"),
    "sim_pose_stall": ("milestone_timeout", "device_disconnect"),
    "door_unlock_not_acked": ("pickup_geofence_mismatch", "milestone_timeout"),
    "dropoff_geofence_blocked": ("pickup_geofence_mismatch", "sim_pose_stall"),
}


# ---------------------------------------------------------------------------
# Table catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    name: str
    type_name: str
    comment: str = ""

    @property
    def type_text(self) -> str:
        return {
            "STRING": "string",
            "LONG": "bigint",
            "INT": "int",
            "DOUBLE": "double",
            "BOOLEAN": "boolean",
            "DATE": "date",
            "TIMESTAMP": "timestamp",
            "ARRAY": "array<string>",
        }[self.type_name]


@dataclass(frozen=True)
class TableSpec:
    schema: str
    name: str
    comment: str
    columns: tuple[Column, ...]
    table_type: str = "MANAGED"
    partitioned_by: tuple[str, ...] = ()
    # Where the row count comes from: a key into the counts dict, or "data" when
    # the table's rows are emitted by this script.
    row_source: str = "data"

    @property
    def full_name(self) -> str:
        return f"{CATALOG}.{self.schema}.{self.name}"

    @property
    def relative_path(self) -> str:
        return f"tables/{self.schema}/{self.name}.jsonl"


def cols(*pairs: tuple[str, str, str]) -> tuple[Column, ...]:
    return tuple(Column(name, type_name, comment) for name, type_name, comment in pairs)


TABLE_SPECS: tuple[TableSpec, ...] = (
    # ---- bronze: landed as-is from the source systems -------------------
    TableSpec(
        schema="bronze",
        name="artifact_manifests",
        comment="Artifacts SQS messages emitted by the TEC device service, landed unchanged.",
        columns=cols(
            ("trigger_id", "STRING", "TEC trigger id, the ride identifier"),
            ("s3_uri", "STRING", "Artifact bundle location"),
            ("ingested_at", "TIMESTAMP", ""),
            ("payload", "STRING", "Raw manifest JSON"),
        ),
        table_type="EXTERNAL",
        row_source="s3_objects",
    ),
    TableSpec(
        schema="bronze",
        name="ctm_logs_raw",
        comment="CTM execution log rows copied out of ClickHouse.",
        columns=cols(
            ("timestamp", "TIMESTAMP", ""),
            ("message", "STRING", ""),
            ("severity", "STRING", ""),
            ("component", "STRING", ""),
            ("trigger_id", "STRING", ""),
            ("ingested_at", "TIMESTAMP", ""),
        ),
        partitioned_by=("ingest_date",),
        row_source="clickhouse_rows",
    ),
    TableSpec(
        schema="bronze",
        name="testrail_results_raw",
        comment="TestRail result payloads pulled from the TestRail API.",
        columns=cols(
            ("result_id", "LONG", ""),
            ("test_id", "LONG", ""),
            ("status_id", "INT", ""),
            ("payload", "STRING", "Raw result JSON including custom_step_results"),
            ("ingested_at", "TIMESTAMP", ""),
        ),
        row_source="testrail_results",
    ),
    TableSpec(
        schema="bronze",
        name="jira_defects_raw",
        comment="ZPT issue payloads pulled from the Jira API.",
        columns=cols(
            ("issue_key", "STRING", ""),
            ("payload", "STRING", "Raw issue JSON including custom fields"),
            ("ingested_at", "TIMESTAMP", ""),
        ),
        row_source="jira_issues",
    ),
    # ---- silver: conformed, one row per business entity ------------------
    TableSpec(
        schema="silver",
        name="rides",
        comment="One row per SOTF ride, conformed across test-mgmt, TestRail and Jira.",
        columns=cols(
            ("trigger_id", "STRING", "Primary key"),
            ("execution_id", "LONG", ""),
            ("scenario_name", "STRING", ""),
            ("scenario_version", "DOUBLE", ""),
            ("ride_status", "STRING", "Passed / Failed / Running / Queued / Invalid"),
            ("environment", "STRING", ""),
            ("ride_service", "STRING", ""),
            ("pudo_zone", "STRING", ""),
            ("pickup_poi", "STRING", ""),
            ("dropoff_poi", "STRING", ""),
            ("started_at", "TIMESTAMP", ""),
            ("ended_at", "TIMESTAMP", ""),
            ("duration_seconds", "DOUBLE", ""),
            ("failure_mode", "STRING", "Null unless ride_status = Failed"),
            ("testrail_result_id", "LONG", ""),
            ("jira_key", "STRING", ""),
            ("s3_key", "STRING", ""),
        ),
        partitioned_by=("ride_date",),
        row_source="rides",
    ),
    TableSpec(
        schema="silver",
        name="executions",
        comment="One row per execution, the parent of a set of rides.",
        columns=cols(
            ("execution_id", "LONG", "Primary key"),
            ("scenario_name", "STRING", ""),
            ("environment", "STRING", ""),
            ("status", "STRING", ""),
            ("created_at", "TIMESTAMP", ""),
            ("testrail_run_id", "LONG", ""),
        ),
        row_source="executions",
    ),
    TableSpec(
        schema="silver",
        name="scenarios",
        comment="Scenario definitions with their TestRail case ids.",
        columns=cols(
            ("scenario_name", "STRING", "Primary key"),
            ("version", "DOUBLE", ""),
            ("testrail_case_id", "LONG", ""),
            ("step_count", "INT", ""),
            ("labels", "ARRAY", ""),
        ),
        row_source="scenarios",
    ),
    TableSpec(
        schema="silver",
        name="ctm_logs",
        comment="Parsed CTM log rows with the severity and component lifted out of the message.",
        columns=cols(
            ("log_timestamp", "TIMESTAMP", ""),
            ("trigger_id", "STRING", ""),
            ("severity", "STRING", ""),
            ("component", "STRING", ""),
            ("event", "STRING", ""),
            ("message", "STRING", ""),
        ),
        partitioned_by=("log_date",),
        row_source="clickhouse_rows",
    ),
    TableSpec(
        schema="silver",
        name="test_results",
        comment="TestRail results joined to the ride that produced them.",
        columns=cols(
            ("result_id", "LONG", "Primary key"),
            ("test_id", "LONG", ""),
            ("run_id", "LONG", ""),
            ("trigger_id", "STRING", ""),
            ("status", "STRING", "PASSED / FAILED / RETEST / BLOCKED / UNTESTED"),
            ("defects", "STRING", "ZPT key, when one was filed"),
            ("created_at", "TIMESTAMP", ""),
        ),
        row_source="testrail_results",
    ),
    TableSpec(
        schema="silver",
        name="defects",
        comment="ZPT defects with their occurrence count and linked rides.",
        columns=cols(
            ("issue_key", "STRING", "Primary key"),
            ("summary", "STRING", ""),
            ("failure_mode", "STRING", ""),
            ("occurrences", "INT", ""),
            ("linked_trigger_ids", "ARRAY", ""),
            ("created_at", "TIMESTAMP", ""),
        ),
        row_source="jira_issues",
    ),
    # ---- gold: the tables this script actually populates ------------------
    TableSpec(
        schema="gold",
        name="daily_execution_metrics",
        comment="Daily pass rate by environment and ride service.",
        columns=cols(
            ("metric_date", "DATE", ""),
            ("environment", "STRING", ""),
            ("ride_service", "STRING", ""),
            ("rides", "LONG", ""),
            ("passed", "LONG", ""),
            ("failed", "LONG", ""),
            ("pass_rate", "DOUBLE", ""),
            ("avg_duration_seconds", "DOUBLE", ""),
            ("p90_duration_seconds", "DOUBLE", ""),
            ("artifacts_captured", "LONG", ""),
            ("defects_filed", "LONG", "Distinct ZPT keys touched that day"),
        ),
        partitioned_by=("metric_date",),
    ),
    TableSpec(
        schema="gold",
        name="scenario_health",
        comment="Per-scenario reliability, including a flip-rate based flakiness score.",
        columns=cols(
            ("scenario_name", "STRING", "Primary key"),
            ("scenario_version", "DOUBLE", ""),
            ("testrail_case_id", "LONG", ""),
            ("total_runs", "LONG", ""),
            ("passed", "LONG", ""),
            ("failed", "LONG", ""),
            ("pass_rate", "DOUBLE", ""),
            ("flakiness_score", "DOUBLE", "Share of consecutive runs that flipped outcome"),
            ("consecutive_failures", "INT", "Trailing failure streak"),
            ("distinct_failure_modes", "INT", ""),
            ("top_failure_mode", "STRING", ""),
            ("environments_covered", "ARRAY", ""),
            ("last_run_at", "TIMESTAMP", ""),
            ("last_failure_at", "TIMESTAMP", ""),
        ),
    ),
    TableSpec(
        schema="gold",
        name="failure_mode_trends",
        comment="Daily occurrence count per failure mode, with the defects it produced.",
        columns=cols(
            ("metric_date", "DATE", ""),
            ("failure_mode", "STRING", ""),
            ("component", "STRING", ""),
            ("occurrences", "LONG", ""),
            ("distinct_scenarios", "INT", ""),
            ("distinct_zones", "INT", ""),
            ("distinct_environments", "INT", ""),
            ("zpt_keys", "ARRAY", ""),
            ("first_seen_at", "TIMESTAMP", ""),
            ("last_seen_at", "TIMESTAMP", ""),
        ),
        partitioned_by=("metric_date",),
    ),
    TableSpec(
        schema="gold",
        name="pudo_zone_reliability",
        comment="Reliability rollup by pick-up / drop-off zone.",
        columns=cols(
            ("pudo_zone", "STRING", "Primary key"),
            ("rides", "LONG", ""),
            ("passed", "LONG", ""),
            ("failed", "LONG", ""),
            ("pass_rate", "DOUBLE", ""),
            ("distinct_scenarios", "INT", ""),
            ("distinct_pickup_pois", "INT", ""),
            ("top_failure_mode", "STRING", ""),
            ("avg_duration_seconds", "DOUBLE", ""),
        ),
    ),
    TableSpec(
        schema="gold",
        name="ride_failure_predictions",
        comment="Batch scoring output of the failure-mode classifier, one row per failed ride.",
        columns=cols(
            ("trigger_id", "STRING", "Primary key"),
            ("scenario_name", "STRING", ""),
            ("environment", "STRING", ""),
            ("pudo_zone", "STRING", ""),
            ("actual_failure_mode", "STRING", "Label from the SOTF failure taxonomy"),
            ("predicted_failure_mode", "STRING", ""),
            ("confidence", "DOUBLE", ""),
            ("is_correct", "BOOLEAN", ""),
            ("model_name", "STRING", ""),
            ("model_version", "INT", ""),
            ("mlflow_run_id", "STRING", ""),
            ("job_run_id", "LONG", "Databricks job run that wrote the row"),
            ("predicted_at", "TIMESTAMP", ""),
        ),
    ),
    # ---- ml: model inputs -------------------------------------------------
    TableSpec(
        schema="ml",
        name="ride_features",
        comment="Feature table for the failure-mode classifier, one row per completed ride.",
        columns=cols(
            ("trigger_id", "STRING", "Primary key"),
            ("scenario_name", "STRING", ""),
            ("environment", "STRING", ""),
            ("ride_service", "STRING", ""),
            ("pudo_zone", "STRING", ""),
            ("hour_of_day", "INT", ""),
            ("day_of_week", "INT", "Monday = 0"),
            ("duration_seconds", "DOUBLE", ""),
            ("log_row_count", "INT", ""),
            ("has_artifact", "BOOLEAN", ""),
            ("has_defect", "BOOLEAN", ""),
            ("label", "STRING", "Failure mode, or 'none' for a passing ride"),
        ),
    ),
    TableSpec(
        schema="ml",
        name="scenario_flakiness_features",
        comment="Feature table for the scenario flakiness model, one row per scenario.",
        columns=cols(
            ("scenario_name", "STRING", "Primary key"),
            ("runs", "LONG", ""),
            ("pass_rate", "DOUBLE", ""),
            ("flip_rate", "DOUBLE", ""),
            ("distinct_failure_modes", "INT", ""),
            ("avg_duration_seconds", "DOUBLE", ""),
            ("duration_stddev_seconds", "DOUBLE", ""),
            ("is_flaky", "BOOLEAN", "flip_rate >= 0.3 and the scenario both passed and failed"),
        ),
    ),
)

GOLD_AND_ML = tuple(spec for spec in TABLE_SPECS if spec.row_source == "data")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobTaskSpec:
    task_key: str
    notebook: str
    depends_on: tuple[str, ...] = ()
    duration_range: tuple[int, int] = (60, 900)
    can_fail: bool = False
    error: str = ""
    parameters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class JobSpec:
    name: str
    description: str
    quartz_cron: str
    interval_hours: int
    tasks: tuple[JobTaskSpec, ...]
    cluster_workers: tuple[int, int] = (2, 8)


JOB_SPECS: tuple[JobSpec, ...] = (
    JobSpec(
        name="sotf_bronze_ingest",
        description="Land raw artifacts, CTM logs, TestRail results and ZPT defects into bronze.",
        quartz_cron="0 0 1 * * ?",
        interval_hours=24,
        tasks=(
            JobTaskSpec(
                task_key="ingest_artifact_manifests",
                notebook="01_bronze_ingest",
                duration_range=(90, 600),
                can_fail=True,
                error="com.amazonaws.SdkClientException: Unable to execute HTTP request: Read timed out",
                parameters=(("source", "s3"),),
            ),
            JobTaskSpec(
                task_key="ingest_ctm_logs",
                notebook="01_bronze_ingest",
                duration_range=(120, 1200),
                can_fail=True,
                error="java.net.SocketTimeoutException: Read timed out reading from ClickHouse",
                parameters=(("source", "clickhouse"),),
            ),
            JobTaskSpec(
                task_key="ingest_testrail_results",
                notebook="01_bronze_ingest",
                duration_range=(45, 400),
                can_fail=True,
                error="requests.exceptions.HTTPError: 429 Client Error: Too Many Requests",
                parameters=(("source", "testrail"),),
            ),
            JobTaskSpec(
                task_key="ingest_jira_defects",
                notebook="01_bronze_ingest",
                duration_range=(30, 300),
                can_fail=True,
                error="requests.exceptions.HTTPError: 401 Client Error: Unauthorized",
                parameters=(("source", "jira"),),
            ),
        ),
    ),
    JobSpec(
        name="sotf_silver_transform",
        description="Conform bronze into the silver rides, logs, results and defects tables.",
        quartz_cron="0 0 2 * * ?",
        interval_hours=24,
        tasks=(
            JobTaskSpec(task_key="build_rides", notebook="02_silver_transform", duration_range=(90, 700)),
            JobTaskSpec(
                task_key="build_ctm_logs",
                notebook="02_silver_transform",
                duration_range=(180, 1500),
                can_fail=True,
                error="org.apache.spark.SparkException: Job aborted due to stage failure: "
                "Task 41 in stage 12.0 failed 4 times, most recent failure: ExecutorLostFailure",
            ),
            JobTaskSpec(
                task_key="build_test_results",
                notebook="02_silver_transform",
                depends_on=("build_rides",),
                duration_range=(60, 400),
            ),
            JobTaskSpec(
                task_key="quality_checks",
                notebook="02_silver_transform",
                depends_on=("build_rides", "build_ctm_logs", "build_test_results"),
                duration_range=(30, 200),
                can_fail=True,
                error="AssertionError: silver.rides has orphan trigger_ids not present in bronze",
            ),
        ),
    ),
    JobSpec(
        name="sotf_gold_aggregates",
        description="Rebuild the gold reporting tables from silver.",
        quartz_cron="0 0 3 * * ?",
        interval_hours=24,
        tasks=(
            JobTaskSpec(task_key="daily_execution_metrics", notebook="03_gold_aggregates", duration_range=(60, 500)),
            JobTaskSpec(task_key="scenario_health", notebook="03_gold_aggregates", duration_range=(60, 500)),
            JobTaskSpec(task_key="failure_mode_trends", notebook="03_gold_aggregates", duration_range=(45, 400)),
            JobTaskSpec(
                task_key="pudo_zone_reliability",
                notebook="03_gold_aggregates",
                duration_range=(45, 400),
                can_fail=True,
                error="org.apache.spark.sql.AnalysisException: Column 'pudo_zone' is not nullable "
                "but contains null values",
            ),
        ),
    ),
    JobSpec(
        name="sotf_failure_mode_inference",
        description="Score failed rides with the champion failure-mode classifier.",
        quartz_cron="0 0 4 * * ?",
        interval_hours=24,
        cluster_workers=(2, 4),
        tasks=(
            JobTaskSpec(task_key="build_features", notebook="04_failure_mode_inference", duration_range=(60, 400)),
            JobTaskSpec(
                task_key="score_batch",
                notebook="04_failure_mode_inference",
                depends_on=("build_features",),
                duration_range=(90, 700),
                can_fail=True,
                error="mlflow.exceptions.MlflowException: Registered model alias 'champion' not found",
            ),
            JobTaskSpec(
                task_key="write_predictions",
                notebook="04_failure_mode_inference",
                depends_on=("score_batch",),
                duration_range=(30, 240),
            ),
        ),
    ),
    JobSpec(
        name="sotf_flakiness_training",
        description="Retrain and register the scenario flakiness model.",
        quartz_cron="0 0 5 ? * MON",
        interval_hours=168,
        cluster_workers=(2, 6),
        tasks=(
            JobTaskSpec(task_key="build_training_set", notebook="05_flakiness_training", duration_range=(120, 900)),
            JobTaskSpec(
                task_key="train_model",
                notebook="05_flakiness_training",
                depends_on=("build_training_set",),
                duration_range=(300, 2400),
                can_fail=True,
                error="ValueError: Found array with 0 sample(s) while a minimum of 1 is required",
            ),
            JobTaskSpec(
                task_key="evaluate_model",
                notebook="05_flakiness_training",
                depends_on=("train_model",),
                duration_range=(60, 400),
            ),
            JobTaskSpec(
                task_key="register_model",
                notebook="05_flakiness_training",
                depends_on=("evaluate_model",),
                duration_range=(20, 120),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------

NOTEBOOK_SOURCE: dict[str, str] = {
    "01_bronze_ingest": '''# Databricks notebook source
# MAGIC %md
# MAGIC # SOTF bronze ingest
# MAGIC Lands one source system per task run. The `source` widget selects which.

# COMMAND ----------

dbutils.widgets.text("source", "s3")
dbutils.widgets.text("ingest_date", "")

source = dbutils.widgets.get("source")
ingest_date = dbutils.widgets.get("ingest_date")

# COMMAND ----------

from pyspark.sql import functions as F

if source == "s3":
    manifests = (
        spark.read.json("s3://zoox-test-artifacts/tec/dev/manifests/")
        .withColumn("ingested_at", F.current_timestamp())
    )
    manifests.write.mode("append").saveAsTable("sotf.bronze.artifact_manifests")

elif source == "clickhouse":
    # Bounded pull: the CTM log table is large and we only want the day.
    logs = (
        spark.read.format("jdbc")
        .option("url", "jdbc:clickhouse://10.1.101.35:8123/default")
        .option("user", dbutils.secrets.get("sotf", "clickhouse_user"))
        .option("password", dbutils.secrets.get("sotf", "clickhouse_password"))
        .option(
            "query",
            f"SELECT * FROM default.logs WHERE toDate(timestamp) = '{ingest_date}'",
        )
        .load()
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("ingest_date", F.lit(ingest_date).cast("date"))
    )
    logs.write.mode("append").partitionBy("ingest_date").saveAsTable("sotf.bronze.ctm_logs_raw")

elif source == "testrail":
    results = spark.createDataFrame(fetch_testrail_results(ingest_date))
    results.write.mode("append").saveAsTable("sotf.bronze.testrail_results_raw")

elif source == "jira":
    issues = spark.createDataFrame(fetch_zpt_issues(ingest_date))
    issues.write.mode("append").saveAsTable("sotf.bronze.jira_defects_raw")

else:
    raise ValueError(f"unknown source: {source}")
''',
    "02_silver_transform": '''# Databricks notebook source
# MAGIC %md
# MAGIC # SOTF silver transform
# MAGIC Conforms bronze into one row per business entity. `trigger_id` is the join key
# MAGIC across every source system.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE sotf.silver.rides AS
# MAGIC SELECT
# MAGIC   r.trigger_id,
# MAGIC   r.execution_id,
# MAGIC   r.scenario_name,
# MAGIC   r.scenario_version,
# MAGIC   r.ride_status,
# MAGIC   r.environment,
# MAGIC   r.ride_service,
# MAGIC   r.pudo_zone,
# MAGIC   r.pickup_poi,
# MAGIC   r.dropoff_poi,
# MAGIC   r.started_at,
# MAGIC   r.ended_at,
# MAGIC   unix_timestamp(r.ended_at) - unix_timestamp(r.started_at) AS duration_seconds,
# MAGIC   r.failure_mode,
# MAGIC   t.result_id AS testrail_result_id,
# MAGIC   d.issue_key AS jira_key,
# MAGIC   m.s3_uri    AS s3_key,
# MAGIC   to_date(r.started_at) AS ride_date
# MAGIC FROM sotf.bronze.rides_raw r
# MAGIC LEFT JOIN sotf.bronze.testrail_results_raw t ON t.trigger_id = r.trigger_id
# MAGIC LEFT JOIN sotf.bronze.jira_defects_raw    d ON array_contains(d.linked_trigger_ids, r.trigger_id)
# MAGIC LEFT JOIN sotf.bronze.artifact_manifests  m ON m.trigger_id = r.trigger_id

# COMMAND ----------

# The CTM message column carries "SEVERITY [component] text"; lift the parts out so
# downstream queries filter on columns instead of scanning the message.
logs = (
    spark.table("sotf.bronze.ctm_logs_raw")
    .withColumn("severity", F.regexp_extract("message", r"^(\\w+) ", 1))
    .withColumn("component", F.regexp_extract("message", r"^\\w+ \\[([^\\]]+)\\]", 1))
    .withColumnRenamed("timestamp", "log_timestamp")
    .withColumn("log_date", F.to_date("log_timestamp"))
)
logs.write.mode("overwrite").partitionBy("log_date").saveAsTable("sotf.silver.ctm_logs")

# COMMAND ----------

# Quality gate: every silver ride must trace back to a bronze manifest or an
# explicitly Queued ride that never produced artifacts.
orphans = spark.sql("""
    SELECT r.trigger_id
    FROM sotf.silver.rides r
    LEFT JOIN sotf.bronze.artifact_manifests m ON m.trigger_id = r.trigger_id
    WHERE m.trigger_id IS NULL AND r.ride_status IN ('Passed', 'Failed')
""")
assert orphans.count() == 0, "silver.rides has orphan trigger_ids not present in bronze"
''',
    "03_gold_aggregates": '''# Databricks notebook source
# MAGIC %md
# MAGIC # SOTF gold aggregates
# MAGIC Rebuilds the reporting tables. Every task in the job maps to one cell below.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE sotf.gold.daily_execution_metrics AS
# MAGIC SELECT
# MAGIC   to_date(started_at)                                   AS metric_date,
# MAGIC   environment,
# MAGIC   ride_service,
# MAGIC   count(*)                                              AS rides,
# MAGIC   sum(if(ride_status = 'Passed', 1, 0))                 AS passed,
# MAGIC   sum(if(ride_status = 'Failed', 1, 0))                 AS failed,
# MAGIC   round(avg(if(ride_status = 'Passed', 1.0, 0.0)), 4)   AS pass_rate,
# MAGIC   round(avg(duration_seconds), 1)                       AS avg_duration_seconds,
# MAGIC   round(percentile_approx(duration_seconds, 0.9), 1)    AS p90_duration_seconds,
# MAGIC   count(s3_key)                                         AS artifacts_captured,
# MAGIC   count(DISTINCT jira_key)                              AS defects_filed
# MAGIC FROM sotf.silver.rides
# MAGIC WHERE ride_status IN ('Passed', 'Failed')
# MAGIC GROUP BY 1, 2, 3

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Flakiness is the share of consecutive runs that flipped outcome, which
# MAGIC -- separates a genuinely broken scenario (low pass rate, low flip rate) from
# MAGIC -- a flaky one (middling pass rate, high flip rate).
# MAGIC CREATE OR REPLACE TABLE sotf.gold.scenario_health AS
# MAGIC WITH ordered AS (
# MAGIC   SELECT
# MAGIC     scenario_name,
# MAGIC     ride_status,
# MAGIC     started_at,
# MAGIC     lag(ride_status) OVER (PARTITION BY scenario_name ORDER BY started_at) AS prev_status
# MAGIC   FROM sotf.silver.rides
# MAGIC   WHERE ride_status IN ('Passed', 'Failed')
# MAGIC )
# MAGIC SELECT
# MAGIC   scenario_name,
# MAGIC   count(*)                                                            AS total_runs,
# MAGIC   sum(if(ride_status = 'Passed', 1, 0))                               AS passed,
# MAGIC   sum(if(ride_status = 'Failed', 1, 0))                               AS failed,
# MAGIC   round(avg(if(ride_status = 'Passed', 1.0, 0.0)), 4)                 AS pass_rate,
# MAGIC   round(sum(if(prev_status IS NOT NULL AND prev_status <> ride_status, 1, 0))
# MAGIC         / nullif(count(*) - 1, 0), 4)                                 AS flakiness_score,
# MAGIC   max(started_at)                                                     AS last_run_at
# MAGIC FROM ordered
# MAGIC GROUP BY scenario_name

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE sotf.gold.failure_mode_trends AS
# MAGIC SELECT
# MAGIC   to_date(started_at)                 AS metric_date,
# MAGIC   failure_mode,
# MAGIC   count(*)                            AS occurrences,
# MAGIC   count(DISTINCT scenario_name)       AS distinct_scenarios,
# MAGIC   count(DISTINCT pudo_zone)           AS distinct_zones,
# MAGIC   count(DISTINCT environment)         AS distinct_environments,
# MAGIC   collect_set(jira_key)               AS zpt_keys,
# MAGIC   min(started_at)                     AS first_seen_at,
# MAGIC   max(started_at)                     AS last_seen_at
# MAGIC FROM sotf.silver.rides
# MAGIC WHERE ride_status = 'Failed'
# MAGIC GROUP BY 1, 2

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE sotf.gold.pudo_zone_reliability AS
# MAGIC SELECT
# MAGIC   pudo_zone,
# MAGIC   count(*)                                            AS rides,
# MAGIC   sum(if(ride_status = 'Passed', 1, 0))               AS passed,
# MAGIC   sum(if(ride_status = 'Failed', 1, 0))               AS failed,
# MAGIC   round(avg(if(ride_status = 'Passed', 1.0, 0.0)), 4) AS pass_rate,
# MAGIC   count(DISTINCT scenario_name)                       AS distinct_scenarios,
# MAGIC   count(DISTINCT pickup_poi)                          AS distinct_pickup_pois,
# MAGIC   round(avg(duration_seconds), 1)                     AS avg_duration_seconds
# MAGIC FROM sotf.silver.rides
# MAGIC WHERE ride_status IN ('Passed', 'Failed')
# MAGIC GROUP BY pudo_zone
''',
    "04_failure_mode_inference": '''# Databricks notebook source
# MAGIC %md
# MAGIC # SOTF failure-mode inference
# MAGIC Scores failed rides with the champion classifier so triage starts from a
# MAGIC suggested cause instead of a blank page. The prediction is advisory: the
# MAGIC actual failure mode still comes from the CTM logs.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

MODEL_URI = "models:/sotf_failure_mode_classifier/Production"

# COMMAND ----------

features = (
    spark.table("sotf.silver.rides")
    .where("ride_status IN ('Passed', 'Failed')")
    .select(
        "trigger_id",
        "scenario_name",
        "environment",
        "ride_service",
        "pudo_zone",
        F.hour("started_at").alias("hour_of_day"),
        (F.dayofweek("started_at") + 5).cast("int").mod(7).alias("day_of_week"),
        "duration_seconds",
        F.col("s3_key").isNotNull().alias("has_artifact"),
        F.col("jira_key").isNotNull().alias("has_defect"),
        F.coalesce("failure_mode", F.lit("none")).alias("label"),
    )
    .join(
        spark.table("sotf.silver.ctm_logs").groupBy("trigger_id").count()
        .withColumnRenamed("count", "log_row_count"),
        on="trigger_id",
        how="left",
    )
)
features.write.mode("overwrite").saveAsTable("sotf.ml.ride_features")

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="string")
confidence = mlflow.pyfunc.spark_udf(spark, MODEL_URI + "/confidence", result_type="double")

scored = (
    spark.table("sotf.ml.ride_features")
    .where("label <> 'none'")
    .withColumn("predicted_failure_mode", predict(F.struct("*")))
    .withColumn("confidence", F.round(confidence(F.struct("*")), 4))
)

# COMMAND ----------

(
    scored.withColumnRenamed("label", "actual_failure_mode")
    .withColumn("is_correct", F.col("predicted_failure_mode") == F.col("actual_failure_mode"))
    .withColumn("model_name", F.lit("sotf_failure_mode_classifier"))
    .withColumn("job_run_id", F.lit(dbutils.widgets.get("job_run_id")).cast("long"))
    .withColumn("predicted_at", F.current_timestamp())
    .write.mode("overwrite")
    .saveAsTable("sotf.gold.ride_failure_predictions")
)
''',
    "05_flakiness_training": '''# Databricks notebook source
# MAGIC %md
# MAGIC # SOTF scenario flakiness training
# MAGIC Weekly retrain. A scenario is labelled flaky when it both passes and fails
# MAGIC and flips outcome on at least 30% of consecutive runs.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("/Shared/sotf/scenario_flakiness")

# COMMAND ----------

training = spark.table("sotf.ml.scenario_flakiness_features").toPandas()
X = training[["runs", "pass_rate", "flip_rate", "distinct_failure_modes",
              "avg_duration_seconds", "duration_stddev_seconds"]]
y = training["is_flaky"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# COMMAND ----------

with mlflow.start_run() as run:
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params, random_state=42).fit(X_train, y_train)
    predictions = model.predict(X_test)

    mlflow.log_metric("f1", f1_score(y_test, predictions))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    mlflow.sklearn.log_model(model, "model", input_example=X_train.head())

    mlflow.register_model(f"runs:/{run.info.run_id}/model", "sotf_scenario_flakiness")
''',
}


# ---------------------------------------------------------------------------
# SQL warehouse queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedQuery:
    name: str
    description: str
    sql: str
    # Result size, computed from the gold tables and the source rides so the
    # query history reports what the query would really have returned.
    rows: Callable[[dict[str, list[dict[str, Any]]], "SotfData"], int]


SAVED_QUERIES: tuple[SavedQuery, ...] = (
    SavedQuery(
        name="Scenarios below the pass-rate bar",
        description="Scenarios passing less than 80% of the time, worst first.",
        sql=(
            "SELECT scenario_name, total_runs, pass_rate, flakiness_score, top_failure_mode\n"
            "FROM sotf.gold.scenario_health\n"
            "WHERE pass_rate < 0.8 AND total_runs >= 3\n"
            "ORDER BY pass_rate ASC"
        ),
        rows=lambda g, s: sum(
            1 for r in g["scenario_health"] if r["pass_rate"] < 0.8 and r["total_runs"] >= 3
        ),
    ),
    SavedQuery(
        name="Flaky scenarios",
        description="High flip rate with a middling pass rate: flaky rather than broken.",
        sql=(
            "SELECT scenario_name, total_runs, pass_rate, flakiness_score\n"
            "FROM sotf.gold.scenario_health\n"
            "WHERE flakiness_score >= 0.3\n"
            "ORDER BY flakiness_score DESC"
        ),
        rows=lambda g, s: sum(1 for r in g["scenario_health"] if r["flakiness_score"] >= 0.3),
    ),
    SavedQuery(
        name="Failure modes this week",
        description="Which failure modes are driving the current failure volume.",
        sql=(
            "SELECT failure_mode, component, sum(occurrences) AS occurrences,\n"
            "       max(distinct_scenarios) AS scenarios\n"
            "FROM sotf.gold.failure_mode_trends\n"
            "WHERE metric_date >= current_date() - INTERVAL 7 DAYS\n"
            "GROUP BY failure_mode, component\n"
            "ORDER BY occurrences DESC"
        ),
        rows=lambda g, s: len({r["failure_mode"] for r in g["failure_mode_trends"]}),
    ),
    SavedQuery(
        name="Pass rate by environment",
        description="Daily pass rate per environment for the trend chart.",
        sql=(
            "SELECT metric_date, environment,\n"
            "       sum(passed) AS passed, sum(failed) AS failed,\n"
            "       round(sum(passed) / sum(rides), 4) AS pass_rate\n"
            "FROM sotf.gold.daily_execution_metrics\n"
            "GROUP BY metric_date, environment\n"
            "ORDER BY metric_date DESC"
        ),
        rows=lambda g, s: len(
            {(r["metric_date"], r["environment"]) for r in g["daily_execution_metrics"]}
        ),
    ),
    SavedQuery(
        name="Worst PUDO zones",
        description="Zones where rides fail most often.",
        sql=(
            "SELECT pudo_zone, rides, pass_rate, top_failure_mode\n"
            "FROM sotf.gold.pudo_zone_reliability\n"
            "ORDER BY pass_rate ASC"
        ),
        rows=lambda g, s: len(g["pudo_zone_reliability"]),
    ),
    SavedQuery(
        name="Classifier accuracy by failure mode",
        description="Where the failure-mode model is reliable and where it is not.",
        sql=(
            "SELECT actual_failure_mode,\n"
            "       count(*) AS rides,\n"
            "       round(avg(if(is_correct, 1.0, 0.0)), 4) AS accuracy,\n"
            "       round(avg(confidence), 4) AS avg_confidence\n"
            "FROM sotf.gold.ride_failure_predictions\n"
            "GROUP BY actual_failure_mode\n"
            "ORDER BY accuracy ASC"
        ),
        rows=lambda g, s: len({r["actual_failure_mode"] for r in g["ride_failure_predictions"]}),
    ),
    SavedQuery(
        name="Low-confidence predictions to review",
        description="Predictions a human should check before trusting.",
        sql=(
            "SELECT trigger_id, scenario_name, actual_failure_mode,\n"
            "       predicted_failure_mode, confidence\n"
            "FROM sotf.gold.ride_failure_predictions\n"
            "WHERE confidence < 0.6\n"
            "ORDER BY confidence ASC"
        ),
        rows=lambda g, s: sum(1 for r in g["ride_failure_predictions"] if r["confidence"] < 0.6),
    ),
    SavedQuery(
        name="Rides with no artifact",
        description="Completed rides whose artifact bundle never landed.",
        sql=(
            "SELECT trigger_id, scenario_name, environment, ride_status, started_at\n"
            "FROM sotf.silver.rides\n"
            "WHERE ride_status IN ('Passed', 'Failed') AND s3_key IS NULL\n"
            "ORDER BY started_at DESC"
        ),
        rows=lambda g, s: sum(1 for r in s.completed if not r.get("s3", {}).get("key")),
    ),
    SavedQuery(
        name="Recurring defects",
        description="ZPT issues that keep collecting new occurrences.",
        sql=(
            "SELECT issue_key, failure_mode, occurrences, size(linked_trigger_ids) AS rides\n"
            "FROM sotf.silver.defects\n"
            "WHERE occurrences > 1\n"
            "ORDER BY occurrences DESC"
        ),
        rows=lambda g, s: s.recurring_defect_count,
    ),
    SavedQuery(
        name="Error log volume by component",
        description="Which subsystem is producing the ERROR lines.",
        sql=(
            "SELECT component, count(*) AS error_rows, count(DISTINCT trigger_id) AS rides\n"
            "FROM sotf.silver.ctm_logs\n"
            "WHERE severity = 'ERROR' AND log_date >= current_date() - INTERVAL 7 DAYS\n"
            "GROUP BY component\n"
            "ORDER BY error_rows DESC"
        ),
        rows=lambda g, s: len(
            {FAILURE_COMPONENT.get(mode, "unknown") for mode in s.failure_modes}
        ),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            written += 1
    return written


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[index], 1)


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    variance = sum((v - average) ** 2 for v in values) / (len(values) - 1)
    return round(variance**0.5, 1)


def top_key(counter: dict[str, int]) -> str | None:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def hex_id(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(length))


class Counter:
    def __init__(self, start: int) -> None:
        self._value = start

    def next(self) -> int:
        self._value += 1
        return self._value


# ---------------------------------------------------------------------------
# SOTF dataset input
# ---------------------------------------------------------------------------


@dataclass
class SotfData:
    root: Path
    now: datetime
    days: int
    counts: dict[str, Any]
    rides: list[dict[str, Any]]

    @property
    def completed(self) -> list[dict[str, Any]]:
        return [r for r in self.rides if r["ride_status"] in ("Passed", "Failed") and r["_started"]]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.completed if r["ride_status"] == "Failed" and r["failure_mode"]]

    @property
    def failure_modes(self) -> list[str]:
        return sorted({r["failure_mode"] for r in self.failed})

    @property
    def recurring_defect_count(self) -> int:
        """ZPT keys linked to more than one ride."""
        per_key: dict[str, int] = defaultdict(int)
        for ride in self.rides:
            if ride.get("jira_key"):
                per_key[ride["jira_key"]] += 1
        return sum(1 for count in per_key.values() if count > 1)


def load_sotf_dataset(root: Path) -> SotfData:
    cross_reference = root / "cross_reference.json"
    summary_file = root / "summary.json"
    missing = [f.name for f in (cross_reference, summary_file) if not f.is_file()]
    if missing:
        raise SystemExit(
            f"SOTF dataset at {root} is missing {', '.join(missing)}\n"
            "Run generate_sotf_mock_data.py first, or pass --sotf-data."
        )

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    params = summary.get("parameters", {})

    rides = json.loads(cross_reference.read_text(encoding="utf-8"))
    for ride in rides:
        ride["_started"] = parse_dt(ride.get("started_at"))
        ride["_ended"] = parse_dt(ride.get("ended_at"))
        if ride["_started"] and ride["_ended"]:
            ride["_duration"] = (ride["_ended"] - ride["_started"]).total_seconds()
        else:
            ride["_duration"] = None

    return SotfData(
        root=root,
        now=parse_dt(params.get("now")) or datetime.now(timezone.utc),
        days=int(params.get("days", 30)),
        counts=summary.get("counts", {}),
        rides=rides,
    )


# ---------------------------------------------------------------------------
# Gold and ML table contents
# ---------------------------------------------------------------------------


def build_daily_execution_metrics(sotf: SotfData) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for ride in sotf.completed:
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
                "environment": environment,
                "ride_service": ride_service,
                "rides": len(members),
                "passed": passed,
                "failed": len(members) - passed,
                "pass_rate": round(passed / len(members), 4),
                "avg_duration_seconds": mean(durations),
                "p90_duration_seconds": percentile(durations, 0.9),
                "artifacts_captured": sum(1 for r in members if r.get("s3", {}).get("key")),
                "defects_filed": len({r["jira_key"] for r in members if r.get("jira_key")}),
            }
        )
    return rows


def build_scenario_health(sotf: SotfData) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ride in sotf.completed:
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
                "scenario_name": scenario,
                "scenario_version": ordered[-1]["scenario_version"],
                "testrail_case_id": ordered[-1].get("testrail", {}).get("case_id"),
                "total_runs": len(ordered),
                "passed": passed,
                "failed": len(ordered) - passed,
                "pass_rate": round(passed / len(ordered), 4),
                "flakiness_score": flakiness,
                "consecutive_failures": streak,
                "distinct_failure_modes": len(modes),
                "top_failure_mode": top_key(dict(modes)),
                "environments_covered": sorted({r["environment"] for r in ordered}),
                "last_run_at": iso(ordered[-1]["_started"]),
                "last_failure_at": iso(failures[-1]["_started"]) if failures else None,
            }
        )
    return rows


def build_failure_mode_trends(sotf: SotfData) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ride in sotf.failed:
        groups[(ride["_started"].strftime("%Y-%m-%d"), ride["failure_mode"])].append(ride)

    rows = []
    for (metric_date, failure_mode), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda r: r["_started"])
        rows.append(
            {
                "metric_date": metric_date,
                "failure_mode": failure_mode,
                "component": FAILURE_COMPONENT.get(failure_mode, "unknown"),
                "occurrences": len(ordered),
                "distinct_scenarios": len({r["scenario"] for r in ordered}),
                "distinct_zones": len({r["pudo_zone"] for r in ordered if r["pudo_zone"]}),
                "distinct_environments": len({r["environment"] for r in ordered}),
                "zpt_keys": sorted({r["jira_key"] for r in ordered if r.get("jira_key")}),
                "first_seen_at": iso(ordered[0]["_started"]),
                "last_seen_at": iso(ordered[-1]["_started"]),
            }
        )
    return rows


def build_pudo_zone_reliability(sotf: SotfData) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ride in sotf.completed:
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
                "pudo_zone": zone,
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


def build_ride_features(sotf: SotfData) -> list[dict[str, Any]]:
    rows = []
    for ride in sorted(sotf.completed, key=lambda r: r["_started"]):
        rows.append(
            {
                "trigger_id": ride["trigger_id"],
                "scenario_name": ride["scenario"],
                "environment": ride["environment"],
                "ride_service": ride["ride_service"],
                "pudo_zone": ride["pudo_zone"],
                "hour_of_day": ride["_started"].hour,
                "day_of_week": ride["_started"].weekday(),
                "duration_seconds": ride["_duration"],
                "log_row_count": ride.get("clickhouse_log_rows", 0),
                "has_artifact": bool(ride.get("s3", {}).get("key")),
                "has_defect": bool(ride.get("jira_key")),
                "label": ride["failure_mode"] or "none",
            }
        )
    return rows


def build_flakiness_features(
    sotf: SotfData, scenario_health: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    durations: dict[str, list[float]] = defaultdict(list)
    for ride in sotf.completed:
        if ride["_duration"] is not None:
            durations[ride["scenario"]].append(ride["_duration"])

    rows = []
    for health in scenario_health:
        scenario = health["scenario_name"]
        values = durations.get(scenario, [])
        mixed = health["passed"] > 0 and health["failed"] > 0
        rows.append(
            {
                "scenario_name": scenario,
                "runs": health["total_runs"],
                "pass_rate": health["pass_rate"],
                "flip_rate": health["flakiness_score"],
                "distinct_failure_modes": health["distinct_failure_modes"],
                "avg_duration_seconds": mean(values),
                "duration_stddev_seconds": stddev(values),
                "is_flaky": mixed and health["flakiness_score"] >= 0.3,
            }
        )
    return rows


def build_predictions(
    sotf: SotfData,
    rng: random.Random,
    accuracy_target: float,
    model_version: int,
    mlflow_run_id: str,
    job_run_by_date: dict[str, int],
) -> list[dict[str, Any]]:
    labels = sotf.failure_modes
    rows = []
    for ride in sorted(sotf.failed, key=lambda r: r["_started"]):
        actual = ride["failure_mode"]
        correct = rng.random() < accuracy_target
        if correct:
            predicted = actual
            confidence = round(rng.uniform(0.72, 0.98), 4)
        else:
            candidates = [c for c in CONFUSABLE.get(actual, ()) if c in labels]
            if not candidates:
                candidates = [label for label in labels if label != actual]
            predicted = rng.choice(candidates) if candidates else actual
            confidence = round(rng.uniform(0.41, 0.71), 4)

        # Scoring happens in the next morning's inference run, which is the run
        # whose data window is the day the ride started.
        ride_date = ride["_started"].strftime("%Y-%m-%d")
        predicted_at = (ride["_started"] + timedelta(days=1)).replace(
            hour=4, minute=12, second=0, microsecond=0
        )
        rows.append(
            {
                "trigger_id": ride["trigger_id"],
                "scenario_name": ride["scenario"],
                "environment": ride["environment"],
                "pudo_zone": ride["pudo_zone"],
                "actual_failure_mode": actual,
                "predicted_failure_mode": predicted,
                "confidence": confidence,
                "is_correct": predicted == actual,
                "model_name": MODEL_NAME,
                "model_version": model_version,
                "mlflow_run_id": mlflow_run_id,
                "job_run_id": job_run_by_date.get(ride_date),
                "predicted_at": iso(predicted_at),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Jobs and runs
# ---------------------------------------------------------------------------


@dataclass
class TaskRun:
    spec: JobTaskSpec
    run_id: int
    state: str
    result: str | None
    start: datetime | None
    end: datetime | None
    error: str = ""


@dataclass
class JobRun:
    spec: JobSpec
    job_id: int
    run_id: int
    number_in_job: int
    window_start: datetime
    start: datetime
    end: datetime
    result: str
    cluster_id: str
    tasks: list[TaskRun] = field(default_factory=list)


def iter_days(start: datetime, now: datetime, interval_hours: int) -> list[datetime]:
    """Schedule boundaries covered by the window."""
    step = timedelta(hours=interval_hours)
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if interval_hours == 168:
        cursor -= timedelta(days=cursor.weekday())
    boundaries = []
    while cursor + step <= now:
        boundaries.append(cursor)
        cursor += step
    return boundaries


def build_job_runs(
    sotf: SotfData, args: argparse.Namespace, rng: random.Random
) -> tuple[dict[str, int], list[JobRun]]:
    job_ids = {spec.name: 800000 + index * 137 for index, spec in enumerate(JOB_SPECS)}
    run_counter = Counter(4_500_000)
    window_start = sotf.now - timedelta(days=sotf.days)

    runs: list[JobRun] = []
    for spec in JOB_SPECS:
        hour = int(spec.quartz_cron.split()[2])
        number = 0
        for day in iter_days(window_start, sotf.now, spec.interval_hours):
            # A run covering data date `day` launches one interval later, at the
            # hour named in the cron expression.
            launch = (day + timedelta(hours=spec.interval_hours)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            if launch > sotf.now:
                continue
            number += 1

            cluster_id = f"{launch.strftime('%m%d')}-{hex_id(rng, 6)}-{hex_id(rng, 8)}"
            fail_candidates = [t for t in spec.tasks if t.can_fail]
            failing = (
                rng.choice(fail_candidates).task_key
                if fail_candidates and rng.random() < args.failure_rate
                else None
            )

            run_start = launch + timedelta(seconds=rng.randint(2, 120))
            ends: dict[str, datetime] = {}
            tasks: list[TaskRun] = []
            skipped = False

            for task_spec in spec.tasks:
                if skipped and task_spec.depends_on:
                    tasks.append(
                        TaskRun(task_spec, run_counter.next(), "SKIPPED", "SKIPPED", None, None)
                    )
                    continue

                upstream = [ends[key] for key in task_spec.depends_on if key in ends]
                ready = max(upstream) if upstream else run_start
                start = ready + timedelta(seconds=rng.randint(5, 40))
                end = start + timedelta(seconds=rng.randint(*task_spec.duration_range))
                ends[task_spec.task_key] = end

                if task_spec.task_key == failing:
                    tasks.append(
                        TaskRun(
                            task_spec,
                            run_counter.next(),
                            "INTERNAL_ERROR",
                            "FAILED",
                            start,
                            end,
                            task_spec.error,
                        )
                    )
                    skipped = True
                else:
                    tasks.append(
                        TaskRun(task_spec, run_counter.next(), "TERMINATED", "SUCCESS", start, end)
                    )

            finished = [t.end for t in tasks if t.end]
            runs.append(
                JobRun(
                    spec=spec,
                    job_id=job_ids[spec.name],
                    run_id=run_counter.next(),
                    number_in_job=number,
                    window_start=day,
                    start=run_start,
                    end=max(finished) + timedelta(seconds=rng.randint(3, 30)),
                    result="FAILED" if failing else "SUCCESS",
                    cluster_id=cluster_id,
                    tasks=tasks,
                )
            )
    return job_ids, runs


def new_cluster_spec(spec: JobSpec) -> dict[str, Any]:
    return {
        "spark_version": SPARK_VERSION,
        "node_type_id": NODE_TYPE,
        "driver_node_type_id": DRIVER_NODE_TYPE,
        "autoscale": {"min_workers": spec.cluster_workers[0], "max_workers": spec.cluster_workers[1]},
        "aws_attributes": {
            "availability": "SPOT_WITH_FALLBACK",
            "zone_id": "us-west-2a",
            "first_on_demand": 1,
            "instance_profile_arn": "arn:aws:iam::000000000000:instance-profile/sotf-lakehouse",
        },
        "spark_conf": {
            "spark.databricks.delta.preview.enabled": "true",
            "spark.sql.shuffle.partitions": "auto",
            # Credentials are referenced, never inlined. dbutils.secrets resolves
            # these at runtime and the API returns the reference, not the value.
            "spark.sotf.clickhouse.password": "{{secrets/sotf/clickhouse_password}}",
            "spark.sotf.testrail.apiKey": "{{secrets/sotf/testrail_api_key}}",
        },
        "custom_tags": {"team": "sotf", "layer": "lakehouse"},
        "data_security_mode": "SINGLE_USER",
        "runtime_engine": "PHOTON",
    }


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def emit_unity_catalog(
    out: Path, sotf: SotfData, row_counts: dict[str, int], rng: random.Random
) -> None:
    created = ms(sotf.now - timedelta(days=sotf.days + 60))
    updated = ms(sotf.now - timedelta(hours=6))

    write_json(
        out / "unity_catalog/catalogs.json",
        {
            "catalogs": [
                {
                    "name": CATALOG,
                    "owner": OWNER,
                    "comment": "Scenario-on-the-fly test telemetry and reporting.",
                    "metastore_id": METASTORE_ID,
                    "created_at": created,
                    "created_by": OWNER,
                    "updated_at": updated,
                    "updated_by": OWNER,
                    "catalog_type": "MANAGED_CATALOG",
                    "isolation_mode": "OPEN",
                }
            ]
        },
    )

    schema_comments = {
        "bronze": "Raw landings from S3, ClickHouse, TestRail and Jira. Append only.",
        "silver": "Conformed entities. trigger_id is the join key across every source.",
        "gold": "Reporting tables. Rebuilt daily by sotf_gold_aggregates.",
        "ml": "Feature tables and model inputs.",
    }
    write_json(
        out / "unity_catalog/schemas.json",
        {
            "schemas": [
                {
                    "name": schema,
                    "catalog_name": CATALOG,
                    "full_name": f"{CATALOG}.{schema}",
                    "owner": OWNER,
                    "comment": schema_comments[schema],
                    "metastore_id": METASTORE_ID,
                    "created_at": created,
                    "created_by": OWNER,
                    "updated_at": updated,
                    "updated_by": OWNER,
                }
                for schema in SCHEMAS
            ]
        },
    )

    tables = []
    for spec in TABLE_SPECS:
        columns = []
        for position, column in enumerate(spec.columns):
            columns.append(
                {
                    "name": column.name,
                    "type_text": column.type_text,
                    "type_name": column.type_name,
                    "type_json": json.dumps(
                        {
                            "name": column.name,
                            "type": column.type_text,
                            "nullable": True,
                            "metadata": {},
                        }
                    ),
                    "position": position,
                    "comment": column.comment,
                    "nullable": True,
                }
            )
        row_count = row_counts.get(spec.full_name, 0)
        tables.append(
            {
                "name": spec.name,
                "catalog_name": CATALOG,
                "schema_name": spec.schema,
                "full_name": spec.full_name,
                "table_type": spec.table_type,
                "data_source_format": "DELTA",
                "columns": columns,
                "storage_location": f"{STORAGE_ROOT}/{spec.schema}/{spec.name}",
                "owner": OWNER,
                "comment": spec.comment,
                "properties": {
                    "delta.minReaderVersion": "1",
                    "delta.minWriterVersion": "2",
                    "delta.enableChangeDataFeed": "true",
                    "sotf.row_count": str(row_count),
                    "sotf.partitioned_by": ",".join(spec.partitioned_by),
                },
                "metastore_id": METASTORE_ID,
                "table_id": "-".join(
                    hex_id(rng, length) for length in (8, 4, 4, 4, 12)
                ),
                "created_at": created,
                "created_by": OWNER,
                "updated_at": updated,
                "updated_by": OWNER,
                "generation": 1,
            }
        )
    write_json(out / "unity_catalog/tables.json", {"tables": tables})

    lineage = {
        f"{CATALOG}.silver.rides": [
            f"{CATALOG}.bronze.artifact_manifests",
            f"{CATALOG}.bronze.testrail_results_raw",
            f"{CATALOG}.bronze.jira_defects_raw",
        ],
        f"{CATALOG}.silver.ctm_logs": [f"{CATALOG}.bronze.ctm_logs_raw"],
        f"{CATALOG}.silver.test_results": [f"{CATALOG}.bronze.testrail_results_raw"],
        f"{CATALOG}.silver.defects": [f"{CATALOG}.bronze.jira_defects_raw"],
        f"{CATALOG}.gold.daily_execution_metrics": [f"{CATALOG}.silver.rides"],
        f"{CATALOG}.gold.scenario_health": [f"{CATALOG}.silver.rides"],
        f"{CATALOG}.gold.failure_mode_trends": [f"{CATALOG}.silver.rides"],
        f"{CATALOG}.gold.pudo_zone_reliability": [f"{CATALOG}.silver.rides"],
        f"{CATALOG}.ml.ride_features": [f"{CATALOG}.silver.rides", f"{CATALOG}.silver.ctm_logs"],
        f"{CATALOG}.ml.scenario_flakiness_features": [f"{CATALOG}.gold.scenario_health"],
        f"{CATALOG}.gold.ride_failure_predictions": [f"{CATALOG}.ml.ride_features"],
    }
    write_json(
        out / "unity_catalog/table_lineage.json",
        {
            "note": "Upstream tables for each derived table, as the lineage system would report it.",
            "lineage": [
                {"table_name": table, "upstreams": upstreams}
                for table, upstreams in sorted(lineage.items())
            ],
        },
    )


def emit_jobs(
    out: Path, job_ids: dict[str, int], runs: list[JobRun], sotf: SotfData
) -> None:
    jobs = []
    for spec in JOB_SPECS:
        tasks = []
        for task_spec in spec.tasks:
            tasks.append(
                {
                    "task_key": task_spec.task_key,
                    "depends_on": [{"task_key": key} for key in task_spec.depends_on],
                    "notebook_task": {
                        "notebook_path": f"{NOTEBOOK_ROOT}/{task_spec.notebook}",
                        "source": "WORKSPACE",
                        "base_parameters": {
                            **{key: value for key, value in task_spec.parameters},
                            "ingest_date": "{{ds}}",
                            "job_run_id": "{{job.run_id}}",
                        },
                    },
                    "job_cluster_key": f"{spec.name}_cluster",
                    "timeout_seconds": 7200,
                    "max_retries": 2,
                    "min_retry_interval_millis": 300000,
                    "retry_on_timeout": False,
                    "email_notifications": {},
                }
            )
        jobs.append(
            {
                "job_id": job_ids[spec.name],
                "creator_user_name": OWNER,
                "run_as_user_name": OWNER,
                "created_time": ms(sotf.now - timedelta(days=sotf.days + 45)),
                "settings": {
                    "name": spec.name,
                    "description": spec.description,
                    "tags": {"team": "sotf", "layer": "lakehouse"},
                    "email_notifications": {"on_failure": ["sotf-data@example.com"]},
                    "timeout_seconds": 14400,
                    "max_concurrent_runs": 1,
                    "schedule": {
                        "quartz_cron_expression": spec.quartz_cron,
                        "timezone_id": "UTC",
                        "pause_status": "UNPAUSED",
                    },
                    "format": "MULTI_TASK",
                    "tasks": tasks,
                    "job_clusters": [
                        {
                            "job_cluster_key": f"{spec.name}_cluster",
                            "new_cluster": new_cluster_spec(spec),
                        }
                    ],
                },
            }
        )
    write_json(out / "jobs/jobs.json", {"jobs": jobs, "has_more": False})

    def run_payload(run: JobRun, include_tasks: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": run.job_id,
            "run_id": run.run_id,
            "creator_user_name": OWNER,
            "number_in_job": run.number_in_job,
            "original_attempt_run_id": run.run_id,
            "state": {
                "life_cycle_state": "TERMINATED",
                "result_state": run.result,
                "state_message": "" if run.result == "SUCCESS" else "A task failed.",
                "user_cancelled_or_timedout": False,
            },
            "schedule": {
                "quartz_cron_expression": run.spec.quartz_cron,
                "timezone_id": "UTC",
                "pause_status": "UNPAUSED",
            },
            "start_time": ms(run.start),
            "setup_duration": 62000,
            "execution_duration": int((run.end - run.start).total_seconds() * 1000),
            "cleanup_duration": 8000,
            "end_time": ms(run.end),
            "trigger": "PERIODIC",
            "run_name": run.spec.name,
            "run_page_url": f"{WORKSPACE_URL}/?o=0#job/{run.job_id}/run/{run.run_id}",
            "run_type": "JOB_RUN",
            "attempt_number": 0,
            "format": "MULTI_TASK",
        }
        if include_tasks:
            payload["tasks"] = [
                {
                    "run_id": task.run_id,
                    "task_key": task.spec.task_key,
                    "depends_on": [{"task_key": key} for key in task.spec.depends_on],
                    "state": {
                        "life_cycle_state": task.state,
                        "result_state": task.result,
                        "state_message": task.error,
                        "user_cancelled_or_timedout": False,
                    },
                    "start_time": ms(task.start) if task.start else 0,
                    "setup_duration": 0,
                    "execution_duration": (
                        int((task.end - task.start).total_seconds() * 1000)
                        if task.start and task.end
                        else 0
                    ),
                    "cleanup_duration": 0,
                    "end_time": ms(task.end) if task.end else 0,
                    "cluster_instance": {
                        "cluster_id": run.cluster_id,
                        "spark_context_id": str(abs(hash(task.run_id)) % 10**19),
                    },
                    "notebook_task": {
                        "notebook_path": f"{NOTEBOOK_ROOT}/{task.spec.notebook}",
                        "source": "WORKSPACE",
                    },
                    "attempt_number": 0,
                }
                for task in run.tasks
            ]
            payload["job_clusters"] = [
                {
                    "job_cluster_key": f"{run.spec.name}_cluster",
                    "new_cluster": new_cluster_spec(run.spec),
                }
            ]
        return payload

    ordered = sorted(runs, key=lambda r: r.start, reverse=True)
    write_json(
        out / "jobs/runs.json",
        {"runs": [run_payload(run, include_tasks=False) for run in ordered], "has_more": False},
    )
    for run in runs:
        write_json(out / f"jobs/runs/{run.run_id}.json", run_payload(run, include_tasks=True))

    for run in runs:
        for task in run.tasks:
            if task.result != "FAILED":
                continue
            write_json(
                out / f"jobs/run_output/{task.run_id}.json",
                {
                    "error": task.error,
                    "error_trace": (
                        "---------------------------------------------------------------------------\n"
                        f"{task.error.split(':')[0]}                    Traceback (most recent call last)\n"
                        f"File {NOTEBOOK_ROOT}/{task.spec.notebook}, in cell 4\n"
                        f"----> 1 run_task('{task.spec.task_key}', ingest_date)\n\n"
                        f"{task.error}"
                    ),
                    "metadata": run_payload(run, include_tasks=True),
                    "notebook_output": {"result": None, "truncated": False},
                },
            )


def emit_clusters(out: Path, runs: list[JobRun], sotf: SotfData, rng: random.Random) -> None:
    shared_id = f"0417-{hex_id(rng, 6)}-{hex_id(rng, 8)}"
    recent = sorted(runs, key=lambda r: r.start, reverse=True)[:2]

    clusters = [
        {
            "cluster_id": shared_id,
            "cluster_name": "sotf-analytics-shared",
            "creator_user_name": OWNER,
            "spark_version": SPARK_VERSION,
            "node_type_id": NODE_TYPE,
            "driver_node_type_id": DRIVER_NODE_TYPE,
            "autoscale": {"min_workers": 1, "max_workers": 6},
            "autotermination_minutes": 60,
            "state": "RUNNING",
            "state_message": "",
            "start_time": ms(sotf.now - timedelta(hours=5)),
            "cluster_source": "UI",
            "data_security_mode": "USER_ISOLATION",
            "runtime_engine": "PHOTON",
            "custom_tags": {"team": "sotf", "purpose": "adhoc-analysis"},
            "spark_conf": {"spark.databricks.delta.preview.enabled": "true"},
            "aws_attributes": {"availability": "ON_DEMAND", "zone_id": "us-west-2a"},
        }
    ]
    for run in recent:
        clusters.append(
            {
                "cluster_id": run.cluster_id,
                "cluster_name": f"job-{run.job_id}-run-{run.run_id}",
                "creator_user_name": OWNER,
                "spark_version": SPARK_VERSION,
                "node_type_id": NODE_TYPE,
                "driver_node_type_id": DRIVER_NODE_TYPE,
                "autoscale": {
                    "min_workers": run.spec.cluster_workers[0],
                    "max_workers": run.spec.cluster_workers[1],
                },
                "autotermination_minutes": 0,
                "state": "TERMINATED",
                "state_message": "Job finished.",
                "start_time": ms(run.start),
                "terminated_time": ms(run.end),
                "cluster_source": "JOB",
                "data_security_mode": "SINGLE_USER",
                "runtime_engine": "PHOTON",
                "custom_tags": {"team": "sotf", "layer": "lakehouse"},
                "spark_conf": new_cluster_spec(run.spec)["spark_conf"],
                "aws_attributes": new_cluster_spec(run.spec)["aws_attributes"],
                "termination_reason": {"code": "JOB_FINISHED", "type": "SUCCESS"},
            }
        )
    write_json(out / "clusters/clusters.json", {"clusters": clusters})

    events = []
    base = sotf.now - timedelta(hours=5)
    for offset, (event_type, detail) in enumerate(
        (
            ("CREATING", {"cluster_size": {"autoscale": {"min_workers": 1, "max_workers": 6}}}),
            ("STARTING", {}),
            ("RUNNING", {"current_num_workers": 1, "target_num_workers": 1}),
            ("UPSIZE_COMPLETED", {"current_num_workers": 4, "target_num_workers": 4}),
            ("AUTOSCALING_STATS_REPORT", {"autoscaling_stats": {"total_instances": 4}}),
            ("DRIVER_HEALTHY", {}),
        )
    ):
        events.append(
            {
                "cluster_id": shared_id,
                "timestamp": ms(base + timedelta(minutes=offset * 7)),
                "type": event_type,
                "details": detail,
            }
        )
    write_json(
        out / f"clusters/events/{shared_id}.json",
        {"events": events, "total_count": len(events)},
    )


def emit_sql(
    out: Path,
    gold: dict[str, list[dict[str, Any]]],
    sotf: SotfData,
    rng: random.Random,
) -> None:
    warehouse_id = hex_id(rng, 16)
    write_json(
        out / "sql/warehouses.json",
        {
            "warehouses": [
                {
                    "id": warehouse_id,
                    "name": "sotf-sql-pro",
                    "cluster_size": "Medium",
                    "min_num_clusters": 1,
                    "max_num_clusters": 4,
                    "auto_stop_mins": 30,
                    "creator_name": OWNER,
                    "state": "RUNNING",
                    "num_clusters": 1,
                    "num_active_sessions": 3,
                    "enable_photon": True,
                    "enable_serverless_compute": True,
                    "warehouse_type": "PRO",
                    "spot_instance_policy": "COST_OPTIMIZED",
                    "jdbc_url": f"jdbc:spark://sotf.cloud.databricks.com:443/default;httpPath=/sql/1.0/warehouses/{warehouse_id}",
                    "tags": {"custom_tags": [{"key": "team", "value": "sotf"}]},
                }
            ]
        },
    )

    queries = []
    for index, query in enumerate(SAVED_QUERIES):
        queries.append(
            {
                "id": f"q{index + 1:03d}-{hex_id(rng, 8)}",
                "name": query.name,
                "description": query.description,
                "query": query.sql,
                "owner_user_name": rng.choice(USERS),
                "warehouse_id": warehouse_id,
                "created_at": iso(sotf.now - timedelta(days=rng.randint(20, 90))),
                "updated_at": iso(sotf.now - timedelta(days=rng.randint(0, 19))),
                "run_as_role": "viewer",
                "tags": ["sotf"],
            }
        )
    write_json(out / "sql/queries.json", {"results": queries, "count": len(queries)})

    (out / "sql").mkdir(parents=True, exist_ok=True)
    (out / "sql/sample_queries.sql").write_text(
        "\n\n".join(
            f"-- {query.name}\n-- {query.description}\n{query.sql};" for query in SAVED_QUERIES
        )
        + "\n",
        encoding="utf-8",
    )

    history = []
    for _ in range(QUERY_HISTORY_ENTRIES):
        query = rng.choice(SAVED_QUERIES)
        started = sotf.now - timedelta(
            days=rng.randint(0, 13), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        duration = rng.randint(280, 18000)
        failed = rng.random() < 0.05
        rows = 0 if failed else query.rows(gold, sotf)
        history.append(
            {
                "query_id": hex_id(rng, 16),
                "status": "FAILED" if failed else "FINISHED",
                "query_text": query.sql,
                "query_start_time_ms": ms(started),
                "execution_end_time_ms": ms(started) + duration,
                "query_end_time_ms": ms(started) + duration + rng.randint(10, 200),
                "user_name": rng.choice(USERS),
                "warehouse_id": warehouse_id,
                "duration": duration,
                "statement_type": "SELECT",
                "rows_produced": rows,
                "error_message": (
                    "[TABLE_OR_VIEW_NOT_FOUND] The table or view cannot be found."
                    if failed
                    else None
                ),
                "metrics": {
                    "total_time_ms": duration,
                    "read_bytes": rng.randint(200_000, 90_000_000),
                    "rows_produced_count": rows,
                    "photon_total_time_ms": int(duration * 0.7),
                },
            }
        )
    history.sort(key=lambda entry: entry["query_start_time_ms"], reverse=True)
    write_json(out / "sql/query_history.json", {"res": history, "has_next_page": False})


def emit_mlflow(
    out: Path,
    sotf: SotfData,
    rng: random.Random,
    predictions: list[dict[str, Any]],
    champion_version: int,
    champion_run_id: str,
    flakiness_rows: list[dict[str, Any]],
) -> None:
    accuracy = (
        round(sum(1 for p in predictions if p["is_correct"]) / len(predictions), 4)
        if predictions
        else 0.0
    )

    experiments = [
        {
            "experiment_id": "3021884471209376",
            "name": f"{NOTEBOOK_ROOT}/failure_mode_classifier",
            "artifact_location": "dbfs:/databricks/mlflow-tracking/3021884471209376",
            "lifecycle_stage": "active",
            "creation_time": ms(sotf.now - timedelta(days=sotf.days + 40)),
            "last_update_time": ms(sotf.now - timedelta(days=1)),
            "tags": [{"key": "mlflow.experimentType", "value": "MLFLOW_EXPERIMENT"}],
        },
        {
            "experiment_id": "3021884471209412",
            "name": f"{NOTEBOOK_ROOT}/scenario_flakiness",
            "artifact_location": "dbfs:/databricks/mlflow-tracking/3021884471209412",
            "lifecycle_stage": "active",
            "creation_time": ms(sotf.now - timedelta(days=sotf.days + 20)),
            "last_update_time": ms(sotf.now - timedelta(days=3)),
            "tags": [{"key": "mlflow.experimentType", "value": "MLFLOW_EXPERIMENT"}],
        },
    ]
    write_json(out / "mlflow/experiments.json", {"experiments": experiments})

    def make_run(
        experiment_id: str,
        run_id: str,
        run_name: str,
        started: datetime,
        params: dict[str, Any],
        metrics: dict[str, float],
        dataset: str,
        status: str = "FINISHED",
    ) -> dict[str, Any]:
        start_ms = ms(started)
        end_ms = start_ms + rng.randint(180_000, 2_400_000)
        return {
            "info": {
                "run_id": run_id,
                "run_uuid": run_id,
                "run_name": run_name,
                "experiment_id": experiment_id,
                "user_id": OWNER,
                "status": status,
                "start_time": start_ms,
                "end_time": end_ms,
                "artifact_uri": f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts",
                "lifecycle_stage": "active",
            },
            "data": {
                "params": [{"key": k, "value": str(v)} for k, v in sorted(params.items())],
                "metrics": [
                    {"key": k, "value": v, "timestamp": end_ms, "step": 0}
                    for k, v in sorted(metrics.items())
                ],
                "tags": [
                    {"key": "mlflow.source.type", "value": "NOTEBOOK"},
                    {"key": "mlflow.source.name", "value": f"{NOTEBOOK_ROOT}/05_flakiness_training"},
                    {"key": "mlflow.user", "value": OWNER},
                    {"key": "sotf.training_table", "value": dataset},
                ],
            },
            "inputs": {
                "dataset_inputs": [
                    {
                        "dataset": {
                            "name": dataset,
                            "digest": hex_id(rng, 8),
                            "source_type": "delta_table",
                            "source": json.dumps({"delta_table_name": dataset}),
                        },
                        "tags": [{"key": "mlflow.data.context", "value": "training"}],
                    }
                ]
            },
        }

    classifier_runs = []
    label_count = len(sotf.failure_modes) or 1
    for index in range(CLASSIFIER_RUN_COUNT):
        version = index + 1
        run_id = hex_id(rng, 32)
        is_champion = version == champion_version
        started = sotf.now - timedelta(
            days=(CLASSIFIER_RUN_COUNT - index) * 3, hours=rng.randint(0, 10)
        )
        # The champion's accuracy is the accuracy its predictions actually have.
        run_accuracy = accuracy if is_champion else round(max(accuracy - rng.uniform(0.02, 0.22), 0.15), 4)
        classifier_runs.append(
            make_run(
                experiment_id="3021884471209376",
                run_id=champion_run_id if is_champion else run_id,
                run_name=f"failure_mode_classifier_v{version}",
                started=started,
                params={
                    "model_type": "GradientBoostingClassifier",
                    "n_estimators": 100 + index * 25,
                    "max_depth": 3 + index % 4,
                    "learning_rate": round(0.15 - index * 0.01, 3),
                    "class_weight": "balanced",
                    "n_classes": label_count,
                    "feature_count": 11,
                },
                metrics={
                    "accuracy": run_accuracy,
                    "f1_macro": round(max(run_accuracy - rng.uniform(0.01, 0.06), 0.1), 4),
                    "log_loss": round(rng.uniform(0.35, 1.45), 4),
                    "training_rows": float(len(sotf.completed)),
                },
                dataset=f"{CATALOG}.ml.ride_features",
            )
        )

    flakiness_runs = []
    flaky_count = sum(1 for row in flakiness_rows if row["is_flaky"])
    for index in range(FLAKINESS_RUN_COUNT):
        started = sotf.now - timedelta(
            days=(FLAKINESS_RUN_COUNT - index) * 7, hours=rng.randint(0, 8)
        )
        flakiness_runs.append(
            make_run(
                experiment_id="3021884471209412",
                run_id=hex_id(rng, 32),
                run_name=f"scenario_flakiness_w{index + 1}",
                started=started,
                params={
                    "model_type": "GradientBoostingClassifier",
                    "n_estimators": 200,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "test_size": 0.25,
                },
                metrics={
                    "f1": round(rng.uniform(0.62, 0.89), 4),
                    "roc_auc": round(rng.uniform(0.71, 0.95), 4),
                    "positive_rate": round(flaky_count / len(flakiness_rows), 4)
                    if flakiness_rows
                    else 0.0,
                    "training_rows": float(len(flakiness_rows)),
                },
                dataset=f"{CATALOG}.ml.scenario_flakiness_features",
            )
        )

    all_runs = classifier_runs + flakiness_runs
    write_json(out / "mlflow/runs.json", {"runs": all_runs})
    for run in all_runs:
        write_json(out / f"mlflow/runs/{run['info']['run_id']}.json", {"run": run})

    classifier_versions = []
    for index, run in enumerate(classifier_runs, start=1):
        if index == champion_version:
            stage = "Production"
        elif index == len(classifier_runs) and index != champion_version:
            stage = "Staging"
        else:
            stage = "Archived"
        classifier_versions.append(
            {
                "name": MODEL_NAME,
                "version": str(index),
                "creation_timestamp": run["info"]["end_time"],
                "last_updated_timestamp": run["info"]["end_time"] + 60000,
                "user_id": OWNER,
                "current_stage": stage,
                "description": "Predicts the SOTF failure mode from ride features.",
                "source": f"{run['info']['artifact_uri']}/model",
                "run_id": run["info"]["run_id"],
                "status": "READY",
                "tags": [{"key": "sotf.label_space", "value": ",".join(sotf.failure_modes)}],
            }
        )

    flakiness_versions = [
        {
            "name": FLAKINESS_MODEL_NAME,
            "version": str(index),
            "creation_timestamp": run["info"]["end_time"],
            "last_updated_timestamp": run["info"]["end_time"] + 60000,
            "user_id": OWNER,
            "current_stage": "Production" if index == len(flakiness_runs) else "Archived",
            "description": "Predicts whether a scenario is flaky rather than broken.",
            "source": f"{run['info']['artifact_uri']}/model",
            "run_id": run["info"]["run_id"],
            "status": "READY",
            "tags": [],
        }
        for index, run in enumerate(flakiness_runs, start=1)
    ]

    versions = classifier_versions + flakiness_versions
    write_json(out / "mlflow/model_versions.json", {"model_versions": versions})

    def latest(name: str, all_versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_stage: dict[str, dict[str, Any]] = {}
        for version in all_versions:
            if version["name"] != name:
                continue
            stage = version["current_stage"]
            if stage not in by_stage or int(version["version"]) > int(by_stage[stage]["version"]):
                by_stage[stage] = version
        return [by_stage[stage] for stage in sorted(by_stage)]

    write_json(
        out / "mlflow/registered_models.json",
        {
            "registered_models": [
                {
                    "name": MODEL_NAME,
                    "creation_timestamp": classifier_versions[0]["creation_timestamp"],
                    "last_updated_timestamp": classifier_versions[-1]["last_updated_timestamp"],
                    "user_id": OWNER,
                    "description": (
                        "Suggests a failure mode for a failed ride so triage starts from a "
                        "hypothesis. Advisory only: the CTM logs remain the source of truth."
                    ),
                    "latest_versions": latest(MODEL_NAME, versions),
                    "tags": [{"key": "team", "value": "sotf"}],
                },
                {
                    "name": FLAKINESS_MODEL_NAME,
                    "creation_timestamp": flakiness_versions[0]["creation_timestamp"],
                    "last_updated_timestamp": flakiness_versions[-1]["last_updated_timestamp"],
                    "user_id": OWNER,
                    "description": "Separates flaky scenarios from consistently broken ones.",
                    "latest_versions": latest(FLAKINESS_MODEL_NAME, versions),
                    "tags": [{"key": "team", "value": "sotf"}],
                },
            ]
        },
    )


def emit_workspace(out: Path, sotf: SotfData) -> None:
    objects = [
        {
            "object_type": "DIRECTORY",
            "path": NOTEBOOK_ROOT,
            "object_id": 4210001,
            "resource_id": "4210001",
        }
    ]
    for index, name in enumerate(sorted(NOTEBOOK_SOURCE), start=1):
        objects.append(
            {
                "object_type": "NOTEBOOK",
                "path": f"{NOTEBOOK_ROOT}/{name}",
                "language": "PYTHON",
                "object_id": 4210001 + index,
                "resource_id": str(4210001 + index),
                "created_at": ms(sotf.now - timedelta(days=sotf.days + 30)),
                "modified_at": ms(sotf.now - timedelta(days=index * 2)),
            }
        )
    write_json(out / "workspace/objects.json", {"objects": objects})

    for name, source in NOTEBOOK_SOURCE.items():
        path = out / f"workspace/notebooks/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def emit_cross_reference(
    out: Path,
    sotf: SotfData,
    gold: dict[str, list[dict[str, Any]]],
    runs: list[JobRun],
) -> None:
    predictions = {row["trigger_id"]: row for row in gold["ride_failure_predictions"]}
    features = {row["trigger_id"] for row in gold["ride_features"]}

    inference_runs = {
        run.window_start.strftime("%Y-%m-%d"): run.run_id
        for run in runs
        if run.spec.name == "sotf_failure_mode_inference"
    }

    entries = []
    for ride in sotf.rides:
        trigger_id = ride["trigger_id"]
        prediction = predictions.get(trigger_id)
        started = ride["_started"]
        entries.append(
            {
                "trigger_id": trigger_id,
                "ride_status": ride["ride_status"],
                "in_ride_features": trigger_id in features,
                "daily_metrics_key": (
                    {
                        "metric_date": started.strftime("%Y-%m-%d"),
                        "environment": ride["environment"],
                        "ride_service": ride["ride_service"],
                    }
                    if started and ride["ride_status"] in ("Passed", "Failed")
                    else None
                ),
                "scenario_health_key": ride["scenario"],
                "pudo_zone_key": ride["pudo_zone"],
                "failure_trend_key": (
                    {"metric_date": started.strftime("%Y-%m-%d"), "failure_mode": ride["failure_mode"]}
                    if started and ride["failure_mode"]
                    else None
                ),
                "prediction": (
                    {
                        "predicted_failure_mode": prediction["predicted_failure_mode"],
                        "actual_failure_mode": prediction["actual_failure_mode"],
                        "confidence": prediction["confidence"],
                        "is_correct": prediction["is_correct"],
                        "job_run_id": prediction["job_run_id"],
                    }
                    if prediction
                    else None
                ),
                "inference_job_run_id": (
                    inference_runs.get(started.strftime("%Y-%m-%d")) if started else None
                ),
            }
        )

    write_json(
        out / "cross_reference_databricks.json",
        {
            "note": (
                "Where each SOTF ride shows up in the lakehouse. Only completed rides reach "
                "ml.ride_features, and only failed ones get a prediction. A prediction may be "
                "wrong on purpose: is_correct records whether the model got it right."
            ),
            "rides": entries,
        },
    )


API_INDEX = {
    "workspace_url": WORKSPACE_URL,
    "endpoints": {
        "GET /api/2.1/unity-catalog/catalogs": "unity_catalog/catalogs.json",
        "GET /api/2.1/unity-catalog/schemas": "unity_catalog/schemas.json",
        "GET /api/2.1/unity-catalog/tables": "unity_catalog/tables.json",
        "GET /api/2.0/lineage-tracking/table-lineage": "unity_catalog/table_lineage.json",
        "GET /api/2.1/jobs/list": "jobs/jobs.json",
        "GET /api/2.1/jobs/runs/list": "jobs/runs.json",
        "GET /api/2.1/jobs/runs/get?run_id=": "jobs/runs/{run_id}.json",
        "GET /api/2.1/jobs/runs/get-output?run_id=": "jobs/run_output/{task_run_id}.json",
        "GET /api/2.1/clusters/list": "clusters/clusters.json",
        "GET /api/2.1/clusters/events": "clusters/events/{cluster_id}.json",
        "GET /api/2.0/sql/warehouses": "sql/warehouses.json",
        "GET /api/2.0/sql/history/queries": "sql/query_history.json",
        "GET /api/2.0/preview/sql/queries": "sql/queries.json",
        "GET /api/2.0/mlflow/experiments/search": "mlflow/experiments.json",
        "GET /api/2.0/mlflow/runs/search": "mlflow/runs.json",
        "GET /api/2.0/mlflow/runs/get?run_id=": "mlflow/runs/{run_id}.json",
        "GET /api/2.0/mlflow/registered-models/search": "mlflow/registered_models.json",
        "GET /api/2.0/mlflow/model-versions/search": "mlflow/model_versions.json",
        "GET /api/2.0/workspace/list": "workspace/objects.json",
        "GET /api/2.0/workspace/export?path=": "workspace/notebooks/{name}.py",
    },
    "table_contents": {
        "note": "Delta table contents as JSONL, one object per row. Bronze and silver are "
        "metadata-only because their rows already exist in the SOTF dataset; the "
        "sotf.row_count table property records what the count would be.",
        "populated": [spec.full_name for spec in GOLD_AND_ML],
    },
    "notes": [
        "Read-only fixtures. No job-submit, cluster-create or model-transition payloads.",
        "Credentials appear only as {{secrets/scope/key}} references, never as values.",
        "Model versions use the workspace registry's current_stage; a Unity Catalog "
        "registered model would use aliases instead.",
    ],
}


# ---------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------

TABLE_PATTERN = re.compile(rf"{CATALOG}\.(\w+)\.(\w+)")


def run_self_check(
    out: Path,
    sotf: SotfData,
    gold: dict[str, list[dict[str, Any]]],
    runs: list[JobRun],
    job_ids: dict[str, int],
) -> list[str]:
    problems: list[str] = []

    tables = json.loads((out / "unity_catalog/tables.json").read_text(encoding="utf-8"))["tables"]
    known_tables = {table["full_name"] for table in tables}
    schemas = {
        schema["full_name"]
        for schema in json.loads((out / "unity_catalog/schemas.json").read_text(encoding="utf-8"))[
            "schemas"
        ]
    }

    for table in tables:
        parent = f"{table['catalog_name']}.{table['schema_name']}"
        if parent not in schemas:
            problems.append(f"{table['full_name']}: schema {parent} is not registered")

    # Row counts declared in the catalog must match what was actually written.
    by_name = {table["full_name"]: table for table in tables}
    for spec in GOLD_AND_ML:
        path = out / spec.relative_path
        if not path.is_file():
            problems.append(f"{spec.full_name}: missing {spec.relative_path}")
            continue
        actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        declared = int(by_name[spec.full_name]["properties"]["sotf.row_count"])
        if actual != declared:
            problems.append(
                f"{spec.full_name}: wrote {actual} rows but the catalog declares {declared}"
            )

    expected_rides = sotf.counts.get("rides")
    if expected_rides is not None:
        declared = int(by_name[f"{CATALOG}.silver.rides"]["properties"]["sotf.row_count"])
        if declared != expected_rides:
            problems.append(
                f"silver.rides declares {declared} rows, SOTF dataset has {expected_rides}"
            )

    # Predictions must be real failed rides, and only failed rides.
    failed_ids = {ride["trigger_id"] for ride in sotf.failed}
    predicted_ids = {row["trigger_id"] for row in gold["ride_failure_predictions"]}
    if predicted_ids - failed_ids:
        problems.append(
            f"{len(predicted_ids - failed_ids)} prediction(s) reference a ride that did not fail"
        )
    if failed_ids - predicted_ids:
        problems.append(f"{len(failed_ids - predicted_ids)} failed ride(s) were never scored")

    # The accuracy logged in MLflow must be the accuracy the predictions have.
    predictions = gold["ride_failure_predictions"]
    if predictions:
        actual_accuracy = round(
            sum(1 for p in predictions if p["is_correct"]) / len(predictions), 4
        )
        versions = json.loads(
            (out / "mlflow/model_versions.json").read_text(encoding="utf-8")
        )["model_versions"]
        champion = next(
            (
                v
                for v in versions
                if v["name"] == MODEL_NAME and v["current_stage"] == "Production"
            ),
            None,
        )
        if champion is None:
            problems.append(f"{MODEL_NAME} has no Production version")
        else:
            run_file = out / f"mlflow/runs/{champion['run_id']}.json"
            if not run_file.is_file():
                problems.append(f"{MODEL_NAME} v{champion['version']} points at a missing run")
            else:
                run = json.loads(run_file.read_text(encoding="utf-8"))["run"]
                logged = next(
                    (m["value"] for m in run["data"]["metrics"] if m["key"] == "accuracy"), None
                )
                if logged != actual_accuracy:
                    problems.append(
                        f"champion logs accuracy {logged} but its predictions score "
                        f"{actual_accuracy}"
                    )
                if {row["mlflow_run_id"] for row in predictions} != {champion["run_id"]}:
                    problems.append("predictions do not all cite the champion run id")

    # Metrics coverage must match the rides that could be aggregated.
    metric_rides = sum(row["rides"] for row in gold["daily_execution_metrics"])
    if metric_rides != len(sotf.completed):
        problems.append(
            f"daily_execution_metrics covers {metric_rides} rides, expected {len(sotf.completed)}"
        )

    # Job runs must be internally consistent.
    known_job_ids = set(job_ids.values())
    for run in runs:
        if run.job_id not in known_job_ids:
            problems.append(f"run {run.run_id}: unknown job_id {run.job_id}")
        if not (out / f"jobs/runs/{run.run_id}.json").is_file():
            problems.append(f"run {run.run_id}: missing jobs/runs/{run.run_id}.json")
        failed_tasks = [t for t in run.tasks if t.result == "FAILED"]
        if run.result == "FAILED" and not failed_tasks:
            problems.append(f"run {run.run_id}: marked FAILED but no task failed")
        if run.result == "SUCCESS" and failed_tasks:
            problems.append(f"run {run.run_id}: marked SUCCESS but a task failed")
        for task in failed_tasks:
            if not (out / f"jobs/run_output/{task.run_id}.json").is_file():
                problems.append(f"task run {task.run_id}: failed with no run output")

    # Every table named in a query or a notebook must exist.
    sources = {"query_history": (out / "sql/query_history.json").read_text(encoding="utf-8")}
    for name in NOTEBOOK_SOURCE:
        sources[name] = (out / f"workspace/notebooks/{name}.py").read_text(encoding="utf-8")
    # The silver notebook reads a bronze landing table this catalog does not model.
    allowed_extra = {f"{CATALOG}.bronze.rides_raw"}
    for origin, text in sources.items():
        for schema, table in set(TABLE_PATTERN.findall(text)):
            full_name = f"{CATALOG}.{schema}.{table}"
            if full_name not in known_tables and full_name not in allowed_extra:
                problems.append(f"{origin} references unknown table {full_name}")

    # Credentials must be references, not values.
    secret_pattern = re.compile(r"(password|api_?key|token|secret)", re.IGNORECASE)
    clusters = json.loads((out / "clusters/clusters.json").read_text(encoding="utf-8"))["clusters"]
    jobs = json.loads((out / "jobs/jobs.json").read_text(encoding="utf-8"))["jobs"]
    specs = [cluster.get("spark_conf", {}) for cluster in clusters]
    specs += [
        job_cluster["new_cluster"].get("spark_conf", {})
        for job in jobs
        for job_cluster in job["settings"]["job_clusters"]
    ]
    for conf in specs:
        for key, value in conf.items():
            if secret_pattern.search(key) and not str(value).startswith("{{secrets/"):
                problems.append(f"spark_conf {key} holds a literal value instead of a secret ref")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mock Databricks data aligned with an existing SOTF mock dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sotf-data", type=Path, default=HERE / "out", help="output of generate_sotf_mock_data.py"
    )
    parser.add_argument("--out", type=Path, default=HERE / "out_databricks")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    parser.add_argument(
        "--failure-rate", type=float, default=0.06, help="share of job runs that fail"
    )
    parser.add_argument(
        "--model-accuracy",
        type=float,
        default=0.78,
        help="target accuracy of the failure-mode classifier",
    )
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

    job_ids, job_runs = build_job_runs(sotf, args, rng)
    inference_run_by_date = {
        run.window_start.strftime("%Y-%m-%d"): run.run_id
        for run in job_runs
        if run.spec.name == "sotf_failure_mode_inference"
    }

    champion_run_id = hex_id(rng, 32)
    champion_version = CHAMPION_VERSION

    scenario_health = build_scenario_health(sotf)
    gold: dict[str, list[dict[str, Any]]] = {
        "daily_execution_metrics": build_daily_execution_metrics(sotf),
        "scenario_health": scenario_health,
        "failure_mode_trends": build_failure_mode_trends(sotf),
        "pudo_zone_reliability": build_pudo_zone_reliability(sotf),
        "ride_features": build_ride_features(sotf),
        "scenario_flakiness_features": build_flakiness_features(sotf, scenario_health),
        "ride_failure_predictions": build_predictions(
            sotf,
            rng,
            args.model_accuracy,
            champion_version,
            champion_run_id,
            inference_run_by_date,
        ),
    }

    row_counts: dict[str, int] = {}
    for spec in TABLE_SPECS:
        if spec.row_source == "data":
            row_counts[spec.full_name] = len(gold[spec.name])
        else:
            row_counts[spec.full_name] = int(sotf.counts.get(spec.row_source, 0))

    for spec in GOLD_AND_ML:
        write_jsonl(out / spec.relative_path, gold[spec.name])

    emit_unity_catalog(out, sotf, row_counts, rng)
    emit_jobs(out, job_ids, job_runs, sotf)
    emit_clusters(out, job_runs, sotf, rng)
    emit_sql(out, gold, sotf, rng)
    emit_mlflow(
        out,
        sotf,
        rng,
        gold["ride_failure_predictions"],
        champion_version,
        champion_run_id,
        gold["scenario_flakiness_features"],
    )
    emit_workspace(out, sotf)
    emit_cross_reference(out, sotf, gold, job_runs)
    write_json(out / "api_index.json", API_INDEX)

    predictions = gold["ride_failure_predictions"]
    accuracy = (
        round(sum(1 for p in predictions if p["is_correct"]) / len(predictions), 4)
        if predictions
        else 0.0
    )
    counts = {
        "catalogs": 1,
        "schemas": len(SCHEMAS),
        "tables": len(TABLE_SPECS),
        "tables_populated": len(GOLD_AND_ML),
        "gold_rows": {spec.name: len(gold[spec.name]) for spec in GOLD_AND_ML},
        "jobs": len(JOB_SPECS),
        "job_runs": len(job_runs),
        "job_runs_failed": sum(1 for run in job_runs if run.result == "FAILED"),
        "task_runs": sum(len(run.tasks) for run in job_runs),
        "mlflow_runs": CLASSIFIER_RUN_COUNT + FLAKINESS_RUN_COUNT,
        "model_versions": CLASSIFIER_RUN_COUNT + FLAKINESS_RUN_COUNT,
        "notebooks": len(NOTEBOOK_SOURCE),
        "saved_queries": len(SAVED_QUERIES),
        "query_history_entries": QUERY_HISTORY_ENTRIES,
        "rides_referenced": len(gold["ride_features"]),
        "classifier_accuracy": accuracy,
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
                "model_accuracy_target": args.model_accuracy,
                "catalog": CATALOG,
            },
            "counts": counts,
        },
    )

    print(f"Wrote Databricks mock data to {out}")
    print(
        "  {tables} tables ({tables_populated} populated), {jobs} jobs, {job_runs} runs, "
        "{task_runs} task runs, {notebooks} notebooks, "
        "{rides_referenced} SOTF rides in the feature table".format(**counts)
    )
    print(f"  failure-mode classifier accuracy: {accuracy}")

    if args.skip_self_check:
        return 0

    problems = run_self_check(out, sotf, gold, job_runs, job_ids)
    if problems:
        print(f"  self-check FAILED with {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    - {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return 1

    print("  self-check passed: catalog row counts, predictions and MLflow metrics agree")
    print(f"  join table: {out / 'cross_reference_databricks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
