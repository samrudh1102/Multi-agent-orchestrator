#!/usr/bin/env python3
"""Generate mock Amazon RDS for PostgreSQL data for the SOTF test-mgmt database.

This script mocks **Postgres only**. It does not regenerate S3, ClickHouse, Jira
or TestRail data -- it *reads* the dataset produced by
``generate_sotf_mock_data.py`` and projects it into the relational shape the SOTF
application actually persists, which is where the scenarios authored in the app
live:

    schema/       DDL: types, tables, indexes, routines and NOTIFY triggers
    tables/       one CSV per table, ready for psql \\copy
    seed.sql      the same rows as batched INSERT statements
    notify/       sample LISTEN/NOTIFY payloads the app's db_listener receives
    pg_catalog/   pg_stat_user_tables, pg_stat_statements, pg_indexes, sizes
    rds/          describe-db-instances / parameters / snapshots / events
                  and CloudWatch datapoints for the instance

The scenario is the centre of gravity: ``test_scenarios`` holds what an author
created in the SOTF UI, ``test_scenario_version_history`` holds every earlier
version of it, ``test_suite_scenarios`` links it into suites, and every
execution and ride in the SOTF dataset traces back to one scenario row.
``cross_reference_postgres.json`` is the join table.

Table and column names come from ``tcc/api-py/database/init_db.sh`` and the
SQLAlchemy models under ``test-mgmt/app/model/``; see README.md for the source of
each and for where those two disagree.

Usage:
    python3 generate_postgres_mock_data.py --clean
    python3 generate_postgres_mock_data.py --sotf-data ./out --out ./out_postgres
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import uuid
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COPY ... WITH (FORMAT csv, NULL '\N'): the token has no quotes or commas, so
# csv.writer leaves it bare and Postgres reads it back as NULL.
NULL_TOKEN = "\\N"

# Dollar-quote tag used for JSONB literals in seed.sql. The self-check fails if
# any payload contains it.
DOLLAR_TAG = "$sotf$"

# Status literals as the SQLAlchemy models write them (Title Case). init_db.sh
# declares UPPER_SNAKE enum *types* but gives every status column VARCHAR(50),
# so the values the app writes are these.
SCENARIO_STATUS = ("Active", "Inactive", "Deprecated")
EXECUTION_STATUS = (
    "Queued",
    "Running",
    "Passed",
    "Invalid",
    "Partially Failed",
    "Failed",
    "Scheduled",
)
RIDE_STATUS = ("Queued", "Invalid", "Running", "Passed", "Failed", "Scheduled")

# execution_dto.ExecutionStates: the lifecycle value carried in the
# executions.scenario NOTIFY payload, separate from the result.
EXECUTION_STATE_BY_STATUS = {
    "Queued": "QUEUED",
    "Scheduled": "SCHEDULED",
    "Running": "RUNNING",
    "Invalid": "INVALID",
    "Passed": "COMPLETED",
    "Failed": "COMPLETED",
    "Partially Failed": "COMPLETED",
}

# Versions the SOTF scenario catalog draws from, in release order.
VERSION_LADDER = (1.0, 1.1, 1.2, 2.0)

VERSION_COMMENTS = (
    "Initial version created in the SOTF scenario builder",
    "Split the pickup wait into an explicit step",
    "Added a Mission Control assertion after assignment",
    "Reworked for the new rider app navigation",
    "Tightened the drop-off verification wording",
)

RECURRENCE_PATTERNS = ("MON,WED,FRI", "TUE,THU", "MON,TUE,WED,THU,FRI", "SAT", "MON")

AUTHORS = (
    "a.patel@example.com",
    "j.nguyen@example.com",
    "m.oconnor@example.com",
    "s.desai@example.com",
    "l.zhang@example.com",
    "r.gomez@example.com",
)

# DatabaseConnections is a pure function of the day's ride volume so the
# self-check can recompute it. Everything else carries jitter.
CONNECTIONS_BASE = 12.0
CONNECTIONS_PER_RIDE = 0.8


# ---------------------------------------------------------------------------
# Table catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Col:
    name: str
    sql_type: str
    nullable: bool = True
    default: str | None = None
    comment: str = ""

    @property
    def is_json(self) -> bool:
        return self.sql_type in ("JSONB", "JSON")


@dataclass(frozen=True)
class Table:
    name: str
    comment: str
    columns: tuple[Col, ...]
    primary_key: str = "id"
    # (column, referenced_table, on_delete)
    foreign_keys: tuple[tuple[str, str, str], ...] = ()
    # (constraint_name, column tuple)
    uniques: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # (constraint_name, expression)
    checks: tuple[tuple[str, str], ...] = ()
    # (index_name, "ON <table> (...)" suffix)
    indexes: tuple[tuple[str, str], ...] = ()

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def json_columns(self) -> set[str]:
        return {c.name for c in self.columns if c.is_json}


def col(name: str, sql_type: str, **kwargs: Any) -> Col:
    return Col(name, sql_type, **kwargs)


TABLES: tuple[Table, ...] = (
    Table(
        name="test_scenarios",
        comment="Scenarios authored in the SOTF app. One row per scenario, holding the current version.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("name", "VARCHAR(255)", nullable=False),
            col("metadata", "JSONB", comment="{testrail_id, labels}"),
            col("test_steps", "JSONB", comment="Ordered steps with action_category and action_name"),
            col("version", "NUMERIC(3,1)", default="1"),
            col("version_comment", "VARCHAR(500)"),
            col("status", "VARCHAR(50)", nullable=False, default="'Active'"),
            col("created_at", "TIMESTAMPTZ", default="CURRENT_TIMESTAMP"),
            col("updated_at", "TIMESTAMPTZ", default="CURRENT_TIMESTAMP"),
            col("deleted_at", "TIMESTAMPTZ", comment="Set when the scenario is deprecated"),
            col("created_by", "VARCHAR(100)"),
            col("updated_by", "VARCHAR(100)"),
        ),
        checks=(
            ("chk_test_scenarios_status", "status IN ('Active', 'Inactive', 'Deprecated')"),
        ),
        indexes=(
            ("idx_test_scenarios_status", "ON test_scenarios (status)"),
            ("idx_test_scenarios_created_by", "ON test_scenarios (created_by)"),
            ("idx_test_scenarios_updated_at", "ON test_scenarios (updated_at DESC)"),
            ("idx_metadata_jsonb_gin", "ON test_scenarios USING GIN (metadata)"),
            ("idx_teststeps_jsonb_gin", "ON test_scenarios USING GIN (test_steps)"),
        ),
    ),
    Table(
        name="test_suites",
        comment="Named collections of scenarios.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("name", "VARCHAR(255)", nullable=False),
            col("description", "VARCHAR(500)"),
            col("created_at", "TIMESTAMPTZ", default="CURRENT_TIMESTAMP"),
            col("created_by", "VARCHAR(100)"),
            col("updated_at", "TIMESTAMPTZ"),
            col("updated_by", "VARCHAR(100)"),
        ),
    ),
    Table(
        name="test_scenario_version_history",
        comment="Every earlier version of a scenario, snapshotted when the author saved a new one.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("scenario_id", "UUID", nullable=False),
            col("name", "VARCHAR(255)", nullable=False),
            col("metadata", "JSONB"),
            col("test_steps", "JSONB"),
            col("version", "NUMERIC(3,1)"),
            col("version_comment", "VARCHAR(500)"),
            col("status", "VARCHAR(50)"),
            col("versioned_by", "VARCHAR(100)"),
            col("versioned_at", "TIMESTAMPTZ", default="CURRENT_TIMESTAMP"),
        ),
        foreign_keys=(("scenario_id", "test_scenarios", "CASCADE"),),
        uniques=(("uq_scenario_version", ("scenario_id", "version")),),
        indexes=(
            (
                "idx_scenario_version_history_scenario_id",
                "ON test_scenario_version_history (scenario_id, version DESC)",
            ),
        ),
    ),
    Table(
        name="test_suite_scenarios",
        comment="Suite membership. Rows are soft deleted rather than removed.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("suite_id", "UUID", nullable=False),
            col("scenario_id", "UUID", nullable=False),
            col("deleted", "BOOLEAN", nullable=False, default="FALSE"),
            col("created_at", "TIMESTAMPTZ", default="CURRENT_TIMESTAMP"),
            col("created_by", "VARCHAR(100)"),
            col("deleted_at", "TIMESTAMPTZ"),
            col("deleted_by", "VARCHAR(100)"),
        ),
        foreign_keys=(
            ("suite_id", "test_suites", "CASCADE"),
            ("scenario_id", "test_scenarios", "CASCADE"),
        ),
        uniques=(("uq_suite_scenario", ("suite_id", "scenario_id")),),
        indexes=(
            ("idx_test_suite_scenarios_suite_id", "ON test_suite_scenarios (suite_id)"),
        ),
    ),
    Table(
        name="test_executions",
        comment="One row per scenario execution request. Status is derived from its rides.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("scenario_id", "UUID", nullable=False),
            col("suite_id", "UUID", comment="Null for an adhoc single-scenario execution"),
            col("scenario_version", "NUMERIC(3,1)"),
            col("payload", "JSONB", comment="ExecutionParameters as submitted by the UI"),
            col("status", "VARCHAR(50)", nullable=False, default="'Queued'"),
            col("no_of_triggers", "INTEGER", default="1"),
            col("schedule_from", "TIMESTAMPTZ"),
            col("schedule_to", "TIMESTAMPTZ"),
            col("recurrence_days", "VARCHAR(50)"),
            col("created_at", "TIMESTAMPTZ", nullable=False, default="CURRENT_TIMESTAMP"),
            col("created_by", "VARCHAR(150)", nullable=False),
            col("scheduled_at", "TIMESTAMPTZ", nullable=False, default="CURRENT_TIMESTAMP"),
        ),
        foreign_keys=(
            ("scenario_id", "test_scenarios", "CASCADE"),
            ("suite_id", "test_suites", "CASCADE"),
        ),
        checks=(
            (
                "chk_test_executions_status",
                "status IN ('Queued', 'Running', 'Passed', 'Invalid', "
                "'Partially Failed', 'Failed', 'Scheduled')",
            ),
        ),
        indexes=(
            ("idx_test_executions_scenario_id", "ON test_executions (scenario_id)"),
            ("idx_test_executions_suite_id", "ON test_executions (suite_id)"),
            ("idx_test_executions_status", "ON test_executions (status)"),
            ("idx_test_executions_created_at", "ON test_executions (created_at DESC)"),
            ("idx_payload_jsonb_gin", "ON test_executions USING GIN (payload)"),
        ),
    ),
    Table(
        name="scheduled_test_executions",
        comment="Recurring executions the scheduler enqueues on a cron-like recurrence.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("scenario_id", "UUID", nullable=False),
            col("suite_id", "UUID"),
            col("payload", "JSONB"),
            col("no_of_triggers", "INTEGER"),
            col("schedule_from", "TIMESTAMPTZ"),
            col("schedule_to", "TIMESTAMPTZ"),
            col("recurrence_days", "VARCHAR(50)"),
            col("created_at", "TIMESTAMPTZ", nullable=False, default="CURRENT_TIMESTAMP"),
            col("created_by", "VARCHAR(150)", nullable=False),
            col("is_deleted", "BOOLEAN", nullable=False, default="FALSE"),
            col("last_enqueued_at", "TIMESTAMPTZ"),
        ),
        foreign_keys=(
            ("scenario_id", "test_scenarios", "CASCADE"),
            ("suite_id", "test_suites", "CASCADE"),
        ),
        indexes=(
            (
                "idx_scheduled_test_executions_active",
                "ON scheduled_test_executions (schedule_from) WHERE is_deleted = FALSE",
            ),
        ),
    ),
    Table(
        name="rides",
        comment="One row per trigger. id is the trigger_id every other SOTF system keys on.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()", comment="trigger_id"),
            col("execution_id", "UUID", nullable=False),
            col("status", "VARCHAR(50)", nullable=False, default="'Queued'"),
            col("metadata", "JSONB", comment="{actual_pick_up, actual_drop_off, vehicle_id, testrail{...}}"),
            col("summary", "JSONB", comment="RideSummaryLiteDto: steps, artifacts, timings"),
            col("device_id", "VARCHAR(100)"),
            col("host_machine", "VARCHAR(100)"),
            col("reattempts_count", "INTEGER", default="1"),
            col("start_time", "TIMESTAMPTZ"),
            col("end_time", "TIMESTAMPTZ"),
            col("mission_id", "UUID"),
            col("created_at", "TIMESTAMPTZ", nullable=False, default="CURRENT_TIMESTAMP"),
            col("updated_at", "TIMESTAMPTZ", nullable=False, default="CURRENT_TIMESTAMP"),
        ),
        foreign_keys=(("execution_id", "test_executions", "CASCADE"),),
        checks=(
            (
                "chk_rides_status",
                "status IN ('Queued', 'Invalid', 'Running', 'Passed', 'Failed', 'Scheduled')",
            ),
        ),
        indexes=(
            ("idx_rides_execution_id", "ON rides (execution_id)"),
            ("idx_rides_status", "ON rides (status)"),
            ("idx_rides_start_time", "ON rides (start_time DESC)"),
            ("idx_summary_jsonb_gin", "ON rides USING GIN (summary)"),
            ("idx_rides_metadata_jsonb_gin", "ON rides USING GIN (metadata)"),
        ),
    ),
    Table(
        name="ride_events",
        comment="Append-only lifecycle events for a ride, written as the trigger progresses.",
        columns=(
            col("id", "UUID", nullable=False, default="gen_random_uuid()"),
            col("ride_id", "UUID", nullable=False),
            col("event", "VARCHAR(255)"),
            col("event_time", "TIMESTAMPTZ"),
            col("event_data", "JSONB"),
        ),
        foreign_keys=(("ride_id", "rides", "CASCADE"),),
        indexes=(
            ("idx_ride_events_ride_id", "ON ride_events (ride_id)"),
            ("idx_ride_events_event_time", "ON ride_events (event_time DESC)"),
            ("idx_eventdata_jsonb_gin", "ON ride_events USING GIN (event_data)"),
        ),
    ),
)

TABLE_BY_NAME = {table.name: table for table in TABLES}


# ---------------------------------------------------------------------------
# Routines the application calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Routine:
    name: str
    kind: str  # PROCEDURE or FUNCTION
    signature: str
    called_from: str
    emitted: bool


ROUTINES: tuple[Routine, ...] = (
    Routine(
        "update_ride_summary_proc",
        "PROCEDURE",
        "(p_trigger_id uuid, p_summary jsonb)",
        "rides_repository.update_ride_summary",
        True,
    ),
    Routine(
        "update_ride_invalid_status_proc",
        "PROCEDURE",
        "(p_trigger_id uuid, p_status text)",
        "rides_repository.update_ride_invalid_status",
        True,
    ),
    Routine(
        "update_ride_realtime_step_status_proc",
        "PROCEDURE",
        "(p_trigger_id uuid, p_step_status text, p_step_id text, p_duration time, "
        "p_location text, p_msg text)",
        "rides_repository.update_ride_realtime_step_status",
        True,
    ),
    Routine(
        "update_ride_metadata_proc",
        "PROCEDURE",
        "(p_trigger_id uuid, p_host_machine text, p_device_id text, p_metadata jsonb, "
        "p_mission_id uuid)",
        "rides_repository.update_ride_metadata",
        True,
    ),
    Routine(
        "update_ride_artifacts_proc",
        "PROCEDURE",
        "(p_trigger_id uuid, p_steps_artifacts jsonb, p_artifacts jsonb)",
        "rides_repository.update_ride_artifacts",
        True,
    ),
    Routine(
        "get_test_scenario_by_id",
        "FUNCTION",
        "(p_id uuid) RETURNS SETOF test_scenarios",
        "scenario_repository.find_by_id",
        True,
    ),
    Routine(
        "get_test_scenarios_by_ids",
        "FUNCTION",
        "(p_ids uuid[]) RETURNS SETOF test_scenarios",
        "scenario_repository.find_by_ids",
        True,
    ),
    Routine(
        "get_zpt_details",
        "FUNCTION",
        "(p_ride_id uuid) RETURNS TABLE (...)",
        "rides_repository.get_zpt_details",
        True,
    ),
    Routine(
        "get_suite_scenarios_with_pagination",
        "FUNCTION",
        "(p_suite_id uuid, p_skip int, p_limit int, p_sort_by text, p_sort_order text, "
        "p_search text) RETURNS TABLE (id uuid, name varchar, total_count bigint)",
        "suite_repository.get_suite_scenarios_with_pagination",
        True,
    ),
    Routine(
        "find_all_test_executions_with_pagination",
        "FUNCTION",
        "(...) RETURNS TABLE (...)",
        "execution_repository.find_all_with_pagination",
        False,
    ),
    Routine(
        "get_scenario_executions_with_pagination",
        "FUNCTION",
        "(...) RETURNS TABLE (...)",
        "execution_repository.get_scenario_executions_with_pagination",
        False,
    ),
    Routine(
        "get_scenario_executions_by_execution_id",
        "FUNCTION",
        "(...) RETURNS TABLE (...)",
        "execution_repository.get_scenario_executions_by_execution_id_with_pagination",
        False,
    ),
    Routine(
        "get_executions_with_scenarios_with_pagination",
        "FUNCTION",
        "(...) RETURNS TABLE (...)",
        "execution_repository.get_executions_with_scenarios_with_pagination",
        False,
    ),
    Routine(
        "get_suite_executions_by_execution_id_with_pagination",
        "FUNCTION",
        "(...) RETURNS TABLE (...)",
        "execution_repository.get_suite_executions_by_execution_id_with_pagination",
        False,
    ),
    Routine(
        "get_suite_executions_by_suite_id_with_pagination",
        "FUNCTION",
        "(...) RETURNS TABLE (...)",
        "execution_repository.get_suite_executions_by_suite_id_with_pagination",
        False,
    ),
    Routine(
        "get_workflows_by_ids",
        "FUNCTION",
        "(p_ids uuid[]) RETURNS TABLE (...)",
        "workflow_repository.find_by_ids",
        False,
    ),
)


# ---------------------------------------------------------------------------
# Static SQL
# ---------------------------------------------------------------------------

# No .format() is applied to these: they contain '{}' JSONB literals and $$ bodies.

TYPES_SQL = """-- SOTF test-mgmt: enum types.
--
-- init_db.sh creates these types with UPPER_SNAKE labels, but every status
-- column it creates is VARCHAR(50), so nothing is actually typed by them. The
-- SQLAlchemy models declare PG_ENUM over the Title Case *values*, which is what
-- the running application writes. The labels below are the Title Case set so the
-- types agree with the data; see README.md for the divergence.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'scenario_status') THEN
        CREATE TYPE public.scenario_status AS ENUM ('Active', 'Inactive', 'Deprecated');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'execution_status') THEN
        CREATE TYPE public.execution_status AS ENUM
            ('Queued', 'Running', 'Passed', 'Invalid', 'Partially Failed', 'Failed', 'Scheduled');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ride_status') THEN
        CREATE TYPE public.ride_status AS ENUM
            ('Queued', 'Invalid', 'Running', 'Passed', 'Failed', 'Scheduled');
    END IF;
END
$$;
"""

FUNCTIONS_SQL = r"""-- SOTF test-mgmt: routines and NOTIFY triggers.
--
-- Only update_ride_summary_proc has a body in the repo (init_db.sh), and its
-- signature there takes six arguments while rides_repository calls it with two.
-- The bodies below are reconstructed from the call sites and from what
-- app/db_listener.py expects to receive on each channel. They are here so the
-- fixture database behaves like the real one, not because the repo defines them.

-- 'Partially Failed' -> 'PARTIALLY_FAILED'. db_listener resolves the payload
-- status with getattr(Enum, value), so payloads carry the member name, not the
-- value.
CREATE OR REPLACE FUNCTION public.sotf_enum_name(p_value text)
RETURNS text
LANGUAGE sql IMMUTABLE
AS $$
    SELECT upper(replace(p_value, ' ', '_'));
$$;

-- execution_dto.ExecutionStates: the lifecycle state, which is separate from
-- the result. Anything terminal reports COMPLETED and carries a result.
CREATE OR REPLACE FUNCTION public.sotf_execution_state(p_status text)
RETURNS text
LANGUAGE sql IMMUTABLE
AS $$
    SELECT CASE p_status
        WHEN 'Queued'    THEN 'QUEUED'
        WHEN 'Scheduled' THEN 'SCHEDULED'
        WHEN 'Running'   THEN 'RUNNING'
        WHEN 'Invalid'   THEN 'INVALID'
        ELSE 'COMPLETED'
    END;
$$;

CREATE OR REPLACE FUNCTION public.sotf_derive_execution_status(p_execution_id uuid)
RETURNS text
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_total     int;
    v_passed    int;
    v_failed    int;
    v_running   int;
    v_queued    int;
    v_scheduled int;
    v_invalid   int;
BEGIN
    SELECT count(*),
           count(*) FILTER (WHERE status = 'Passed'),
           count(*) FILTER (WHERE status = 'Failed'),
           count(*) FILTER (WHERE status = 'Running'),
           count(*) FILTER (WHERE status = 'Queued'),
           count(*) FILTER (WHERE status = 'Scheduled'),
           count(*) FILTER (WHERE status = 'Invalid')
      INTO v_total, v_passed, v_failed, v_running, v_queued, v_scheduled, v_invalid
      FROM public.rides
     WHERE execution_id = p_execution_id;

    IF v_total = 0 OR v_queued = v_total THEN
        RETURN 'Queued';
    ELSIF v_scheduled = v_total THEN
        RETURN 'Scheduled';
    ELSIF v_running > 0 THEN
        RETURN 'Running';
    ELSIF v_invalid = v_total THEN
        RETURN 'Invalid';
    ELSIF v_passed = v_total THEN
        RETURN 'Passed';
    ELSIF v_failed = v_total THEN
        RETURN 'Failed';
    END IF;
    RETURN 'Partially Failed';
END;
$$;

CREATE OR REPLACE FUNCTION public.sotf_derive_suite_status(p_suite_id uuid)
RETURNS text
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_total     int;
    v_passed    int;
    v_failed    int;
    v_running   int;
    v_queued    int;
    v_scheduled int;
BEGIN
    SELECT count(*),
           count(*) FILTER (WHERE status = 'Passed'),
           count(*) FILTER (WHERE status = 'Failed'),
           count(*) FILTER (WHERE status = 'Running'),
           count(*) FILTER (WHERE status = 'Queued'),
           count(*) FILTER (WHERE status = 'Scheduled')
      INTO v_total, v_passed, v_failed, v_running, v_queued, v_scheduled
      FROM public.test_executions
     WHERE suite_id = p_suite_id;

    IF v_total = 0 OR v_queued = v_total THEN
        RETURN 'Queued';
    ELSIF v_scheduled = v_total THEN
        RETURN 'Scheduled';
    ELSIF v_running > 0 THEN
        RETURN 'Running';
    ELSIF v_passed = v_total THEN
        RETURN 'Passed';
    ELSIF v_failed = v_total THEN
        RETURN 'Failed';
    END IF;
    RETURN 'Partially Failed';
END;
$$;

-- ---------------------------------------------------------------------------
-- Ride mutation procedures
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE public.update_ride_summary_proc(
    p_trigger_id uuid,
    p_summary    jsonb)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE public.rides
       SET summary    = p_summary,
           updated_at = now()
     WHERE id = p_trigger_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ride % not found', p_trigger_id;
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE public.update_ride_invalid_status_proc(
    p_trigger_id uuid,
    p_status     text)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE public.rides
       SET status     = p_status,
           end_time   = COALESCE(end_time, now()),
           updated_at = now()
     WHERE id = p_trigger_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ride % not found', p_trigger_id;
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE public.update_ride_realtime_step_status_proc(
    p_trigger_id  uuid,
    p_step_status text,
    p_step_id     text,
    p_duration    time DEFAULT NULL,
    p_location    text DEFAULT NULL,
    p_msg         text DEFAULT NULL)
LANGUAGE plpgsql
AS $$
DECLARE
    v_summary jsonb;
    v_steps   jsonb;
    v_index   int;
BEGIN
    SELECT COALESCE(summary, '{}'::jsonb) INTO v_summary
      FROM public.rides WHERE id = p_trigger_id FOR UPDATE;

    IF v_summary IS NULL THEN
        RAISE EXCEPTION 'ride % not found', p_trigger_id;
    END IF;

    v_steps := COALESCE(v_summary -> 'steps', '[]'::jsonb);

    SELECT ordinality - 1 INTO v_index
      FROM jsonb_array_elements(v_steps) WITH ORDINALITY AS s(value, ordinality)
     WHERE s.value ->> 'step_id' = p_step_id
     LIMIT 1;

    IF v_index IS NULL THEN
        v_steps := v_steps || jsonb_build_array(jsonb_build_object(
            'step_id', p_step_id,
            'status',  p_step_status,
            'message', p_msg,
            'logs',    '[]'::jsonb));
    ELSE
        v_steps := jsonb_set(
            v_steps,
            ARRAY[v_index::text],
            (v_steps -> v_index)
                || jsonb_build_object('status', p_step_status, 'message', p_msg));
    END IF;

    v_summary := v_summary
        || jsonb_build_object('steps', v_steps)
        || CASE WHEN p_duration IS NULL
                THEN '{}'::jsonb
                ELSE jsonb_build_object('ride_duration', to_char(p_duration, 'HH24:MI:SS'))
           END
        || CASE WHEN p_location IS NULL
                THEN '{}'::jsonb
                ELSE jsonb_build_object('last_location', p_location)
           END;

    UPDATE public.rides
       SET summary = v_summary, updated_at = now()
     WHERE id = p_trigger_id;
END;
$$;

CREATE OR REPLACE PROCEDURE public.update_ride_metadata_proc(
    p_trigger_id   uuid,
    p_host_machine text,
    p_device_id    text,
    p_metadata     jsonb,
    p_mission_id   uuid DEFAULT NULL)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE public.rides
       SET host_machine = p_host_machine,
           device_id    = p_device_id,
           metadata     = COALESCE(metadata, '{}'::jsonb) || COALESCE(p_metadata, '{}'::jsonb),
           mission_id   = COALESCE(p_mission_id, mission_id),
           updated_at   = now()
     WHERE id = p_trigger_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ride % not found', p_trigger_id;
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE public.update_ride_artifacts_proc(
    p_trigger_id      uuid,
    p_steps_artifacts jsonb,
    p_artifacts       jsonb)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE public.rides
       SET summary = COALESCE(summary, '{}'::jsonb)
                     || jsonb_build_object(
                            'artifacts',       COALESCE(p_artifacts, '[]'::jsonb),
                            'steps_artifacts', COALESCE(p_steps_artifacts, '[]'::jsonb),
                            'has_artifacts',   jsonb_array_length(COALESCE(p_artifacts, '[]'::jsonb)) > 0),
           metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{testrail,artifacts_uploaded}',
                            'true'::jsonb,
                            true),
           updated_at = now()
     WHERE id = p_trigger_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'ride % not found', p_trigger_id;
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Read functions
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.get_test_scenario_by_id(p_id uuid)
RETURNS SETOF public.test_scenarios
LANGUAGE sql STABLE
AS $$
    SELECT * FROM public.test_scenarios WHERE id = p_id AND deleted_at IS NULL;
$$;

CREATE OR REPLACE FUNCTION public.get_test_scenarios_by_ids(p_ids uuid[])
RETURNS SETOF public.test_scenarios
LANGUAGE sql STABLE
AS $$
    SELECT * FROM public.test_scenarios WHERE id = ANY(p_ids) ORDER BY name;
$$;

CREATE OR REPLACE FUNCTION public.get_suite_scenarios_with_pagination(
    p_suite_id   uuid,
    p_skip       int,
    p_limit      int,
    p_sort_by    text DEFAULT 'name',
    p_sort_order text DEFAULT 'asc',
    p_search     text DEFAULT NULL)
RETURNS TABLE (id uuid, name varchar, total_count bigint)
LANGUAGE sql STABLE
AS $$
    WITH members AS (
        SELECT s.id, s.name, s.updated_at
          FROM public.test_suite_scenarios ss
          JOIN public.test_scenarios s ON s.id = ss.scenario_id
         WHERE ss.suite_id = p_suite_id
           AND ss.deleted = FALSE
           AND (p_search IS NULL OR s.name ILIKE '%' || p_search || '%')
    )
    SELECT m.id, m.name, (SELECT count(*) FROM members) AS total_count
      FROM members m
     ORDER BY
        CASE WHEN p_sort_by = 'name'       AND lower(p_sort_order) = 'asc'  THEN m.name END ASC,
        CASE WHEN p_sort_by = 'name'       AND lower(p_sort_order) = 'desc' THEN m.name END DESC,
        CASE WHEN p_sort_by = 'updated_at' AND lower(p_sort_order) = 'asc'  THEN m.updated_at END ASC,
        CASE WHEN p_sort_by = 'updated_at' AND lower(p_sort_order) = 'desc' THEN m.updated_at END DESC,
        m.name ASC
     OFFSET GREATEST(p_skip, 0)
     LIMIT NULLIF(GREATEST(p_limit, 0), 0);
$$;

-- Feeds rides_service.generate_zpt. 'occurance_counter' is spelled that way at
-- the call site, so the column keeps the spelling.
CREATE OR REPLACE FUNCTION public.get_zpt_details(p_ride_id uuid)
RETURNS TABLE (
    test_id            text,
    scenario_name      text,
    environment        text,
    device             text,
    vehicle_type       text,
    vehicle_subtype    text,
    location           text,
    found_during       text,
    mission_phase      text,
    occurance_counter  int,
    failed_event       text,
    updated_at         timestamptz,
    artifacts          jsonb)
LANGUAGE sql STABLE
AS $$
    SELECT
        r.metadata -> 'testrail' ->> 'test_id',
        s.name::text,
        e.payload ->> 'environment',
        r.device_id::text,
        'Container Sim'::text,
        r.metadata ->> 'vehicle_id',
        e.payload ->> 'ride_service',
        'Testing - Milestone testing'::text,
        r.summary -> 'system' ->> 'mission_phase',
        (SELECT count(*)::int
           FROM public.rides peer
           JOIN public.test_executions pe ON pe.id = peer.execution_id
          WHERE pe.scenario_id = e.scenario_id AND peer.status = 'Failed'),
        (SELECT step ->> 'message'
           FROM jsonb_array_elements(COALESCE(r.summary -> 'steps', '[]'::jsonb)) AS step
          WHERE step ->> 'status' IN ('FAILED', 'STUCK')
          LIMIT 1),
        r.updated_at,
        COALESCE(r.summary -> 'artifacts', '[]'::jsonb)
      FROM public.rides r
      JOIN public.test_executions e ON e.id = r.execution_id
      JOIN public.test_scenarios s  ON s.id = e.scenario_id
     WHERE r.id = p_ride_id;
$$;

-- ---------------------------------------------------------------------------
-- NOTIFY triggers
--
-- app/db_listener.py listens on 'rides', 'executions.scenario' and
-- 'executions.suite'. pg_notify caps a payload at 8000 bytes; a ride with many
-- steps can exceed that, which is why the SUMMARY payload carries only the
-- step list and duration rather than the whole summary document.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.trg_rides_notify()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    v_ride_state       text := public.sotf_enum_name(NEW.status);
    v_execution_status text;
    v_execution_state  text;
    v_suite_id         uuid;
    v_scenario_id      uuid;
    v_last_run_at      timestamptz;
    v_batch_status     text;
BEGIN
    IF NEW.summary IS DISTINCT FROM OLD.summary THEN
        PERFORM pg_notify('rides', jsonb_build_object(
            'id',            NEW.id,
            'status',        v_ride_state,
            'event_type',    'SUMMARY',
            'steps',         COALESCE(NEW.summary -> 'steps', '[]'::jsonb),
            'ride_duration', NEW.summary ->> 'ride_duration'
        )::text);
    END IF;

    IF NEW.metadata IS DISTINCT FROM OLD.metadata
       OR NEW.device_id IS DISTINCT FROM OLD.device_id
       OR NEW.host_machine IS DISTINCT FROM OLD.host_machine THEN
        PERFORM pg_notify('rides', jsonb_build_object(
            'id',           NEW.id,
            'status',       v_ride_state,
            'event_type',   'METADATA',
            'device_id',    NEW.device_id,
            'host_machine', NEW.host_machine,
            'metadata',     COALESCE(NEW.metadata, '{}'::jsonb),
            'mission_id',   NEW.mission_id
        )::text);
    END IF;

    IF NEW.status IS DISTINCT FROM OLD.status
       OR NEW.start_time IS DISTINCT FROM OLD.start_time THEN

        v_execution_status := public.sotf_derive_execution_status(NEW.execution_id);
        v_execution_state  := public.sotf_execution_state(v_execution_status);

        UPDATE public.test_executions
           SET status = v_execution_status
         WHERE id = NEW.execution_id;

        SELECT e.suite_id, e.scenario_id
          INTO v_suite_id, v_scenario_id
          FROM public.test_executions e
         WHERE e.id = NEW.execution_id;

        SELECT max(r.start_time) INTO v_last_run_at
          FROM public.rides r
         WHERE r.execution_id = NEW.execution_id;

        PERFORM pg_notify('executions.scenario', jsonb_build_object(
            'execution_id', NEW.execution_id,
            'last_run_at',  v_last_run_at,
            'status',       v_execution_state,
            'result',       CASE WHEN v_execution_state = 'COMPLETED'
                                 THEN public.sotf_enum_name(v_execution_status)
                                 ELSE '' END,
            'event_type',   'SCENARIO_EXECUTION'
        )::text);

        IF v_suite_id IS NOT NULL THEN
            v_batch_status := public.sotf_derive_suite_status(v_suite_id);

            SELECT max(r.start_time) INTO v_last_run_at
              FROM public.rides r
              JOIN public.test_executions e ON e.id = r.execution_id
             WHERE e.suite_id = v_suite_id;

            PERFORM pg_notify('executions.suite', jsonb_build_object(
                'ride_id',      NEW.id,
                'execution_id', NEW.execution_id,
                'suite_id',     v_suite_id,
                'scenario_id',  v_scenario_id,
                'batch_status', public.sotf_execution_state(v_batch_status),
                'batch_result', CASE WHEN public.sotf_execution_state(v_batch_status) = 'COMPLETED'
                                     THEN public.sotf_enum_name(v_batch_status)
                                     ELSE '' END,
                'last_run_at',  v_last_run_at,
                'event_type',   'SUITE_EXECUTION'
            )::text);
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS rides_notify ON public.rides;
CREATE TRIGGER rides_notify
AFTER UPDATE ON public.rides
FOR EACH ROW
EXECUTE FUNCTION public.trg_rides_notify();

CREATE OR REPLACE FUNCTION public.trg_scenario_version_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.version IS DISTINCT FROM OLD.version THEN
        INSERT INTO public.test_scenario_version_history (
            scenario_id, name, metadata, test_steps, version,
            version_comment, status, versioned_by, versioned_at)
        VALUES (
            OLD.id, OLD.name, OLD.metadata, OLD.test_steps, OLD.version,
            OLD.version_comment, OLD.status, NEW.updated_by, now())
        ON CONFLICT ON CONSTRAINT uq_scenario_version DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS test_scenarios_version_snapshot ON public.test_scenarios;
CREATE TRIGGER test_scenarios_version_snapshot
BEFORE UPDATE ON public.test_scenarios
FOR EACH ROW
EXECUTE FUNCTION public.trg_scenario_version_snapshot();
"""

SAMPLE_QUERIES_SQL = """-- Queries a read-only SOTF assistant would run against the test-mgmt database.
-- The scenario table is the entry point: everything else hangs off it.

-- 1. Find a scenario by name, with its current version and TestRail case id.
SELECT id, name, version, status,
       metadata ->> 'testrail_id' AS testrail_case_id,
       metadata -> 'labels'       AS labels,
       jsonb_array_length(test_steps) AS step_count,
       updated_at, updated_by
  FROM test_scenarios
 WHERE name ILIKE '%unlock%'
 ORDER BY updated_at DESC;

-- 2. The full step list of a scenario, one row per step.
SELECT s.name,
       step ->> 'order'           AS step_order,
       step ->> 'action_category' AS action_category,
       step ->> 'action_name'     AS action_name,
       step ->> 'operational_code' AS operational_code
  FROM test_scenarios s,
       LATERAL jsonb_array_elements(s.test_steps) AS step
 WHERE s.id = '00000000-0000-0000-0000-000000000000'
 ORDER BY (step ->> 'order')::int;

-- 3. Version history of a scenario, newest first.
SELECT version, version_comment, versioned_by, versioned_at,
       jsonb_array_length(test_steps) AS step_count
  FROM test_scenario_version_history
 WHERE scenario_id = '00000000-0000-0000-0000-000000000000'
 UNION ALL
SELECT version, version_comment, updated_by, updated_at,
       jsonb_array_length(test_steps)
  FROM test_scenarios
 WHERE id = '00000000-0000-0000-0000-000000000000'
 ORDER BY version DESC;

-- 4. Which suites contain a scenario, ignoring soft-deleted membership.
SELECT su.name AS suite_name, ss.created_at, ss.created_by
  FROM test_suite_scenarios ss
  JOIN test_suites su ON su.id = ss.suite_id
 WHERE ss.scenario_id = '00000000-0000-0000-0000-000000000000'
   AND ss.deleted = FALSE
 ORDER BY su.name;

-- 5. Pass rate per scenario over the last 30 days. This is the query the
--    warehouse aggregates replace once volume grows.
SELECT s.name,
       count(*)                                              AS rides,
       count(*) FILTER (WHERE r.status = 'Passed')           AS passed,
       count(*) FILTER (WHERE r.status = 'Failed')           AS failed,
       round(avg(EXTRACT(EPOCH FROM (r.end_time - r.start_time)))::numeric, 1)
                                                             AS avg_duration_seconds
  FROM rides r
  JOIN test_executions e ON e.id = r.execution_id
  JOIN test_scenarios  s ON s.id = e.scenario_id
 WHERE r.status IN ('Passed', 'Failed')
   AND r.start_time >= now() - INTERVAL '30 days'
 GROUP BY s.name
 ORDER BY failed DESC, s.name;

-- 6. The most recent failures with the ids needed to pivot into every other
--    system: TestRail, Jira and the artifact bundle.
SELECT r.id                                        AS trigger_id,
       s.name                                      AS scenario,
       e.payload ->> 'environment'                 AS environment,
       e.payload ->> 'pudo_zone'                   AS pudo_zone,
       r.metadata -> 'testrail' ->> 'run_id'       AS testrail_run_id,
       r.metadata -> 'testrail' ->> 'result_id'    AS testrail_result_id,
       (SELECT step ->> 'message'
          FROM jsonb_array_elements(COALESCE(r.summary -> 'steps', '[]'::jsonb)) AS step
         WHERE step ->> 'status' IN ('FAILED', 'STUCK')
         LIMIT 1)                                  AS failed_step_message,
       r.summary  -> 'artifacts' -> 0 ->> 'path'   AS artifact_uri,
       r.start_time, r.end_time
  FROM rides r
  JOIN test_executions e ON e.id = r.execution_id
  JOIN test_scenarios  s ON s.id = e.scenario_id
 WHERE r.status = 'Failed'
 ORDER BY r.start_time DESC
 LIMIT 50;

-- 7. Step-level failures across all rides, for "which step breaks most often".
SELECT s.name                       AS scenario,
       step ->> 'step_id'           AS step_id,
       step ->> 'message'           AS message,
       count(*)                     AS occurrences
  FROM rides r
  JOIN test_executions e ON e.id = r.execution_id
  JOIN test_scenarios  s ON s.id = e.scenario_id,
       LATERAL jsonb_array_elements(COALESCE(r.summary -> 'steps', '[]'::jsonb)) AS step
 WHERE step ->> 'status' IN ('FAILED', 'STUCK')
 GROUP BY s.name, step ->> 'step_id', step ->> 'message'
 ORDER BY occurrences DESC
 LIMIT 25;

-- 8. Executions still in flight, and how far their rides have got.
SELECT e.id, s.name, e.status, e.created_by, e.created_at,
       count(r.id)                                    AS rides,
       count(r.id) FILTER (WHERE r.status = 'Running') AS running
  FROM test_executions e
  JOIN test_scenarios s ON s.id = e.scenario_id
  LEFT JOIN rides r ON r.execution_id = e.id
 WHERE e.status IN ('Queued', 'Running', 'Scheduled')
 GROUP BY e.id, s.name, e.status, e.created_by, e.created_at
 ORDER BY e.created_at DESC;

-- 9. The ride event stream, which is the cheapest way to see what happened
--    without unpacking the summary document.
SELECT event, event_time, event_data
  FROM ride_events
 WHERE ride_id = '00000000-0000-0000-0000-000000000000'
 ORDER BY event_time;

-- 10. Recurring schedules that are still active.
SELECT sc.id, s.name AS scenario, su.name AS suite,
       sc.recurrence_days, sc.schedule_from, sc.schedule_to, sc.last_enqueued_at
  FROM scheduled_test_executions sc
  JOIN test_scenarios s ON s.id = sc.scenario_id
  LEFT JOIN test_suites su ON su.id = sc.suite_id
 WHERE sc.is_deleted = FALSE
 ORDER BY sc.schedule_from;
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rand_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def enum_name(value: str) -> str:
    """'Partially Failed' -> 'PARTIALLY_FAILED', the enum member db_listener resolves."""
    return value.upper().replace(" ", "_")


def json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def csv_cell(value: Any) -> str:
    if value is None:
        return NULL_TOKEN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json_compact(value)
    return str(value)


def sql_literal(value: Any, is_json: bool) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if is_json:
        return f"{DOLLAR_TAG}{json_compact(value)}{DOLLAR_TAG}::jsonb"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# SOTF dataset input
# ---------------------------------------------------------------------------


@dataclass
class SotfData:
    root: Path
    now: datetime
    days: int
    counts: dict[str, Any]
    scenarios: list[dict[str, Any]]
    suites: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    rides: list[dict[str, Any]]
    cross_reference: dict[str, dict[str, Any]]


def load_sotf_dataset(root: Path) -> SotfData:
    required = [
        root / "summary.json",
        root / "cross_reference.json",
        root / "domain/scenarios.json",
        root / "domain/suites.json",
        root / "domain/executions.json",
        root / "domain/rides.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            f"SOTF dataset at {root} is missing {', '.join(missing)}\n"
            "Run generate_sotf_mock_data.py first, or pass --sotf-data."
        )

    def load(relative: str) -> Any:
        return json.loads((root / relative).read_text(encoding="utf-8"))

    summary = load("summary.json")
    params = summary.get("parameters", {})

    return SotfData(
        root=root,
        now=parse_dt(params.get("now")) or datetime.now(timezone.utc),
        days=int(params.get("days", 30)),
        counts=summary.get("counts", {}),
        scenarios=load("domain/scenarios.json"),
        suites=load("domain/suites.json"),
        executions=load("domain/executions.json"),
        rides=load("domain/rides.json"),
        cross_reference={
            entry["trigger_id"]: entry for entry in load("cross_reference.json")
        },
    )


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def build_scenario_rows(sotf: SotfData) -> list[dict[str, Any]]:
    rows = []
    for scenario in sotf.scenarios:
        status = scenario["status"]
        rows.append(
            {
                "id": scenario["id"],
                "name": scenario["name"],
                "metadata": scenario["metadata"],
                "test_steps": scenario["test_steps"],
                "version": scenario["version"],
                "version_comment": scenario["version_comment"],
                "status": status,
                "created_at": scenario["created_at"],
                "updated_at": scenario["updated_at"],
                "deleted_at": scenario["updated_at"] if status == "Deprecated" else None,
                "created_by": scenario["created_by"],
                "updated_by": scenario["updated_by"],
            }
        )
    return rows


def build_version_history_rows(
    rng: random.Random, sotf: SotfData
) -> list[dict[str, Any]]:
    """One row per version that came before the scenario's current version."""
    rows = []
    for scenario in sotf.scenarios:
        current = float(scenario["version"])
        earlier = [v for v in VERSION_LADDER if v < current]
        if not earlier:
            continue

        created = parse_dt(scenario["created_at"])
        updated = parse_dt(scenario["updated_at"])
        if created is None or updated is None:
            continue
        span = max((updated - created).total_seconds(), 3600.0)
        steps = scenario["test_steps"]

        for position, version in enumerate(earlier, start=1):
            # The oldest version is the shortest: steps were added over time.
            age = len(earlier) - position + 1
            kept = steps[: max(2, len(steps) - age)]
            offset = span * position / (len(earlier) + 1)
            rows.append(
                {
                    "id": rand_uuid(rng),
                    "scenario_id": scenario["id"],
                    "name": scenario["name"],
                    "metadata": {"testrail_id": None, "labels": scenario["metadata"]["labels"]},
                    "test_steps": kept,
                    "version": version,
                    "version_comment": VERSION_COMMENTS[position % len(VERSION_COMMENTS)],
                    "status": "Active",
                    "versioned_by": rng.choice(AUTHORS),
                    "versioned_at": iso(created + timedelta(seconds=offset)),
                }
            )
    return rows


def build_suite_rows(sotf: SotfData) -> list[dict[str, Any]]:
    return [
        {
            "id": suite["id"],
            "name": suite["name"],
            "description": suite["description"],
            "created_at": suite["created_at"],
            "created_by": suite["created_by"],
            "updated_at": suite["updated_at"],
            "updated_by": suite["updated_by"],
        }
        for suite in sotf.suites
    ]


def build_suite_scenario_rows(
    rng: random.Random, sotf: SotfData, soft_delete_rate: float
) -> list[dict[str, Any]]:
    """Expand suite.scenario_ids into the join table, soft deleting a few unused links."""
    in_use = {
        (execution["suite_id"], execution["scenario_id"])
        for execution in sotf.executions
        if execution["suite_id"]
    }
    known_scenarios = {scenario["id"] for scenario in sotf.scenarios}

    rows = []
    for suite in sotf.suites:
        created = parse_dt(suite["created_at"]) or sotf.now
        for scenario_id in suite["scenario_ids"]:
            if scenario_id not in known_scenarios:
                continue
            pair = (suite["id"], scenario_id)
            # A link an execution used must stay live, or that execution would
            # reference a suite it is no longer a member of.
            deleted = pair not in in_use and rng.random() < soft_delete_rate
            removed_at = created + timedelta(days=rng.randint(5, 40))
            rows.append(
                {
                    "id": rand_uuid(rng),
                    "suite_id": suite["id"],
                    "scenario_id": scenario_id,
                    "deleted": deleted,
                    "created_at": iso(created),
                    "created_by": suite["created_by"],
                    "deleted_at": iso(removed_at) if deleted else None,
                    "deleted_by": rng.choice(AUTHORS) if deleted else None,
                }
            )
    return rows


def build_execution_rows(sotf: SotfData) -> list[dict[str, Any]]:
    return [
        {
            "id": execution["id"],
            "scenario_id": execution["scenario_id"],
            "suite_id": execution["suite_id"],
            "scenario_version": execution["scenario_version"],
            "payload": execution["payload"],
            "status": execution["status"],
            "no_of_triggers": execution["no_of_triggers"],
            "schedule_from": None,
            "schedule_to": None,
            "recurrence_days": None,
            "created_at": execution["created_at"],
            "created_by": execution["created_by"],
            "scheduled_at": execution["scheduled_at"],
        }
        for execution in sotf.executions
    ]


def build_scheduled_execution_rows(
    rng: random.Random, sotf: SotfData, count: int
) -> list[dict[str, Any]]:
    """Recurring schedules, modelled on executions that already exist."""
    if not sotf.executions:
        return []

    rows = []
    samples = rng.sample(sotf.executions, k=min(count, len(sotf.executions)))
    for index, execution in enumerate(samples):
        start = sotf.now - timedelta(days=sotf.days + rng.randint(5, 30))
        deleted = index % 5 == 4
        payload = dict(execution["payload"])
        payload["scheduled"] = True
        rows.append(
            {
                "id": rand_uuid(rng),
                "scenario_id": execution["scenario_id"],
                "suite_id": execution["suite_id"],
                "payload": payload,
                "no_of_triggers": execution["no_of_triggers"],
                "schedule_from": iso(start),
                "schedule_to": iso(start + timedelta(days=90)),
                "recurrence_days": RECURRENCE_PATTERNS[index % len(RECURRENCE_PATTERNS)],
                "created_at": iso(start - timedelta(days=1)),
                "created_by": rng.choice(AUTHORS),
                "is_deleted": deleted,
                "last_enqueued_at": (
                    None if deleted else iso(sotf.now - timedelta(hours=rng.randint(2, 60)))
                ),
            }
        )
    return rows


def build_ride_rows(sotf: SotfData) -> list[dict[str, Any]]:
    """Project domain/rides.json onto the rides table.

    created_at and updated_at are NOT NULL in the model but the SOTF fixture
    leaves them null for a ride that never started, so they fall back to the
    parent execution's creation time.
    """
    execution_created = {
        execution["id"]: execution["created_at"] for execution in sotf.executions
    }

    rows = []
    for ride in sotf.rides:
        fallback = execution_created.get(ride["execution_id"])
        rows.append(
            {
                "id": ride["id"],
                "execution_id": ride["execution_id"],
                "status": ride["status"],
                "metadata": ride["metadata"],
                "summary": ride.get("summary") or None,
                "device_id": ride["device_id"],
                "host_machine": ride["host_machine"],
                "reattempts_count": ride["reattempts_count"],
                "start_time": ride["start_time"],
                "end_time": ride["end_time"],
                "mission_id": ride["mission_id"],
                "created_at": ride["created_at"] or fallback,
                "updated_at": ride["updated_at"] or fallback,
            }
        )
    return rows


def failed_step_message(summary: dict[str, Any] | None) -> str | None:
    for step in (summary or {}).get("steps", []):
        if step.get("status") in ("FAILED", "STUCK"):
            return step.get("message")
    return None


def build_ride_event_rows(
    rng: random.Random, sotf: SotfData, ride_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ride in ride_rows:
        joined = sotf.cross_reference.get(ride["id"], {})
        created = parse_dt(ride["created_at"])
        started = parse_dt(ride["start_time"])
        ended = parse_dt(ride["end_time"])
        anchor = created or started or sotf.now

        def event(offset_from: datetime, name: str, data: dict[str, Any]) -> None:
            rows.append(
                {
                    "id": rand_uuid(rng),
                    "ride_id": ride["id"],
                    "event": name,
                    "event_time": iso(offset_from),
                    "event_data": data,
                }
            )

        event(
            anchor,
            "RideCreated",
            {
                "execution_id": ride["execution_id"],
                "status": ride["status"],
                "reattempts_count": ride["reattempts_count"],
            },
        )

        if started is None:
            continue

        event(
            started,
            "RideStarted",
            {
                "host_machine": ride["host_machine"],
                "device_id": ride["device_id"],
                "mission_id": ride["mission_id"],
            },
        )

        failure_mode = joined.get("failure_mode")
        failed_step_id = joined.get("failed_step_id")
        if failure_mode and ended:
            event(
                ended - timedelta(seconds=2),
                "StepFailed",
                {
                    "step_id": failed_step_id,
                    "failure_mode": failure_mode,
                    "message": failed_step_message(ride.get("summary")),
                },
            )

        s3_key = (joined.get("s3") or {}).get("key")
        if s3_key and ended:
            event(
                ended + timedelta(seconds=rng.randint(20, 240)),
                "ArtifactsUploaded",
                {"s3_key": s3_key, "s3_uri": (joined.get("s3") or {}).get("uri")},
            )

        testrail = joined.get("testrail") or {}
        if testrail.get("result_id") and ended:
            event(
                ended + timedelta(seconds=rng.randint(30, 300)),
                "TestRailResultRecorded",
                {
                    "run_id": testrail.get("run_id"),
                    "test_id": testrail.get("test_id"),
                    "result_id": testrail.get("result_id"),
                },
            )

        if joined.get("jira_key") and ended:
            event(
                ended + timedelta(seconds=rng.randint(300, 3600)),
                "DefectLinked",
                {"issue_key": joined["jira_key"], "project": "ZPT"},
            )

        terminal = {
            "Passed": "RideCompleted",
            "Failed": "RideFailed",
            "Invalid": "RideInvalidated",
        }.get(ride["status"])
        if terminal and ended:
            event(
                ended,
                terminal,
                {
                    "status": ride["status"],
                    "duration_seconds": int((ended - started).total_seconds()),
                },
            )

    rows.sort(key=lambda r: (r["event_time"], r["ride_id"], r["event"]))
    return rows


# ---------------------------------------------------------------------------
# DDL emitters
# ---------------------------------------------------------------------------


def render_create_table(table: Table) -> str:
    lines = [f"CREATE TABLE IF NOT EXISTS public.{table.name} ("]
    parts: list[str] = []
    width = max(len(c.name) for c in table.columns) + 2

    for column in table.columns:
        piece = f"    {column.name.ljust(width)}{column.sql_type}"
        if not column.nullable:
            piece += " NOT NULL"
        if column.default:
            piece += f" DEFAULT {column.default}"
        parts.append(piece)

    parts.append(f"    CONSTRAINT pk_{table.name} PRIMARY KEY ({table.primary_key})")
    for column_name, referenced, on_delete in table.foreign_keys:
        parts.append(
            f"    CONSTRAINT fk_{table.name}_{column_name} FOREIGN KEY ({column_name}) "
            f"REFERENCES public.{referenced} (id) ON DELETE {on_delete}"
        )
    for name, columns in table.uniques:
        parts.append(f"    CONSTRAINT {name} UNIQUE ({', '.join(columns)})")
    for name, expression in table.checks:
        parts.append(f"    CONSTRAINT {name} CHECK ({expression})")

    lines.append(",\n".join(parts))
    lines.append(");")

    lines.append(f"COMMENT ON TABLE public.{table.name} IS '{table.comment}';")
    for column in table.columns:
        if column.comment:
            escaped = column.comment.replace("'", "''")
            lines.append(
                f"COMMENT ON COLUMN public.{table.name}.{column.name} IS '{escaped}';"
            )
    return "\n".join(lines)


def emit_schema(out: Path) -> None:
    write_text(out / "schema/001_types.sql", TYPES_SQL)

    body = [
        "-- SOTF test-mgmt: tables.",
        "--",
        "-- Column names and types follow tcc/api-py/database/init_db.sh where it",
        "-- defines them, and the SQLAlchemy models under test-mgmt/app/model/ where",
        "-- init_db.sh is behind (rides.mission_id, rides.created_at/updated_at,",
        "-- test_executions.scheduled_at and scenario_version, and the whole",
        "-- scheduled_test_executions table). Timestamps are TIMESTAMPTZ because the",
        "-- models declare DateTime(timezone=True); init_db.sh uses bare TIMESTAMP.",
        "",
    ]
    for table in TABLES:
        body.append(render_create_table(table))
        body.append("")
    write_text(out / "schema/002_tables.sql", "\n".join(body))

    index_lines = ["-- SOTF test-mgmt: indexes.", ""]
    for table in TABLES:
        for name, definition in table.indexes:
            index_lines.append(f"CREATE INDEX IF NOT EXISTS {name} {definition};")
    index_lines.append("")
    write_text(out / "schema/003_indexes.sql", "\n".join(index_lines))

    write_text(out / "schema/004_routines.sql", FUNCTIONS_SQL)


# ---------------------------------------------------------------------------
# Data emitters
# ---------------------------------------------------------------------------


def emit_csv(out: Path, table: Table, rows: list[dict[str, Any]]) -> None:
    path = out / f"tables/{table.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(table.column_names)
        for row in rows:
            writer.writerow([csv_cell(row.get(name)) for name in table.column_names])


def emit_seed_sql(
    out: Path, data: dict[str, list[dict[str, Any]]], batch_size: int
) -> None:
    chunks = [
        "-- SOTF test-mgmt seed data.",
        "-- Tables are inserted parent-first so the foreign keys resolve. JSONB values",
        f"-- are dollar-quoted with {DOLLAR_TAG} to avoid escaping every embedded quote.",
        "",
        "BEGIN;",
        "",
    ]
    for table in TABLES:
        rows = data[table.name]
        chunks.append(f"-- {table.name}: {len(rows)} row(s)")
        if not rows:
            chunks.append("")
            continue
        json_columns = table.json_columns
        columns = table.column_names
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            values = ",\n".join(
                "    ("
                + ", ".join(
                    sql_literal(row.get(name), name in json_columns) for name in columns
                )
                + ")"
                for row in batch
            )
            chunks.append(
                f"INSERT INTO public.{table.name} ({', '.join(columns)}) VALUES\n"
                f"{values}\nON CONFLICT DO NOTHING;"
            )
        chunks.append("")

    chunks.append("COMMIT;")
    chunks.append("")
    write_text(out / "seed.sql", "\n".join(chunks))


def emit_notify_samples(
    out: Path,
    rng: random.Random,
    ride_rows: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    sample_size: int,
) -> dict[str, int]:
    """Payloads exactly as db_listener.py receives them on each channel."""
    execution_by_id = {row["id"]: row for row in execution_rows}
    completed = [r for r in ride_rows if r["status"] in ("Passed", "Failed") and r["summary"]]
    picks = rng.sample(completed, k=min(sample_size, len(completed))) if completed else []

    ride_payloads: list[dict[str, Any]] = []
    scenario_payloads: list[dict[str, Any]] = []
    suite_payloads: list[dict[str, Any]] = []

    for ride in picks:
        execution = execution_by_id.get(ride["execution_id"])
        if execution is None:
            continue
        summary = ride["summary"]

        ride_payloads.append(
            {
                "channel": "rides",
                "payload": {
                    "id": ride["id"],
                    "status": enum_name(ride["status"]),
                    "event_type": "SUMMARY",
                    "steps": summary.get("steps", []),
                    "ride_duration": summary.get("ride_duration"),
                },
            }
        )
        ride_payloads.append(
            {
                "channel": "rides",
                "payload": {
                    "id": ride["id"],
                    "status": enum_name(ride["status"]),
                    "event_type": "METADATA",
                    "device_id": ride["device_id"],
                    "host_machine": ride["host_machine"],
                    "metadata": ride["metadata"],
                    "mission_id": ride["mission_id"],
                },
            }
        )

        state = EXECUTION_STATE_BY_STATUS[execution["status"]]
        scenario_payloads.append(
            {
                "channel": "executions.scenario",
                "payload": {
                    "execution_id": execution["id"],
                    "last_run_at": ride["start_time"],
                    "status": state,
                    "result": enum_name(execution["status"]) if state == "COMPLETED" else "",
                    "event_type": "SCENARIO_EXECUTION",
                },
            }
        )

        if execution["suite_id"]:
            suite_payloads.append(
                {
                    "channel": "executions.suite",
                    "payload": {
                        "ride_id": ride["id"],
                        "execution_id": execution["id"],
                        "suite_id": execution["suite_id"],
                        "scenario_id": execution["scenario_id"],
                        "batch_status": state,
                        "batch_result": (
                            enum_name(execution["status"]) if state == "COMPLETED" else ""
                        ),
                        "last_run_at": ride["start_time"],
                        "event_type": "SUITE_EXECUTION",
                    },
                }
            )

    note = (
        "Payload status fields carry the enum *member name* (PARTIALLY_FAILED), not the "
        "value, because db_listener resolves them with getattr(Enum, value). 'status' on "
        "executions.scenario is an ExecutionStates member and 'result' is an "
        "ExecutionStatus member; result is empty until the execution is COMPLETED."
    )
    write_json(
        out / "notify/rides.json",
        {"channel": "rides", "note": note, "notifications": ride_payloads},
    )
    write_json(
        out / "notify/executions_scenario.json",
        {"channel": "executions.scenario", "note": note, "notifications": scenario_payloads},
    )
    write_json(
        out / "notify/executions_suite.json",
        {"channel": "executions.suite", "note": note, "notifications": suite_payloads},
    )
    return {
        "rides": len(ride_payloads),
        "executions.scenario": len(scenario_payloads),
        "executions.suite": len(suite_payloads),
    }


# ---------------------------------------------------------------------------
# Catalog and instance metadata
# ---------------------------------------------------------------------------

PAGE_SIZE = 8192
ROW_OVERHEAD_BYTES = 28


def table_bytes(table: Table, rows: list[dict[str, Any]]) -> int:
    """Heap size rounded up to whole pages, from the real serialized row width."""
    payload = sum(
        len(json_compact(row.get(name)).encode("utf-8")) for row in rows for name in table.column_names
    )
    payload += ROW_OVERHEAD_BYTES * len(rows)
    return max(1, math.ceil(payload / PAGE_SIZE)) * PAGE_SIZE


def emit_pg_catalog(
    out: Path,
    rng: random.Random,
    sotf: SotfData,
    data: dict[str, list[dict[str, Any]]],
) -> None:
    stat_tables = []
    sizes = []
    for table in TABLES:
        rows = data[table.name]
        heap = table_bytes(table, rows)
        index_count = len(table.indexes) + 1  # plus the primary key
        index_size = max(PAGE_SIZE, int(heap * 0.22)) * index_count
        updates = len(rows) * (3 if table.name == "rides" else 1)
        dead = int(updates * rng.uniform(0.02, 0.12))

        stat_tables.append(
            {
                "schemaname": "public",
                "relname": table.name,
                "seq_scan": rng.randint(20, 400),
                "seq_tup_read": len(rows) * rng.randint(3, 40),
                "idx_scan": rng.randint(400, 40000),
                "idx_tup_fetch": len(rows) * rng.randint(5, 60),
                "n_tup_ins": len(rows),
                "n_tup_upd": updates,
                "n_tup_del": 0,
                "n_live_tup": len(rows),
                "n_dead_tup": dead,
                "last_vacuum": None,
                "last_autovacuum": iso(sotf.now - timedelta(hours=rng.randint(2, 40))),
                "last_analyze": None,
                "last_autoanalyze": iso(sotf.now - timedelta(hours=rng.randint(1, 30))),
                "autovacuum_count": rng.randint(1, 25),
            }
        )
        sizes.append(
            {
                "schemaname": "public",
                "relname": table.name,
                "rows": len(rows),
                "heap_bytes": heap,
                "index_bytes": index_size,
                "total_bytes": heap + index_size,
                "total_pretty": f"{(heap + index_size) / 1024 / 1024:.1f} MB",
            }
        )

    write_json(out / "pg_catalog/pg_stat_user_tables.json", stat_tables)
    write_json(out / "pg_catalog/table_sizes.json", sizes)

    indexes = [
        {
            "schemaname": "public",
            "tablename": table.name,
            "indexname": name,
            "indexdef": f"CREATE INDEX {name} {definition}",
        }
        for table in TABLES
        for name, definition in table.indexes
    ] + [
        {
            "schemaname": "public",
            "tablename": table.name,
            "indexname": f"pk_{table.name}",
            "indexdef": f"CREATE UNIQUE INDEX pk_{table.name} ON {table.name} ({table.primary_key})",
        }
        for table in TABLES
    ]
    write_json(out / "pg_catalog/pg_indexes.json", indexes)

    ride_count = max(len(data["rides"]), 1)
    # (statement, calls per ride over the whole window, mean_exec_time in ms)
    statements = [
        ("SELECT * FROM get_test_scenario_by_id($1)", 14, 1.9),
        ("SELECT * FROM get_test_scenarios_by_ids($1)", 6, 3.4),
        ("SELECT * FROM get_executions_with_scenarios_with_pagination($1,$2,$3,$4,$5,$6)", 9, 168.0),
        ("SELECT * FROM get_suite_scenarios_with_pagination($1,$2,$3,$4,$5,$6)", 4, 22.5),
        (
            "SELECT id, device_id, reattempts_count, execution_id, status, "
            "(COALESCE(metadata, $1::jsonb) || jsonb_build_object($2, metadata->$3->$4)) "
            "AS ride_metadata, host_machine, summary, start_time, end_time, mission_id "
            "FROM rides WHERE id = $5",
            22,
            2.6,
        ),
        ("CALL update_ride_summary_proc($1, $2)", 3, 11.8),
        ("CALL update_ride_metadata_proc($1, $2, $3, $4, $5)", 3, 9.1),
        ("CALL update_ride_realtime_step_status_proc($1, $2, $3, $4, $5, $6)", 18, 6.7),
        ("CALL update_ride_artifacts_proc($1, $2, $3)", 1, 8.2),
        ("SELECT * FROM get_zpt_details($1)", 1, 41.3),
        (
            "UPDATE rides SET metadata = jsonb_set(COALESCE(metadata, $1::jsonb), $2, $3, true) "
            "WHERE execution_id IN ($4)",
            2,
            4.4,
        ),
        ("SELECT count(*) FROM rides WHERE execution_id = $1", 8, 1.1),
    ]
    statement_rows = []
    for query, per_ride, mean_ms in statements:
        calls = max(1, int(per_ride * ride_count))
        statement_rows.append(
            {
                "queryid": zlib.crc32(query.encode("utf-8")),
                "query": query,
                "calls": calls,
                "total_exec_time": round(calls * mean_ms, 2),
                "mean_exec_time": mean_ms,
                "max_exec_time": round(mean_ms * rng.uniform(2.5, 9.0), 2),
                "rows": calls * rng.randint(1, 40),
                "shared_blks_hit": calls * rng.randint(20, 900),
                "shared_blks_read": calls * rng.randint(0, 40),
            }
        )
    statement_rows.sort(key=lambda row: -row["total_exec_time"])
    write_json(out / "pg_catalog/pg_stat_statements.json", statement_rows)

    write_json(
        out / "pg_catalog/pg_settings.json",
        [
            {"name": "max_connections", "setting": "410", "unit": None, "source": "database"},
            {"name": "shared_buffers", "setting": "1048576", "unit": "8kB", "source": "database"},
            {"name": "work_mem", "setting": "8192", "unit": "kB", "source": "database"},
            {"name": "maintenance_work_mem", "setting": "262144", "unit": "kB", "source": "database"},
            {"name": "log_min_duration_statement", "setting": "1000", "unit": "ms", "source": "database"},
            {"name": "ssl", "setting": "on", "unit": None, "source": "override"},
            {"name": "rds.force_ssl", "setting": "1", "unit": None, "source": "database"},
            {"name": "timezone", "setting": "UTC", "unit": None, "source": "database"},
            {"name": "shared_preload_libraries", "setting": "pg_stat_statements", "unit": None, "source": "database"},
        ],
    )


def emit_rds(
    out: Path,
    rng: random.Random,
    args: argparse.Namespace,
    sotf: SotfData,
    rides_by_day: dict[str, int],
) -> None:
    endpoint = f"{args.rds_instance_id}.c9xk2plm0abc.{args.region}.rds.amazonaws.com"
    replica_id = f"{args.rds_instance_id}-ro"

    instance = {
        "DBInstanceIdentifier": args.rds_instance_id,
        "DBInstanceClass": "db.r6g.xlarge",
        "Engine": "postgres",
        "EngineVersion": "15.7",
        "DBInstanceStatus": "available",
        "MasterUsername": args.db_user,
        # The API never returns a password. The instance is managed by Secrets
        # Manager, so only the secret ARN is visible.
        "MasterUserSecret": {
            "SecretArn": (
                f"arn:aws:secretsmanager:{args.region}:123456789012:secret:"
                f"rds!db-{args.rds_instance_id}-9f2a"
            ),
            "SecretStatus": "active",
            "KmsKeyId": f"arn:aws:kms:{args.region}:123456789012:key/aws/secretsmanager",
        },
        "DBName": args.db_name,
        "Endpoint": {"Address": endpoint, "Port": 5432, "HostedZoneId": "Z1PVIF0B656C1W"},
        "AllocatedStorage": 200,
        "MaxAllocatedStorage": 1000,
        "StorageType": "gp3",
        "Iops": 3000,
        "StorageEncrypted": True,
        "KmsKeyId": f"arn:aws:kms:{args.region}:123456789012:key/8f1c0a5e-3f4b-4c22-9c11-2f6d4a7e0b31",
        "MultiAZ": True,
        "AvailabilityZone": f"{args.region}a",
        "SecondaryAvailabilityZone": f"{args.region}b",
        "BackupRetentionPeriod": 7,
        "PreferredBackupWindow": "09:12-09:42",
        "PreferredMaintenanceWindow": "sun:11:03-sun:11:33",
        "AutoMinorVersionUpgrade": True,
        "DeletionProtection": True,
        "PubliclyAccessible": False,
        "PerformanceInsightsEnabled": True,
        "PerformanceInsightsRetentionPeriod": 7,
        "EnabledCloudwatchLogsExports": ["postgresql", "upgrade"],
        "IAMDatabaseAuthenticationEnabled": True,
        "DBParameterGroups": [
            {"DBParameterGroupName": f"{args.rds_instance_id}-pg15", "ParameterApplyStatus": "in-sync"}
        ],
        "VpcSecurityGroups": [{"VpcSecurityGroupId": "sg-0a41c9f2be7d5310c", "Status": "active"}],
        "DBSubnetGroup": {
            "DBSubnetGroupName": "sotf-private",
            "VpcId": "vpc-04ad2e91c7b3f6a10",
            "SubnetGroupStatus": "Complete",
        },
        "ReadReplicaDBInstanceIdentifiers": [replica_id],
        "TagList": [
            {"Key": "service", "Value": "sotf-test-mgmt"},
            {"Key": "environment", "Value": args.environment},
            {"Key": "owner", "Value": "sotf-platform"},
        ],
        "InstanceCreateTime": iso(sotf.now - timedelta(days=420)),
    }

    replica = dict(instance)
    replica.update(
        {
            "DBInstanceIdentifier": replica_id,
            "DBInstanceClass": "db.r6g.large",
            "MultiAZ": False,
            "AvailabilityZone": f"{args.region}c",
            "Endpoint": {
                "Address": f"{replica_id}.c9xk2plm0abc.{args.region}.rds.amazonaws.com",
                "Port": 5432,
                "HostedZoneId": "Z1PVIF0B656C1W",
            },
            "ReadReplicaSourceDBInstanceIdentifier": args.rds_instance_id,
            "ReadReplicaDBInstanceIdentifiers": [],
            "BackupRetentionPeriod": 0,
            "DeletionProtection": False,
        }
    )
    write_json(out / "rds/db_instances.json", {"DBInstances": [instance, replica]})

    write_json(
        out / "rds/db_parameters.json",
        {
            "DBParameterGroupName": f"{args.rds_instance_id}-pg15",
            "Parameters": [
                {"ParameterName": "max_connections", "ParameterValue": "410", "Source": "user", "ApplyType": "static"},
                {"ParameterName": "shared_buffers", "ParameterValue": "{DBInstanceClassMemory/32768}", "Source": "engine-default", "ApplyType": "static"},
                {"ParameterName": "work_mem", "ParameterValue": "8192", "Source": "user", "ApplyType": "dynamic"},
                {"ParameterName": "log_min_duration_statement", "ParameterValue": "1000", "Source": "user", "ApplyType": "dynamic"},
                {"ParameterName": "rds.force_ssl", "ParameterValue": "1", "Source": "user", "ApplyType": "static"},
                {"ParameterName": "shared_preload_libraries", "ParameterValue": "pg_stat_statements,pg_cron", "Source": "user", "ApplyType": "static"},
                {"ParameterName": "autovacuum_vacuum_scale_factor", "ParameterValue": "0.05", "Source": "user", "ApplyType": "dynamic"},
                {"ParameterName": "max_notify_queue_pages", "ParameterValue": "1048576", "Source": "engine-default", "ApplyType": "static"},
            ],
        },
    )

    snapshots = []
    for day_offset in range(min(sotf.days, 7), 0, -1):
        taken = (sotf.now - timedelta(days=day_offset)).replace(
            hour=9, minute=18, second=0, microsecond=0
        )
        snapshots.append(
            {
                "DBSnapshotIdentifier": f"rds:{args.rds_instance_id}-{taken:%Y-%m-%d-%H-%M}",
                "DBInstanceIdentifier": args.rds_instance_id,
                "SnapshotCreateTime": iso(taken),
                "Engine": "postgres",
                "EngineVersion": "15.7",
                "AllocatedStorage": 200,
                "Status": "available",
                "SnapshotType": "automated",
                "Encrypted": True,
                "PercentProgress": 100,
            }
        )
    write_json(out / "rds/db_snapshots.json", {"DBSnapshots": snapshots})

    events = [
        {
            "SourceIdentifier": args.rds_instance_id,
            "SourceType": "db-instance",
            "Message": "Backing up DB instance",
            "EventCategories": ["backup"],
            "Date": snapshot["SnapshotCreateTime"],
        }
        for snapshot in snapshots
    ]
    events.append(
        {
            "SourceIdentifier": args.rds_instance_id,
            "SourceType": "db-instance",
            "Message": "Applying modification to database instance class",
            "EventCategories": ["configuration change"],
            "Date": iso(sotf.now - timedelta(days=min(sotf.days, 12))),
        }
    )
    events.append(
        {
            "SourceIdentifier": replica_id,
            "SourceType": "db-instance",
            "Message": "Read replica has fallen behind the master by 41 seconds",
            "EventCategories": ["read replica"],
            "Date": iso(sotf.now - timedelta(days=min(sotf.days, 4), hours=3)),
        }
    )
    events.sort(key=lambda entry: entry["Date"])
    write_json(out / "rds/events.json", {"Events": events})

    # DatabaseConnections is a pure function of the day's ride volume, so a
    # question like "why did connections spike on the 14th" has an answer in the
    # rides table. The self-check recomputes it.
    connections = []
    cpu = []
    free_storage = []
    read_iops = []
    write_iops = []
    replica_lag = []
    total_bytes = 200 * 1024**3

    for day_offset in range(sotf.days, -1, -1):
        day = (sotf.now - timedelta(days=day_offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        key = day.strftime("%Y-%m-%d")
        rides = rides_by_day.get(key, 0)
        stamp = iso(day)

        connections.append(
            {
                "Timestamp": stamp,
                "Maximum": round(CONNECTIONS_BASE + rides * CONNECTIONS_PER_RIDE, 2),
                "Unit": "Count",
            }
        )
        cpu.append(
            {
                "Timestamp": stamp,
                "Average": round(min(92.0, 6.0 + rides * 0.9 + rng.uniform(0.0, 4.0)), 2),
                "Unit": "Percent",
            }
        )
        used = int(total_bytes * (0.31 + 0.004 * (sotf.days - day_offset)))
        free_storage.append(
            {"Timestamp": stamp, "Average": total_bytes - used, "Unit": "Bytes"}
        )
        read_iops.append(
            {"Timestamp": stamp, "Average": round(40 + rides * 3.2 + rng.uniform(0, 25), 1), "Unit": "Count/Second"}
        )
        write_iops.append(
            {"Timestamp": stamp, "Average": round(18 + rides * 5.6 + rng.uniform(0, 30), 1), "Unit": "Count/Second"}
        )
        replica_lag.append(
            {"Timestamp": stamp, "Maximum": round(rng.uniform(0.2, 3.5) + rides * 0.35, 2), "Unit": "Seconds"}
        )

    write_json(
        out / "rds/cloudwatch_metrics.json",
        {
            "namespace": "AWS/RDS",
            "dimensions": {"DBInstanceIdentifier": args.rds_instance_id},
            "period_seconds": 86400,
            "note": (
                "DatabaseConnections is exactly "
                f"{CONNECTIONS_BASE} + {CONNECTIONS_PER_RIDE} * (rides started that day), "
                "so a connection spike traces back to a specific day's rides."
            ),
            "metrics": {
                "DatabaseConnections": connections,
                "CPUUtilization": cpu,
                "FreeStorageSpace": free_storage,
                "ReadIOPS": read_iops,
                "WriteIOPS": write_iops,
                "ReplicaLag": replica_lag,
            },
        },
    )


def emit_catalog(
    out: Path, args: argparse.Namespace, data: dict[str, list[dict[str, Any]]]
) -> None:
    write_json(
        out / "catalog.json",
        {
            "engine": "postgres",
            "database": args.db_name,
            "schema": "public",
            "note": (
                "One CSV per table under tables/, header row included, NULL written as "
                f"'{NULL_TOKEN}'. Load with \\copy ... WITH (FORMAT csv, HEADER true, "
                f"NULL '{NULL_TOKEN}'), or run seed.sql instead."
            ),
            "load_order": [table.name for table in TABLES],
            "tables": [
                {
                    "name": table.name,
                    "comment": table.comment,
                    "rows": len(data[table.name]),
                    "csv": f"tables/{table.name}.csv",
                    "primary_key": table.primary_key,
                    "foreign_keys": [
                        {"column": column, "references": f"public.{referenced}(id)", "on_delete": on_delete}
                        for column, referenced, on_delete in table.foreign_keys
                    ],
                    "columns": [
                        {
                            "name": column.name,
                            "type": column.sql_type,
                            "nullable": column.nullable,
                            "comment": column.comment,
                        }
                        for column in table.columns
                    ],
                }
                for table in TABLES
            ],
            "routines": [
                {
                    "name": routine.name,
                    "kind": routine.kind,
                    "signature": routine.signature,
                    "called_from": routine.called_from,
                    "body_emitted": routine.emitted,
                }
                for routine in ROUTINES
            ],
            "notify_channels": {
                "rides": "notify/rides.json",
                "executions.scenario": "notify/executions_scenario.json",
                "executions.suite": "notify/executions_suite.json",
            },
            "notes": [
                "Read-only fixtures. The DDL and seed data are here to stand up a scratch "
                "database, never to be pointed at a real SOTF environment.",
                "No credentials appear anywhere. rds/db_instances.json carries a Secrets "
                "Manager ARN, which is what describe-db-instances actually returns.",
            ],
        },
    )


def emit_cross_reference(
    out: Path,
    sotf: SotfData,
    data: dict[str, list[dict[str, Any]]],
) -> None:
    execution_by_id = {row["id"]: row for row in data["test_executions"]}
    scenario_by_id = {row["id"]: row for row in data["test_scenarios"]}

    events_per_ride: dict[str, int] = defaultdict(int)
    for event in data["ride_events"]:
        events_per_ride[event["ride_id"]] += 1

    history_versions: dict[str, list[float]] = defaultdict(list)
    for row in data["test_scenario_version_history"]:
        history_versions[row["scenario_id"]].append(float(row["version"]))

    suites_per_scenario: dict[str, list[str]] = defaultdict(list)
    for link in data["test_suite_scenarios"]:
        if not link["deleted"]:
            suites_per_scenario[link["scenario_id"]].append(link["suite_id"])

    executions_per_scenario: dict[str, list[str]] = defaultdict(list)
    for execution in data["test_executions"]:
        executions_per_scenario[execution["scenario_id"]].append(execution["id"])

    rides_per_scenario: dict[str, int] = defaultdict(int)
    failed_per_scenario: dict[str, int] = defaultdict(int)
    for ride in data["rides"]:
        execution = execution_by_id.get(ride["execution_id"])
        if execution is None:
            continue
        rides_per_scenario[execution["scenario_id"]] += 1
        if ride["status"] == "Failed":
            failed_per_scenario[execution["scenario_id"]] += 1

    scenarios = [
        {
            "scenario_id": scenario["id"],
            "name": scenario["name"],
            "current_version": scenario["version"],
            "status": scenario["status"],
            "testrail_case_id": (scenario["metadata"] or {}).get("testrail_id"),
            "step_count": len(scenario["test_steps"] or []),
            "history_versions": sorted(history_versions.get(scenario["id"], [])),
            "suite_ids": sorted(suites_per_scenario.get(scenario["id"], [])),
            "execution_ids": sorted(executions_per_scenario.get(scenario["id"], [])),
            "rides": rides_per_scenario.get(scenario["id"], 0),
            "failed_rides": failed_per_scenario.get(scenario["id"], 0),
        }
        for scenario in data["test_scenarios"]
    ]

    rides = []
    for ride in data["rides"]:
        execution = execution_by_id.get(ride["execution_id"])
        scenario = scenario_by_id.get(execution["scenario_id"]) if execution else None
        joined = sotf.cross_reference.get(ride["id"], {})
        testrail = (ride["metadata"] or {}).get("testrail") or {}
        rides.append(
            {
                "trigger_id": ride["id"],
                "ride_status": ride["status"],
                "execution_id": ride["execution_id"],
                "scenario_id": execution["scenario_id"] if execution else None,
                "scenario_name": scenario["name"] if scenario else None,
                "suite_id": execution["suite_id"] if execution else None,
                "ride_events": events_per_ride.get(ride["id"], 0),
                "testrail": {
                    "run_id": testrail.get("run_id"),
                    "test_id": testrail.get("test_id"),
                    "result_id": testrail.get("result_id"),
                },
                "jira_key": joined.get("jira_key"),
                "s3_key": (joined.get("s3") or {}).get("key"),
                "failure_mode": joined.get("failure_mode"),
            }
        )

    write_json(
        out / "cross_reference_postgres.json",
        {
            "note": (
                "How the Postgres rows join to the rest of the SOTF dataset. A scenario "
                "row is the anchor: it fans out to suites through test_suite_scenarios, to "
                "executions through test_executions.scenario_id, and to rides through "
                "rides.execution_id. The trigger_id in the rides table is the same id S3, "
                "ClickHouse, TestRail and Jira key on."
            ),
            "scenarios": scenarios,
            "rides": rides,
        },
    )


LOADER_TEMPLATE = """#!/usr/bin/env bash
# Load the generated SOTF test-mgmt fixtures into a throwaway Postgres.
# Point PGHOST at a local container or a scratch RDS instance -- never at a real
# SOTF environment.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

: "${{PGHOST:=localhost}}"
: "${{PGPORT:=5432}}"
: "${{PGUSER:={db_user}}}"
: "${{PGDATABASE:={db_name}}}"

PSQL=(psql --no-psqlrc --set ON_ERROR_STOP=1 -h "${{PGHOST}}" -p "${{PGPORT}}" \\
      -U "${{PGUSER}}" -d "${{PGDATABASE}}")

echo "==> applying schema"
for file in "${{HERE}}"/schema/*.sql; do
  echo "    ${{file##*/}}"
  "${{PSQL[@]}}" -f "${{file}}"
done

echo "==> loading table data"
{copy_lines}

echo -n "==> scenarios loaded: "
"${{PSQL[@]}}" -tAc 'SELECT count(*) FROM test_scenarios'
echo -n "==> rides loaded: "
"${{PSQL[@]}}" -tAc 'SELECT count(*) FROM rides'

echo "seed.sql contains the same rows as INSERT statements if \\copy is unavailable."
"""


def emit_loader(out: Path, args: argparse.Namespace) -> None:
    copy_lines = "\n".join(
        f'"${{PSQL[@]}}" -c "\\\\copy public.{table.name} '
        f'({", ".join(table.column_names)}) FROM \'${{HERE}}/tables/{table.name}.csv\' '
        f"WITH (FORMAT csv, HEADER true, NULL '{NULL_TOKEN}')\""
        for table in TABLES
    )
    path = out / "load_postgres.sh"
    write_text(
        path,
        LOADER_TEMPLATE.format(
            db_user=args.db_user, db_name=args.db_name, copy_lines=copy_lines
        ),
    )
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------

STATUS_DOMAIN = {
    "test_scenarios": SCENARIO_STATUS,
    "test_scenario_version_history": SCENARIO_STATUS,
    "test_executions": EXECUTION_STATUS,
    "rides": RIDE_STATUS,
}

SECRET_MARKERS = ("password", "secret", "token", "apikey", "api_key")

# Real API fields whose name matches a marker but whose value is status metadata,
# not a credential.
SECRET_KEY_ALLOWLIST = {"secretstatus"}


def find_literal_secrets(node: Any, origin: str) -> list[str]:
    """Any credential-shaped key must hold an ARN, not a value."""
    problems: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            looks_sensitive = (
                lowered not in SECRET_KEY_ALLOWLIST
                and any(marker in lowered for marker in SECRET_MARKERS)
            )
            if looks_sensitive and isinstance(value, str) and not value.startswith("arn:aws:"):
                problems.append(f"{origin}: {key} holds a literal value instead of an ARN")
            problems.extend(find_literal_secrets(value, origin))
    elif isinstance(node, list):
        for item in node:
            problems.extend(find_literal_secrets(item, origin))
    return problems


def run_self_check(
    out: Path,
    sotf: SotfData,
    data: dict[str, list[dict[str, Any]]],
    rides_by_day: dict[str, int],
) -> list[str]:
    problems: list[str] = []

    ids_by_table = {name: {row["id"] for row in rows} for name, rows in data.items()}

    # Foreign keys resolve.
    for table in TABLES:
        for column, referenced, _ in table.foreign_keys:
            targets = ids_by_table[referenced]
            for row in data[table.name]:
                value = row.get(column)
                if value is None:
                    continue
                if value not in targets:
                    problems.append(
                        f"{table.name}.{column} = {value} has no row in {referenced}"
                    )
                    break

    # Primary keys are unique and non-null.
    for table in TABLES:
        rows = data[table.name]
        if len(ids_by_table[table.name]) != len(rows):
            problems.append(f"{table.name}: duplicate primary keys")
        if any(row.get(table.primary_key) is None for row in rows):
            problems.append(f"{table.name}: null primary key")

    # NOT NULL columns really are populated.
    for table in TABLES:
        for column in table.columns:
            if column.nullable or column.default is not None:
                continue
            if any(row.get(column.name) is None for row in data[table.name]):
                problems.append(f"{table.name}.{column.name} is NOT NULL but has null rows")

    # Status values stay inside the CHECK constraints.
    for table_name, allowed in STATUS_DOMAIN.items():
        for row in data[table_name]:
            if row["status"] not in allowed:
                problems.append(f"{table_name}: status {row['status']!r} is outside the domain")
                break

    # uq_suite_scenario.
    pairs = [(row["suite_id"], row["scenario_id"]) for row in data["test_suite_scenarios"]]
    if len(set(pairs)) != len(pairs):
        problems.append("test_suite_scenarios violates uq_suite_scenario")

    # No execution points at a suite/scenario pair whose membership was removed.
    removed = {
        (row["suite_id"], row["scenario_id"])
        for row in data["test_suite_scenarios"]
        if row["deleted"]
    }
    for execution in data["test_executions"]:
        if execution["suite_id"] and (execution["suite_id"], execution["scenario_id"]) in removed:
            problems.append(
                f"execution {execution['id']} runs a scenario removed from its suite"
            )
            break

    # Version history sits strictly below the current version, once per version.
    current_version = {row["id"]: float(row["version"]) for row in data["test_scenarios"]}
    seen: set[tuple[str, float]] = set()
    for row in data["test_scenario_version_history"]:
        key = (row["scenario_id"], float(row["version"]))
        if key in seen:
            problems.append(f"duplicate history row for scenario {row['scenario_id']} v{row['version']}")
        seen.add(key)
        if float(row["version"]) >= current_version.get(row["scenario_id"], 0.0):
            problems.append(
                f"history v{row['version']} is not older than the current version of "
                f"scenario {row['scenario_id']}"
            )

    # Row counts agree with the SOTF dataset.
    for table_name, count_key in (
        ("test_scenarios", "scenarios"),
        ("test_suites", "suites"),
        ("test_executions", "executions"),
        ("rides", "rides"),
    ):
        expected = sotf.counts.get(count_key)
        if expected is not None and len(data[table_name]) != expected:
            problems.append(
                f"{table_name} has {len(data[table_name])} rows, SOTF summary says {expected}"
            )

    # Every completed ride produced a terminal event.
    events_by_ride: dict[str, set[str]] = defaultdict(set)
    for event in data["ride_events"]:
        events_by_ride[event["ride_id"]].add(event["event"])
    terminal = {"RideCompleted", "RideFailed", "RideInvalidated"}
    for ride in data["rides"]:
        if ride["status"] in ("Passed", "Failed", "Invalid") and ride["end_time"]:
            if not (events_by_ride[ride["id"]] & terminal):
                problems.append(f"ride {ride['id']} finished with no terminal ride_event")
                break

    # CSV files match the in-memory rows.
    for table in TABLES:
        path = out / f"tables/{table.name}.csv"
        if not path.is_file():
            problems.append(f"missing tables/{table.name}.csv")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            written = sum(1 for _ in reader)
        if header != table.column_names:
            problems.append(f"tables/{table.name}.csv header does not match the table definition")
        if written != len(data[table.name]):
            problems.append(
                f"tables/{table.name}.csv has {written} rows, expected {len(data[table.name])}"
            )

    # Dollar quoting in seed.sql is safe.
    for table in TABLES:
        for row in data[table.name]:
            for name in table.json_columns:
                if DOLLAR_TAG in json_compact(row.get(name)):
                    problems.append(f"{table.name}.{name} contains the {DOLLAR_TAG} quote tag")
                    break

    # NOTIFY payload statuses resolve to real enum members.
    ride_members = {enum_name(value) for value in RIDE_STATUS}
    state_members = set(EXECUTION_STATE_BY_STATUS.values())
    result_members = {enum_name(value) for value in EXECUTION_STATUS} | {""}
    for relative, key, allowed in (
        ("notify/rides.json", "status", ride_members),
        ("notify/executions_scenario.json", "status", state_members),
        ("notify/executions_scenario.json", "result", result_members),
        ("notify/executions_suite.json", "batch_status", state_members),
        ("notify/executions_suite.json", "batch_result", result_members),
    ):
        path = out / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
            continue
        for entry in json.loads(path.read_text(encoding="utf-8"))["notifications"]:
            value = entry["payload"].get(key)
            if value not in allowed:
                problems.append(f"{relative}: {key} = {value!r} is not an enum member name")
                break

    # CloudWatch connections must be recomputable from the rides table.
    metrics_file = out / "rds/cloudwatch_metrics.json"
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))["metrics"]
        for point in metrics["DatabaseConnections"]:
            day = point["Timestamp"][:10]
            expected = round(CONNECTIONS_BASE + rides_by_day.get(day, 0) * CONNECTIONS_PER_RIDE, 2)
            if point["Maximum"] != expected:
                problems.append(
                    f"DatabaseConnections on {day} is {point['Maximum']}, "
                    f"but the rides table implies {expected}"
                )
                break

    # No credentials anywhere in the RDS fixtures.
    for path in sorted((out / "rds").glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        problems.extend(find_literal_secrets(blob, path.name))

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mock Amazon RDS for PostgreSQL data for the SOTF test-mgmt "
            "database, aligned with an existing SOTF mock dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sotf-data", type=Path, default=HERE / "out", help="output of generate_sotf_mock_data.py"
    )
    parser.add_argument("--out", type=Path, default=HERE / "out_postgres")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    parser.add_argument("--db-name", default="tcc_db_dev", help="init_db.sh DB_NAME")
    parser.add_argument("--db-user", default="tcc_db_dev", help="init_db.sh DB_USER")
    parser.add_argument("--rds-instance-id", default="sotf-tcc-db-dev")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--environment", default="dev")
    parser.add_argument(
        "--schedules", type=int, default=8, help="rows in scheduled_test_executions"
    )
    parser.add_argument(
        "--soft-delete-rate",
        type=float,
        default=0.12,
        help="share of unused suite-scenario links marked deleted",
    )
    parser.add_argument(
        "--notify-samples", type=int, default=6, help="rides to render NOTIFY payloads for"
    )
    parser.add_argument("--insert-batch", type=int, default=25, help="rows per INSERT in seed.sql")
    parser.add_argument("--no-seed-sql", action="store_true", help="skip writing seed.sql")
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

    ride_rows = build_ride_rows(sotf)
    execution_rows = build_execution_rows(sotf)
    data: dict[str, list[dict[str, Any]]] = {
        "test_scenarios": build_scenario_rows(sotf),
        "test_suites": build_suite_rows(sotf),
        "test_scenario_version_history": build_version_history_rows(rng, sotf),
        "test_suite_scenarios": build_suite_scenario_rows(rng, sotf, args.soft_delete_rate),
        "test_executions": execution_rows,
        "scheduled_test_executions": build_scheduled_execution_rows(rng, sotf, args.schedules),
        "rides": ride_rows,
        "ride_events": build_ride_event_rows(rng, sotf, ride_rows),
    }

    rides_by_day: dict[str, int] = defaultdict(int)
    for ride in ride_rows:
        started = parse_dt(ride["start_time"])
        if started:
            rides_by_day[started.strftime("%Y-%m-%d")] += 1

    emit_schema(out)
    for table in TABLES:
        emit_csv(out, table, data[table.name])
    if not args.no_seed_sql:
        emit_seed_sql(out, data, args.insert_batch)
    notify_counts = emit_notify_samples(
        out, rng, ride_rows, execution_rows, args.notify_samples
    )
    emit_pg_catalog(out, rng, sotf, data)
    emit_rds(out, rng, args, sotf, rides_by_day)
    emit_catalog(out, args, data)
    emit_cross_reference(out, sotf, data)
    write_text(out / "sample_queries.sql", SAMPLE_QUERIES_SQL)
    emit_loader(out, args)

    counts = {
        "tables": len(TABLES),
        "rows": {table.name: len(data[table.name]) for table in TABLES},
        "total_rows": sum(len(rows) for rows in data.values()),
        "routines_declared": len(ROUTINES),
        "routines_with_bodies": sum(1 for routine in ROUTINES if routine.emitted),
        "notify_payloads": notify_counts,
        "scenarios_with_history": len(
            {row["scenario_id"] for row in data["test_scenario_version_history"]}
        ),
        "suite_links_deleted": sum(1 for row in data["test_suite_scenarios"] if row["deleted"]),
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
                "database": args.db_name,
                "rds_instance_id": args.rds_instance_id,
                "region": args.region,
                "schedules": args.schedules,
                "soft_delete_rate": args.soft_delete_rate,
            },
            "counts": counts,
        },
    )

    print(f"Wrote Postgres mock data to {out}")
    print(
        "  {tables} tables, {total_rows} rows "
        "({rows[test_scenarios]} scenarios, {rows[test_executions]} executions, "
        "{rows[rides]} rides, {rows[ride_events]} ride events)".format(**counts)
    )

    if args.skip_self_check:
        return 0

    problems = run_self_check(out, sotf, data, rides_by_day)
    if problems:
        print(f"  self-check FAILED with {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    - {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return 1

    print("  self-check passed: foreign keys, status domains, CSV counts and metrics agree")
    print(f"  join table: {out / 'cross_reference_postgres.json'}")
    print(f"  loader:     {out / 'load_postgres.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
