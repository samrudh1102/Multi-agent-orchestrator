# SOTF mock data generator

`generate_sotf_mock_data.py` produces a synthetic but realistically-shaped SOTF
dataset across the four systems that SOTF actually integrates with today: **S3**,
**ClickHouse**, **Jira** and **TestRail**. It is meant for building and testing
things like a read-only SOTF assistant without touching a real environment.

Standard library only — no dependencies to install.

## Run it

```bash
cd mock_data
python3 generate_sotf_mock_data.py --clean

# reproducible output (timestamps are relative to "now" unless you pin it)
python3 generate_sotf_mock_data.py --clean --seed 7 --now 2026-07-27T05:00:00+00:00

# bigger, failure-heavy dataset
python3 generate_sotf_mock_data.py --clean --executions 200 --failure-rate 0.5
```

Reference data (PUDO locations, environments, devices, Jira custom-field option
ids) is read from the SOTF repo at generation time:

- `driving/qa/siv/sotf/tcc/api-py/test-mgmt/metadata.json`
- `driving/qa/siv/sotf/tcc/api-py/test-mgmt/app/util/jira_issue_mappings.json`

Override with `--metadata` / `--jira-mappings`. If the mappings file is missing,
an embedded copy of the field ids is used.

After writing, the script runs a **self-check** that verifies every ride joins
across all five views (TestRail case/run/test/result exist, the artifact zip
exists and is readable, the manifest exists, the Jira issue file exists, log rows
were written). It exits non-zero and lists the problems if anything fails to
line up. Use `--skip-self-check` to bypass.

## What you get

```
out/
  summary.json            counts + the exact parameters used
  cross_reference.json    the join table: one row per ride, ids for every system
  load_mocks.sh           uploads the zips to S3 and inserts the log rows
  domain/                 test-mgmt Postgres rows
    scenarios.json  suites.json  executions.json  rides.json
    pudo_locations.json  metadata.json
  testrail/               TestRail API v2 response bodies
    suites.json  sections.json  cases.json  runs.json
    tests_by_run/<run_id>.json
    results_by_test/<test_id>.json
    attachments.json  api_index.json
  jira/                   Jira REST v2 bodies
    issues/ZPT-<n>.json
    search_issues.json  createmeta_ZPT_Problem.json  api_index.json
  s3/
    zoox-test-artifacts/tec/dev/artifacts/<trigger_id>.zip    real zip files
    manifests/<trigger_id>.json                               SQS Artifacts message
    objects.json  api_index.json
  clickhouse/
    schema.sql  logs.jsonl  sample_queries.sql  api_index.json
```

Each `api_index.json` maps real API routes to the fixture files, so you can serve
`jira/` and `testrail/` from any mock HTTP server.

## The cross-reference is the point

A single failed ride is reachable from every system, which is what makes the
dataset useful for failure-analysis work:

```
ride (trigger_id)
  ├─ domain/rides.json          metadata.testrail = {run_id, test_id, result_id}
  ├─ testrail/results_by_test/  the failing result, step results, defects = ZPT key
  ├─ s3/.../<trigger_id>.zip    snapshots/, video/, logs/, summary.json
  ├─ clickhouse/logs.jsonl      rows carrying trigger_id, with WARN/ERROR lines
  └─ jira/issues/ZPT-<n>.json   the defect, if one was filed
```

Failures are generated from a taxonomy of nine modes (pickup geofence mismatch,
milestone M4 timeout, unassigned vehicle, rider app crash, webapi timeout, device
disconnect, sim pose stall, door unlock not acknowledged, blocked drop-off). One
mode drives the failing step, the log lines, the TestRail comment and the Jira
description consistently, so a correct answer exists for every "why did this
fail?" question.

Recurring failures matter too: the same failure mode in the same ride service
usually links to the **existing** ZPT with an incremented occurrence count rather
than filing a new one, so "is this a known issue?" has a real answer.

## Where each shape comes from

| Emitted data | Source in the repo |
|---|---|
| Ride / execution / scenario status literals | `app/model/rides_model.py`, `execution_model.py`, `scenario_model.py` (Title Case, not the older `init_db.sh` UPPER_SNAKE) |
| Step statuses, artifact types, failure reasons | `app/dto/rides_dto.py` (`StepStatusType`, `ArtifactType`, `FailureReason`) |
| `metadata.testrail` = `{run_id, test_id, result_id, artifacts_uploaded}` | `app/dto/testrail_request_dto.py`, `rides_repository.py` |
| Ride `summary` shape | `RideSummaryLiteDto` in `app/dto/rides_dto.py` |
| Execution `payload` (`apl`/`adl`/`ppl`/`pdl`, coordinates, `webapi_domain`) | `ExecutionParameters` in `app/dto/execution_dto.py` |
| TestRail case title format | `execution_service._format_tr_title` |
| TestRail suite `SOTF` / section `Automated Scenarios` / `Adhoc Executions` | `app/config.py`, `execution_service.py` |
| TestRail status ids (1/2/3/4/5) and case defaults (`template_id` 2, `type_id` 3, `priority_id` 2) | `app/dto/testrail_request_dto.py` |
| TestRail result comment text | `integration_utils.build_testrail_comment` |
| Jira project `ZPT`, issue type `Problem` | `jira_integration/app/config.py` |
| Jira custom field ids and option ids | `app/util/jira_issue_mappings.json` |
| ZPT description + `customfield_20316` narrative | `rides_service.generate_zpt`, `_format_zpt_jira_description` |
| `customfield_21008` date format (`%d/%b/%y %I:%M %p`) | `rides_service.generate_zpt` |
| S3 key `<prefix>/<trigger_id>.zip`, prefix `tec/${ENV}/artifacts` | `device_service/app/artifact_pipeline/uploader.py`, `config.py` |
| Zip interior (`snapshots/`, `video/`, `logs/`) | `device_service/DEVELOPER_GUIDE.md`, staging layout in the artifact pipeline |
| Artifacts SQS manifest | `ArtifactManifestDto`, `ride_artifacts_handler.py` |
| ClickHouse `timestamp` / `message` columns, `JSONEachRow` | `ctm_logs_service/app/clickhouse_service.py` |
| PUDO locations, environments, devices, host machines | `test-mgmt/metadata.json` |

Three things are invented because the repo does not define them:

- **`pudo_zone` values** (`LV-Strip-South`, `SF-Downtown`, …). `ExecutionParameters.pudo_zone`
  exists but has no source of truth in the repo.
- **The ClickHouse table schema.** `ctm_logs_service` never creates the table and
  only assumes `timestamp` and `message`; `schema.sql` adds the columns an
  assistant would want to filter on rather than parse out of the message text.
- **Scenario names and steps.** Written to look like SOTF rider-app and
  operations flows, using the real `action_category` values that
  `scenario_mapper.py` maps to `APP` / `RD` / `TD` / `MC`.

Two fidelity notes:

- `StepStatus` in TestRail has no `SKIPPED` or `STUCK` member, so those step
  statuses map to `RETEST` (4) in `custom_step_results`.
- All artifacts land under one `${ENV}`-expanded prefix, matching the real TEC
  service where the deployment environment (not the execution's environment)
  determines the prefix. Executions still span all five environments.

## Loading it somewhere

```bash
cd out
S3_BUCKET_NAME=my-scratch-bucket CLICKHOUSE_URL=http://localhost:8123 ./load_mocks.sh
```

Point it at local stand-ins — MinIO and a scratch ClickHouse. **Never at a real
SOTF environment.** Jira and TestRail are static JSON on purpose; serve them
read-only rather than writing into a real tracker.

## A note on the ClickHouse tool boundary

`clickhouse_service.build_search_query` interpolates the table name, column
names, search patterns and `additional_conditions` directly into SQL. That is
fine when your own code is the caller, but if an assistant ever supplies those
arguments it becomes an injection path. `sample_queries.sql` shows the shape a
tool wrapper should expose instead: an identifier plus a bounded time window,
with table and column names resolved from a server-side allow-list.

## Not covered

Looker, Airflow, Redshift, Grafana and Databricks have no SOTF implementation to
imitate — they appear only in the `SOTF_Chatbot_*` planning documents. Nothing is
generated for them here.
