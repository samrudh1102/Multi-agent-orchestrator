#!/usr/bin/env python3
"""Generate a cross-referenced mock dataset shaped like SOTF production data.

One synthetic set of test executions is emitted as five consistent views:

    domain/      test-mgmt Postgres rows (scenarios, suites, executions, rides)
    testrail/    TestRail API v2 response bodies (suites/cases/runs/tests/results)
    jira/        Jira REST v2 issue bodies for ZPT defects
    s3/          real .zip artifact bundles under the TEC artifacts prefix
    clickhouse/  CTM log rows in JSONEachRow format, plus DDL and sample queries

Every ride is reachable from every system, which is what makes the dataset
useful for failure-analysis work: a ride carries its TestRail ids in
metadata.testrail, its artifact bundle lives at <prefix>/<trigger_id>.zip, its
CTM log rows carry trigger_id, and a failed ride may link to a ZPT Jira issue.
cross_reference.json is the join table.

Field names, enum literals, id formats and title formats are taken from the SOTF
services rather than invented; see README.md for the source of each.

Usage:
    python3 generate_sotf_mock_data.py --clean
    python3 generate_sotf_mock_data.py --seed 7 --executions 120 --failure-rate 0.4
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import uuid
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MGMT = REPO_ROOT / "driving/qa/siv/sotf/tcc/api-py/test-mgmt"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The real SOTF test-mgmt checkout isn't present in this repo, so the defaults
# point at the bundled fixtures/metadata.json instead. Pass --metadata / to
# point at a real test-mgmt checkout's metadata.json if you have one (e.g. the
# SOTF automation-internal repo) instead of the bundled fixture.
DEFAULT_METADATA = FIXTURES / "metadata.json" if not (TEST_MGMT / "metadata.json").is_file() else TEST_MGMT / "metadata.json"
DEFAULT_JIRA_MAPPINGS = TEST_MGMT / "app/util/jira_issue_mappings.json"

# ---------------------------------------------------------------------------
# Literals lifted from the SOTF services
# ---------------------------------------------------------------------------

# rides_model.RideStatus / execution_model.ExecutionStatus / scenario_model.ScenarioStatus
RIDE_STATUS = ("Queued", "Invalid", "Running", "Passed", "Failed", "Scheduled")
EXECUTION_STATUS = (
    "Queued",
    "Running",
    "Passed",
    "Invalid",
    "Partially Failed",
    "Failed",
    "Scheduled",
)
SCENARIO_STATUS = ("Active", "Inactive", "Deprecated")

# environment_util slugs keyed by the metadata.json display name
ENV_SLUGS = {
    "Dev": "dev",
    "Staging": "staging",
    "SDET": "sdet",
    "Prod 0": "prod",
    "Prod 1": "prod1",
}

# scenario_mapper.py action_category -> SQS entity
ENTITY_BY_CATEGORY = {
    "mobile rider app": "APP",
    "riderdash": "RD",
    "teledash": "TD",
    "mission control": "MC",
}

# testrail_request_dto.StepStatus
TESTRAIL_STATUS_ID = {"PASSED": 1, "BLOCKED": 2, "UNTESTED": 3, "RETEST": 4, "FAILED": 5}
TESTRAIL_REGRESSION_SUITE = {"id": 8, "section_id": 5, "name": "SOTF", "description": "Automation Test Suite"}
TESTRAIL_ADHOC_SUITE = {
    "id": 9,
    "section_id": 6,
    "name": "Adhoc Executions",
    "description": "Suite for single scenario executions",
}
TESTRAIL_SECTION_NAME = "Automated Scenarios"

JIRA_PROJECT_KEY = "ZPT"
JIRA_ISSUE_TYPE = "Problem"
FOUND_DURING = "Testing - Milestone testing"
VEHICLE_TYPE = "Container Sim"
VEHICLE_SUBTYPES = (
    "kitt_cs_wink1",
    "kitt_cs_wink2",
    "kitt_cs_wink3",
    "vh6_cs_alpha2",
    "vh6_cs_alpha3",
)

HOST_MACHINES = ("Host machine 1", "Host machine 2")
AUTHORS = (
    "a.patel@example.com",
    "j.nguyen@example.com",
    "m.oconnor@example.com",
    "s.desai@example.com",
    "l.zhang@example.com",
    "r.gomez@example.com",
)

# pudo_zone is not modelled in the repo, so zones are assigned per ride service.
ZONES_BY_RIDE_SERVICE = {
    "Las Vegas": ("LV-Strip-North", "LV-Strip-South", "LV-Westside"),
    "San Francisco": ("SF-Downtown", "SF-Marina", "SF-Mission"),
}

MILESTONES = (
    ("M1", "RideRequested"),
    ("M2", "VehicleAssigned"),
    ("M3", "EnRouteToPickup"),
    ("M4", "ArrivedAtPickup"),
    ("M5", "TripStarted"),
    ("M6", "ArrivedAtDropoff"),
    ("M7", "TripCompleted"),
)
MILESTONE_ORDER = [code for code, _ in MILESTONES]

# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------

SCENARIO_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "name": "Rider requests ride and completes trip",
        "labels": ["regression", "rider-app", "happy-path"],
        "steps": [
            ("Mobile Rider App", "Launch rider app and sign in"),
            ("Mobile Rider App", "Set pickup and drop-off locations"),
            ("Mobile Rider App", "Confirm ride request"),
            ("Mission Control", "Verify vehicle assignment for the request"),
            ("Mobile Rider App", "Wait for vehicle arrival at pickup"),
            ("Mobile Rider App", "Unlock vehicle and start trip"),
            ("Mobile Rider App", "Verify arrival at drop-off"),
            ("RiderDash", "Verify ride is marked complete"),
        ],
    },
    {
        "name": "Rider cancels ride before vehicle arrival",
        "labels": ["regression", "rider-app", "cancellation"],
        "steps": [
            ("Mobile Rider App", "Request a ride from the selected pickup"),
            ("Mission Control", "Verify vehicle assignment for the request"),
            ("Mobile Rider App", "Cancel the ride while the vehicle is en route"),
            ("Mobile Rider App", "Verify cancellation confirmation is shown"),
            ("RiderDash", "Verify ride is marked cancelled"),
        ],
    },
    {
        "name": "Rider changes drop-off during ride",
        "labels": ["regression", "rider-app", "in-ride"],
        "steps": [
            ("Mobile Rider App", "Request a ride and board the vehicle"),
            ("Mobile Rider App", "Open the drop-off editor during the trip"),
            ("Mobile Rider App", "Select a new drop-off location"),
            ("Mission Control", "Verify the route update is accepted"),
            ("Mobile Rider App", "Verify arrival at the updated drop-off"),
        ],
    },
    {
        "name": "Rider unlocks vehicle at pickup",
        "labels": ["smoke", "rider-app", "pickup"],
        "steps": [
            ("Mobile Rider App", "Request a ride from the selected pickup"),
            ("Mobile Rider App", "Wait for vehicle arrival at pickup"),
            ("Mobile Rider App", "Tap unlock and verify the door opens"),
            ("TeleDash", "Verify door state is reported as unlocked"),
        ],
    },
    {
        "name": "TeleDash operator monitors active mission",
        "labels": ["regression", "teledash", "operations"],
        "steps": [
            ("Mission Control", "Start a mission with an assigned vehicle"),
            ("TeleDash", "Open the active mission view"),
            ("TeleDash", "Verify live pose and telemetry are streaming"),
            ("TeleDash", "Verify the milestone timeline matches mission state"),
        ],
    },
    {
        "name": "Mission Control assigns vehicle to ride request",
        "labels": ["regression", "mission-control", "dispatch"],
        "steps": [
            ("Mobile Rider App", "Submit a ride request"),
            ("Mission Control", "Verify the request appears in the dispatch queue"),
            ("Mission Control", "Assign an available vehicle to the request"),
            ("Mobile Rider App", "Verify the assigned vehicle is shown to the rider"),
        ],
    },
    {
        "name": "RiderDash operator resolves stuck ride",
        "labels": ["regression", "riderdash", "recovery"],
        "steps": [
            ("Mission Control", "Start a ride and hold the vehicle en route"),
            ("RiderDash", "Locate the stuck ride in the active list"),
            ("RiderDash", "Apply the recovery action for a stuck ride"),
            ("RiderDash", "Verify the ride returns to an active state"),
        ],
    },
    {
        "name": "Rider requests ride with accessibility option",
        "labels": ["regression", "rider-app", "accessibility"],
        "steps": [
            ("Mobile Rider App", "Enable the accessibility preference"),
            ("Mobile Rider App", "Request a ride from the selected pickup"),
            ("Mission Control", "Verify an accessible vehicle is assigned"),
            ("Mobile Rider App", "Verify the extended boarding window is applied"),
            ("Mobile Rider App", "Complete the trip to drop-off"),
        ],
    },
    {
        "name": "Rider reports issue during ride",
        "labels": ["regression", "rider-app", "support"],
        "steps": [
            ("Mobile Rider App", "Request a ride and board the vehicle"),
            ("Mobile Rider App", "Open in-ride support and report an issue"),
            ("RiderDash", "Verify the reported issue appears for the operator"),
            ("TeleDash", "Verify the operator can view the affected mission"),
        ],
    },
    {
        "name": "Rider completes multi-stop ride",
        "labels": ["regression", "rider-app", "multi-stop"],
        "steps": [
            ("Mobile Rider App", "Request a ride with an intermediate stop"),
            ("Mission Control", "Verify the multi-stop route is accepted"),
            ("Mobile Rider App", "Verify arrival at the intermediate stop"),
            ("Mobile Rider App", "Resume the trip to the final drop-off"),
            ("RiderDash", "Verify both stops are recorded on the ride"),
        ],
    },
    {
        "name": "Vehicle rebalances to charging after drop-off",
        "labels": ["regression", "mission-control", "fleet"],
        "steps": [
            ("Mobile Rider App", "Complete a trip to drop-off"),
            ("Mission Control", "Verify the vehicle enters the rebalancing state"),
            ("Mission Control", "Verify a charging destination is assigned"),
            ("TeleDash", "Verify the vehicle reaches the charging location"),
        ],
    },
    {
        "name": "Rider account sign-in and payment setup",
        "labels": ["smoke", "rider-app", "account"],
        "steps": [
            ("Mobile Rider App", "Sign in with a test rider account"),
            ("Mobile Rider App", "Add a test payment method"),
            ("Mobile Rider App", "Verify the payment method is listed as default"),
            ("Mobile Rider App", "Request a ride and verify no payment prompt"),
        ],
    },
)

SUITE_CATALOG: tuple[dict[str, Any], ...] = (
    {"name": "SOTF Rider App Regression", "description": "Full rider app regression pass"},
    {"name": "SOTF Smoke", "description": "Pre-deploy smoke checks"},
    {
        "name": "SOTF Operations Regression",
        "description": "RiderDash, TeleDash and Mission Control flows",
    },
)

# ---------------------------------------------------------------------------
# Failure taxonomy: one place that drives step results, logs, TestRail and Jira
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureMode:
    key: str
    component: str
    failure_reason: str  # FailureReason literal
    mission_phase: str  # customfield_19416 option label
    step_fraction: float  # where in the step list the failure lands
    last_milestone: str  # last milestone reached before the failure
    failed_event: str
    expected_result: str
    actual_result: str
    step_message: str
    warn: tuple[str, ...]
    error: tuple[str, ...]


FAILURE_MODES: tuple[FailureMode, ...] = (
    FailureMode(
        key="pickup_geofence_mismatch",
        component="geofence_monitor",
        failure_reason="SIMULATOR",
        mission_phase="Pick-Up",
        step_fraction=0.55,
        last_milestone="M3",
        failed_event="Vehicle stopped outside the pickup geofence at {pickup}",
        expected_result="Vehicle comes to a stop inside the pickup geofence at {pickup} and the app enables the unlock control.",
        actual_result="Vehicle stopped 47 m outside the pickup geofence and the unlock control stayed disabled.",
        step_message="Pickup geofence validation failed; unlock control never enabled",
        warn=(
            "geofence boundary deviation of 47 m detected for zone {zone}",
            "pickup arrival confidence dropped to 0.42 for mission {mission_id}",
        ),
        error=(
            "pickup geofence validation failed for zone {zone}: vehicle pose outside polygon",
            "milestone M4 (ArrivedAtPickup) rejected by geofence_monitor for trigger {trigger_id}",
        ),
    ),
    FailureMode(
        key="milestone_timeout",
        component="mission_monitor",
        failure_reason="SIMULATOR",
        mission_phase="Hailing",
        step_fraction=0.5,
        last_milestone="M3",
        failed_event="Milestone M4 (ArrivedAtPickup) not reached within 300s",
        expected_result="Milestone M4 (ArrivedAtPickup) is reached within the 300s window after assignment.",
        actual_result="Milestone M4 was still pending when the 300s window expired and the ride was marked failed.",
        step_message="Milestone M4 (ArrivedAtPickup) not reached within 300s",
        warn=(
            "milestone M4 still pending after 240s of a 300s window for mission {mission_id}",
            "vehicle {vehicle} average speed 1.8 m/s is below the 4.0 m/s route estimate",
        ),
        error=(
            "milestone timeout: M4 (ArrivedAtPickup) not reached within 300s for mission {mission_id}",
        ),
    ),
    FailureMode(
        key="vehicle_unassigned",
        component="dispatch",
        failure_reason="SIMULATOR",
        mission_phase="Hailing",
        step_fraction=0.3,
        last_milestone="M1",
        failed_event="No vehicle assigned to the ride request within 120s",
        expected_result="A vehicle is assigned to the ride request within 120s of submission.",
        actual_result="Dispatch returned no candidate vehicles and the request expired unassigned.",
        step_message="No vehicle assigned within 120s",
        warn=(
            "no candidate vehicle within 2.5 km of {pickup}; widening search radius",
            "fleet availability in zone {zone} is 0 of 4 vehicles",
        ),
        error=(
            "dispatch returned zero candidates for trigger {trigger_id}; assignment timed out after 120s",
        ),
    ),
    FailureMode(
        key="rider_app_crash_on_confirm",
        component="rider_app",
        failure_reason="DEVICE",
        mission_phase="Hailing",
        step_fraction=0.25,
        last_milestone="M1",
        failed_event="Rider app crashed while confirming the ride request",
        expected_result="Ride request confirmation returns to the tracking screen without terminating the app.",
        actual_result="The rider app terminated on confirm and the session was lost before assignment.",
        step_message="Rider app terminated unexpectedly on confirm",
        warn=("rider app main thread blocked for 4.8s during confirm on {device}",),
        error=(
            "rider app terminated unexpectedly (SIGABRT) on device {device}",
            "session lost for trigger {trigger_id}; remaining steps skipped",
        ),
    ),
    FailureMode(
        key="webapi_timeout",
        component="webapi_client",
        failure_reason="NETWORK",
        mission_phase="Hailing",
        step_fraction=0.2,
        last_milestone="M1",
        failed_event="webapi POST /rides timed out after 30s",
        expected_result="webapi POST /rides returns 201 within the 5s latency budget.",
        actual_result="webapi POST /rides timed out after 30s across three retries.",
        step_message="webapi POST /rides timed out after 30s",
        warn=("webapi latency 12.4s exceeds the 5s budget for POST /rides on {env}",),
        error=(
            "webapi request POST /rides timed out after 30000 ms",
            "retry 3 of 3 failed for trigger {trigger_id}; aborting ride",
        ),
    ),
    FailureMode(
        key="device_disconnect",
        component="device_agent",
        failure_reason="DEVICE",
        mission_phase="On-Mission - Rider",
        step_fraction=0.75,
        last_milestone="M5",
        failed_event="Device {device} disconnected from the host during the trip",
        expected_result="The device stays connected to the host for the full trip so screen capture continues.",
        actual_result="The device dropped its host connection mid-trip and screen capture stopped.",
        step_message="Device disconnected from host during the trip",
        warn=("connection to {device} unstable on {host}; 2 reconnect attempts",),
        error=(
            "device {device} disconnected from host {host}; screen capture stopped",
            "no further step updates received for trigger {trigger_id}",
        ),
    ),
    FailureMode(
        key="sim_pose_stall",
        component="sim_bridge",
        failure_reason="SIMULATOR",
        mission_phase="On-Mission - No Rider",
        step_fraction=0.5,
        last_milestone="M3",
        failed_event="Container sim stopped publishing vehicle pose",
        expected_result="The container sim publishes vehicle pose continuously for the whole mission.",
        actual_result="Pose updates stopped and the mission stalled with the vehicle en route.",
        step_message="Container sim stopped publishing vehicle pose",
        warn=("pose updates stale for 6.0s on {vehicle}",),
        error=(
            "container sim {vehicle} stopped publishing pose; mission {mission_id} stalled",
        ),
    ),
    FailureMode(
        key="door_unlock_not_acked",
        component="vehicle_gateway",
        failure_reason="SIMULATOR",
        mission_phase="Pick-Up",
        step_fraction=0.6,
        last_milestone="M4",
        failed_event="Door unlock command was not acknowledged by the vehicle",
        expected_result="The vehicle acknowledges the door unlock command within 15s of the rider tapping unlock.",
        actual_result="The unlock command was sent twice and never acknowledged, so the rider could not board.",
        step_message="Door unlock not acknowledged within 15s",
        warn=("unlock command retried once for {vehicle}",),
        error=("vehicle {vehicle} did not acknowledge door unlock within 15s",),
    ),
    FailureMode(
        key="dropoff_geofence_blocked",
        component="geofence_monitor",
        failure_reason="SIMULATOR",
        mission_phase="Drop-off",
        step_fraction=0.9,
        last_milestone="M5",
        failed_event="Drop-off blocked by an active construction geofence at {dropoff}",
        expected_result="Vehicle completes the drop-off at {dropoff} and the ride is marked complete.",
        actual_result="The drop-off was rejected because it fell inside an active construction geofence and the vehicle rerouted.",
        step_message="Drop-off rejected by an active construction geofence",
        warn=("construction geofence overlaps requested drop-off {dropoff}",),
        error=(
            "drop-off rejected: {dropoff} is inside an active construction geofence",
            "vehicle {vehicle} rerouted; milestone M6 (ArrivedAtDropoff) not recorded",
        ),
    ),
)

HAPPY_TIMELINE = (
    ("M1", "RideRequested", "dispatch", "ride request accepted: {pickup} -> {dropoff}"),
    ("M2", "VehicleAssigned", "dispatch", "vehicle {vehicle} assigned to mission {mission_id}"),
    ("M3", "EnRouteToPickup", "mission_monitor", "vehicle {vehicle} en route to {pickup}"),
    (
        "M4",
        "ArrivedAtPickup",
        "geofence_monitor",
        "vehicle {vehicle} arrived inside the pickup geofence at {pickup}",
    ),
    ("M4", "DoorUnlocked", "vehicle_gateway", "door unlock acknowledged by {vehicle}"),
    ("M5", "TripStarted", "mission_monitor", "trip started, destination {dropoff}"),
    ("M6", "ArrivedAtDropoff", "geofence_monitor", "vehicle {vehicle} arrived at {dropoff}"),
    ("M7", "TripCompleted", "mission_monitor", "trip completed for trigger {trigger_id}"),
)

NOISE_LINES = (
    ("telemetry", "pose update lat {lat:.6f} lng {lng:.6f} speed {speed:.1f} m/s"),
    ("mission_monitor", "eta to {pickup} recalculated to {eta} s"),
    ("rider_app", "ride status polled: {status}"),
    ("webapi_client", "GET /rides/{trigger_id} 200 in {latency} ms"),
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def rand_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def ch_ts(dt: datetime) -> str:
    """The format clickhouse_service.build_search_query filters on."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def zpt_ts(dt: datetime) -> str:
    """The format rides_service.generate_zpt parses: %Y-%m-%dT%H:%M:%S.%fZ."""
    utc = dt.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def jira_datetime_field(dt: datetime) -> str:
    """customfield_21008 as generate_zpt renders it: %d/%b/%y %I:%M %p."""
    return dt.astimezone(timezone.utc).strftime("%d/%b/%y %I:%M %p")


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def milestone_index(code: str) -> int:
    return MILESTONE_ORDER.index(code)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Smallest honest PNG: a solid colour, so the artifact is a real image."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 9))
        + chunk(b"IEND", b"")
    )


def mp4_bytes(note: str) -> bytes:
    """Placeholder MP4 with a valid ftyp box and a text mdat payload."""
    ftyp = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    payload = note.encode("utf-8")
    return ftyp + struct.pack(">I", len(payload) + 8) + b"mdat" + payload


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


@dataclass
class Pudo:
    id: str
    ride_service_id: str
    ride_service: str
    poi: str
    address: str
    pickup_location: bool
    dropoff_location: bool
    latitude: float
    longitude: float
    zone: str


@dataclass
class Reference:
    metadata: dict[str, Any]
    jira_mappings: dict[str, Any]
    environments: list[str]
    ride_services: list[dict[str, Any]]
    devices: list[dict[str, Any]]
    pudos: dict[str, list[Pudo]]

    def pickups(self, ride_service: str) -> list[Pudo]:
        return [p for p in self.pudos[ride_service] if p.pickup_location]

    def dropoffs(self, ride_service: str) -> list[Pudo]:
        return [p for p in self.pudos[ride_service] if p.dropoff_location]


# Used only when app/util/jira_issue_mappings.json is unavailable.
FALLBACK_JIRA_MAPPINGS: dict[str, Any] = {
    "customfield_19362": {
        "name": "Severity",
        "type": "single_select",
        "options": {"S0": {"id": "42416"}, "S1": {"id": "42417"}, "S2": {"id": "42418"}},
    },
    "customfield_24302": {
        "name": "Location",
        "type": "single_select",
        "options": {"San Francisco": {"id": "33562"}, "Las Vegas": {"id": "33553"}},
    },
    "customfield_20317": {
        "name": "Found During",
        "type": "cascading",
        "options": {
            "Testing": {"value": "Testing"},
            "Milestone testing": {"value": "Milestone testing"},
        },
    },
    "customfield_14617": {
        "name": "Vehicle Selector",
        "type": "cascading",
        "options": {"Container Sim": {"id": "55500"}},
    },
    "customfield_19416": {
        "name": "Mission Phase",
        "type": "single_select",
        "options": {
            "On-Mission - No Rider": {"id": "23423"},
            "On-Mission - Rider": {"id": "42117"},
            "Other/Break": {"id": "26009"},
            "Pick-Up": {"id": "42116"},
            "Shutdown": {"id": "42119"},
            "Charging": {"id": "26007"},
            "Drop-off": {"id": "42118"},
            "Hailing": {"id": "42115"},
        },
    },
}


def load_reference(metadata_path: Path, mappings_path: Path) -> Reference:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    jira_mappings = (
        json.loads(mappings_path.read_text(encoding="utf-8"))
        if mappings_path.is_file()
        else FALLBACK_JIRA_MAPPINGS
    )

    block = metadata["test_execution"]
    pudos: dict[str, list[Pudo]] = {}
    for service in block["ride_service"]:
        zones = ZONES_BY_RIDE_SERVICE.get(service["name"], ("ZONE-1", "ZONE-2"))
        entries: list[Pudo] = []
        for index, loc in enumerate(service["pudo_locations"]):
            entries.append(
                Pudo(
                    id=loc["id"],
                    ride_service_id=service["id"],
                    ride_service=service["name"],
                    poi=loc["poi"].strip(),
                    address=loc.get("address", ""),
                    pickup_location=bool(loc.get("pickup_location")),
                    dropoff_location=bool(loc.get("dropoff_location")),
                    latitude=float(loc["latitude"]),
                    longitude=float(loc["longitude"]),
                    zone=zones[index % len(zones)],
                )
            )
        pudos[service["name"]] = entries

    return Reference(
        metadata=metadata,
        jira_mappings=jira_mappings,
        environments=[e["name"] for e in block["environment"]],
        ride_services=block["ride_service"],
        devices=block["device"],
        pudos=pudos,
    )


# ---------------------------------------------------------------------------
# Dataset model
# ---------------------------------------------------------------------------


@dataclass
class Ride:
    id: str  # also the trigger_id
    execution_id: str
    scenario_name: str
    scenario_version: float
    status: str
    environment: str
    ride_service: str
    host_machine: str
    device_id: str
    device_host: str
    vehicle_id: str
    pickup: Pudo
    dropoff: Pudo
    zone: str
    mission_id: str | None
    start_time: datetime | None
    end_time: datetime | None
    steps: list[dict[str, Any]]
    failure: FailureMode | None
    failed_step_id: int | None
    testrail_case_id: int | None = None
    testrail_run_id: int | None = None
    testrail_test_id: int | None = None
    testrail_result_id: int | None = None
    artifacts_uploaded: bool = False
    s3_key: str | None = None
    s3_uri: str | None = None
    jira_key: str | None = None
    log_rows: list[dict[str, Any]] = field(default_factory=list)
    zip_members: list[tuple[str, bytes]] = field(default_factory=list)

    @property
    def is_android(self) -> bool:
        return "Android" in self.device_host

    @property
    def has_artifacts(self) -> bool:
        return self.status in ("Passed", "Failed")

    @property
    def duration_seconds(self) -> int:
        if not (self.start_time and self.end_time):
            return 0
        return int((self.end_time - self.start_time).total_seconds())

    def context(self) -> dict[str, Any]:
        return {
            "pickup": self.pickup.poi,
            "dropoff": self.dropoff.poi,
            "vehicle": self.vehicle_id,
            "trigger_id": self.id,
            "mission_id": self.mission_id or "-",
            "device": self.device_id,
            "host": self.host_machine,
            "env": ENV_SLUGS.get(self.environment, self.environment.lower()),
            "zone": self.zone,
        }


@dataclass
class Execution:
    id: str
    suite_id: str | None
    suite_name: str | None
    scenario_id: str
    scenario_name: str
    scenario_version: float
    status: str
    environment: str
    ride_service: str
    host_machine: str
    device_id: str
    pickup: Pudo
    dropoff: Pudo
    zone: str
    apl: str
    adl: str
    no_of_triggers: int
    created_by: str
    created_at: datetime
    scheduled_at: datetime
    testrail_case_id: int
    testrail_run_id: int
    testrail_test_id: int
    testrail_title: str
    rides: list[Ride] = field(default_factory=list)


@dataclass
class TestRailRun:
    id: int
    name: str
    suite_id: int
    case_ids: list[int]
    created_on: datetime
    is_completed: bool


@dataclass
class Dataset:
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    suites: list[dict[str, Any]] = field(default_factory=list)
    executions: list[Execution] = field(default_factory=list)
    rides: list[Ride] = field(default_factory=list)
    tr_cases: list[dict[str, Any]] = field(default_factory=list)
    tr_runs: list[TestRailRun] = field(default_factory=list)
    tr_tests: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    tr_results: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    tr_attachments: list[dict[str, Any]] = field(default_factory=list)
    jira_issues: list[dict[str, Any]] = field(default_factory=list)


class Counters:
    """Id sequences chosen to look like established systems rather than start at 1."""

    def __init__(self) -> None:
        self.case_id = 12000
        self.run_id = 4200
        self.test_id = 90000
        self.result_id = 700000
        self.attachment_id = 31000
        self.jira_number = 24500
        self.jira_id = 1500000

    def next(self, name: str) -> int:
        value = getattr(self, name) + 1
        setattr(self, name, value)
        return value


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_scenarios(rng: random.Random, count: int, now: datetime) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for index in range(min(count, len(SCENARIO_CATALOG))):
        spec = SCENARIO_CATALOG[index]
        version = rng.choice([1.0, 1.1, 1.2, 2.0])
        created = now - timedelta(days=rng.randint(45, 200), minutes=rng.randint(0, 1440))
        steps = [
            {
                "order": order,
                "action_category": category,
                "action_name": action,
                "type": "UI",
                "operational_code": f"{ENTITY_BY_CATEGORY[category.lower()]}-{order:03d}",
                "notes": None,
            }
            for order, (category, action) in enumerate(spec["steps"], start=1)
        ]
        scenarios.append(
            {
                "id": rand_uuid(rng),
                "name": spec["name"],
                "metadata": {"testrail_id": None, "labels": list(spec["labels"])},
                "test_steps": steps,
                "version": version,
                "version_comment": "Generated by the SOTF mock data generator",
                "status": SCENARIO_STATUS[0] if index % 7 else rng.choice(SCENARIO_STATUS[1:]),
                "created_by": rng.choice(AUTHORS),
                "updated_by": rng.choice(AUTHORS),
                "created_at": iso(created),
                "updated_at": iso(created + timedelta(days=rng.randint(1, 30))),
            }
        )
    return scenarios


def build_suites(
    rng: random.Random, scenarios: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    suites: list[dict[str, Any]] = []
    for spec in SUITE_CATALOG:
        members = rng.sample(scenarios, k=min(len(scenarios), rng.randint(3, 6)))
        created = now - timedelta(days=rng.randint(60, 180))
        suites.append(
            {
                "id": rand_uuid(rng),
                "name": spec["name"],
                "description": spec["description"],
                "scenario_ids": [s["id"] for s in members],
                "created_by": rng.choice(AUTHORS),
                "updated_by": rng.choice(AUTHORS),
                "created_at": iso(created),
                "updated_at": iso(created + timedelta(days=rng.randint(1, 40))),
            }
        )
    return suites


def format_tr_title(
    scenario_name: str,
    version: float,
    environment: str,
    ride_service: str,
    pickup: str,
    dropoff: str,
    host_machine: str | None,
) -> str:
    """Mirrors execution_service._format_tr_title."""
    host = host_machine.strip() if host_machine else "Default"
    ver = f"{version:g}" if version else "1.0"
    return f"{scenario_name}_{ver}_{environment}_{ride_service}_{pickup}_{dropoff}_{host}"


def build_steps(
    rng: random.Random,
    scenario: dict[str, Any],
    failure: FailureMode | None,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], int | None]:
    raw_steps = scenario["test_steps"]
    total = len(raw_steps)
    failed_index = (
        min(total - 1, max(0, round(failure.step_fraction * (total - 1))))
        if failure is not None
        else None
    )

    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if failed_index is None:
            status = "PASSED" if rng.random() > 0.06 else "PASSED_WITH_ISSUES"
            message = None
        elif index < failed_index:
            status, message = "PASSED", None
        elif index == failed_index:
            assert failure is not None
            status = "STUCK" if failure.key == "sim_pose_stall" else "FAILED"
            message = failure.step_message.format(**context)
        else:
            status = "SKIPPED"
            message = "Skipped after the preceding step failed"
        steps.append(
            {
                "step_id": str(index + 1),
                "status": status,
                "message": message,
                "logs": [],
                "action_name": raw["action_name"],
                "action_category": raw["action_category"],
            }
        )
    return steps, (failed_index + 1 if failed_index is not None else None)


def build_log_rows(rng: random.Random, ride: Ride) -> list[dict[str, Any]]:
    if not ride.start_time:
        return []

    context = ride.context()

    def row(
        offset: float,
        severity: str,
        component: str,
        text: str,
        event: str,
        step_id: int | None,
    ) -> dict[str, Any]:
        assert ride.start_time is not None
        return {
            "timestamp": ch_ts(ride.start_time + timedelta(seconds=offset)),
            "message": f"{severity} [{component}] {text}",
            "severity": severity,
            "component": component,
            "event": event,
            "step_id": step_id,
            "trigger_id": ride.id,
            "ride_id": ride.id,
            "execution_id": ride.execution_id,
            "mission_id": ride.mission_id or "",
            "scenario_name": ride.scenario_name,
            "environment": ride.environment,
            "ride_service": ride.ride_service,
            "pudo_zone": ride.zone,
            "pickup_poi": ride.pickup.poi,
            "dropoff_poi": ride.dropoff.poi,
            "host_machine": ride.host_machine,
            "device_id": ride.device_id,
            "vehicle_id": ride.vehicle_id,
            "testrail_run_id": ride.testrail_run_id or 0,
            "testrail_test_id": ride.testrail_test_id or 0,
        }

    duration = max(ride.duration_seconds, 60)
    rows = [
        row(
            0.0,
            "INFO",
            "orchestrator",
            f"trigger {ride.id} started on {ride.host_machine} with device {ride.device_id}",
            "TriggerStarted",
            None,
        )
    ]

    if ride.status == "Invalid":
        rows.append(
            row(
                2.0,
                "ERROR",
                "orchestrator",
                f"ride payload rejected for trigger {ride.id}: scenario steps failed validation",
                "PayloadRejected",
                None,
            )
        )
        return rows

    if ride.failure:
        cut = milestone_index(ride.failure.last_milestone)
    elif ride.status == "Running":
        cut = milestone_index("M3")  # in flight, so no arrival or completion milestones yet
    else:
        cut = len(MILESTONE_ORDER) - 1
    reached = [entry for entry in HAPPY_TIMELINE if milestone_index(entry[0]) <= cut]
    span = duration * (0.6 if cut < len(MILESTONE_ORDER) - 1 else 0.9)

    for position, (milestone, event, component, template) in enumerate(reached, start=1):
        offset = span * position / max(len(reached), 1)
        rows.append(
            row(
                offset,
                "INFO",
                component,
                f"milestone {milestone} ({event}): " + template.format(**context),
                event,
                None,
            )
        )
        for _ in range(rng.randint(0, 2)):
            noise_component, noise_template = rng.choice(NOISE_LINES)
            rows.append(
                row(
                    offset + rng.uniform(0.5, 6.0),
                    "INFO",
                    noise_component,
                    noise_template.format(
                        lat=ride.pickup.latitude + rng.uniform(-0.004, 0.004),
                        lng=ride.pickup.longitude + rng.uniform(-0.004, 0.004),
                        speed=rng.uniform(0.0, 11.0),
                        pickup=ride.pickup.poi,
                        eta=rng.randint(30, 480),
                        status=ride.status,
                        trigger_id=ride.id,
                        latency=rng.randint(40, 900),
                    ),
                    "Telemetry",
                    None,
                )
            )

    if ride.failure:
        for index, template in enumerate(ride.failure.warn):
            rows.append(
                row(
                    span + 4.0 + index * 3.0,
                    "WARN",
                    ride.failure.component,
                    template.format(**context),
                    "FailureWarning",
                    ride.failed_step_id,
                )
            )
        for index, template in enumerate(ride.failure.error):
            rows.append(
                row(
                    span + 12.0 + index * 2.5,
                    "ERROR",
                    ride.failure.component,
                    template.format(**context),
                    "FailureDetected",
                    ride.failed_step_id,
                )
            )
        rows.append(
            row(
                duration,
                "ERROR",
                "orchestrator",
                f"trigger {ride.id} finished with status Failed ({ride.failure.key})",
                "TriggerFinished",
                ride.failed_step_id,
            )
        )
    elif ride.status == "Running":
        rows.append(
            row(duration, "INFO", "orchestrator", f"trigger {ride.id} still running", "TriggerRunning", None)
        )
    else:
        rows.append(
            row(
                duration,
                "INFO",
                "orchestrator",
                f"trigger {ride.id} finished with status Passed",
                "TriggerFinished",
                None,
            )
        )

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def build_zip_members(ride: Ride) -> list[tuple[str, bytes]]:
    """Contents mirror the TEC staging layout that uploader.py zips."""
    members: list[tuple[str, bytes]] = []

    for step in [s for s in ride.steps if s["status"] in ("PASSED", "FAILED", "STUCK")][:4]:
        failed = step["status"] in ("FAILED", "STUCK")
        members.append(
            (
                f"snapshots/step_{step['step_id']}_{step['status'].lower()}.png",
                png_bytes(160, 100, (200, 60, 60) if failed else (60, 140, 90)),
            )
        )

    members.append(
        (
            f"video/{ride.id}.mp4",
            mp4_bytes(f"ride {ride.id} {ride.scenario_name} {ride.status} {ride.duration_seconds}s"),
        )
    )

    stamp = (ride.start_time or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    device_log_name = (
        f"logs/logcat_{ride.id}_{stamp}.log"
        if ride.is_android
        else f"logs/xcodebuild_{ride.id}_{stamp}.log"
    )
    device_lines = [
        f"{r['timestamp']} {r['message']}"
        for r in ride.log_rows
        if r["component"] in ("rider_app", "device_agent", "orchestrator")
    ]
    members.append(
        (device_log_name, ("\n".join(device_lines) or "no device log lines captured").encode("utf-8"))
    )
    members.append(
        (
            f"logs/ctm_{ride.id}.log",
            "\n".join(f"{r['timestamp']} {r['message']}" for r in ride.log_rows).encode("utf-8"),
        )
    )

    summary = {
        "trigger_id": ride.id,
        "execution_id": ride.execution_id,
        "scenario": ride.scenario_name,
        "scenario_version": ride.scenario_version,
        "status": ride.status,
        "environment": ride.environment,
        "ride_service": ride.ride_service,
        "pudo_zone": ride.zone,
        "pickup": {"poi": ride.pickup.poi, "address": ride.pickup.address},
        "dropoff": {"poi": ride.dropoff.poi, "address": ride.dropoff.address},
        "device_id": ride.device_id,
        "host_machine": ride.host_machine,
        "vehicle_id": ride.vehicle_id,
        "duration": hms(ride.duration_seconds),
        "failed_event": ride.failure.failed_event.format(**ride.context()) if ride.failure else None,
        "steps": [
            {"step_id": s["step_id"], "status": s["status"], "message": s["message"]}
            for s in ride.steps
        ],
    }
    members.append(("summary.json", (json.dumps(summary, indent=2) + "\n").encode("utf-8")))
    return members


def ride_summary_payload(ride: Ride) -> dict[str, Any]:
    """Shaped like RideSummaryLiteDto."""
    if not (ride.start_time and ride.end_time):
        return {}

    eta_pu = ride.start_time + timedelta(seconds=int(ride.duration_seconds * 0.35))
    eta_do = ride.start_time + timedelta(seconds=int(ride.duration_seconds * 0.9))
    reached_pickup = ride.failure is None or milestone_index(ride.failure.last_milestone) >= milestone_index("M4")
    completed = ride.failure is None and ride.status == "Passed"
    last = ride.dropoff if completed else ride.pickup

    artifacts = (
        [{"name": f"{ride.id}.zip", "path": ride.s3_uri, "type": "Zip"}] if ride.s3_uri else []
    )

    return {
        "ride_duration": hms(ride.duration_seconds),
        "eta_pu": iso(eta_pu),
        "eta_do": iso(eta_do),
        "ata_pu": iso(eta_pu + timedelta(seconds=12)) if reached_pickup else None,
        "ata_do": iso(ride.end_time) if completed else None,
        "last_location": last.poi,
        "last_location_coords": {"lat": last.latitude, "lng": last.longitude},
        "steps": [
            {
                "step_id": s["step_id"],
                "status": s["status"],
                "message": s["message"],
                "logs": s["logs"],
                "artifacts": [],
            }
            for s in ride.steps
        ],
        "artifacts": artifacts,
        "has_artifacts": bool(artifacts),
        "system": {
            "failure_reason": ride.failure.failure_reason if ride.failure else None,
            "device": {
                "type": "SIMULATOR" if ride.is_android else "PHYSICAL",
                "platform": "ANDROID" if ride.is_android else "IOS",
                "model": "Pixel 7" if ride.is_android else "iPhone 14",
                "os_version": "14" if ride.is_android else "17.4",
            },
        },
    }


def derive_execution_status(rides: Iterable[Ride]) -> str:
    statuses = [r.status for r in rides]
    if not statuses:
        status = "Queued"
    elif all(s == "Queued" for s in statuses):
        status = "Queued"
    elif all(s == "Scheduled" for s in statuses):
        status = "Scheduled"
    elif any(s == "Running" for s in statuses):
        status = "Running"
    elif all(s == "Invalid" for s in statuses):
        status = "Invalid"
    elif all(s == "Passed" for s in statuses):
        status = "Passed"
    elif all(s == "Failed" for s in statuses):
        status = "Failed"
    else:
        status = "Partially Failed"
    assert status in EXECUTION_STATUS
    return status


def make_ride(
    rng: random.Random,
    args: argparse.Namespace,
    scenario: dict[str, Any],
    execution: Execution,
    device: dict[str, Any],
    prefix: str,
    started: datetime,
    now: datetime,
) -> Ride:
    roll = rng.random()
    if roll < 0.04:
        status = "Queued"
    elif roll < 0.07:
        status = "Running"
    elif roll < 0.10:
        status = "Invalid"
    elif roll < 0.12:
        status = "Scheduled"
    else:
        status = "Failed" if rng.random() < args.failure_rate else "Passed"

    failure = rng.choice(FAILURE_MODES) if status == "Failed" else None
    vehicle_id = rng.choice(VEHICLE_SUBTYPES)

    step_context = {
        "pickup": execution.pickup.poi,
        "dropoff": execution.dropoff.poi,
        "device": device["name"],
        "vehicle": vehicle_id,
    }
    if status in ("Queued", "Scheduled"):
        steps: list[dict[str, Any]] = []
        failed_step_id: int | None = None
    else:
        steps, failed_step_id = build_steps(rng, scenario, failure, step_context)
        if status == "Running":
            active = rng.randint(1, max(1, len(steps) - 1))
            for index, step in enumerate(steps):
                if index < active - 1:
                    step["status"] = "PASSED"
                elif index == active - 1:
                    step["status"] = "RUNNING"
                else:
                    step["status"] = "SKIPPED"
                step["message"] = None
        elif status == "Invalid":
            for step in steps:
                step["status"] = "SKIPPED"
                step["message"] = "Ride payload rejected before execution"

    if status in ("Queued", "Scheduled"):
        start_time: datetime | None = None
        end_time: datetime | None = None
    elif status == "Running":
        start_time = min(started, now - timedelta(minutes=2))
        end_time = None
    elif status == "Invalid":
        start_time = started
        end_time = started + timedelta(seconds=rng.randint(5, 30))
    else:
        start_time = started
        span = rng.randint(240, 1500) if failure is None else rng.randint(90, 900)
        end_time = started + timedelta(seconds=span)

    ride = Ride(
        id=rand_uuid(rng),
        execution_id=execution.id,
        scenario_name=execution.scenario_name,
        scenario_version=execution.scenario_version,
        status=status,
        environment=execution.environment,
        ride_service=execution.ride_service,
        host_machine=execution.host_machine,
        device_id=device["name"],
        device_host=device["host_machine"],
        vehicle_id=vehicle_id,
        pickup=execution.pickup,
        dropoff=execution.dropoff,
        zone=execution.zone,
        mission_id=rand_uuid(rng) if status in ("Running", "Passed", "Failed") else None,
        start_time=start_time,
        end_time=end_time,
        steps=steps,
        failure=failure,
        failed_step_id=failed_step_id,
    )

    if status in ("Running", "Passed", "Failed"):
        ride.testrail_case_id = execution.testrail_case_id
        ride.testrail_run_id = execution.testrail_run_id
        ride.testrail_test_id = execution.testrail_test_id

    if ride.has_artifacts:
        ride.s3_key = f"{prefix}/{ride.id}.zip" if prefix else f"{ride.id}.zip"
        ride.s3_uri = f"s3://{args.bucket}/{ride.s3_key}"
        ride.artifacts_uploaded = True

    ride.log_rows = build_log_rows(rng, ride)
    if ride.has_artifacts:
        ride.zip_members = build_zip_members(ride)
    return ride


def attach_jira_issue(
    data: Dataset,
    counters: Counters,
    rng: random.Random,
    ref: Reference,
    ride: Ride,
    known_defect: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Create a ZPT issue, or link the ride to the existing issue for this signature."""
    assert ride.failure is not None
    signature = (ride.failure.key, ride.ride_service)

    existing = known_defect.get(signature)
    if existing is not None and rng.random() < 0.7:
        existing["occurrences"] += 1
        existing["linked_rides"].append(ride.id)
        ride.jira_key = existing["key"]
        return

    mappings = ref.jira_mappings
    context = ride.context()
    failed_event = ride.failure.failed_event.format(**context)
    expected_result = ride.failure.expected_result.format(**context)
    actual_result = ride.failure.actual_result.format(**context)
    occurred_at = ride.end_time or ride.start_time or datetime.now(timezone.utc)

    key = f"{JIRA_PROJECT_KEY}-{counters.next('jira_number')}"
    issue_id = str(counters.next("jira_id"))
    defect_title = f"{failed_event} [{ride.environment}]"
    parent, child = [part.strip() for part in FOUND_DURING.split("-", 1)]

    description = (
        f"Test ID: {ride.testrail_test_id}\n"
        f"Scenario: {ride.scenario_name}\n"
        f"Environment: {ride.environment}\n"
        f"Device: {ride.device_id}\n"
        f"Vehicle Type: {VEHICLE_TYPE}\n"
        f"Vehicle Subtype: {ride.vehicle_id}\n"
        f"Location: {ride.ride_service}\n"
        f"Found During: {FOUND_DURING}\n"
        f"Mission Phase: {ride.failure.mission_phase}\n"
        "Occurrence Count: 1\n"
        f"Failed Event: {failed_event}\n"
        f"Expected Result: {expected_result}\n"
        f"Actual Result: {actual_result}\n"
    )

    details = format_zpt_details(
        test_id=ride.testrail_test_id,
        scenario_name=ride.scenario_name,
        environment=ride.environment,
        device=ride.device_id,
        found_during=FOUND_DURING,
        failed_event=failed_event,
        actual_result=actual_result,
        expected_result=expected_result,
        mission_phase=ride.failure.mission_phase,
        occurrence_counter=1,
        location=ride.ride_service,
        vehicle_type=VEHICLE_TYPE,
        vehicle_subtype=ride.vehicle_id,
        defect_title=defect_title,
        artifact_paths=[ride.s3_uri] if ride.s3_uri else [],
    )

    issue = {
        "id": issue_id,
        "key": key,
        "occurrences": 1,
        "linked_rides": [ride.id],
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY, "name": "Problem Tracker"},
            "issuetype": {"name": JIRA_ISSUE_TYPE},
            "summary": defect_title,
            "description": description,
            "status": {"name": rng.choice(["Open", "In Progress", "Under Review", "Resolved"])},
            "reporter": {
                "emailAddress": "sotf-automation@example.com",
                "displayName": "SOTF Automation",
            },
            "created": iso(occurred_at),
            "updated": iso(occurred_at + timedelta(hours=rng.randint(1, 72))),
            "labels": ["sotf", "automated-report", ride.failure.key],
            "customfield_19362": mappings["customfield_19362"]["options"]["S2"],
            "customfield_24302": mappings["customfield_24302"]["options"][ride.ride_service],
            "customfield_20317": {"value": parent, "child": {"value": child}},
            "customfield_14617": mappings["customfield_14617"]["options"][VEHICLE_TYPE],
            "customfield_32900": ride.vehicle_id,
            "customfield_21008": jira_datetime_field(occurred_at),
            "customfield_19416": (
                mappings["customfield_19416"]["options"]["Other/Break"]
                if parent == "Testing"
                else None
            ),
            "customfield_20316": details,
            "attachment": (
                [{"filename": f"{ride.id}.zip", "content": ride.s3_uri}] if ride.s3_uri else []
            ),
        },
        "sotf_context": {
            "failure_mode": ride.failure.key,
            "trigger_id": ride.id,
            "execution_id": ride.execution_id,
            "testrail_run_id": ride.testrail_run_id,
            "testrail_test_id": ride.testrail_test_id,
            "s3_uri": ride.s3_uri,
            "occurred_at_zpt_format": zpt_ts(occurred_at),
        },
    }

    data.jira_issues.append(issue)
    known_defect[signature] = issue
    ride.jira_key = key


def attach_testrail_result(
    data: Dataset, counters: Counters, ride: Ride, args: argparse.Namespace
) -> None:
    assert ride.testrail_test_id is not None
    result_id = counters.next("result_id")
    ride.testrail_result_id = result_id

    # build_testrail_comment in integration_utils.py
    lines = ["Steps:"]
    for step in ride.steps:
        line = f"- Step {step['step_id']}: {step['status']}"
        if step["message"]:
            line += f" | {step['message']}"
        lines.append(line)
    lines.append(f"\nFor detailed screenshots and videos, refer to {ride.id}")
    if ride.s3_uri:
        lines.append(f"Artifacts link: {args.artifacts_base_url}/{ride.id}.zip")

    data.tr_results.setdefault(ride.testrail_test_id, []).append(
        {
            "id": result_id,
            "test_id": ride.testrail_test_id,
            "status_id": TESTRAIL_STATUS_ID["PASSED" if ride.status == "Passed" else "FAILED"],
            "comment": "\n".join(lines),
            "version": f"{ride.scenario_version:g}",
            "elapsed": f"{ride.duration_seconds}s",
            "defects": ride.jira_key,
            "assignedto_id": None,
            "created_by": 7,
            "created_on": int((ride.end_time or ride.start_time or datetime.now(timezone.utc)).timestamp()),
            "custom_step_results": [
                {
                    "content": f"{step['action_category']}: {step['action_name']}",
                    "expected": "Step should execute successfully.",
                    "actual": step["message"] or step["status"],
                    # StepStatus has no SKIPPED/STUCK member, so those fall back to RETEST.
                    "status_id": TESTRAIL_STATUS_ID.get(step["status"], TESTRAIL_STATUS_ID["RETEST"]),
                }
                for step in ride.steps
            ],
        }
    )

    if ride.s3_uri:
        data.tr_attachments.append(
            {
                "attachment_id": str(counters.next("attachment_id")),
                "result_id": result_id,
                "test_id": ride.testrail_test_id,
                "filename": f"{ride.id}.zip",
                "source_s3_uri": ride.s3_uri,
            }
        )


def format_zpt_details(**kwargs: Any) -> str:
    """Mirrors rides_service._format_zpt_jira_description."""

    def nonempty(*parts: object) -> str:
        text = " ".join(str(p) for p in parts if p is not None and str(p).strip())
        return text if text.strip() else "Not provided in automated test report."

    how_observed = nonempty(
        f"Automated test execution (Test ID: {kwargs['test_id']}).",
        f"Scenario: {kwargs['scenario_name']}.",
        f"Environment: {kwargs['environment']}.",
        f"Device: {kwargs['device']}.",
        f"Found during: {kwargs['found_during']}.",
    )
    what_observed = nonempty(
        f"Failed event: {kwargs['failed_event']}.",
        f"Actual result: {kwargs['actual_result']}.",
    )
    why_issue = nonempty(f"Expected result: {kwargs['expected_result']}.")
    impact = nonempty(
        f"Mission phase: {kwargs['mission_phase']}.",
        f"Occurrence count: {kwargs['occurrence_counter']}.",
        f"Location: {kwargs['location']}.",
    )
    other = nonempty(
        f"Vehicle type: {kwargs['vehicle_type']}.",
        f"Vehicle subtype: {kwargs['vehicle_subtype']}.",
    )
    executive = nonempty(
        kwargs["defect_title"],
        f"Scenario: {kwargs['scenario_name']}.",
        f"Failed event: {kwargs['failed_event']}.",
    )
    attachments = "\n".join(kwargs["artifact_paths"])

    return (
        "-- CONTACTS --\n"
        "(Notable contacts or stakeholders of the issue)\n\n"
        "No contact fields are collected in the automated test report.\n\n"
        "-- ISSUE REPORTING --\n"
        "(Details of the issue at the time of initial observation)\n\n"
        "How was the issue observed? (the procedure that was followed)\n"
        f"{how_observed}\n\n"
        "What is the issue observed? "
        "(the observed behavior; add any and all relevant faults and error messages)\n"
        f"{what_observed}\n\n"
        "Why is this an issue? "
        "(the expected behavior; spec/requirements not being met, not matching CAD, "
        "opportunity for design improvement/DFM, etc.)\n"
        f"{why_issue}\n\n"
        "Impact to operations / test activity: (estimated downtime)\n"
        f"{impact}\n\n"
        "Other relevant info: (time range, intermittent issue, occurring periodically)\n"
        f"{other}\n\n"
        "Mobile App info (if Mobile App was used)\n"
        "Email address or phone number used to sign-in to Mobile App: "
        "Not applicable (no Mobile App data in test report).\n"
        "Version of Mobile App used: "
        "Not applicable (no Mobile App data in test report).\n\n"
        "-- EXECUTIVE SUMMARY --\n"
        "(Summary of the latest status, or what is currently known about the issue)\n\n"
        f"{executive}\n"
        "Error screenshots:\n"
        f"{attachments}\n"
    )


def build_dataset(
    rng: random.Random, args: argparse.Namespace, ref: Reference, now: datetime
) -> Dataset:
    data = Dataset()
    counters = Counters()

    data.scenarios = build_scenarios(rng, args.scenarios, now)
    data.suites = build_suites(rng, data.scenarios, now)
    scenario_by_id = {s["id"]: s for s in data.scenarios}

    case_id_by_key: dict[tuple[int, str], int] = {}
    known_defect: dict[tuple[str, str], dict[str, Any]] = {}
    prefix = args.s3_prefix.replace("${ENV}", args.tec_env).strip("/")

    remaining = args.executions
    while remaining > 0:
        is_suite_run = rng.random() < 0.3 and len(data.scenarios) >= 3
        suite = rng.choice(data.suites) if is_suite_run else None
        tr_suite = TESTRAIL_REGRESSION_SUITE if is_suite_run else TESTRAIL_ADHOC_SUITE

        if suite is not None:
            member_ids = [sid for sid in suite["scenario_ids"] if sid in scenario_by_id]
            picked = rng.sample(member_ids, k=min(len(member_ids), rng.randint(2, 4)))
        else:
            picked = [rng.choice(data.scenarios)["id"]]
        picked = picked[:remaining]
        if not picked:
            break
        remaining -= len(picked)

        environment = rng.choice(ref.environments)
        ride_service = rng.choice([s["name"] for s in ref.ride_services])
        host_machine = rng.choice(HOST_MACHINES)
        device = rng.choice(ref.devices)
        created_by = rng.choice(AUTHORS)
        created_at = now - timedelta(
            days=rng.randint(0, max(args.days - 1, 0)),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )

        run_id = counters.next("run_id")
        run_case_ids: list[int] = []
        group: list[Execution] = []

        for scenario_id in picked:
            scenario = scenario_by_id[scenario_id]
            pickup = rng.choice(ref.pickups(ride_service))
            candidates = [p for p in ref.dropoffs(ride_service) if p.id != pickup.id]
            dropoff = rng.choice(candidates or ref.dropoffs(ride_service))

            title = format_tr_title(
                scenario["name"],
                scenario["version"],
                environment,
                ride_service,
                pickup.poi,
                dropoff.poi,
                host_machine,
            )
            case_key = (tr_suite["id"], title)
            if case_key in case_id_by_key:
                case_id = case_id_by_key[case_key]
            else:
                case_id = counters.next("case_id")
                case_id_by_key[case_key] = case_id
                data.tr_cases.append(
                    {
                        "id": case_id,
                        "title": title,
                        "section_id": tr_suite["section_id"],
                        "suite_id": tr_suite["id"],
                        "template_id": 2,
                        "type_id": 3,
                        "priority_id": 2,
                        "labels": scenario["metadata"]["labels"],
                        "custom_case_is_automated": False,
                        "custom_steps_separated": [
                            {
                                "content": f"{step['action_category']}: {step['action_name']}",
                                "expected": "Step should execute successfully.",
                            }
                            for step in scenario["test_steps"]
                        ],
                        "created_by": 7,
                        "created_on": int(created_at.timestamp()),
                        "updated_on": int(created_at.timestamp()),
                    }
                )
            scenario["metadata"]["testrail_id"] = str(case_id)
            run_case_ids.append(case_id)

            test_id = counters.next("test_id")
            no_of_triggers = rng.randint(1, 3)

            execution = Execution(
                id=rand_uuid(rng),
                suite_id=suite["id"] if suite else None,
                suite_name=suite["name"] if suite else None,
                scenario_id=scenario_id,
                scenario_name=scenario["name"],
                scenario_version=scenario["version"],
                status="Queued",
                environment=environment,
                ride_service=ride_service,
                host_machine=host_machine,
                device_id=device["name"],
                pickup=pickup,
                dropoff=dropoff,
                zone=pickup.zone,
                apl=str(rng.randint(0, 10)),
                adl=str(rng.randint(0, 10)),
                no_of_triggers=no_of_triggers,
                created_by=created_by,
                created_at=created_at,
                scheduled_at=created_at + timedelta(hours=2),
                testrail_case_id=case_id,
                testrail_run_id=run_id,
                testrail_test_id=test_id,
                testrail_title=title,
            )

            for trigger_index in range(no_of_triggers):
                ride = make_ride(
                    rng,
                    args,
                    scenario,
                    execution,
                    device,
                    prefix,
                    created_at + timedelta(minutes=3 + trigger_index * 9),
                    now,
                )
                execution.rides.append(ride)
                data.rides.append(ride)

            execution.status = derive_execution_status(execution.rides)
            group.append(execution)
            data.executions.append(execution)

            data.tr_tests.setdefault(run_id, []).append(
                {
                    "id": test_id,
                    "case_id": case_id,
                    "run_id": run_id,
                    "title": title,
                    "status_id": TESTRAIL_STATUS_ID["UNTESTED"],
                    "template_id": 2,
                    "type_id": 3,
                    "priority_id": 2,
                }
            )

        all_rides = [r for execution in group for r in execution.rides]
        data.tr_runs.append(
            TestRailRun(
                id=run_id,
                name=(suite["name"] if suite else group[0].scenario_name),
                suite_id=tr_suite["id"],
                case_ids=run_case_ids,
                created_on=created_at,
                is_completed=all(r.status in ("Passed", "Failed", "Invalid") for r in all_rides),
            )
        )

        # Jira first: the TestRail result's `defects` field references the ZPT key.
        for ride in all_rides:
            if ride.status != "Failed":
                continue
            if rng.random() < args.defect_rate:
                attach_jira_issue(data, counters, rng, ref, ride, known_defect)
        for ride in all_rides:
            if ride.status in ("Passed", "Failed"):
                attach_testrail_result(data, counters, ride, args)

        for test in data.tr_tests.get(run_id, []):
            results = data.tr_results.get(test["id"], [])
            if results:
                test["status_id"] = results[-1]["status_id"]

    return data


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def emit_domain(out: Path, data: Dataset, ref: Reference) -> None:
    write_json(out / "domain/scenarios.json", data.scenarios)
    write_json(out / "domain/suites.json", data.suites)
    write_json(
        out / "domain/executions.json",
        [
            {
                "id": e.id,
                "suite_id": e.suite_id,
                "suite_name": e.suite_name,
                "scenario_id": e.scenario_id,
                "scenario_name": e.scenario_name,
                "scenario_version": e.scenario_version,
                "status": e.status,
                "no_of_triggers": e.no_of_triggers,
                "created_by": e.created_by,
                "created_at": iso(e.created_at),
                "scheduled_at": iso(e.scheduled_at),
                "payload": {
                    "environment": e.environment,
                    "host_machine": e.host_machine,
                    "ride_service": e.ride_service,
                    "pudo_zone": e.zone,
                    "num_rides": e.no_of_triggers,
                    "apl": e.apl,
                    "adl": e.adl,
                    "ppl": f"{e.pickup.poi}, {e.pickup.address}",
                    "pdl": f"{e.dropoff.poi}, {e.dropoff.address}",
                    "ppl_coordinates": {"lat": e.pickup.latitude, "lng": e.pickup.longitude},
                    "pdl_coordinates": {"lat": e.dropoff.latitude, "lng": e.dropoff.longitude},
                    "testrail_case_id": str(e.testrail_case_id),
                    "testrail_test_id": str(e.testrail_test_id),
                    "webapi_domain": f"https://webapi.{ENV_SLUGS.get(e.environment, 'dev')}.internal",
                },
                "testrail": {
                    "case_id": e.testrail_case_id,
                    "run_id": e.testrail_run_id,
                    "test_id": e.testrail_test_id,
                    "title": e.testrail_title,
                },
            }
            for e in data.executions
        ],
    )
    write_json(
        out / "domain/rides.json",
        [
            {
                "id": r.id,
                "execution_id": r.execution_id,
                "status": r.status,
                "device_id": r.device_id,
                "host_machine": r.host_machine,
                "reattempts_count": 1,
                "mission_id": r.mission_id,
                "start_time": iso(r.start_time) if r.start_time else None,
                "end_time": iso(r.end_time) if r.end_time else None,
                "created_at": iso(r.start_time) if r.start_time else None,
                "updated_at": iso(r.end_time or r.start_time) if r.start_time else None,
                "metadata": {
                    "actual_pick_up": r.pickup.poi,
                    "actual_drop_off": r.dropoff.poi,
                    "actual_request_time": iso(r.start_time) if r.start_time else None,
                    "vehicle_id": r.vehicle_id,
                    "test_id": r.testrail_test_id,
                    "testrail": {
                        "run_id": r.testrail_run_id,
                        "test_id": r.testrail_test_id,
                        "result_id": r.testrail_result_id,
                        "artifacts_uploaded": r.artifacts_uploaded,
                    },
                },
                "summary": ride_summary_payload(r),
            }
            for r in data.rides
        ],
    )
    write_json(out / "domain/metadata.json", ref.metadata)
    write_json(
        out / "domain/pudo_locations.json",
        [
            {
                "id": p.id,
                "ride_service_id": p.ride_service_id,
                "ride_service": p.ride_service,
                "poi": p.poi,
                "address": p.address,
                "pickup_location": p.pickup_location,
                "dropoff_location": p.dropoff_location,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "pudo_zone": p.zone,
            }
            for entries in ref.pudos.values()
            for p in entries
        ],
    )


def emit_testrail(out: Path, data: Dataset, args: argparse.Namespace) -> None:
    root = out / "testrail"
    project_id = args.testrail_project_id

    write_json(
        root / "suites.json",
        [
            {
                "id": suite["id"],
                "name": suite["name"],
                "description": suite["description"],
                "project_id": project_id,
                "is_master": False,
                "is_completed": False,
            }
            for suite in (TESTRAIL_REGRESSION_SUITE, TESTRAIL_ADHOC_SUITE)
        ],
    )
    write_json(
        root / "sections.json",
        [
            {
                "id": suite["section_id"],
                "suite_id": suite["id"],
                "name": TESTRAIL_SECTION_NAME,
                "parent_id": None,
                "depth": 0,
            }
            for suite in (TESTRAIL_REGRESSION_SUITE, TESTRAIL_ADHOC_SUITE)
        ],
    )
    write_json(root / "cases.json", data.tr_cases)
    write_json(
        root / "runs.json",
        [
            {
                "id": run.id,
                "suite_id": run.suite_id,
                "project_id": project_id,
                "name": run.name,
                "include_all": False,
                "case_ids": run.case_ids,
                "is_completed": run.is_completed,
                "created_on": int(run.created_on.timestamp()),
                "created_by": 7,
                "url": f"{args.testrail_base_url}/index.php?/runs/view/{run.id}",
            }
            for run in data.tr_runs
        ],
    )

    for run_id, tests in data.tr_tests.items():
        write_json(
            root / f"tests_by_run/{run_id}.json",
            {"offset": 0, "limit": 250, "size": len(tests), "tests": tests},
        )
    for test_id, results in data.tr_results.items():
        write_json(
            root / f"results_by_test/{test_id}.json",
            {"offset": 0, "limit": 250, "size": len(results), "results": results},
        )
    write_json(root / "attachments.json", data.tr_attachments)
    write_json(
        root / "api_index.json",
        {
            "base_url": args.testrail_base_url,
            "project_id": project_id,
            "endpoints": {
                f"GET index.php?/api/v2/get_suites/{project_id}": "suites.json",
                f"GET index.php?/api/v2/get_sections/{project_id}": "sections.json",
                f"GET index.php?/api/v2/get_cases/{project_id}": "cases.json",
                f"GET index.php?/api/v2/get_runs/{project_id}": "runs.json",
                "GET index.php?/api/v2/get_tests/{run_id}": "tests_by_run/{run_id}.json",
                "GET index.php?/api/v2/get_results/{test_id}": "results_by_test/{test_id}.json",
            },
            "note": "Read-only fixtures. add_result / update_case payloads are deliberately absent.",
        },
    )


def emit_jira(out: Path, data: Dataset, ref: Reference, args: argparse.Namespace) -> None:
    root = out / "jira"

    for issue in data.jira_issues:
        fields = dict(issue["fields"])
        fields["description"] = fields["description"].replace(
            "Occurrence Count: 1", f"Occurrence Count: {issue['occurrences']}"
        )
        context = dict(issue["sotf_context"])
        context["occurrences"] = issue["occurrences"]
        context["linked_rides"] = issue["linked_rides"]
        write_json(
            root / f"issues/{issue['key']}.json",
            {
                "id": issue["id"],
                "key": issue["key"],
                "self": f"{args.jira_base_url}/rest/api/2/issue/{issue['id']}",
                "fields": fields,
                "sotf_context": context,
            },
        )

    write_json(
        root / "search_issues.json",
        {
            "startAt": 0,
            "maxResults": len(data.jira_issues),
            "total": len(data.jira_issues),
            "jql": f'project = {JIRA_PROJECT_KEY} AND labels = "sotf" ORDER BY created DESC',
            "issues": [
                {
                    "id": issue["id"],
                    "key": issue["key"],
                    "fields": {
                        "summary": issue["fields"]["summary"],
                        "status": issue["fields"]["status"],
                        "created": issue["fields"]["created"],
                        "updated": issue["fields"]["updated"],
                        "labels": issue["fields"]["labels"],
                        "customfield_24302": issue["fields"]["customfield_24302"],
                    },
                    "occurrences": issue["occurrences"],
                    "linked_rides": issue["linked_rides"],
                }
                for issue in data.jira_issues
            ],
        },
    )

    write_json(
        root / f"createmeta_{JIRA_PROJECT_KEY}_{JIRA_ISSUE_TYPE}.json",
        {
            "maxResults": 50,
            "startAt": 0,
            "fields": [
                {"fieldId": "summary", "name": "Summary", "required": True, "schema": {"type": "string"}},
                {
                    "fieldId": "description",
                    "name": "Description",
                    "required": False,
                    "schema": {"type": "string"},
                },
            ]
            + [
                {
                    "fieldId": field_id,
                    "name": spec.get("name", field_id),
                    "required": False,
                    "schema": {"type": spec.get("type", "string")},
                    "allowedValues": [
                        {"id": option.get("id"), "value": label}
                        for label, option in spec.get("options", {}).items()
                    ],
                }
                for field_id, spec in ref.jira_mappings.items()
            ],
        },
    )
    write_json(
        root / "api_index.json",
        {
            "base_url": args.jira_base_url,
            "project_key": JIRA_PROJECT_KEY,
            "issue_type": JIRA_ISSUE_TYPE,
            "endpoints": {
                "GET /rest/api/2/issue/{issue_key}": "issues/{issue_key}.json",
                f"GET /rest/api/2/issue/createmeta/{JIRA_PROJECT_KEY}/issuetypes/{JIRA_ISSUE_TYPE}": f"createmeta_{JIRA_PROJECT_KEY}_{JIRA_ISSUE_TYPE}.json",
                "GET /rest/api/2/search?jql=...": "search_issues.json",
            },
            "note": "Read-only fixtures. No create-issue payloads are emitted.",
        },
    )


def emit_s3(out: Path, data: Dataset, args: argparse.Namespace) -> None:
    root = out / "s3" / args.bucket
    objects: list[dict[str, Any]] = []

    for ride in data.rides:
        if not (ride.s3_key and ride.zip_members):
            continue
        target = root / ride.s3_key
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in ride.zip_members:
                archive.writestr(name, payload)

        objects.append(
            {
                "bucket": args.bucket,
                "key": ride.s3_key,
                "uri": ride.s3_uri,
                "size": target.stat().st_size,
                "last_modified": iso(ride.end_time or ride.start_time or datetime.now(timezone.utc)),
                "trigger_id": ride.id,
                "contents": [name for name, _ in ride.zip_members],
            }
        )

        write_json(
            out / f"s3/manifests/{ride.id}.json",
            {
                "trigger_id": ride.id,
                "message_type": "Artifacts",
                "artifacts": [{"name": f"{ride.id}.zip", "path": ride.s3_uri, "type": "Zip"}],
                "steps": [
                    {
                        "step_id": step["step_id"],
                        "artifacts": [
                            {
                                "name": f"step_{step['step_id']}_{step['status'].lower()}.png",
                                "path": ride.s3_uri,
                                "type": "Screenshot",
                            }
                        ],
                    }
                    for step in ride.steps
                    if step["status"] in ("PASSED", "FAILED", "STUCK")
                ][:4],
            },
        )

    write_json(out / "s3/objects.json", objects)
    write_json(
        out / "s3/api_index.json",
        {
            "bucket": args.bucket,
            "prefix": args.s3_prefix.replace("${ENV}", args.tec_env),
            "key_format": "<prefix>/<trigger_id>.zip",
            "local_root": f"s3/{args.bucket}/",
            "manifests": "s3/manifests/<trigger_id>.json is the SQS Artifacts message TEC publishes",
            "note": "One zip per trigger, matching device_service uploader._s3_key_for_zip.",
        },
    )


CLICKHOUSE_DDL = """-- CTM log table for the SOTF mock dataset.
-- `timestamp` and `message` are the two columns ctm_logs_service assumes
-- (clickhouse_service.build_search_query defaults). The rest give an assistant
-- something to filter on without parsing the message text.

CREATE DATABASE IF NOT EXISTS {database};

CREATE TABLE IF NOT EXISTS {database}.{table}
(
    `timestamp`        DateTime,
    `message`          String,
    `severity`         LowCardinality(String),
    `component`        LowCardinality(String),
    `event`            LowCardinality(String),
    `step_id`          Nullable(UInt16),
    `trigger_id`       String,
    `ride_id`          String,
    `execution_id`     String,
    `mission_id`       String,
    `scenario_name`    String,
    `environment`      LowCardinality(String),
    `ride_service`     LowCardinality(String),
    `pudo_zone`        LowCardinality(String),
    `pickup_poi`       String,
    `dropoff_poi`      String,
    `host_machine`     LowCardinality(String),
    `device_id`        LowCardinality(String),
    `vehicle_id`       LowCardinality(String),
    `testrail_run_id`  UInt32,
    `testrail_test_id` UInt32
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(`timestamp`)
ORDER BY (`timestamp`, `trigger_id`);
"""

CLICKHOUSE_SAMPLE_QUERIES = """-- Queries a read-only SOTF assistant would run against the mock log table.

-- 1. Everything for one trigger, newest first: the failure-analysis entry point.
SELECT * FROM {database}.{table}
WHERE trigger_id = '<trigger_id>'
ORDER BY timestamp DESC
FORMAT JSONEachRow;

-- 2. Errors in a bounded window. A tool wrapper should accept an id and a window
--    like this instead of free-form SQL.
SELECT timestamp, component, message
FROM {database}.{table}
WHERE severity = 'ERROR'
  AND timestamp BETWEEN '2026-07-01 00:00:00' AND '2026-07-31 23:59:59'
ORDER BY timestamp DESC
LIMIT 200;

-- 3. Which components fail most often, per environment.
SELECT environment, component, count() AS errors
FROM {database}.{table}
WHERE severity = 'ERROR'
GROUP BY environment, component
ORDER BY errors DESC;

-- 4. Failure volume by PUDO zone, for "failures at PUDO Y last week" questions.
SELECT pudo_zone, pickup_poi, count() AS errors
FROM {database}.{table}
WHERE severity = 'ERROR' AND timestamp >= now() - INTERVAL 7 DAY
GROUP BY pudo_zone, pickup_poi
ORDER BY errors DESC;

-- 5. The ILIKE search the service issues today.
SELECT * FROM {database}.{table}
WHERE message ILIKE '%geofence%'
ORDER BY timestamp DESC
FORMAT JSONEachRow;
"""


def emit_clickhouse(out: Path, data: Dataset, args: argparse.Namespace) -> int:
    root = out / "clickhouse"
    root.mkdir(parents=True, exist_ok=True)

    rows = [row for ride in data.rides for row in ride.log_rows]
    rows.sort(key=lambda r: (r["timestamp"], r["trigger_id"]))

    with (root / "logs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    (root / "schema.sql").write_text(
        CLICKHOUSE_DDL.format(database=args.clickhouse_database, table=args.clickhouse_table),
        encoding="utf-8",
    )
    (root / "sample_queries.sql").write_text(
        CLICKHOUSE_SAMPLE_QUERIES.format(
            database=args.clickhouse_database, table=args.clickhouse_table
        ),
        encoding="utf-8",
    )
    write_json(
        root / "api_index.json",
        {
            "database": args.clickhouse_database,
            "table": args.clickhouse_table,
            "row_count": len(rows),
            "format": "JSONEachRow",
            "columns_assumed_by_service": ["timestamp", "message"],
            "severities": ["INFO", "WARN", "ERROR"],
        },
    )
    return len(rows)


def emit_cross_reference(out: Path, data: Dataset) -> None:
    write_json(
        out / "cross_reference.json",
        [
            {
                "trigger_id": r.id,
                "ride_status": r.status,
                "execution_id": r.execution_id,
                "scenario": r.scenario_name,
                "scenario_version": r.scenario_version,
                "environment": r.environment,
                "ride_service": r.ride_service,
                "pudo_zone": r.zone,
                "pickup_poi": r.pickup.poi,
                "dropoff_poi": r.dropoff.poi,
                "host_machine": r.host_machine,
                "device_id": r.device_id,
                "vehicle_id": r.vehicle_id,
                "started_at": iso(r.start_time) if r.start_time else None,
                "ended_at": iso(r.end_time) if r.end_time else None,
                "failure_mode": r.failure.key if r.failure else None,
                "failed_step_id": r.failed_step_id,
                "testrail": {
                    "case_id": r.testrail_case_id,
                    "run_id": r.testrail_run_id,
                    "test_id": r.testrail_test_id,
                    "result_id": r.testrail_result_id,
                },
                "s3": {"key": r.s3_key, "uri": r.s3_uri},
                "jira_key": r.jira_key,
                "clickhouse_log_rows": len(r.log_rows),
            }
            for r in data.rides
        ],
    )


LOADER_TEMPLATE = """#!/usr/bin/env bash
# Load the generated mock dataset into a throwaway S3 bucket and ClickHouse.
# Point these at local stand-ins (MinIO, a scratch ClickHouse) -- never at a real
# SOTF environment.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

: "${{S3_BUCKET_NAME:={bucket}}}"
: "${{CLICKHOUSE_URL:=http://localhost:8123}}"
: "${{CLICKHOUSE_DATABASE:={database}}}"
: "${{CLICKHOUSE_TABLE:={table}}}"
: "${{CLICKHOUSE_USER:=default}}"
: "${{CLICKHOUSE_PASSWORD:=}}"

# ClickHouse Cloud requires auth; a bare curl (no headers) 401s. Local/plain
# instances generally ignore these headers, so it's safe to always send them.
CH_HEADERS=(-H "X-ClickHouse-User: ${{CLICKHOUSE_USER}}" -H "X-ClickHouse-Database: ${{CLICKHOUSE_DATABASE}}")
if [[ -n "${{CLICKHOUSE_PASSWORD}}" ]]; then
  CH_HEADERS+=(-H "X-ClickHouse-Key: ${{CLICKHOUSE_PASSWORD}}")
fi

echo "==> uploading artifact bundles to s3://${{S3_BUCKET_NAME}}"
aws s3 sync "${{HERE}}/s3/{bucket}/" "s3://${{S3_BUCKET_NAME}}/" --no-progress

echo "==> creating ClickHouse table ${{CLICKHOUSE_DATABASE}}.${{CLICKHOUSE_TABLE}}"
# The HTTP interface takes one statement per request, so split schema.sql on ';'.
while IFS= read -r -d ';' stmt || [[ -n "${{stmt//[[:space:]]/}}" ]]; do
  if [[ -n "${{stmt//[[:space:]]/}}" ]]; then
    curl -sS --fail "${{CH_HEADERS[@]}}" --data-binary "${{stmt}};" "${{CLICKHOUSE_URL}}/" > /dev/null
  fi
done < "${{HERE}}/clickhouse/schema.sql"

echo "==> inserting log rows"
curl -sS --fail "${{CH_HEADERS[@]}}" \\
  --data-binary @"${{HERE}}/clickhouse/logs.jsonl" \\
  "${{CLICKHOUSE_URL}}/?query=INSERT%20INTO%20${{CLICKHOUSE_DATABASE}}.${{CLICKHOUSE_TABLE}}%20FORMAT%20JSONEachRow" > /dev/null

echo -n "==> row count: "
curl -sS "${{CH_HEADERS[@]}}" "${{CLICKHOUSE_URL}}/?query=SELECT%20count()%20FROM%20${{CLICKHOUSE_DATABASE}}.${{CLICKHOUSE_TABLE}}"

echo "Jira and TestRail fixtures are static JSON under jira/ and testrail/."
echo "Serve them with any mock HTTP server; see each api_index.json for the route map."
"""


def emit_loader(out: Path, args: argparse.Namespace) -> None:
    path = out / "load_mocks.sh"
    path.write_text(
        LOADER_TEMPLATE.format(
            bucket=args.bucket,
            database=args.clickhouse_database,
            table=args.clickhouse_table,
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------


def run_self_check(out: Path, data: Dataset, args: argparse.Namespace) -> list[str]:
    """Verify the emitted views actually join up. Returns a list of problems."""
    problems: list[str] = []

    cases = {c["id"] for c in json.loads((out / "testrail/cases.json").read_text())}
    runs = {r["id"] for r in json.loads((out / "testrail/runs.json").read_text())}

    for ride in data.rides:
        if ride.testrail_case_id and ride.testrail_case_id not in cases:
            problems.append(f"{ride.id}: TestRail case {ride.testrail_case_id} missing from cases.json")
        if ride.testrail_run_id and ride.testrail_run_id not in runs:
            problems.append(f"{ride.id}: TestRail run {ride.testrail_run_id} missing from runs.json")

        if ride.testrail_test_id:
            tests_file = out / f"testrail/tests_by_run/{ride.testrail_run_id}.json"
            if not tests_file.is_file():
                problems.append(f"{ride.id}: missing {tests_file.name}")
            else:
                ids = {t["id"] for t in json.loads(tests_file.read_text())["tests"]}
                if ride.testrail_test_id not in ids:
                    problems.append(f"{ride.id}: test {ride.testrail_test_id} not in run {ride.testrail_run_id}")

        if ride.testrail_result_id:
            results_file = out / f"testrail/results_by_test/{ride.testrail_test_id}.json"
            if not results_file.is_file():
                problems.append(f"{ride.id}: missing {results_file.name}")
            else:
                ids = {r["id"] for r in json.loads(results_file.read_text())["results"]}
                if ride.testrail_result_id not in ids:
                    problems.append(f"{ride.id}: result {ride.testrail_result_id} not recorded")

        if ride.s3_key:
            zip_path = out / "s3" / args.bucket / ride.s3_key
            if not zip_path.is_file():
                problems.append(f"{ride.id}: artifact zip missing at {ride.s3_key}")
            else:
                try:
                    with zipfile.ZipFile(zip_path) as archive:
                        if archive.testzip() is not None:
                            problems.append(f"{ride.id}: corrupt zip at {ride.s3_key}")
                except zipfile.BadZipFile:
                    problems.append(f"{ride.id}: unreadable zip at {ride.s3_key}")
            if not (out / f"s3/manifests/{ride.id}.json").is_file():
                problems.append(f"{ride.id}: artifact manifest missing")

        if ride.jira_key and not (out / f"jira/issues/{ride.jira_key}.json").is_file():
            problems.append(f"{ride.id}: Jira issue {ride.jira_key} missing")

        if ride.status in ("Passed", "Failed") and not ride.log_rows:
            problems.append(f"{ride.id}: completed ride has no CTM log rows")

    expected_rows = sum(len(r.log_rows) for r in data.rides)
    actual_rows = sum(1 for _ in (out / "clickhouse/logs.jsonl").open(encoding="utf-8"))
    if expected_rows != actual_rows:
        problems.append(f"log row count mismatch: expected {expected_rows}, wrote {actual_rows}")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-referenced mock S3, ClickHouse, Jira and TestRail data for SOTF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    parser.add_argument("--now", default=None, help="ISO timestamp used as 'now' for reproducible runs")
    parser.add_argument("--days", type=int, default=30, help="how far back executions are spread")
    parser.add_argument("--scenarios", type=int, default=len(SCENARIO_CATALOG))
    parser.add_argument("--executions", type=int, default=60, help="number of test_executions rows")
    parser.add_argument("--failure-rate", type=float, default=0.35, help="share of completed rides that fail")
    parser.add_argument("--defect-rate", type=float, default=0.6, help="share of failed rides that get a ZPT")
    parser.add_argument("--bucket", default="zoox-test-artifacts", help="test-mgmt S3_BUCKET_NAME default")
    parser.add_argument("--s3-prefix", default="tec/${ENV}/artifacts", help="TEC S3_ARTIFACTS_PREFIX")
    parser.add_argument("--tec-env", default="dev", help="value substituted for ${ENV} in the prefix")
    parser.add_argument(
        "--artifacts-base-url",
        default="http://s3.amazonaws.com/tec-artifacts-s3-bucket/tec-artifacts",
        help="TEC_ARTIFACTS_BASE_URL used in TestRail comments",
    )
    parser.add_argument("--testrail-project-id", type=int, default=42)
    parser.add_argument("--testrail-base-url", default="https://testrail.example.com")
    parser.add_argument("--jira-base-url", default="https://jira.example.com")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--jira-mappings", type=Path, default=DEFAULT_JIRA_MAPPINGS)
    parser.add_argument("--clickhouse-database", default="default")
    parser.add_argument("--clickhouse-table", default="logs")
    parser.add_argument("--skip-self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.metadata.is_file():
        raise SystemExit(
            f"metadata file not found: {args.metadata}\n"
            "Pass --metadata to point at a SOTF test-mgmt metadata.json."
        )

    now = (
        datetime.fromisoformat(args.now).astimezone(timezone.utc)
        if args.now
        else datetime.now(timezone.utc).replace(microsecond=0)
    )

    rng = random.Random(args.seed)
    ref = load_reference(args.metadata, args.jira_mappings)

    out: Path = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    data = build_dataset(rng, args, ref, now)

    emit_domain(out, data, ref)
    emit_testrail(out, data, args)
    emit_jira(out, data, ref, args)
    emit_s3(out, data, args)
    log_rows = emit_clickhouse(out, data, args)
    emit_cross_reference(out, data)
    emit_loader(out, args)

    failed = [r for r in data.rides if r.status == "Failed"]
    counts = {
        "scenarios": len(data.scenarios),
        "suites": len(data.suites),
        "executions": len(data.executions),
        "rides": len(data.rides),
        "rides_by_status": {s: sum(1 for r in data.rides if r.status == s) for s in RIDE_STATUS},
        "failure_modes": {
            mode.key: sum(1 for r in failed if r.failure and r.failure.key == mode.key)
            for mode in FAILURE_MODES
        },
        "testrail_cases": len(data.tr_cases),
        "testrail_runs": len(data.tr_runs),
        "testrail_tests": sum(len(v) for v in data.tr_tests.values()),
        "testrail_results": sum(len(v) for v in data.tr_results.values()),
        "testrail_attachments": len(data.tr_attachments),
        "jira_issues": len(data.jira_issues),
        "s3_objects": sum(1 for r in data.rides if r.s3_key),
        "clickhouse_rows": log_rows,
    }
    write_json(
        out / "summary.json",
        {
            "generated_at": iso(datetime.now(timezone.utc)),
            "parameters": {
                "seed": args.seed,
                "now": iso(now),
                "days": args.days,
                "executions_requested": args.executions,
                "failure_rate": args.failure_rate,
                "defect_rate": args.defect_rate,
                "bucket": args.bucket,
                "s3_prefix": args.s3_prefix.replace("${ENV}", args.tec_env),
                "clickhouse": f"{args.clickhouse_database}.{args.clickhouse_table}",
                "testrail_project_id": args.testrail_project_id,
            },
            "counts": counts,
        },
    )

    print(f"Wrote mock dataset to {out}")
    print(
        "  {executions} executions, {rides} rides, {jira_issues} ZPT issues, "
        "{testrail_results} TestRail results, {s3_objects} artifact zips, "
        "{clickhouse_rows} log rows".format(**counts)
    )

    if args.skip_self_check:
        return 0

    problems = run_self_check(out, data, args)
    if problems:
        print(f"  self-check FAILED with {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"    - {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        return 1

    print("  self-check passed: every ride joins across domain, TestRail, S3, ClickHouse and Jira")
    print(f"  join table: {out / 'cross_reference.json'}")
    print(f"  loader:     {out / 'load_mocks.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
