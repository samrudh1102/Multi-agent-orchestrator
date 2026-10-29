"""Local FastMCP server exposing TestRail as MCP tools.

There is no official TestRail MCP server, so this is a self-hosted one built on
FastMCP. It speaks the MCP protocol over **stdio**: the chatbot's MCP client
(app/langgraph_agent/fastmcp_client.py) launches it as a subprocess and passes
credentials via the TESTRAIL_URL / TESTRAIL_USERNAME / TESTRAIL_API_KEY
environment variables. Tools call into `TestRailService`, which wraps the async
TestRail REST client.

Run standalone for a quick smoke test:

    TESTRAIL_URL=... TESTRAIL_USERNAME=... TESTRAIL_API_KEY=... \
        python app/testrail/mcp_server.py

It can also be launched over any other transport FastMCP supports, but the app
wires it up over stdio.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

# When launched as a script (python app/testrail/mcp_server.py) the project root
# is not on sys.path, so `import app.*` would fail. Add it before app imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Best-effort: use the OS trust store so HTTPS to a corporate TestRail with an
# internal CA works without extra config. Harmless if unavailable.
try:  # pragma: no cover - environment dependent
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

from app.testrail.exceptions import TestRailError  # noqa: E402
from app.testrail.model import (  # noqa: E402
    AddCasePayload,
    AddMilestonePayload,
    AddResultPayload,
    AddRunPayload,
    AddSectionPayload,
    TestRailConfig,
    UpdateCasePayload,
)
from app.testrail.service import TestRailService  # noqa: E402

mcp = FastMCP(
    name="testrail",
    instructions=(
        "Tools for a TestRail instance: browse projects, suites, sections, "
        "cases, runs, tests and results, and create cases/runs/results/"
        "milestones/sections. IDs are numeric TestRail ids. When unsure of a "
        "status_id, priority_id or type_id, call the corresponding reference "
        "tool (get_statuses / get_priorities / get_case_types) first."
    ),
)

_service: TestRailService | None = None


async def _svc() -> TestRailService:
    """Return the shared, connected service (created lazily on first use)."""
    global _service
    if _service is None:
        _service = TestRailService(TestRailConfig.from_env())
    await _service.connect()  # idempotent
    return _service


def _guard(fn):
    """Translate TestRail client errors into MCP ToolErrors so the message
    reaches the model instead of a generic 'error executing tool'."""

    @functools.wraps(fn)  # preserves signature/annotations for FastMCP schema
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except TestRailError as exc:
            raise ToolError(f"TestRail error: {exc}") from exc

    return wrapper


# ----------------------------------------------------------------------
# Projects
# ----------------------------------------------------------------------


@mcp.tool
@_guard
async def get_projects() -> list[dict]:
    """List all TestRail projects (id, name, is_completed, etc.)."""
    return await (await _svc()).list_projects()


@mcp.tool
@_guard
async def get_project(project_id: int) -> dict:
    """Get a single TestRail project by id."""
    return await (await _svc()).get_project(project_id)


# ----------------------------------------------------------------------
# Suites / Sections
# ----------------------------------------------------------------------


@mcp.tool
@_guard
async def get_suites(project_id: int) -> list[dict]:
    """List the test suites in a project."""
    return await (await _svc()).list_suites(project_id)


@mcp.tool
@_guard
async def get_suite(suite_id: int) -> dict:
    """Get a single test suite by id."""
    return await (await _svc()).get_suite(suite_id)


@mcp.tool
@_guard
async def get_sections(project_id: int, suite_id: int | None = None) -> list[dict]:
    """List sections in a project, optionally scoped to a suite."""
    return await (await _svc()).list_sections(project_id, suite_id)


@mcp.tool
@_guard
async def add_section(
    project_id: int,
    name: str,
    suite_id: int | None = None,
    description: str | None = None,
    parent_id: int | None = None,
) -> dict:
    """Create a new section in a project. `suite_id` is required for
    multi-suite projects; `parent_id` nests the section under another."""
    payload = AddSectionPayload(
        name=name,
        suite_id=suite_id,
        description=description,
        parent_id=parent_id,
    )
    return await (await _svc()).add_section(project_id, payload)


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------


@mcp.tool
@_guard
async def get_cases(
    project_id: int,
    suite_id: int | None = None,
    section_id: int | None = None,
) -> list[dict]:
    """List test cases in a project, optionally scoped to a suite/section."""
    return await (await _svc()).list_cases(project_id, suite_id, section_id)


@mcp.tool
@_guard
async def get_case(case_id: int) -> dict:
    """Get a single test case by id."""
    return await (await _svc()).get_case(case_id)


@mcp.tool
@_guard
async def add_case(
    section_id: int,
    title: str,
    type_id: int | None = None,
    priority_id: int | None = None,
    estimate: str | None = None,
    milestone_id: int | None = None,
    refs: str | None = None,
    template_id: int | None = None,
    custom_steps: str | None = None,
    custom_expected: str | None = None,
) -> dict:
    """Create a test case in a section. `title` is required; the rest are
    optional. `refs` is comma-separated requirement references."""
    payload = AddCasePayload(
        title=title,
        type_id=type_id,
        priority_id=priority_id,
        estimate=estimate,
        milestone_id=milestone_id,
        refs=refs,
        template_id=template_id,
        custom_steps=custom_steps,
        custom_expected=custom_expected,
    )
    return await (await _svc()).add_case(section_id, payload)


@mcp.tool
@_guard
async def update_case(
    case_id: int,
    title: str | None = None,
    type_id: int | None = None,
    priority_id: int | None = None,
    estimate: str | None = None,
    milestone_id: int | None = None,
    refs: str | None = None,
    custom_steps: str | None = None,
    custom_expected: str | None = None,
) -> dict:
    """Update an existing test case. Only the provided fields are changed."""
    payload = UpdateCasePayload(
        title=title,
        type_id=type_id,
        priority_id=priority_id,
        estimate=estimate,
        milestone_id=milestone_id,
        refs=refs,
        custom_steps=custom_steps,
        custom_expected=custom_expected,
    )
    return await (await _svc()).update_case(case_id, payload)


# ----------------------------------------------------------------------
# Runs / Tests / Results
# ----------------------------------------------------------------------


@mcp.tool
@_guard
async def get_runs(project_id: int) -> list[dict]:
    """List test runs in a project."""
    return await (await _svc()).list_runs(project_id)


@mcp.tool
@_guard
async def get_run(run_id: int) -> dict:
    """Get a single test run by id (includes pass/fail counts)."""
    return await (await _svc()).get_run(run_id)


@mcp.tool
@_guard
async def add_run(
    project_id: int,
    name: str,
    suite_id: int | None = None,
    description: str | None = None,
    milestone_id: int | None = None,
    include_all: bool = True,
    case_ids: list[int] | None = None,
) -> dict:
    """Create a test run in a project. Set `include_all=False` and pass
    `case_ids` to include only specific cases."""
    payload = AddRunPayload(
        name=name,
        suite_id=suite_id,
        description=description,
        milestone_id=milestone_id,
        include_all=include_all,
        case_ids=case_ids,
    )
    return await (await _svc()).add_run(project_id, payload)


@mcp.tool
@_guard
async def close_run(run_id: int) -> dict:
    """Close a test run, archiving its results. This cannot be undone."""
    return await (await _svc()).close_run(run_id)


@mcp.tool
@_guard
async def get_tests(run_id: int) -> list[dict]:
    """List the tests contained in a run."""
    return await (await _svc()).list_tests(run_id)


@mcp.tool
@_guard
async def get_test(test_id: int) -> dict:
    """Get a single test by id."""
    return await (await _svc()).get_test(test_id)


@mcp.tool
@_guard
async def add_result_for_case(
    run_id: int,
    case_id: int,
    status_id: int,
    comment: str | None = None,
    version: str | None = None,
    elapsed: str | None = None,
    defects: str | None = None,
) -> dict:
    """Record a result for a case in a run. `status_id` defaults: 1=Passed,
    2=Blocked, 3=Untested, 4=Retest, 5=Failed (verify via get_statuses)."""
    payload = AddResultPayload(
        status_id=status_id,
        comment=comment,
        version=version,
        elapsed=elapsed,
        defects=defects,
    )
    return await (await _svc()).add_result_for_case(run_id, case_id, payload)


@mcp.tool
@_guard
async def get_results_for_case(run_id: int, case_id: int) -> list[dict]:
    """List the recorded results for a case within a run."""
    return await (await _svc()).list_results_for_case(run_id, case_id)


@mcp.tool
@_guard
async def get_results_for_run(run_id: int) -> list[dict]:
    """List all recorded results for a run."""
    return await (await _svc()).list_results_for_run(run_id)


# ----------------------------------------------------------------------
# Milestones
# ----------------------------------------------------------------------


@mcp.tool
@_guard
async def get_milestones(project_id: int) -> list[dict]:
    """List milestones in a project."""
    return await (await _svc()).list_milestones(project_id)


@mcp.tool
@_guard
async def add_milestone(
    project_id: int,
    name: str,
    description: str | None = None,
    due_on: int | None = None,
) -> dict:
    """Create a milestone in a project. `due_on` is a UNIX timestamp."""
    payload = AddMilestonePayload(
        name=name, description=description, due_on=due_on
    )
    return await (await _svc()).add_milestone(project_id, payload)


# ----------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------


@mcp.tool
@_guard
async def get_priorities() -> list[dict]:
    """List the available case priorities (id -> name)."""
    return await (await _svc()).list_priorities()


@mcp.tool
@_guard
async def get_case_types() -> list[dict]:
    """List the available case types (id -> name)."""
    return await (await _svc()).list_case_types()


@mcp.tool
@_guard
async def get_statuses() -> list[dict]:
    """List the available test statuses (id -> name/label)."""
    return await (await _svc()).list_statuses()


def main() -> None:
    """Entry point: serve over stdio (the transport the chatbot uses)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
