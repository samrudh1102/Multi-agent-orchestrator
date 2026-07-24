"""Service layer over `TestRailClient`.

This is the API the local FastMCP server (app/testrail/mcp_server.py) calls. It
owns the client's connect/close lifecycle and normalizes TestRail's responses:
newer TestRail versions wrap list endpoints in a pagination envelope
(e.g. ``{"offset": 0, "size": 250, "cases": [...]}``) while older ones return a
bare list — `_unwrap` handles both so tools always get a plain list.
"""

from __future__ import annotations

from typing import Any

from .client import TestRailClient
from .model import (
    AddCasePayload,
    AddMilestonePayload,
    AddResultPayload,
    AddRunPayload,
    AddSectionPayload,
    TestRailConfig,
    UpdateCasePayload,
)


def _unwrap(data: Any, key: str) -> list[dict]:
    """Return the list under `key` from a paginated envelope, or `data`
    itself if it is already a list."""

    if isinstance(data, dict):
        value = data.get(key)
        return value if isinstance(value, list) else []
    if isinstance(data, list):
        return data
    return []


class TestRailService:
    """Higher-level, LLM-friendly operations over the TestRail REST API."""

    def __init__(self, config: TestRailConfig) -> None:
        self._client = TestRailClient(
            base_url=config.url,
            username=config.username,
            api_key=config.api_key,
        )

    async def connect(self) -> None:
        await self._client.connect()

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "TestRailService":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    async def list_projects(self) -> list[dict]:
        return _unwrap(await self._client.get_projects(), "projects")

    async def get_project(self, project_id: int) -> dict:
        return await self._client.get_project(project_id)

    # ------------------------------------------------------------------
    # Suites / Sections
    # ------------------------------------------------------------------

    async def list_suites(self, project_id: int) -> list[dict]:
        return _unwrap(await self._client.get_suites(project_id), "suites")

    async def get_suite(self, suite_id: int) -> dict:
        return await self._client.get_suite(suite_id)

    async def list_sections(
        self, project_id: int, suite_id: int | None = None
    ) -> list[dict]:
        return _unwrap(
            await self._client.get_sections(project_id, suite_id), "sections"
        )

    async def add_section(
        self, project_id: int, payload: AddSectionPayload
    ) -> dict:
        return await self._client.add_section(
            project_id, payload.model_dump(exclude_none=True)
        )

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    async def list_cases(
        self,
        project_id: int,
        suite_id: int | None = None,
        section_id: int | None = None,
    ) -> list[dict]:
        return _unwrap(
            await self._client.get_cases(project_id, suite_id, section_id),
            "cases",
        )

    async def get_case(self, case_id: int) -> dict:
        return await self._client.get_case(case_id)

    async def add_case(self, section_id: int, payload: AddCasePayload) -> dict:
        return await self._client.add_case(
            section_id, payload.model_dump(exclude_none=True)
        )

    async def update_case(
        self, case_id: int, payload: UpdateCasePayload
    ) -> dict:
        return await self._client.update_case(
            case_id, payload.model_dump(exclude_none=True)
        )

    # ------------------------------------------------------------------
    # Runs / Tests / Results
    # ------------------------------------------------------------------

    async def list_runs(self, project_id: int) -> list[dict]:
        return _unwrap(await self._client.get_runs(project_id), "runs")

    async def get_run(self, run_id: int) -> dict:
        return await self._client.get_run(run_id)

    async def add_run(self, project_id: int, payload: AddRunPayload) -> dict:
        return await self._client.add_run(
            project_id, payload.model_dump(exclude_none=True)
        )

    async def close_run(self, run_id: int) -> dict:
        return await self._client.close_run(run_id)

    async def list_tests(self, run_id: int) -> list[dict]:
        return _unwrap(await self._client.get_tests(run_id), "tests")

    async def get_test(self, test_id: int) -> dict:
        return await self._client.get_test(test_id)

    async def add_result_for_case(
        self, run_id: int, case_id: int, payload: AddResultPayload
    ) -> dict:
        return await self._client.add_result_for_case(
            run_id, case_id, payload.model_dump(exclude_none=True)
        )

    async def list_results_for_case(
        self, run_id: int, case_id: int
    ) -> list[dict]:
        return _unwrap(
            await self._client.get_results_for_case(run_id, case_id), "results"
        )

    async def list_results_for_run(self, run_id: int) -> list[dict]:
        return _unwrap(
            await self._client.get_results_for_run(run_id), "results"
        )

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    async def list_milestones(self, project_id: int) -> list[dict]:
        return _unwrap(
            await self._client.get_milestones(project_id), "milestones"
        )

    async def add_milestone(
        self, project_id: int, payload: AddMilestonePayload
    ) -> dict:
        return await self._client.add_milestone(
            project_id, payload.model_dump(exclude_none=True)
        )

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    async def list_priorities(self) -> list[dict]:
        return await self._client.get_priorities()

    async def list_case_types(self) -> list[dict]:
        return await self._client.get_case_types()

    async def list_statuses(self) -> list[dict]:
        return await self._client.get_statuses()
