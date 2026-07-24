from __future__ import annotations

from typing import Any
from urllib.parse import quote
import logging

import httpx

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    RequestError,
    ResourceNotFoundError,
    ServerError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class TestRailClient:
    """Async client for interacting with the TestRail REST API."""

    API_PREFIX = "/index.php?/api/v2"

    def __init__(
        self,
        base_url: str,
        username: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        self._auth = httpx.BasicAuth(
            username=username,
            password=api_key,
        )

        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """
        Initialize the underlying AsyncClient.

        Safe to call multiple times.
        """

        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            auth=self._auth,
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

        logger.info("Connected to TestRail.")

    async def close(self) -> None:
        """Close the underlying AsyncClient."""

        if self._client is None:
            return

        await self._client.aclose()
        self._client = None

        logger.info("Disconnected from TestRail.")

    @property
    def client(self) -> httpx.AsyncClient:
        """
        Return the initialized AsyncClient.

        Raises:
            RuntimeError: If connect() has not been called.
        """
        if self._client is None:
            raise RuntimeError(
                "TestRailClient has not been initialized. Call connect() first."
            )

        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a request against the TestRail REST API.

        Args:
            method: HTTP method.
            endpoint: API endpoint relative to /api/v2.
            params: Query parameters, appended manually since TestRail's
                "?/api/v2/..." URL scheme gets mangled if passed through
                httpx's own query-param merging (it percent-encodes the
                literal "?/api/v2/..." segment, which TestRail no longer
                recognizes as an API request and redirects to the login
                page instead of returning JSON).
            **kwargs: Additional arguments passed to httpx.

        Returns:
            Parsed JSON response.

        Raises:
            AuthenticationError
            AuthorizationError
            ValidationError
            ResourceNotFoundError
            RateLimitError
            ServerError
            RequestError
        """

        url = f"{self.API_PREFIX}/{endpoint}"

        if params:
            query = "&".join(
                f"{quote(str(k))}={quote(str(v))}"
                for k, v in params.items()
                if v is not None
            )
            if query:
                url = f"{url}&{query}"

        logger.debug("%s %s", method, url)

        try:
            response = await self.client.request(
                method=method,
                url=url,
                **kwargs,
            )
        except httpx.RequestError as exc:
            logger.exception("Unable to communicate with TestRail.")
            raise RequestError(str(exc)) from exc

        logger.debug("Response Status: %s", response.status_code)

        match response.status_code:
            case 200 | 201:
                return response.json()

            case 204:
                return None

            case 400:
                raise ValidationError(response.text)

            case 401:
                raise AuthenticationError("Invalid TestRail credentials.")

            case 403:
                raise AuthorizationError("Permission denied.")

            case 404:
                raise ResourceNotFoundError(
                    f"Resource '{endpoint}' not found."
                )

            case 429:
                raise RateLimitError("TestRail API rate limit exceeded.")

            case status if 500 <= status < 600:
                raise ServerError(response.text)

            case _:
                response.raise_for_status()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    async def get_projects(self) -> list[dict]:
        """Retrieve all TestRail projects."""

        return await self._request(
            method="GET",
            endpoint="get_projects",
        )

    async def get_project(self, project_id: int) -> dict:
        """Retrieve a single project."""

        return await self._request(
            method="GET",
            endpoint=f"get_project/{project_id}",
        )

    # ------------------------------------------------------------------
    # Suites
    # ------------------------------------------------------------------

    async def get_suites(self, project_id: int) -> Any:
        """Retrieve all test suites for a project."""

        return await self._request(
            method="GET",
            endpoint=f"get_suites/{project_id}",
        )

    async def get_suite(self, suite_id: int) -> dict:
        """Retrieve a single test suite."""

        return await self._request(
            method="GET",
            endpoint=f"get_suite/{suite_id}",
        )

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    async def get_sections(
        self, project_id: int, suite_id: int | None = None
    ) -> Any:
        """Retrieve sections for a project (and optionally a suite)."""

        params = {} if suite_id is None else {"suite_id": suite_id}
        return await self._request(
            method="GET",
            endpoint=f"get_sections/{project_id}",
            params=params,
        )

    async def add_section(self, project_id: int, data: dict) -> dict:
        """Create a new section in a project."""

        return await self._request(
            method="POST",
            endpoint=f"add_section/{project_id}",
            json=data,
        )

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    async def get_cases(
        self,
        project_id: int,
        suite_id: int | None = None,
        section_id: int | None = None,
    ) -> Any:
        """Retrieve test cases for a project, optionally scoped to a
        suite and/or section."""

        params: dict[str, Any] = {}
        if suite_id is not None:
            params["suite_id"] = suite_id
        if section_id is not None:
            params["section_id"] = section_id
        return await self._request(
            method="GET",
            endpoint=f"get_cases/{project_id}",
            params=params,
        )

    async def get_case(self, case_id: int) -> dict:
        """Retrieve a single test case."""

        return await self._request(
            method="GET",
            endpoint=f"get_case/{case_id}",
        )

    async def add_case(self, section_id: int, data: dict) -> dict:
        """Create a new test case within a section."""

        return await self._request(
            method="POST",
            endpoint=f"add_case/{section_id}",
            json=data,
        )

    async def update_case(self, case_id: int, data: dict) -> dict:
        """Update an existing test case."""

        return await self._request(
            method="POST",
            endpoint=f"update_case/{case_id}",
            json=data,
        )

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def get_runs(self, project_id: int) -> Any:
        """Retrieve test runs for a project."""

        return await self._request(
            method="GET",
            endpoint=f"get_runs/{project_id}",
        )

    async def get_run(self, run_id: int) -> dict:
        """Retrieve a single test run."""

        return await self._request(
            method="GET",
            endpoint=f"get_run/{run_id}",
        )

    async def add_run(self, project_id: int, data: dict) -> dict:
        """Create a new test run in a project."""

        return await self._request(
            method="POST",
            endpoint=f"add_run/{project_id}",
            json=data,
        )

    async def close_run(self, run_id: int) -> dict:
        """Close an existing test run (archives its results)."""

        return await self._request(
            method="POST",
            endpoint=f"close_run/{run_id}",
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def get_tests(self, run_id: int) -> Any:
        """Retrieve the tests contained in a run."""

        return await self._request(
            method="GET",
            endpoint=f"get_tests/{run_id}",
        )

    async def get_test(self, test_id: int) -> dict:
        """Retrieve a single test."""

        return await self._request(
            method="GET",
            endpoint=f"get_test/{test_id}",
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    async def add_result_for_case(
        self, run_id: int, case_id: int, data: dict
    ) -> dict:
        """Add a result for a specific case within a run."""

        return await self._request(
            method="POST",
            endpoint=f"add_result_for_case/{run_id}/{case_id}",
            json=data,
        )

    async def get_results_for_case(self, run_id: int, case_id: int) -> Any:
        """Retrieve the results for a specific case within a run."""

        return await self._request(
            method="GET",
            endpoint=f"get_results_for_case/{run_id}/{case_id}",
        )

    async def get_results_for_run(self, run_id: int) -> Any:
        """Retrieve all results for a run."""

        return await self._request(
            method="GET",
            endpoint=f"get_results_for_run/{run_id}",
        )

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    async def get_milestones(self, project_id: int) -> Any:
        """Retrieve milestones for a project."""

        return await self._request(
            method="GET",
            endpoint=f"get_milestones/{project_id}",
        )

    async def add_milestone(self, project_id: int, data: dict) -> dict:
        """Create a new milestone in a project."""

        return await self._request(
            method="POST",
            endpoint=f"add_milestone/{project_id}",
            json=data,
        )

    # ------------------------------------------------------------------
    # Metadata (reference data)
    # ------------------------------------------------------------------

    async def get_priorities(self) -> list[dict]:
        """Retrieve the available case priorities."""

        return await self._request(method="GET", endpoint="get_priorities")

    async def get_case_types(self) -> list[dict]:
        """Retrieve the available case types."""

        return await self._request(method="GET", endpoint="get_case_types")

    async def get_statuses(self) -> list[dict]:
        """Retrieve the available test statuses."""

        return await self._request(method="GET", endpoint="get_statuses")