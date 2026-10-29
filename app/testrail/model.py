"""Data models for the TestRail integration.

`TestRailConfig` carries the connection settings the local FastMCP server
(app/testrail/mcp_server.py) reads from its environment. The `Add*`/`Update*`
models describe the payloads for the create/update tools; FastMCP turns them
into the JSON schema advertised to the LLM, and pydantic validates whatever the
model sends back before it reaches TestRail.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class TestRailConfig(BaseModel):
    """Connection settings for a TestRail instance."""

    url: str
    username: str
    api_key: str

    @classmethod
    def from_env(cls) -> "TestRailConfig":
        """Build a config from the TESTRAIL_* environment variables passed
        to the stdio server by the MCP client.

        Raises:
            ValueError: if any required variable is missing.
        """

        url = os.environ.get("TESTRAIL_URL")
        username = os.environ.get("TESTRAIL_USERNAME")
        api_key = os.environ.get("TESTRAIL_API_KEY")

        missing = [
            name
            for name, value in (
                ("TESTRAIL_URL", url),
                ("TESTRAIL_USERNAME", username),
                ("TESTRAIL_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required TestRail environment variables: "
                + ", ".join(missing)
            )

        if not url.startswith(("http://", "https://")):
            raise ValueError(
                f"TESTRAIL_URL must include a scheme, e.g. 'https://{url}' "
                f"(got '{url}')."
            )

        return cls(url=url, username=username, api_key=api_key)  # type: ignore[arg-type]


class AddCasePayload(BaseModel):
    """Fields for creating a test case (POST add_case/{section_id})."""

    title: str = Field(..., description="Title of the test case.")
    template_id: int | None = Field(
        None, description="Template (field layout) id."
    )
    type_id: int | None = Field(None, description="Case type id.")
    priority_id: int | None = Field(None, description="Priority id.")
    estimate: str | None = Field(
        None, description="Time estimate, e.g. '30s' or '1m 45s'."
    )
    milestone_id: int | None = Field(None, description="Associated milestone id.")
    refs: str | None = Field(
        None, description="Comma-separated references/requirements."
    )
    custom_steps: str | None = Field(
        None, description="Plain-text test steps (custom_steps field)."
    )
    custom_expected: str | None = Field(
        None, description="Expected result (custom_expected field)."
    )


class UpdateCasePayload(BaseModel):
    """Fields for updating a test case (POST update_case/{case_id}).

    All fields are optional; only those provided are changed.
    """

    title: str | None = None
    type_id: int | None = None
    priority_id: int | None = None
    estimate: str | None = None
    milestone_id: int | None = None
    refs: str | None = None
    custom_steps: str | None = None
    custom_expected: str | None = None


class AddRunPayload(BaseModel):
    """Fields for creating a test run (POST add_run/{project_id})."""

    name: str = Field(..., description="Name of the test run.")
    suite_id: int | None = Field(
        None, description="Suite id (required for multi-suite projects)."
    )
    description: str | None = None
    milestone_id: int | None = None
    include_all: bool = Field(
        True, description="Include all cases; set False to use case_ids."
    )
    case_ids: list[int] | None = Field(
        None, description="Case ids to include when include_all is False."
    )


class AddResultPayload(BaseModel):
    """Fields for adding a result to a case in a run
    (POST add_result_for_case/{run_id}/{case_id})."""

    status_id: int = Field(
        ...,
        description="Status id: 1=Passed, 2=Blocked, 3=Untested, "
        "4=Retest, 5=Failed (defaults; check get_statuses).",
    )
    comment: str | None = Field(None, description="Comment / result notes.")
    version: str | None = Field(None, description="Version or build tested.")
    elapsed: str | None = Field(
        None, description="Time taken, e.g. '30s' or '1m 45s'."
    )
    defects: str | None = Field(
        None, description="Comma-separated defect references."
    )


class AddMilestonePayload(BaseModel):
    """Fields for creating a milestone (POST add_milestone/{project_id})."""

    name: str = Field(..., description="Name of the milestone.")
    description: str | None = None
    due_on: int | None = Field(
        None, description="Due date as a UNIX timestamp."
    )


class AddSectionPayload(BaseModel):
    """Fields for creating a section (POST add_section/{project_id})."""

    name: str = Field(..., description="Name of the section.")
    suite_id: int | None = Field(
        None, description="Suite id (required for multi-suite projects)."
    )
    description: str | None = None
    parent_id: int | None = Field(
        None, description="Parent section id for nesting."
    )
