"""Local FastMCP server exposing Apache Airflow REST API as MCP tools.

This runs locally over stdio and communicates with Airflow using Airflow's
native REST API v2 via `httpx`. Bypasses any need for Airflow dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path so imports work correctly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Best-effort: use corporate truststore if available
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import time

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP(
    name="airflow",
    instructions=(
        "Read-only tools for Apache Airflow: list DAGs, check DAG run states, "
        "and fetch task logs. Uses standard Airflow REST API v2."
    ),
)

# Airflow 3's default auth managers (SimpleAuthManager / FAB in JWT mode) don't accept
# Basic Auth on /api/v2/* — a JWT must first be obtained from /auth/token and sent as a
# Bearer token. Cache it (Airflow's default token TTL is 30 min) to avoid a round trip
# to /auth/token on every tool call.
_TOKEN_TTL_SECONDS = 25 * 60
_token_cache: dict[str, Any] = {"token": None, "fetched_at": 0.0}


def _root_url() -> str:
    url = os.environ.get("AIRFLOW_URL")
    if not url:
        raise ToolError("Missing AIRFLOW_URL configuration.")
    return url.rstrip("/")


async def _get_access_token(root_url: str) -> str:
    # A token copied straight from the Airflow UI (profile page -> API Token)
    # takes priority over the username/password flow -- use it as-is.
    static_token = os.environ.get("AIRFLOW_TOKEN")
    if static_token:
        return static_token

    cached = _token_cache["token"]
    if cached and (time.monotonic() - _token_cache["fetched_at"]) < _TOKEN_TTL_SECONDS:
        return cached

    username = os.environ.get("AIRFLOW_USERNAME", "")
    password = os.environ.get("AIRFLOW_PASSWORD", "")
    if not username or not password:
        raise ToolError("Missing AIRFLOW_USERNAME/AIRFLOW_PASSWORD configuration.")

    try:
        async with httpx.AsyncClient(base_url=root_url, timeout=30.0) as auth_client:
            response = await auth_client.post(
                "/auth/token", json={"username": username, "password": password}
            )
            if response.status_code != 200:
                raise ToolError(
                    f"Airflow auth failed ({response.status_code}): {response.text}"
                )
            token = response.json().get("access_token")
            if not token:
                raise ToolError("Airflow auth response did not include an access_token.")
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to reach Airflow /auth/token endpoint: {exc}") from exc

    _token_cache["token"] = token
    _token_cache["fetched_at"] = time.monotonic()
    return token


async def _get_api_client() -> httpx.AsyncClient:
    root_url = _root_url()
    base_url = root_url if root_url.endswith("/api/v2") else f"{root_url}/api/v2"
    token = await _get_access_token(root_url)
    return httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0
    )

@mcp.tool
async def list_dags(limit: int = 100) -> list[dict]:
    """List all DAGs available in Airflow."""

    all_dags = []
    offset = 0

    try:
        async with (await _get_api_client()) as client:
            while True:
                response = await client.get(
                    "/dags",
                    params={
                        "limit": limit,
                        "offset": offset,
                    },
                )

                if response.status_code != 200:
                    raise ToolError(
                        f"Airflow API failed ({response.status_code}): {response.text}"
                    )

                page = response.json().get("dags", [])

                if not page:
                    break

                all_dags.extend(page)

                if len(page) < limit:
                    break

                offset += limit

        return all_dags

    except httpx.RequestError as exc:
        raise ToolError(
            f"Failed to communicate with Airflow REST endpoint: {exc}"
        ) from exc


@mcp.tool
async def list_dag_runs(dag_id: str, limit: int = 100) -> list[dict]:
    """List recent execution runs of a specific DAG (shows Success, Failed, etc.)."""
    try:
        async with (await _get_api_client()) as client:
            # Sort by execution_date descending to show newest runs first
            response = await client.get(
                f"/dags/{dag_id}/dagRuns", 
                params={"limit": limit}
            )
            if response.status_code != 200:
                raise ToolError(f"Airflow API failed to fetch runs ({response.status_code}): {response.text}")
            return response.json().get("dag_runs", [])
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Airflow REST endpoint: {exc}") from exc

@mcp.tool
async def get_task_logs(dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1) -> str:
    """Fetch the execution log string of a specific task run within a DAG run for debugging."""
    try:
        async with (await _get_api_client()) as client:
            response = await client.get(
                f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}",
                headers={"Accept": "application/json"}
            )
            if response.status_code != 200:
                raise ToolError(f"Airflow API failed to fetch logs ({response.status_code}): {response.text}")
            
            # Limit returned logs to the last 5000 characters to prevent context window overflow
            logs = response.text
            if len(logs) > 5000:
                return f"... [Truncated first part of log due to size] ...\n{logs[-5000:]}"
            return logs
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Airflow REST endpoint: {exc}") from exc

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()