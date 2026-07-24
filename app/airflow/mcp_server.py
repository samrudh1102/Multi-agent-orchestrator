"""Local FastMCP server exposing Apache Airflow REST API as MCP tools.

This runs locally over stdio and communicates with Airflow using Airflow's
native REST API v1 via `httpx`. Bypasses any need for Airflow dependencies.
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

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP(
    name="airflow",
    instructions=(
        "Tools for Apache Airflow: list DAGs, trigger DAG runs, check DAG run states, "
        "and fetch failed task logs. Uses standard Airflow REST API v1."
    ),
)

def _get_api_client() -> httpx.AsyncClient:
    url = os.environ.get("AIRFLOW_URL")
    if not url:
        raise ToolError("Missing AIRFLOW_URL configuration.")

    # Strip trailing slash from base url
    base_url = url.rstrip("/")
    if not base_url.endswith("/api/v1"):
        base_url = f"{base_url}/api/v1"

    username = os.environ.get("AIRFLOW_USERNAME", "")
    password = os.environ.get("AIRFLOW_PASSWORD", "")

    # Configure client with Basic Authentication
    auth = (username, password) if username and password else None
    return httpx.AsyncClient(base_url=base_url, auth=auth, timeout=30.0)

@mcp.tool
async def list_dags(limit: int = 100) -> list[dict]:
    """List all DAGs available in your Airflow environment."""
    try:
        async with _get_api_client() as client:
            response = await client.get("/dags", params={"limit": limit})
            if response.status_code != 200:
                raise ToolError(f"Airflow API failed ({response.status_code}): {response.text}")
            return response.json().get("dags", [])
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Airflow REST endpoint: {exc}") from exc

@mcp.tool
async def trigger_dag(dag_id: str, conf: dict[str, Any] | None = None) -> dict:
    """Trigger a new DAG run for a specific DAG ID. Conf is optional JSON configuration."""
    payload = {}
    if conf:
        payload["conf"] = conf

    try:
        async with _get_api_client() as client:
            response = await client.post(f"/dags/{dag_id}/dagRuns", json=payload)
            if response.status_code not in (200, 201):
                raise ToolError(f"Airflow API failed to trigger DAG ({response.status_code}): {response.text}")
            return response.json()
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Airflow REST endpoint: {exc}") from exc

@mcp.tool
async def list_dag_runs(dag_id: str, limit: int = 20) -> list[dict]:
    """List recent execution runs of a specific DAG (shows Success, Failed, etc.)."""
    try:
        async with _get_api_client() as client:
            # Sort by execution_date descending to show newest runs first
            response = await client.get(
                f"/dags/{dag_id}/dagRuns", 
                params={"limit": limit, "order_by": "-execution_date"}
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
        async with _get_api_client() as client:
            response = await client.get(
                f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}",
                headers={"Accept": "text/plain"}
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