"""Local FastMCP server exposing Databricks REST APIs as MCP tools.

This runs locally over stdio and communicates with Databricks using
Databricks SQL Statement Execution API (v2.0), Unity Catalog API (v2.1),
and Jobs API (v2.1) via `httpx`. Bypasses traditional ODBC/JDBC dependencies.
"""

from __future__ import annotations

import asyncio
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
    name="databricks",
    instructions=(
        "Tools for Databricks: execute SQL queries against SQL Warehouses, browse Unity Catalog "
        "(catalogs, schemas, tables), and inspect or trigger Workflows/Jobs."
    ),
)


def _get_api_client() -> httpx.AsyncClient:
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        raise ToolError("Missing DATABRICKS_HOST or DATABRICKS_TOKEN environment configuration.")

    base_url = host.rstrip("/")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0)


def _is_read_only(sql: str) -> bool:
    """Determine if the query is a read-only operation."""
    clean_sql = sql.strip().upper()
    allowed_prefixes = ("SELECT", "DESCRIBE", "SHOW", "EXPLAIN", "WITH")
    return any(clean_sql.startswith(prefix) for prefix in allowed_prefixes)


@mcp.tool
async def query(
    sql: str,
    catalog: str | None = None,
    schema: str | None = None,
    warehouse_id: str | None = None,
) -> list[dict] | str:
    """Execute a read-only SQL query on a Databricks SQL Warehouse via Statement Execution API."""
    if not _is_read_only(sql):
        raise ToolError("Security Alert: Only read-only queries (SELECT, WITH, EXPLAIN) are permitted.")

    target_warehouse = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not target_warehouse:
        raise ToolError("Missing warehouse_id argument or DATABRICKS_WAREHOUSE_ID environment configuration.")

    payload: dict[str, Any] = {
        "statement": sql,
        "warehouse_id": target_warehouse,
        "wait_timeout": "30s",  # Wait up to 30 seconds synchronously
    }
    if catalog:
        payload["catalog"] = catalog
    if schema:
        payload["schema"] = schema

    try:
        async with _get_api_client() as client:
            response = await client.post("/api/2.0/sql/statements", json=payload)
            if response.status_code not in (200, 202):
                raise ToolError(f"Databricks SQL Execution failed ({response.status_code}): {response.text}")

            res_data = response.json()
            statement_id = res_data.get("statement_id")
            status = res_data.get("status", {}).get("state")

            # Asynchronously poll if status is PENDING or RUNNING
            while status in ("PENDING", "RUNNING"):
                await asyncio.sleep(1)
                poll_resp = await client.get(f"/api/2.0/sql/statements/{statement_id}")
                if poll_resp.status_code != 200:
                    raise ToolError(f"Failed polling statement status ({poll_resp.status_code}): {poll_resp.text}")
                res_data = poll_resp.json()
                status = res_data.get("status", {}).get("state")

            if status == "FAILED":
                err_msg = res_data.get("status", {}).get("error", {}).get("message", "Unknown execution error")
                raise ToolError(f"Databricks query failed: {err_msg}")

            # Parse schema columns and records array
            manifest = res_data.get("manifest", {}).get("schema", {}).get("columns", [])
            col_names = [col["name"] for col in manifest]

            data_array = res_data.get("result", {}).get("data_array", [])
            records = []
            for row in data_array:
                records.append(dict(zip(col_names, row)))

            return records

    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Databricks REST endpoint: {exc}") from exc


@mcp.tool
async def list_catalogs() -> list[dict]:
    """List Unity Catalog catalogs available in your Databricks workspace."""
    try:
        async with _get_api_client() as client:
            response = await client.get("/api/2.1/unity-catalog/catalogs")
            if response.status_code != 200:
                raise ToolError(f"Failed to fetch catalogs ({response.status_code}): {response.text}")
            return response.json().get("catalogs", [])
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Databricks REST endpoint: {exc}") from exc


@mcp.tool
async def list_schemas(catalog_name: str = "main") -> list[dict]:
    """List schemas contained inside a specific Unity Catalog catalog."""
    try:
        async with _get_api_client() as client:
            response = await client.get("/api/2.1/unity-catalog/schemas", params={"catalog_name": catalog_name})
            if response.status_code != 200:
                raise ToolError(f"Failed to fetch schemas ({response.status_code}): {response.text}")
            return response.json().get("schemas", [])
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Databricks REST endpoint: {exc}") from exc


@mcp.tool
async def list_tables(catalog_name: str = "main", schema_name: str = "default") -> list[dict]:
    """List tables contained inside a target catalog and schema."""
    try:
        async with _get_api_client() as client:
            response = await client.get(
                "/api/2.1/unity-catalog/tables",
                params={"catalog_name": catalog_name, "schema_name": schema_name},
            )
            if response.status_code != 200:
                raise ToolError(f"Failed to fetch tables ({response.status_code}): {response.text}")
            return response.json().get("tables", [])
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Databricks REST endpoint: {exc}") from exc


@mcp.tool
async def list_jobs(limit: int = 25) -> list[dict]:
    """List Databricks Workflows/Jobs configured in the workspace."""
    try:
        async with _get_api_client() as client:
            response = await client.get("/api/2.1/jobs/list", params={"limit": limit})
            if response.status_code != 200:
                raise ToolError(f"Failed to fetch jobs ({response.status_code}): {response.text}")
            return response.json().get("jobs", [])
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Databricks REST endpoint: {exc}") from exc


@mcp.tool
async def trigger_job(job_id: int, notebook_params: dict[str, str] | None = None) -> dict:
    """Trigger a run for a specific Databricks Job/Workflow ID."""
    payload: dict[str, Any] = {"job_id": job_id}
    if notebook_params:
        payload["notebook_params"] = notebook_params

    try:
        async with _get_api_client() as client:
            response = await client.post("/api/2.1/jobs/run-now", json=payload)
            if response.status_code != 200:
                raise ToolError(f"Failed to trigger job ({response.status_code}): {response.text}")
            return response.json()
    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with Databricks REST endpoint: {exc}") from exc


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()