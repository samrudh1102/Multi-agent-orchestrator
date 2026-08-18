"""Local FastMCP server exposing ClickHouse as MCP tools.

This runs locally over stdio and communicates with ClickHouse using ClickHouse's
native REST HTTP API via `httpx`. Bypasses any need for database drivers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path so imports work correctly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Best-effort: use truststore if available
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP(
    name="clickhouse",
    instructions=(
        "Tools for ClickHouse: query tables, list databases, list tables, and "
        "describe tables. All queries are restricted to safe, read-only SELECT operations."
    ),
)

def _is_read_only(sql: str) -> bool:
    """Determine if the query is a read-only operation."""
    clean_sql = sql.strip().upper()
    allowed_prefixes = ("SELECT", "DESCRIBE", "SHOW", "EXPLAIN", "WITH")
    return any(clean_sql.startswith(prefix) for prefix in allowed_prefixes)

async def _execute_http_query(sql: str) -> list[dict] | str:
    """Execute query using ClickHouse's raw REST HTTP interface."""
    url = os.environ.get("CLICKHOUSE_URL")
    if not url:
        raise ToolError("Missing CLICKHOUSE_URL configuration.")

    user = os.environ.get("CLICKHOUSE_USER", "default")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")
    database = os.environ.get("CLICKHOUSE_DATABASE", "default")

    # ClickHouse HTTP interface details:
    # We append 'FORMAT JSON' to ensure ClickHouse returns a structured JSON output
    clean_sql = sql.strip()
    if not clean_sql.upper().endswith("FORMAT JSON"):
        clean_sql = f"{clean_sql} FORMAT JSON"

    headers = {
        "X-ClickHouse-User": user,
        "X-ClickHouse-Database": database,
    }
    if password:
        headers["X-ClickHouse-Key"] = password

    try:
        # Connect with a reasonable timeout
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url=url,
                content=clean_sql,
                headers=headers,
            )
            
            if response.status_code != 200:
                raise ToolError(f"ClickHouse query failed ({response.status_code}): {response.text}")
            
            # Parse ClickHouse JSON format: it returns {"meta": [...], "data": [...], "rows": X}
            try:
                res_json = response.json()
                return res_json.get("data", [])
            except ValueError:
                # If it's not JSON for some reason, return the raw text
                return response.text

    except httpx.RequestError as exc:
        raise ToolError(f"Failed to communicate with ClickHouse REST endpoint: {exc}") from exc

@mcp.tool
async def query(sql: str) -> list[dict] | str:
    """Execute a read-only SELECT query against ClickHouse."""
    if not _is_read_only(sql):
        raise ToolError("Security Alert: Only read-only queries (SELECT, WITH, EXPLAIN) are permitted.")
    return await _execute_http_query(sql)

@mcp.tool
async def list_databases() -> list[dict] | str:
    """List all active databases inside your ClickHouse instance."""
    return await _execute_http_query("SHOW DATABASES")

@mcp.tool
async def list_tables() -> list[dict] | str:
    """List all tables in the default/active database."""
    return await _execute_http_query("SHOW TABLES")

@mcp.tool
async def describe_table(table_name: str) -> list[dict] | str:
    """Show the columns, types, and defaults of a target table."""
    return await _execute_http_query(f"DESCRIBE TABLE {table_name}")

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()