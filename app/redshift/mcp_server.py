"""Local FastMCP server exposing AWS Redshift as MCP tools.

This runs locally over stdio using standard AWS Redshift Data API (via `boto3`).
Bypasses traditional JDBC/ODBC and port-forwarding requirements completely.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

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

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP(
    name="redshift",
    instructions=(
        "Tools for AWS Redshift: query tables, list schemas, list tables, and "
        "describe tables. All queries are restricted to safe, read-only SELECT operations."
    ),
)

def _get_client():
    return boto3.client("redshift-data")

def _is_read_only(sql: str) -> bool:
    """Determine if the query is a read-only operation."""
    clean_sql = sql.strip().upper()
    allowed_prefixes = ("SELECT", "DESCRIBE", "SHOW", "EXPLAIN", "WITH")
    return any(clean_sql.startswith(prefix) for prefix in allowed_prefixes)

async def _execute_and_poll(client, sql: str) -> str:
    """Execute a statement against Redshift Data API and poll asynchronously until finished."""
    database = os.environ.get("REDSHIFT_DATABASE")
    if not database:
        raise ToolError("Missing REDSHIFT_DATABASE environment variable configuration.")

    kwargs = {
        "Database": database,
        "Sql": sql,
    }
    
    # Configure serverless vs provisioned vs secrets-manager connection
    workgroup = os.environ.get("REDSHIFT_WORKGROUP_NAME")
    cluster = os.environ.get("REDSHIFT_CLUSTER_IDENTIFIER")
    secret = os.environ.get("REDSHIFT_SECRET_ARN")
    db_user = os.environ.get("REDSHIFT_DB_USER")

    if secret:
        kwargs["SecretArn"] = secret
    if workgroup:
        kwargs["WorkgroupName"] = workgroup
    elif cluster:
        kwargs["ClusterIdentifier"] = cluster
        if db_user:
            kwargs["DbUser"] = db_user

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.execute_statement(**kwargs))
        statement_id = response["Id"]

        # Asynchronously poll the statement execution status
        while True:
            desc = await loop.run_in_executor(None, lambda: client.describe_statement(Id=statement_id))
            status = desc["Status"]
            if status == "FINISHED":
                return statement_id
            elif status in ("FAILED", "ABORTED"):
                error_msg = desc.get("Error", "Unknown execution error")
                raise ToolError(f"Redshift execution failed ({status}): {error_msg}")
            await asyncio.sleep(1)
            
    except Exception as exc:
        if not isinstance(exc, ToolError):
            raise ToolError(f"Redshift communication failed: {exc}") from exc
        raise exc

async def _get_results(client, statement_id: str) -> list[dict]:
    """Download statement query records and map column metadata to row keys."""
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, lambda: client.get_statement_result(Id=statement_id))

        metadata = res.get("ColumnMetadata", [])
        col_names = [col["name"] for col in metadata]

        records = []
        for row in res.get("Records", []):
            record = {}
            for i, val_container in enumerate(row):
                val = None
                if val_container:
                    val = next(iter(val_container.values()))
                record[col_names[i]] = val
            records.append(record)
        return records
    except Exception as exc:
        raise ToolError(f"Failed to read Redshift results: {exc}") from exc

@mcp.tool
async def query(sql: str) -> list[dict]:
    """Execute a read-only SQL SELECT query against AWS Redshift (automatically safety-guarded)."""
    if not _is_read_only(sql):
        raise ToolError("Security Alert: Only read-only queries (SELECT, WITH, EXPLAIN) are allowed.")

    client = _get_client()
    statement_id = await _execute_and_poll(client, sql)
    return await _get_results(client, statement_id)

@mcp.tool
async def list_schemas() -> list[dict]:
    """List all active schemas inside your Redshift database."""
    sql = "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog', 'information_schema') ORDER BY schema_name"
    client = _get_client()
    statement_id = await _execute_and_poll(client, sql)
    return await _get_results(client, statement_id)

@mcp.tool
async def list_tables(schema: str = "public") -> list[dict]:
    """List tables inside a specific database schema."""
    sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE' ORDER BY table_name"
    client = _get_client()
    statement_id = await _execute_and_poll(client, sql)
    return await _get_results(client, statement_id)

@mcp.tool
async def describe_table(table_name: str, schema: str = "public") -> list[dict]:
    """Describe columns, data types, and nullability specs of a specific table."""
    sql = f"SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = '{schema}' AND table_name = '{table_name}' ORDER BY ordinal_position"
    client = _get_client()
    statement_id = await _execute_and_poll(client, sql)
    return await _get_results(client, statement_id)

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()