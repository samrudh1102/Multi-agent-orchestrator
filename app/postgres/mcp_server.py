"""Local FastMCP server exposing PostgreSQL as MCP tools.

This runs locally over stdio and talks to Postgres directly via `asyncpg`,
replacing the external `uvx postgres-mcp` package. Useful when the target
database (e.g. an RDS instance) is unreachable from wherever `postgres-mcp`
would otherwise run, or when you just want the connection logic in-repo.

Run standalone for a quick smoke test:

    DATABASE_URI=postgresql://user:pass@host:5432/dbname \
        python app/postgres/mcp_server.py
"""

from __future__ import annotations

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

import asyncpg
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP(
    name="postgres",
    instructions=(
        "Tools for a PostgreSQL database: run read-only SQL queries, list "
        "schemas/tables, and describe table columns."
    ),
)

# restricted (default) = read-only queries only; unrestricted also permits writes.
_ACCESS_MODE = os.environ.get("POSTGRES_MCP_ACCESS_MODE", "restricted")

_READ_ONLY_PREFIXES = ("SELECT", "WITH", "EXPLAIN", "SHOW")

_pool: asyncpg.Pool | None = None


def _dsn() -> str:
    dsn = os.environ.get("DATABASE_URI")
    if not dsn:
        raise ToolError("Missing DATABASE_URI configuration.")
    return dsn

import traceback
import ssl
from pathlib import Path

async def _get_pool() -> asyncpg.Pool:
    global _pool

    if _pool is None:
        try:
            ssl_ctx = ssl.create_default_context(
                cafile=str(Path("/u/rag-chatbot/data/global-bundle.pem"))
            )

            ssl_ctx.check_hostname = True
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED

            _pool = await asyncpg.create_pool(
                dsn=_dsn(),
                ssl=ssl_ctx,
                min_size=1,
                max_size=5,
                timeout=3,
                command_timeout=5,
            )

        except Exception as exc:
            traceback.print_exc()
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return _pool

def _is_read_only(sql: str) -> bool:
    clean_sql = sql.strip().upper()
    return clean_sql.startswith(_READ_ONLY_PREFIXES)


async def _fetch(sql: str, *args: object) -> list[dict]:
    pool = await _get_pool()
    try:
        async with pool.acquire() as conn:
            if _ACCESS_MODE == "restricted":
                async with conn.transaction(readonly=True):
                    rows = await conn.fetch(sql, *args)
            else:
                rows = await conn.fetch(sql, *args)
        return [dict(row) for row in rows]
    except asyncpg.PostgresError as exc:
        raise ToolError(f"Postgres query failed: {exc}") from exc


@mcp.tool
async def query(sql: str) -> list[dict]:
    """Execute a SQL query against PostgreSQL. Read-only (SELECT/WITH/EXPLAIN/SHOW)
    unless POSTGRES_MCP_ACCESS_MODE=unrestricted."""
    if _ACCESS_MODE == "restricted" and not _is_read_only(sql):
        raise ToolError(
            "Security Alert: Only read-only queries (SELECT, WITH, EXPLAIN, SHOW) "
            "are allowed in restricted access mode."
        )
    return await _fetch(sql)


@mcp.tool
async def list_schemas() -> list[dict]:
    """List all non-system schemas in the database."""
    return await _fetch(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name NOT IN ('pg_catalog', 'information_schema') "
        "ORDER BY schema_name"
    )


@mcp.tool
async def list_tables(schema: str = "public") -> list[dict]:
    """List tables inside a specific database schema."""
    return await _fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = $1 AND table_type = 'BASE TABLE' ORDER BY table_name",
        schema,
    )


@mcp.tool
async def describe_table(table_name: str, schema: str = "public") -> list[dict]:
    """Describe columns, data types, and nullability of a specific table."""
    return await _fetch(
        "SELECT column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position",
        schema,
        table_name,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
