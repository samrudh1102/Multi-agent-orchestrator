"""MCP client to Jira (Atlassian's hosted remote MCP server), Databricks
(Databricks' own hosted remote MCP server), S3 (AWS's official remote MCP
Server, reached via AWS's mcp-proxy-for-aws bridge), and ClickHouse
(ClickHouse Cloud's own hosted remote MCP server), plus the TestRail/
Redshift/Airflow local stdio servers, exposed as LangChain tools for the
LangGraph agent (app/langgraph_agent/graph_agent.py).

Postgres and Grafana have no vendor-hosted remote MCP endpoint for a generic
instance, so they're launched the same way as S3: the official/community
package (postgres-mcp, mcp-grafana) run as a local stdio subprocess via uvx,
rather than a hand-written app/*/mcp_server.py.

S3, Databricks and ClickHouse used to run as local stdio servers
(app/s3/mcp_server.py, app/databricks/mcp_server.py,
app/clickhouse/mcp_server.py); those files are kept in the repo for
reference but are no longer launched — see _build_config below.

Backed by langchain-mcp-adapters' MultiServerMCPClient: it takes a single
multi-server `connections` config, with each server's tool names prefixed by
server (e.g. `jira_getJiraIssue`, `testrail_getCase`) so calls route back to
the right server automatically. get_tools() opens a fresh MCP session (and,
for stdio servers, a fresh subprocess) per tool call — costly for both the
remote streamable_http servers (a full OAuth handshake for jira/clickhouse)
and the local stdio ones (uvx package resolve + Python startup per call), so
every configured server instead gets one session held open for the app's
lifetime — see FastMCPToolProvider.connect below.

Jira and ClickHouse auth follow the same pattern: scripts/setup_jira_auth.py
/ scripts/setup_clickhouse_auth.py are one-time interactive logins that
persist an access + refresh token pair to settings.jira_token_file /
settings.clickhouse_token_file. The running app reuses that same
OAuthClientProvider/FileTokenStorage pairing as its httpx auth, so it
transparently refreshes the access token using the stored refresh token
instead of needing the script re-run. Re-running the script is only needed
if the refresh token itself is
revoked or expired.

Only used when settings.enable_tools is true."""
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from app.config import settings

logger = logging.getLogger(__name__)

# Shared with scripts/setup_jira_auth.py so the one-time interactive login
# registers the same redirect URI / client metadata the running app expects.
JIRA_REDIRECT_URI = "http://localhost:8765/callback"


def jira_client_metadata() -> OAuthClientMetadata:
    return OAuthClientMetadata(
        redirect_uris=[JIRA_REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name="rag-chatbot",
    )


class FileTokenStorage(TokenStorage):
    """Persists OAuth tokens/client info to a JSON file. The file is first
    populated by scripts/setup_jira_auth.py; the running app only reads it
    (via _jira_bearer_token below)."""

    def __init__(self, path: Path):
        self._path = path

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write(self, data: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(data))

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read().get("tokens")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)


def _refuse_interactive_auth(service: str, script: str):
    """The running app must never block on an interactive browser consent
    flow; that only happens once, via the given setup script. If these are
    invoked, the stored refresh token was rejected and a human needs to log
    in again."""

    async def _redirect(url: str) -> None:
        raise RuntimeError(f"{service} OAuth requires interactive re-authorization. Run {script} to log in again.")

    async def _callback() -> tuple[str, str | None]:
        raise RuntimeError(f"{service} OAuth requires interactive re-authorization. Run {script} to log in again.")

    return _redirect, _callback


_refused_redirect, _refused_callback = _refuse_interactive_auth("Jira", "scripts/setup_jira_auth.py")
_clickhouse_refused_redirect, _clickhouse_refused_callback = _refuse_interactive_auth(
    "ClickHouse Cloud", "scripts/setup_clickhouse_auth.py"
)


CLICKHOUSE_REDIRECT_URI = "http://localhost:8766/callback"


def clickhouse_client_metadata() -> OAuthClientMetadata:
    return OAuthClientMetadata(
        redirect_uris=[CLICKHOUSE_REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name="rag-chatbot",
    )


def _clickhouse_auth() -> OAuthClientProvider | None:
    """Build an auth handler backed by the tokens scripts/setup_clickhouse_auth.py
    persisted, same FileTokenStorage pairing as Jira so the access token is
    transparently refreshed."""
    path = settings.clickhouse_token_file
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read ClickHouse token file %s", path)
        return None
    if not data.get("tokens"):
        return None
    return OAuthClientProvider(
        server_url=settings.clickhouse_mcp_url,
        client_metadata=clickhouse_client_metadata(),
        storage=FileTokenStorage(path),
        redirect_handler=_clickhouse_refused_redirect,
        callback_handler=_clickhouse_refused_callback,
    )


def _jira_auth() -> OAuthClientProvider | None:
    """Build an auth handler backed by the tokens scripts/setup_jira_auth.py
    persisted. Reusing the same OAuthClientProvider/FileTokenStorage pairing
    means the app transparently refreshes the access token using the stored
    refresh token instead of needing that script re-run every time the
    (short-lived) access token expires."""
    path = settings.jira_token_file
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read Jira token file %s", path)
        return None
    if not data.get("tokens"):
        return None
    return OAuthClientProvider(
        server_url=settings.jira_mcp_url,
        client_metadata=jira_client_metadata(),
        storage=FileTokenStorage(path),
        redirect_handler=_refused_redirect,
        callback_handler=_refused_callback,
    )


# mcp.client.stdio.get_default_environment() only inherits a small
# allowlist (HOME/LOGNAME/PATH/SHELL/TERM/USER on Unix) into local stdio
# server subprocesses, for security. That strips SSL_CERT_FILE, so those
# subprocesses fall back to their own certifi bundle instead of the OS trust
# store — unlike this process, which trusts the OS store via
# truststore.inject_into_ssl() in app/main.py. On networks where a corporate
# TLS-inspecting proxy issues a root CA that's only in the OS store (not
# certifi), that mismatch surfaces as CERTIFICATE_VERIFY_FAILED when a stdio
# server (or an aws_mcp_proxy_command like uvx fetching its package) makes an
# outbound HTTPS call. Explicitly forward a CA bundle to fix it.
_SYSTEM_CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/Fedora
    "/etc/ssl/cert.pem",  # macOS/Alpine
)


def _stdio_env(env: dict[str, str]) -> dict[str, str]:
    """Extend a stdio MCP server's env with a CA bundle so it can make
    outbound HTTPS calls on networks with a locally-trusted proxy CA."""
    cert_file = os.environ.get("SSL_CERT_FILE") or next(
        (path for path in _SYSTEM_CA_BUNDLES if Path(path).is_file()), None
    )
    if cert_file:
        env = {**env, "SSL_CERT_FILE": cert_file, "REQUESTS_CA_BUNDLE": cert_file}
    return env


def _build_config() -> dict[str, Any]:
    """Assemble a MultiServerMCPClient `connections` config from settings.
    Servers that are not configured/authenticated are simply omitted."""
    servers: dict[str, Any] = {}

    # jira_auth = _jira_auth()
    # if jira_auth:
    #     servers["jira"] = {
    #         "transport": "streamable_http",
    #         "url": settings.jira_mcp_url,
    #         "auth": jira_auth,
    #     }
    # else:
    #     logger.warning(
    #         "No Jira access token found at %s; Jira tools will be unavailable. "
    #         "Run scripts/setup_jira_auth.py to authenticate.",
    #         settings.jira_token_file,
    #     )

    # if settings.testrail_url and settings.testrail_username and settings.testrail_api_key:
    #     servers["testrail"] = {
    #         "transport": "stdio",
    #         "command": settings.testrail_mcp_command,
    #         "args": settings.testrail_mcp_args,
    #         "env": _stdio_env({
    #             "TESTRAIL_URL": settings.testrail_url,
    #             "TESTRAIL_USERNAME": settings.testrail_username,
    #             "TESTRAIL_API_KEY": settings.testrail_api_key,
    #         }),
    #     }
    # else:
    #     logger.warning(
    #         "TestRail is not fully configured (TESTRAIL_URL/USERNAME/API_KEY); "
    #         "TestRail tools will be unavailable."
    #     )
    # if settings.aws_access_key_id and settings.aws_secret_access_key:
    #     # AWS's official remote MCP Server only speaks OAuth+SigV4, which no
    #     # generic MCP client (including MultiServerMCPClient) can do natively.
    #     # AWS's own fix is mcp-proxy-for-aws: a local stdio bridge that signs
    #     # requests with our existing AWS creds and forwards them to the real
    #     # remote server (aws_mcp_endpoint). No S3 logic runs locally anymore;
    #     # this replaces app/s3/mcp_server.py entirely.
    #     proxy_args = [settings.aws_mcp_proxy_package, settings.aws_mcp_endpoint]
    #     region = settings.aws_mcp_region or settings.aws_default_region or "us-east-1"
    #     proxy_args += ["--region", region]
    #     if settings.aws_mcp_read_only:
    #         proxy_args.append("--read-only")
    #     servers["s3"] = {
    #         "transport": "stdio",
    #         "command": settings.aws_mcp_proxy_command,
    #         "args": proxy_args,
    #         "env": {
    #             "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
    #             "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
    #             "AWS_DEFAULT_REGION": region,
    #         },
    #     }
    # else:
    #     logger.warning(
    #         "AWS is not fully configured (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY); "
    #         "S3 tools will be unavailable."
    #     )

    # if settings.aws_access_key_id and settings.aws_secret_access_key and settings.redshift_database:
    #     servers["redshift"] = {
    #         "transport": "stdio",
    #         "command": settings.redshift_mcp_command,
    #         "args": settings.redshift_mcp_args,
    #         "env": {
    #             "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
    #             "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
    #             "AWS_DEFAULT_REGION": settings.aws_default_region or "us-east-1",
    #             "REDSHIFT_DATABASE": settings.redshift_database,
    #             "REDSHIFT_WORKGROUP_NAME": settings.redshift_workgroup_name or "",
    #             "REDSHIFT_CLUSTER_IDENTIFIER": settings.redshift_cluster_identifier or "",
    #             "REDSHIFT_DB_USER": settings.redshift_db_user or "",
    #             "REDSHIFT_SECRET_ARN": settings.redshift_secret_arn or "",
    #         },
    #     }

    # postgres_uri = settings.postgres_mcp_uri
    # if postgres_uri:
    #     # Our own local FastMCP server (app/postgres/mcp_server.py), run as a
    #     # stdio subprocess with the same interpreter as the app -- same
    #     # pattern as TestRail/Airflow above. postgres_mcp_uri builds a DSN
    #     # from PGHOST/PGUSER/PGPASSWORD/PGDATABASE if postgres_database_uri
    #     # isn't set directly.
    #     servers["postgres"] = {
    #         "transport": "stdio",
    #         "command": settings.postgres_mcp_command,
    #         "args": settings.postgres_mcp_args,
    #         "env": _stdio_env(
    #             {
    #                 "DATABASE_URI": postgres_uri,
    #                 "POSTGRES_MCP_ACCESS_MODE": settings.postgres_mcp_access_mode,
    #             }
    #         ),
    #     }
    # else:
    #     logger.warning(
    #         "Postgres is not fully configured (PGHOST/PGUSER/PGPASSWORD/PGDATABASE "
    #         "or POSTGRES_DATABASE_URI); Postgres tools will be unavailable."
    #     )

    # ClickHouse Cloud's own remote MCP server (single global endpoint,
    # OAuth 2.0 only — no API key/password auth). This replaces
    # app/clickhouse/mcp_server.py, which used clickhouse_url/user/password
    # directly; those REST creds are unused for MCP now but still power
    # scripts/seed_jira_mock.py.
    # clickhouse_auth = _clickhouse_auth()
    # if clickhouse_auth:
    #     servers["clickhouse"] = {
    #         "transport": "streamable_http",
    #         "url": settings.clickhouse_mcp_url,
    #         "auth": clickhouse_auth,
    #     }
    # else:
    #     logger.warning(
    #         "No ClickHouse Cloud access token found at %s; ClickHouse tools will be unavailable. "
    #         "Run scripts/setup_clickhouse_auth.py to authenticate.",
    #         settings.clickhouse_token_file,
    #     )

    grafana_token = settings.grafana_mcp_token
    if settings.grafana_url and grafana_token:
        # Grafana Labs' official open-source MCP server (grafana/mcp-grafana).
        # No vendor-hosted remote endpoint exists for a generic Grafana
        # instance, so — like Postgres/S3 above — the official package is
        # launched as a local stdio subprocess via uvx instead of hand-writing
        # tools against Grafana's REST API.
        servers["grafana"] = {
            "transport": "stdio",
            "command": settings.grafana_mcp_command,
            "args": [settings.grafana_mcp_package],
            "env": _stdio_env({
                "GRAFANA_URL": settings.grafana_url,
                "GRAFANA_SERVICE_ACCOUNT_TOKEN": grafana_token,
            }),
        }
    else:
        logger.warning(
            "Grafana is not fully configured (GRAFANA_URL/GRAFANA_API_TOKEN); "
            "Grafana tools will be unavailable."
        )

    # if settings.airflow_url:
    #     servers["airflow"] = {
    #         "transport": "stdio",
    #         "command": settings.airflow_mcp_command,
    #         "args": settings.airflow_mcp_args,
    #         "env": {
    #             "AIRFLOW_URL": settings.airflow_url,
    #             "AIRFLOW_USERNAME": settings.airflow_username or "",
    #             "AIRFLOW_PASSWORD": settings.airflow_password or "",
    #             "AIRFLOW_TOKEN": settings.airflow_token or "",
    #         },
    #     }

    # if settings.databricks_host and settings.databricks_token:
    #     # Databricks' own managed remote MCP server for SQL Warehouses
    #     # (https://<workspace>/api/2.0/mcp/sql), reached directly over
    #     # streamable_http with the same PAT used before, sent as a bearer
    #     # header. This replaces app/databricks/mcp_server.py; note the
    #     # managed endpoint only exposes SQL query tools (execute_sql/
    #     # poll_response) — there is no remote equivalent for the Jobs API,
    #     # so list_jobs/trigger_job are no longer available.
    #     servers["databricks"] = {
    #         "transport": "streamable_http",
    #         "url": settings.databricks_mcp_url,
    #         "headers": {"Authorization": f"Bearer {settings.databricks_token}"},
    #     }
    # else:
    #     logger.warning(
    #         "Databricks is not fully configured (DATABRICKS_HOST/DATABRICKS_TOKEN); "
    #         "Databricks tools will be unavailable."
    #     )


    return servers


# Strict exact-match allowlist of the only tools the agent may ever call.
# Binding the full ~114-tool catalog (every list/get/query/search/describe
# tool across every configured server, plus every write/trigger/delete tool)
# made bind_tools() slow and left the agent able to reach destructive or
# exploratory-discovery tools it was never meant to use. Anything not named
# here — including read-shaped tools like `postgres_list_tables` or
# `clickhouse_get_organizations` — is dropped. Names are post-prefix
# ("{server}_{toolName}", tool_name_prefix=True in _build_config), matching
# what load_mcp_tools actually produces. Adding a new tool requires adding
# its exact name here, on purpose.
STRICT_TOOL_ALLOWLIST: set[str] = {
    # ClickHouse (1 tool - direct SQL only)
    "clickhouse_run_select_query",

    # Jira (2 tools)
    "jira_searchJiraIssuesUsingJql",
    "jira_getJiraIssue",

    # PostgreSQL (2 tools)
    "postgres_query",
    "postgres_describe_table",

    # Databricks (2 tools)
    "databricks_execute_sql",
    "databricks_poll_response",

    # Airflow (3 tools)
    "airflow_list_dags",
    "airflow_list_dag_runs",
    "airflow_get_task_logs",

    # TestRail (5 tools)
    "testrail_get_projects",
    "testrail_get_runs",
    "testrail_get_tests",
    "testrail_get_results",
    "testrail_get_cases",

}
   # # S3 (2 tools)
    # "s3_list_objects",
    # "s3_get_object",

    # # Grafana (3 tools)
    # "grafana_get_dashboard_by_uid",
    # "grafana_search_dashboards",
    # "grafana_get_panel_data",

    # # Redshift (2 tools)
    # "redshift_query",
    # "redshift_describe_table",


class FastMCPToolProvider:
    """Owns one MultiServerMCPClient for the app's lifetime and the LangChain
    tools it exposes across all configured MCP servers."""

    def __init__(self) -> None:
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        connections = _build_config()
        if not connections:
            logger.warning("No MCP servers configured; LangGraph agent will run with no tools.")
            return

        logger.info("MultiServerMCPClient connecting to servers: %s", sorted(connections))
        t0 = time.monotonic()
        self._client = MultiServerMCPClient(connections, tool_name_prefix=True)
        self._exit_stack = AsyncExitStack()

        # Every configured server (streamable_http and stdio alike) gets one
        # session held open for the app's lifetime. get_tools() opens a fresh
        # MCP session (and, for stdio, a fresh subprocess) per tool call —
        # cheap for remote streamable_http servers but multi-second per call
        # for stdio ones launched via uvx (package resolve + Python startup),
        # which dominated end-to-end latency. Holding one session per server
        # open avoids paying that cost on every single tool invocation.
        tools: list[BaseTool] = []
        for name in connections:
            session = await self._exit_stack.enter_async_context(self._client.session(name))
            tools.extend(await load_mcp_tools(
                session,
                callbacks=self._client.callbacks,
                tool_interceptors=self._client.tool_interceptors,
                server_name=name,
                tool_name_prefix=True,
                handle_tool_errors=self._client.handle_tool_errors,
            ))
        loaded_count = len(tools)
        self._tools = [t for t in tools if t.name in STRICT_TOOL_ALLOWLIST]
        elapsed = time.monotonic() - t0

        per_server_counts: dict[str, int] = {}
        for t in self._tools:
            server = t.name.split("_", 1)[0]
            per_server_counts[server] = per_server_counts.get(server, 0) + 1

        logger.info(
            "MultiServerMCPClient connected in %.2fs: %d/%d tool(s) allowed after strict "
            "allowlist filtering, per server: %s",
            elapsed, len(self._tools), loaded_count, per_server_counts,
        )

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    async def disconnect(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._client = None
        self._tools = []


fastmcp_provider = FastMCPToolProvider()
