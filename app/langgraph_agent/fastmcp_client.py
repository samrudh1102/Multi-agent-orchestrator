"""MCP client to Jira (Atlassian's hosted remote MCP server), Databricks
(Databricks' own hosted remote MCP server), S3 (AWS's official remote MCP
Server, reached via AWS's mcp-proxy-for-aws bridge), and ClickHouse
(ClickHouse Cloud's own hosted remote MCP server), plus the TestRail/
Redshift/Airflow local stdio servers, exposed as LangChain tools for the
LangGraph agent (app/langgraph_agent/graph_agent.py).

S3, Databricks and ClickHouse used to run as local stdio servers
(app/s3/mcp_server.py, app/databricks/mcp_server.py,
app/clickhouse/mcp_server.py); those files are kept in the repo for
reference but are no longer launched — see _build_config below.

Backed by langchain-mcp-adapters' MultiServerMCPClient: it takes a single
multi-server `connections` config and its `get_tools()` returns ready-made
LangChain tools (a fresh MCP session is opened per tool call), with each
server's tool names prefixed by server (e.g. `jira_getJiraIssue`,
`testrail_getCase`) so calls route back to the right server automatically.

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
from pathlib import Path
from typing import Any
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
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


def _build_config() -> dict[str, Any]:
    """Assemble a MultiServerMCPClient `connections` config from settings.
    Servers that are not configured/authenticated are simply omitted."""
    servers: dict[str, Any] = {}

    jira_auth = _jira_auth()
    if jira_auth:
        servers["jira"] = {
            "transport": "streamable_http",
            "url": settings.jira_mcp_url,
            "auth": jira_auth,
        }
    else:
        logger.warning(
            "No Jira access token found at %s; Jira tools will be unavailable. "
            "Run scripts/setup_jira_auth.py to authenticate.",
            settings.jira_token_file,
        )

    if settings.testrail_url and settings.testrail_username and settings.testrail_api_key:
        servers["testrail"] = {
            "transport": "stdio",
            "command": settings.testrail_mcp_command,
            "args": settings.testrail_mcp_args,
            "env": {
                "TESTRAIL_URL": settings.testrail_url,
                "TESTRAIL_USERNAME": settings.testrail_username,
                "TESTRAIL_API_KEY": settings.testrail_api_key,
            },
        }
    else:
        logger.warning(
            "TestRail is not fully configured (TESTRAIL_URL/USERNAME/API_KEY); "
            "TestRail tools will be unavailable."
        )
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        # AWS's official remote MCP Server only speaks OAuth+SigV4, which no
        # generic MCP client (including MultiServerMCPClient) can do natively.
        # AWS's own fix is mcp-proxy-for-aws: a local stdio bridge that signs
        # requests with our existing AWS creds and forwards them to the real
        # remote server (aws_mcp_endpoint). No S3 logic runs locally anymore;
        # this replaces app/s3/mcp_server.py entirely.
        proxy_args = [settings.aws_mcp_proxy_package, settings.aws_mcp_endpoint]
        region = settings.aws_mcp_region or settings.aws_default_region or "us-east-1"
        proxy_args += ["--region", region]
        if settings.aws_mcp_read_only:
            proxy_args.append("--read-only")
        servers["s3"] = {
            "transport": "stdio",
            "command": settings.aws_mcp_proxy_command,
            "args": proxy_args,
            "env": {
                "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
                "AWS_DEFAULT_REGION": region,
            },
        }
    else:
        logger.warning(
            "AWS is not fully configured (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY); "
            "S3 tools will be unavailable."
        )

    if settings.aws_access_key_id and settings.aws_secret_access_key and settings.redshift_database:
        servers["redshift"] = {
            "transport": "stdio",
            "command": settings.redshift_mcp_command,
            "args": settings.redshift_mcp_args,
            "env": {
                "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
                "AWS_DEFAULT_REGION": settings.aws_default_region or "us-east-1",
                "REDSHIFT_DATABASE": settings.redshift_database,
                "REDSHIFT_WORKGROUP_NAME": settings.redshift_workgroup_name or "",
                "REDSHIFT_CLUSTER_IDENTIFIER": settings.redshift_cluster_identifier or "",
                "REDSHIFT_DB_USER": settings.redshift_db_user or "",
                "REDSHIFT_SECRET_ARN": settings.redshift_secret_arn or "",
            },
        }
 
    # ClickHouse Cloud's own remote MCP server (single global endpoint,
    # OAuth 2.0 only — no API key/password auth). This replaces
    # app/clickhouse/mcp_server.py, which used clickhouse_url/user/password
    # directly; those REST creds are unused for MCP now but still power
    # scripts/seed_jira_mock.py.
    clickhouse_auth = _clickhouse_auth()
    if clickhouse_auth:
        servers["clickhouse"] = {
            "transport": "streamable_http",
            "url": settings.clickhouse_mcp_url,
            "auth": clickhouse_auth,
        }
    else:
        logger.warning(
            "No ClickHouse Cloud access token found at %s; ClickHouse tools will be unavailable. "
            "Run scripts/setup_clickhouse_auth.py to authenticate.",
            settings.clickhouse_token_file,
        )

    if settings.airflow_url:
        servers["airflow"] = {
            "transport": "stdio",
            "command": settings.airflow_mcp_command,
            "args": settings.airflow_mcp_args,
            "env": {
                "AIRFLOW_URL": settings.airflow_url,
                "AIRFLOW_USERNAME": settings.airflow_username or "",
                "AIRFLOW_PASSWORD": settings.airflow_password or "",
            },
        }

    if settings.databricks_host and settings.databricks_token:
        # Databricks' own managed remote MCP server for SQL Warehouses
        # (https://<workspace>/api/2.0/mcp/sql), reached directly over
        # streamable_http with the same PAT used before, sent as a bearer
        # header. This replaces app/databricks/mcp_server.py; note the
        # managed endpoint only exposes SQL query tools (execute_sql/
        # poll_response) — there is no remote equivalent for the Jobs API,
        # so list_jobs/trigger_job are no longer available.
        servers["databricks"] = {
            "transport": "streamable_http",
            "url": settings.databricks_mcp_url,
            "headers": {"Authorization": f"Bearer {settings.databricks_token}"},
        }
    else:
        logger.warning(
            "Databricks is not fully configured (DATABRICKS_HOST/DATABRICKS_TOKEN); "
            "Databricks tools will be unavailable."
        )


    return servers


class FastMCPToolProvider:
    """Owns one MultiServerMCPClient for the app's lifetime and the LangChain
    tools it exposes across all configured MCP servers."""

    def __init__(self) -> None:
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []

    async def connect(self) -> None:
        connections = _build_config()
        if not connections:
            logger.warning("No MCP servers configured; LangGraph agent will run with no tools.")
            return

        # MultiServerMCPClient opens a fresh MCP session per tool call, so there
        # is no long-lived connection to enter/exit — the client is just the
        # config holder. tool_name_prefix=True prefixes each tool with its
        # server name (e.g. `jira_getJiraIssue`) so names stay unique and route
        # back to the right server.
        self._client = MultiServerMCPClient(connections, tool_name_prefix=True)
        self._tools = await self._client.get_tools()
        logger.info(
            "MultiServerMCPClient connected: %d tool(s) available: %s",
            len(self._tools),
            ", ".join(t.name for t in self._tools) or "(none)",
        )

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    async def disconnect(self) -> None:
        # Sessions are per-call and closed automatically; nothing to tear down.
        self._client = None
        self._tools = []


fastmcp_provider = FastMCPToolProvider()
