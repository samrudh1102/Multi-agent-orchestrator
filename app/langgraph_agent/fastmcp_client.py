"""MCP client to Jira (Atlassian's hosted remote MCP server) and TestRail (a
self-hosted community stdio server), plus the S3/Redshift/ClickHouse/Airflow/
Databricks stdio servers, exposed as LangChain tools for the LangGraph agent
(app/langgraph_agent/graph_agent.py).

Backed by langchain-mcp-adapters' MultiServerMCPClient: it takes a single
multi-server `connections` config and its `get_tools()` returns ready-made
LangChain tools (a fresh MCP session is opened per tool call), with each
server's tool names prefixed by server (e.g. `jira_getJiraIssue`,
`testrail_getCase`) so calls route back to the right server automatically.

Jira auth: scripts/setup_jira_auth.py is a one-time interactive login that
persists an access + refresh token pair to settings.jira_token_file. The
running app reuses that same OAuthClientProvider/FileTokenStorage pairing as
its httpx auth, so it transparently refreshes the access token (Atlassian's
expire hourly) using the stored refresh token instead of needing the script
re-run. Re-running the script is only needed if the refresh token itself is
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


async def _refused_redirect(url: str) -> None:
    # The running app must never block on an interactive browser consent
    # flow; that only happens once, via scripts/setup_jira_auth.py. If we get
    # here, the stored refresh token was rejected and a human needs to log in
    # again.
    raise RuntimeError(
        "Jira OAuth requires interactive re-authorization. "
        "Run scripts/setup_jira_auth.py to log in again."
    )


async def _refused_callback() -> tuple[str, str | None]:
    raise RuntimeError(
        "Jira OAuth requires interactive re-authorization. "
        "Run scripts/setup_jira_auth.py to log in again."
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
            servers["s3"] = {
                "transport": "stdio",
                "command": settings.s3_mcp_command,
                "args": settings.s3_mcp_args,
                "env": {
                    "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
                    "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
                    "AWS_DEFAULT_REGION": settings.aws_default_region or "us-east-1",
                },
            }
  
        
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
 
    if settings.clickhouse_url:
        servers["clickhouse"] = {
            "transport": "stdio",
            "command": settings.clickhouse_mcp_command,
            "args": settings.clickhouse_mcp_args,
            "env": {
                "CLICKHOUSE_URL": settings.clickhouse_url,
                "CLICKHOUSE_USER": settings.clickhouse_user or "default",
                "CLICKHOUSE_PASSWORD": settings.clickhouse_password or "",
                "CLICKHOUSE_DATABASE": settings.clickhouse_database or "default",
            },
        }
 
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
        servers["databricks"] = {
            "transport": "stdio",
            "command": settings.databricks_mcp_command,
            "args": settings.databricks_mcp_args,
            "env": {
                "DATABRICKS_HOST": settings.databricks_host,
                "DATABRICKS_TOKEN": settings.databricks_token,
                "DATABRICKS_WAREHOUSE_ID": settings.databricks_warehouse_id or "",
            },
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
