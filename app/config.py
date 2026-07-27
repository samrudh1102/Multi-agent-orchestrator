import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to our local FastMCP TestRail server (app/testrail/mcp_server.py).
_TESTRAIL_MCP_SERVER = Path(__file__).resolve().parent / "testrail" / "mcp_server.py"
_REDSHIFT_MCP_SERVER = Path(__file__).resolve().parent / "redshift" / "mcp_server.py"
_AIRFLOW_MCP_SERVER = Path(__file__).resolve().parent / "airflow" / "mcp_server.py"
# NOTE: app/s3/mcp_server.py, app/databricks/mcp_server.py and
# app/clickhouse/mcp_server.py still exist for reference but are no longer
# launched — S3, Databricks and ClickHouse now go through their vendors'
# official remote MCP servers configured below instead of these local ones.

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Google Gemini
    # -------------------------------------------------------------------------

    google_api_key: str

    # Embedding Model
    embedding_model_name: str = "models/gemini-embedding-2"

    # Generation Model
    llm_model_name: str = "gemini-3.1-flash-lite"

    # -------------------------------------------------------------------------
    # RAG Configuration
    # -------------------------------------------------------------------------

    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k: int = 4

    upload_dir: str = "data/uploads"
    index_dir: str = "data/index"

    max_new_tokens: int = 1545
    temperature: float = 0.3

    # -------------------------------------------------------------------------
    # MCP Tool Integration
    # -------------------------------------------------------------------------

    enable_tools: bool = False
    max_tool_iterations: int = 5

    # Gemini will be used for tool calling as well
    tool_model_name: str = "gemini-3.1-flash-lite"

    # -------------------------------------------------------------------------
    # Jira MCP
    # -------------------------------------------------------------------------

    jira_mcp_url: str = "https://mcp.atlassian.com/v1/mcp/authv2"
    jira_token_path: str = "data/mcp_tokens/jira.json"

    # -------------------------------------------------------------------------
    # TestRail MCP
    # -------------------------------------------------------------------------

    # Launch our local FastMCP TestRail server (app/testrail/mcp_server.py)
    # over stdio using the same interpreter that runs the app, so it always
    # picks up this venv's dependencies. sys.executable is absolute, so it does
    # not depend on PATH or the subprocess working directory.
    testrail_mcp_command: str = sys.executable
    testrail_mcp_args: list[str] = [str(_TESTRAIL_MCP_SERVER)]

    testrail_url: str | None = None
    
    testrail_username: str | None = None
    testrail_api_key: str | None = None

    # -------------------------------------------------------------------------
    # S3 MCP (AWS's official remote MCP Server, reached via AWS's official
    # mcp-proxy-for-aws bridge — see app/langgraph_agent/fastmcp_client.py)
    # -------------------------------------------------------------------------

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_default_region: str | None = None

    # mcp-proxy-for-aws is AWS's own published stdio<->SigV4 bridge: the MCP
    # spec only supports OAuth, but the AWS MCP Server is authenticated with
    # IAM SigV4, so AWS ships this proxy instead of a plain streamable_http
    # endpoint. It signs requests locally using the AWS_ACCESS_KEY_ID/
    # AWS_SECRET_ACCESS_KEY/AWS_DEFAULT_REGION above and forwards them to
    # aws_mcp_endpoint. No local S3 business logic runs here anymore.
    aws_mcp_proxy_command: str = "uvx"
    aws_mcp_proxy_package: str = "mcp-proxy-for-aws@latest"
    aws_mcp_endpoint: str = "https://aws-mcp.us-east-1.api.aws/mcp"
    aws_mcp_region: str | None = None
    # The official server exposes a generic `call_aws` tool covering all AWS
    # APIs (not just S3). Keep it read-only since our previous local S3 tools
    # (list_buckets/list_objects/get_object) were read-only too.
    aws_mcp_read_only: bool = True

    # -------------------------------------------------------------------------
    # Redshift MCP
    # -------------------------------------------------------------------------

    redshift_mcp_command: str = sys.executable
    redshift_mcp_args: list[str] = [str(_REDSHIFT_MCP_SERVER)]

    redshift_database: str | None = None
    redshift_workgroup_name: str | None = None
    redshift_cluster_identifier: str | None = None
    redshift_db_user: str | None = None
    redshift_secret_arn: str | None = None

    # ClickHouse MCP
    # -------------------------------------------------------------------------

    # Direct REST credentials for your ClickHouse Cloud service. No longer
    # used to launch the local MCP server (app/clickhouse/mcp_server.py,
    # kept in-repo for reference but unused) — the MCP client now talks to
    # ClickHouse Cloud's own remote MCP server instead (see below). Still
    # used directly by scripts/seed_jira_mock.py.
    clickhouse_url: str | None = None
    clickhouse_user: str | None = None
    clickhouse_password: str | None = None
    clickhouse_database: str | None = None

    # ClickHouse Cloud's remote MCP server (single global endpoint, private
    # preview). Auth is interactive OAuth 2.0 only (no API key/password
    # option), so this mirrors the Jira setup: run
    # scripts/setup_clickhouse_auth.py once to log in, and the running app
    # only loads/refreshes the saved tokens from clickhouse_token_path.
    clickhouse_mcp_url: str = "https://mcp.clickhouse.cloud/mcp"
    clickhouse_token_path: str = "data/mcp_tokens/clickhouse.json"

    # -------------------------------------------------------------------------
    # Airflow MCP
    # -------------------------------------------------------------------------

    # Launch our local FastMCP Airflow server (app/airflow/mcp_server.py) over stdio
    # using the same interpreter that runs the app, so it always picks up
    # this venv's dependencies.
    airflow_mcp_command: str = sys.executable
    airflow_mcp_args: list[str] = [str(_AIRFLOW_MCP_SERVER)]

    airflow_url: str | None = None
    airflow_username: str | None = None
    airflow_password: str | None = None

    # -------------------------------------------------------------------------
    # Databricks MCP (Databricks' own managed remote MCP server, reached over
    # streamable_http — see app/langgraph_agent/fastmcp_client.py)
    # -------------------------------------------------------------------------

    databricks_host: str | None = None
    databricks_token: str | None = None
    # No longer sent by the client: Databricks' managed /api/2.0/mcp/sql
    # endpoint picks the SQL Warehouse server-side (per-user default
    # warehouse, configurable in the Databricks workspace UI) rather than
    # accepting a warehouse_id from the MCP client.
    databricks_warehouse_id: str | None = None

    # -------------------------------------------------------------------------
    # Helper Paths
    # -------------------------------------------------------------------------

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def index_path(self) -> Path:
        p = Path(self.index_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def databricks_mcp_url(self) -> str | None:
        if not self.databricks_host:
            return None
        return f"{self.databricks_host.rstrip('/')}/api/2.0/mcp/sql"

    @property
    def clickhouse_token_file(self) -> Path:
        p = Path(self.clickhouse_token_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def jira_token_file(self) -> Path:
        p = Path(self.jira_token_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()