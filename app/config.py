import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to our local FastMCP TestRail server (app/testrail/mcp_server.py).
_TESTRAIL_MCP_SERVER = Path(__file__).resolve().parent / "testrail" / "mcp_server.py"
_S3_MCP_SERVER = Path(__file__).resolve().parent / "s3" / "mcp_server.py"
_REDSHIFT_MCP_SERVER = Path(__file__).resolve().parent / "redshift" / "mcp_server.py"
_CLICKHOUSE_MCP_SERVER = Path(__file__).resolve().parent / "clickhouse" / "mcp_server.py"
_AIRFLOW_MCP_SERVER = Path(__file__).resolve().parent / "airflow" / "mcp_server.py"
_DATABRICKS_MCP_SERVER = Path(__file__).resolve().parent / "databricks" / "mcp_server.py"

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
    # S3 MCP
    # -------------------------------------------------------------------------

    # Command to run the S3 MCP server. We support 'uvx' (Python-based) or 'npx' (JS-based).
    s3_mcp_command: str = sys.executable
    s3_mcp_args: list[str] = [str(_S3_MCP_SERVER)]

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_default_region: str | None = None
    
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

    # Launch our local FastMCP ClickHouse server (app/clickhouse/mcp_server.py) over stdio
    # using the same interpreter that runs the app, so it always picks up
    # this venv's dependencies.
    clickhouse_mcp_command: str = sys.executable
    clickhouse_mcp_args: list[str] = [str(_CLICKHOUSE_MCP_SERVER)]

    clickhouse_url: str | None = None
    clickhouse_user: str | None = None
    clickhouse_password: str | None = None
    clickhouse_database: str | None = None

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
    # Databricks MCP
    # -------------------------------------------------------------------------

    databricks_mcp_command: str = sys.executable
    databricks_mcp_args: list[str] = [str(_DATABRICKS_MCP_SERVER)]

    databricks_host: str | None = None
    databricks_toke: str | None = None
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
    def jira_token_file(self) -> Path:
        p = Path(self.jira_token_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()