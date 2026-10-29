"""One-time interactive OAuth login for ClickHouse Cloud's remote MCP server.

Run this manually (`python scripts/setup_clickhouse_auth.py`) once before
starting the app with ENABLE_TOOLS=true. It opens your browser for
ClickHouse Cloud's consent screen, completes the OAuth 2.0 + PKCE exchange,
and persists the resulting tokens to settings.clickhouse_token_path. The
running FastAPI app never does this interactive step itself — it only loads
and refreshes the saved tokens.

The remote MCP server must be enabled for your ClickHouse Cloud service
first: Console -> your service -> Connect -> MCP -> enable.
"""
import asyncio
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client

from app.config import settings
from app.langgraph_agent.fastmcp_client import (
    CLICKHOUSE_REDIRECT_URI,
    FileTokenStorage,
    clickhouse_client_metadata,
)


class _CallbackServer:
    """One-shot localhost HTTP server that catches the OAuth redirect."""

    def __init__(self, host: str, port: int, path: str):
        self._path = path
        self._result: tuple[str | None, str | None] | None = None
        self._httpd = HTTPServer((host, port), self._make_handler())

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                params = parse_qs(urlparse(self.path).query)
                outer._result = (params.get("code", [None])[0], params.get("state", [None])[0])
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body>Authorized. You can close this tab.</body></html>")

            def log_message(self, *args) -> None:
                pass

        return Handler

    async def wait_for_callback(self) -> tuple[str, str | None]:
        loop = asyncio.get_event_loop()
        while self._result is None:
            await loop.run_in_executor(None, self._httpd.handle_request)
        return self._result


async def _redirect_handler(url: str) -> None:
    print(f"\nOpening your browser for ClickHouse Cloud consent. If it doesn't open, visit:\n{url}\n")
    webbrowser.open(url)


async def main() -> None:
    redirect_url = urlparse(CLICKHOUSE_REDIRECT_URI)
    callback_server = _CallbackServer(redirect_url.hostname, redirect_url.port, redirect_url.path)
    storage = FileTokenStorage(settings.clickhouse_token_file)

    oauth = OAuthClientProvider(
        server_url=settings.clickhouse_mcp_url,
        client_metadata=clickhouse_client_metadata(),
        storage=storage,
        redirect_handler=_redirect_handler,
        callback_handler=callback_server.wait_for_callback,
    )

    async with httpx.AsyncClient(auth=oauth, timeout=60) as http_client:
        async with streamable_http_client(settings.clickhouse_mcp_url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

    print(f"ClickHouse Cloud MCP login complete. Tokens saved to {settings.clickhouse_token_file}")


if __name__ == "__main__":
    asyncio.run(main())
