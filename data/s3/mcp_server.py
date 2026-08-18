"""Local FastMCP server exposing AWS S3 as MCP tools.

This runs locally over stdio, completely bypassing GitHub and external package
registries. It uses standard `boto3` to interact with S3.
"""

from __future__ import annotations

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
    name="s3",
    instructions=(
        "Tools for AWS S3: list buckets, list objects, and get object contents. "
        "Credentials should be configured via standard AWS environment variables."
    ),
)

def _get_client():
    return boto3.client("s3")

@mcp.tool
async def list_buckets() -> list[str]:
    """List all accessible S3 bucket names in your AWS account."""
    try:
        client = _get_client()
        response = client.list_buckets()
        return [bucket["Name"] for bucket in response.get("Buckets", [])]
    except (BotoCoreError, ClientError) as exc:
        raise ToolError(f"S3 Error listing buckets: {exc}") from exc

@mcp.tool
async def list_objects(bucket: str, prefix: str = "", max_keys: int = 100) -> list[dict]:
    """List objects inside an S3 bucket, with an optional path prefix."""
    try:
        client = _get_client()
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
        results = []
        for obj in response.get("Contents", []):
            results.append({
                "Key": obj["Key"],
                "Size": obj["Size"],
                "LastModified": str(obj["LastModified"])
            })
        return results
    except (BotoCoreError, ClientError) as exc:
        raise ToolError(f"S3 Error listing objects in bucket '{bucket}': {exc}") from exc

@mcp.tool
async def get_object(bucket: str, key: str) -> str:
    """Read the text content of a specific S3 object (file)."""
    try:
        client = _get_client()
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")
    except (BotoCoreError, ClientError) as exc:
        raise ToolError(f"S3 Error reading object '{key}' from bucket '{bucket}': {exc}") from exc

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()