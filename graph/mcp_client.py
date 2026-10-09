import asyncio
import sys

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import (
    ClientSession,
    StdioServerParameters,
)

from mcp.client.stdio import stdio_client
import json


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# MCP Server Configuration
# ============================================================

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[
        "-m",
        "mcp_server.server",
    ],
    cwd=str(PROJECT_ROOT),
)


# ============================================================
# MCP Session
# ============================================================

@asynccontextmanager
async def open_mcp_session():
    """
    기존 Manufacturing MCP Server를 subprocess로 실행하고
    MCP ClientSession을 생성한다.
    """

    async with stdio_client(
        SERVER_PARAMS
    ) as (
        read_stream,
        write_stream,
    ):

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            yield session


# ============================================================
# Tool Discovery
# ============================================================

async def list_mcp_tools():
    """
    MCP Server가 제공하는 tool 목록을 조회한다.
    """

    async with open_mcp_session() as session:

        result = await session.list_tools()

        return result.tools


# ============================================================
# Tool Execution
# ============================================================

async def call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
):
    """
    MCP tool 하나를 실행한다.
    """

    async with open_mcp_session() as session:

        result = await session.call_tool(
            tool_name,
            arguments=arguments,
        )

        return result


async def call_mcp_tool_data(
    tool_name: str,
    arguments: dict[str, Any],
):
    """
    MCP Tool을 실행하고
    LangGraph에서 사용하기 쉬운 Python dict를 반환한다.
    """

    result = await call_mcp_tool(
        tool_name=tool_name,
        arguments=arguments,
    )

    if result.is_error:
        raise RuntimeError(
            f"MCP tool failed: {tool_name}\n"
            f"{result}"
        )

    # 1순위: MCP structured content
    if result.structured_content is not None:
        return result.structured_content

    # fallback:
    # structured content가 없는 tool도 대응
    if result.content:
        first = result.content[0]

        text = getattr(
            first,
            "text",
            None,
        )

        if text:
            try:
                return json.loads(text)

            except json.JSONDecodeError:
                return {
                    "text": text,
                }

    return {}


# ============================================================
# Manual Test
# ============================================================

async def main():

    print(
        "Connecting to Manufacturing MCP Server..."
    )

    tools = await list_mcp_tools()

    print(
        f"\nConnected MCP tools: "
        f"{len(tools)}"
    )

    for index, tool in enumerate(
        tools,
        start=1,
    ):
        print("\n" + "=" * 80)

        print(
            f"{index}. {tool.name}"
        )

        if tool.description:
            print("\nDescription:")
            print(tool.description.strip())

        print("\nInput Schema:")

        input_schema = getattr(
            tool,
            "inputSchema",
            None,
        )

        if input_schema is None:
            input_schema = getattr(
                tool,
                "input_schema",
                None,
            )

        print(input_schema)

if __name__ == "__main__":
    asyncio.run(main())