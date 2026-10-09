import asyncio

from graph.mcp_client import call_mcp_tool


async def main():

    print("Calling MCP tool: process_summary")

    result = await call_mcp_tool(
        tool_name="process_summary",
        arguments={
            "product_type": "L",
        },
    )

    print("\n" + "=" * 80)
    print("RAW MCP RESULT")
    print("=" * 80)

    print("type:", type(result))
    print(result)


if __name__ == "__main__":
    asyncio.run(main())