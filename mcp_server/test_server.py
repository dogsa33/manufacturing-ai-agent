import asyncio
from pprint import pprint

from mcp import Client

from mcp_server.server import mcp


def print_tool_result(result) -> None:
    """
    MCP Tool 호출 결과를 확인하기 위한 공통 출력 함수.
    """

    print("\nis_error:")
    pprint(result.is_error)

    print("\nstructured_content:")
    pprint(result.structured_content)

    print("\ncontent:")
    pprint(result.content)


async def main():

    # =====================================================
    # MCP Client / Server in-process 연결
    # =====================================================

    async with Client(
        mcp,
        raise_exceptions=True,
    ) as client:

        # =================================================
        # 1. MCP 연결 확인
        # =================================================

        print("=" * 60)
        print("MCP CONNECTION")
        print("=" * 60)

        print(
            "Protocol version:",
            client.protocol_version,
        )


        # =================================================
        # 2. Tool 목록 및 Schema 확인
        # =================================================

        tools_result = await client.list_tools()

        print("\n" + "=" * 60)
        print("AVAILABLE MCP TOOLS")
        print("=" * 60)

        for tool in tools_result.tools:

            print("\nTool:", tool.name)

            print("Description:")
            print(tool.description)

            print("Input schema:")
            pprint(tool.input_schema)

            print("Output schema:")
            pprint(tool.output_schema)


        # =================================================
        # 3. Process Summary Tool
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 1 - process_summary")
        print("=" * 60)

        result = await client.call_tool(
            "process_summary",
            {
                "product_type": "L",
            },
        )

        print_tool_result(result)


        # =================================================
        # 4. Condition Comparison Tool
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 2 - compare_process_condition")
        print("=" * 60)

        result = await client.call_tool(
            "compare_process_condition",
            {
                "feature": "Torque",
            },
        )

        print_tool_result(result)


        # =================================================
        # 5. Failure Type Summary Tool
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 3 - failure_type_summary")
        print("=" * 60)

        result = await client.call_tool(
            "failure_type_summary",
            {},
        )

        print_tool_result(result)


        # =================================================
        # 6. Prediction Tool
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 4 - predict_machine_failure")
        print("=" * 60)

        result = await client.call_tool(
            "predict_machine_failure",
            {
                "product_type": "L",
                "air_temperature": 301.0,
                "process_temperature": 310.5,
                "rotational_speed": 1300,
                "torque": 65.0,
                "tool_wear": 200,
            },
        )

        print_tool_result(result)


        # =================================================
        # 7. Explanation Tool
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 5 - explain_machine_failure")
        print("=" * 60)

        result = await client.call_tool(
            "explain_machine_failure",
            {
                "product_type": "L",
                "air_temperature": 301.0,
                "process_temperature": 310.5,
                "rotational_speed": 1300,
                "torque": 65.0,
                "tool_wear": 200,
            },
        )

        print_tool_result(result)


        # =================================================
        # 8. Feature Distribution Chart
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 6 - feature_distribution_chart")
        print("=" * 60)

        result = await client.call_tool(
            "feature_distribution_chart",
            {
                "feature": "Torque",
            },
        )

        print_tool_result(result)


        # =================================================
        # 9. Failure Rate by Type Chart
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 7 - failure_rate_by_type_chart")
        print("=" * 60)

        result = await client.call_tool(
            "failure_rate_by_type_chart",
            {},
        )

        print_tool_result(result)


        # =================================================
        # 10. Risk Driver Chart
        # =================================================

        print("\n" + "=" * 60)
        print("TEST 8 - risk_driver_chart")
        print("=" * 60)

        result = await client.call_tool(
            "risk_driver_chart",
            {
                "product_type": "L",
                "air_temperature": 301.0,
                "process_temperature": 310.5,
                "rotational_speed": 1300,
                "torque": 65.0,
                "tool_wear": 200,
            },
        )

        print_tool_result(result)


        # =================================================
        # Finish
        # =================================================

        print("\n" + "=" * 60)
        print("ALL MCP TESTS COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())