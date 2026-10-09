import asyncio

from graph.manufacturing_graph import (
    manufacturing_graph,
)


async def main():

    # ========================================================
    # 같은 thread_id = 같은 대화
    # ========================================================

    config = {
        "configurable": {
            "thread_id": "memory-test-001",
        }
    }

    # ========================================================
    # Turn 1
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TURN 1"
    )

    print(
        "=" * 100
    )

    result1 = await manufacturing_graph.ainvoke(
        {
            "query": (
                "Product Type L의 "
                "고장 위험을 예측해줘."
            )
        },
        config=config,
    )

    print(
        "\nANSWER 1:"
    )

    print(
        result1.get(
            "answer"
        )
    )

    print(
        "\nPENDING TOOL:"
    )

    print(
        result1.get(
            "pending_tool_name"
        )
    )

    print(
        "\nPENDING ARGUMENTS:"
    )

    print(
        result1.get(
            "pending_tool_arguments"
        )
    )

    # ========================================================
    # Turn 2
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TURN 2"
    )

    print(
        "=" * 100
    )

    result2 = await manufacturing_graph.ainvoke(
        {
            "query": (
                "Air temperature는 301.0 K, "
                "Process temperature는 310.5 K, "
                "Rotational speed는 1300 rpm, "
                "Torque는 65.0 Nm, "
                "Tool wear는 200 min이야."
            )
        },
        config=config,
    )

    print(
        "\nFINAL TOOL:"
    )

    print(
        result2.get(
            "tool_name"
        )
    )

    print(
        "\nFINAL ARGUMENTS:"
    )

    print(
        result2.get(
            "tool_arguments"
        )
    )

    print(
        "\nANSWER 2:"
    )

    print(
        result2.get(
            "answer"
        )
    )

    print(
        "\nPENDING AFTER SUCCESS:"
    )

    print(
        result2.get(
            "pending_tool_arguments"
        )
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )