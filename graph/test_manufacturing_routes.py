import asyncio

from graph.manufacturing_graph import (
    manufacturing_graph,
)


TEST_CASES = [
    {
        "name": "RAG route",
        "thread_id": "route-test-rag",
        "query": (
            "AI4I 데이터셋에서 "
            "열 방산 고장은 어떤 조건에서 발생하는가?"
        ),
    },
    {
        "name": "Prediction tool route",
        "thread_id": "route-test-tool",
        "query": (
            "Product Type L이고, "
            "Air temperature 301.0 K, "
            "Process temperature 310.5 K, "
            "Rotational speed 1300 rpm, "
            "Torque 65.0 Nm, "
            "Tool wear 200 min일 때 "
            "고장 위험을 예측해줘."
        ),
    },
    {
        "name": "Direct route",
        "thread_id": "route-test-direct",
        "query": (
            "안녕. 너는 어떤 제조 분석을 할 수 있어?"
        ),
    },
]


async def main():

    for case in TEST_CASES:

        print("\n" + "=" * 100)
        print(case["name"])
        print("=" * 100)

        print(
            "QUERY:",
            case["query"],
        )

        result = await manufacturing_graph.ainvoke(
            {
                "query": case["query"],
            },
            config={
                "configurable": {
                    "thread_id": case["thread_id"],
                }
            },
        )

        print("\nRESULT")
        print("-" * 100)

        print(
            "route:",
            result.get("route"),
        )

        print(
            "tool_name:",
            result.get("tool_name"),
        )

        print(
            "tool_arguments:",
            result.get("tool_arguments"),
        )

        print("\nanswer:")
        print(
            result.get("answer")
        )


if __name__ == "__main__":
    asyncio.run(main())