import asyncio
import sys

from graph.persistent_graph import (
    CHECKPOINT_DB_PATH,
    invoke_persistent_graph,
)


THREAD_ID = (
    "sqlite-memory-test-001"
)


# ============================================================
# Phase 1
# 일부 입력만 제공
# ============================================================

async def phase1():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SQLITE MEMORY TEST - PHASE 1"
    )

    print(
        "=" * 100
    )

    print(
        "DB:",
        CHECKPOINT_DB_PATH,
    )

    result = await invoke_persistent_graph(
        query=(
            "Product Type L의 "
            "고장 위험을 예측해줘."
        ),
        thread_id=THREAD_ID,
    )

    print(
        "\nANSWER"
    )

    print(
        result.get(
            "answer"
        )
    )

    print(
        "\nPENDING TOOL"
    )

    print(
        result.get(
            "pending_tool_name"
        )
    )

    print(
        "\nPENDING ARGUMENTS"
    )

    print(
        result.get(
            "pending_tool_arguments"
        )
    )

    print(
        "\nPENDING MISSING ARGUMENTS"
    )

    print(
        result.get(
            "pending_missing_arguments"
        )
    )


# ============================================================
# Phase 2
# 새 Python 프로세스에서 나머지 값만 제공
# ============================================================

async def phase2():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SQLITE MEMORY TEST - PHASE 2"
    )

    print(
        "=" * 100
    )

    print(
        "DB:",
        CHECKPOINT_DB_PATH,
    )

    result = await invoke_persistent_graph(
        query=(
            "Air temperature 301.0 K, "
            "Process temperature 310.5 K, "
            "Rotational speed 1300 rpm, "
            "Torque 65.0 Nm, "
            "Tool wear 200 min이야."
        ),
        thread_id=THREAD_ID,
    )

    print(
        "\nANSWER"
    )

    print(
        result.get(
            "answer"
        )
    )

    print(
        "\nTOOL"
    )

    print(
        result.get(
            "tool_name"
        )
    )

    print(
        "\nTOOL ARGUMENTS"
    )

    print(
        result.get(
            "tool_arguments"
        )
    )

    print(
        "\nPENDING AFTER SUCCESS"
    )

    print(
        result.get(
            "pending_tool_name"
        )
    )

    print(
        result.get(
            "pending_tool_arguments"
        )
    )

    print(
        result.get(
            "pending_missing_arguments"
        )
    )


# ============================================================
# Main
# ============================================================

async def main():

    if len(
        sys.argv
    ) != 2:

        print(
            "사용법:"
        )

        print(
            "python -m "
            "graph.test_sqlite_memory "
            "phase1"
        )

        print(
            "또는"
        )

        print(
            "python -m "
            "graph.test_sqlite_memory "
            "phase2"
        )

        return

    phase = (
        sys.argv[1]
        .strip()
        .lower()
    )

    if phase == "phase1":

        await phase1()

    elif phase == "phase2":

        await phase2()

    else:

        print(
            "phase1 또는 phase2를 "
            "입력하세요."
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )