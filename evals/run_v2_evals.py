import asyncio
import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from graph.manufacturing_graph import (
    manufacturing_graph,
)

from evals.eval_cases import (
    EVAL_CASES,
)


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    PROJECT_ROOT
    / "evals"
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Deterministic Checks
# ============================================================

def check_required_tools(
    called_tools: list[str],
    required_tools: list[str],
) -> bool:

    return all(
        tool in called_tools
        for tool in required_tools
    )


def check_forbidden_tools(
    called_tools: list[str],
    forbidden_tools: list[str],
) -> bool:

    return all(
        tool not in called_tools
        for tool in forbidden_tools
    )


def check_answer_patterns(
    answer: str,
    patterns: list[str],
) -> bool:

    for pattern in patterns:

        matched = re.search(
            pattern,
            answer,
            flags=re.IGNORECASE,
        )

        if matched is None:
            return False

    return True


def check_tool_efficiency(
    tool_count: int,
    max_tool_calls: int | None,
) -> bool:

    if max_tool_calls is None:
        return True

    return (
        tool_count
        <= max_tool_calls
    )


# ============================================================
# LangGraph Execution + Trace
# ============================================================

async def run_graph_with_trace(
    case: dict,
) -> tuple[dict, list[str]]:
    """
    LangGraph를 stream_mode='updates'로 실행한다.

    목적:
    - 최종 State 확보
    - 실제 execute_tool Node가 실행된 횟수와 Tool 추적

    Router가 Tool을 선택하기만 하고 clarification에서 종료한 경우는
    실제 MCP 호출로 계산하지 않는다.
    """

    thread_id = (
        f"v2-eval-{case['id']}-"
        f"{uuid.uuid4()}"
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # 최종 State를 직접 재구성
    state = {
        "query": case["question"],
    }

    called_tools = []

    # Router가 마지막으로 선택한 Tool
    current_tool_name = ""

    async for event in manufacturing_graph.astream(
        {
            "query": case["question"],
        },
        config=config,
        stream_mode="updates",
    ):

        if not isinstance(
            event,
            dict,
        ):
            continue

        for node_name, update in event.items():

            if not isinstance(
                update,
                dict,
            ):
                continue

            # -----------------------------------------------
            # Partial State 병합
            # -----------------------------------------------

            state.update(
                update
            )

            # -----------------------------------------------
            # Router가 선택한 Tool 기억
            # -----------------------------------------------

            if node_name == "router":

                route = update.get(
                    "route",
                    "",
                )

                tool_name = update.get(
                    "tool_name",
                    "",
                )

                if (
                    route == "tool"
                    and tool_name
                ):

                    current_tool_name = (
                        tool_name
                    )

            # -----------------------------------------------
            # 실제 MCP Tool 실행만 Count
            # -----------------------------------------------

            if (
                node_name
                == "execute_tool"
                and current_tool_name
            ):

                called_tools.append(
                    current_tool_name
                )

    return (
        state,
        called_tools,
    )


# ============================================================
# Single Eval
# ============================================================

async def evaluate_case(
    case: dict,
) -> dict:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"EVAL: {case['id']}"
    )

    print(
        case["description"]
    )

    print(
        "=" * 70
    )

    print(
        "\nQUESTION"
    )

    print(
        case["question"]
    )

    # ========================================================
    # Run LangGraph
    # ========================================================

    state, called_tools = (
        await run_graph_with_trace(
            case
        )
    )

    answer = state.get(
        "answer",
        "",
    )

    tool_count = len(
        called_tools
    )

    # ========================================================
    # Deterministic Checks
    # ========================================================

    required_tools_ok = (
        check_required_tools(
            called_tools,
            case["required_tools"],
        )
    )

    forbidden_tools_ok = (
        check_forbidden_tools(
            called_tools,
            case["forbidden_tools"],
        )
    )

    answer_patterns_ok = (
        check_answer_patterns(
            answer,
            case["answer_patterns"],
        )
    )

    efficiency_ok = (
        check_tool_efficiency(
            tool_count,
            case.get(
                "max_tool_calls"
            ),
        )
    )

    # ========================================================
    # LangGraph Validator
    # ========================================================

    validation_passed = state.get(
        "validation_passed",
        False,
    )

    validation_feedback = state.get(
        "validation_feedback",
        "",
    )

    needs_clarification = state.get(
        "needs_clarification",
        False,
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    route = state.get(
        "route",
        "",
    )

    rag_rewrite_count = state.get(
        "rag_rewrite_count",
        0,
    )

    # ========================================================
    # Missing-input case
    #
    # Clarification은 정상 종료이며 Validator를 거치지 않는다.
    # 따라서 이 경우 validation_passed=False가 정상일 수 있다.
    # ========================================================

    clarification_expected = (
        len(
            case["required_tools"]
        )
        == 0
        and len(
            case["forbidden_tools"]
        )
        > 0
        and case.get(
            "max_tool_calls"
        )
        == 0
    )

    if clarification_expected:

        validator_ok = (
            needs_clarification
            and forbidden_tools_ok
            and tool_count == 0
        )

    else:

        validator_ok = (
            validation_passed
        )

    # ========================================================
    # Hard Pass Criteria
    # ========================================================

    case_passed = all(
        [
            required_tools_ok,
            forbidden_tools_ok,
            answer_patterns_ok,
            validator_ok,
        ]
    )

    # 기존 V1과 마찬가지로
    # efficiency는 기록하지만 hard fail 조건으로 사용하지 않는다.

    # ========================================================
    # Print
    # ========================================================

    print(
        "\nROUTE"
    )

    print(
        route
    )

    print(
        "\nCALLED TOOLS"
    )

    if called_tools:

        for index, tool in enumerate(
            called_tools,
            start=1,
        ):

            print(
                f"{index}. {tool}"
            )

    else:

        print(
            "(No MCP tools executed)"
        )

    print(
        "\nCHECKS"
    )

    print(
        "required_tools_ok:",
        required_tools_ok,
    )

    print(
        "forbidden_tools_ok:",
        forbidden_tools_ok,
    )

    print(
        "answer_patterns_ok:",
        answer_patterns_ok,
    )

    print(
        "efficiency_ok:",
        efficiency_ok,
    )

    print(
        "validator_ok:",
        validator_ok,
    )

    print(
        "validation_passed:",
        validation_passed,
    )

    print(
        "needs_clarification:",
        needs_clarification,
    )

    print(
        "retry_count:",
        retry_count,
    )

    print(
        "case_passed:",
        case_passed,
    )

    print(
        "\nFINAL ANSWER"
    )

    print(
        answer
    )

    # ========================================================
    # Result Record
    # ========================================================

    return {
        "id":
            case["id"],

        "description":
            case["description"],

        "question":
            case["question"],

        "route":
            route,

        "called_tools":
            called_tools,

        "tool_count":
            tool_count,

        "required_tools":
            case["required_tools"],

        "forbidden_tools":
            case["forbidden_tools"],

        "max_tool_calls":
            case.get(
                "max_tool_calls"
            ),

        "required_tools_ok":
            required_tools_ok,

        "forbidden_tools_ok":
            forbidden_tools_ok,

        "answer_patterns_ok":
            answer_patterns_ok,

        "efficiency_ok":
            efficiency_ok,

        "validator_ok":
            validator_ok,

        "validation_passed":
            validation_passed,

        "validation_feedback":
            validation_feedback,

        "needs_clarification":
            needs_clarification,

        "retry_count":
            retry_count,

        "rag_rewrite_count":
            rag_rewrite_count,

        "case_passed":
            case_passed,

        "final_answer":
            answer,
    }


# ============================================================
# Summary
# ============================================================

def calculate_summary(
    results: list[dict],
) -> dict:

    total = len(
        results
    )

    passed = sum(
        result["case_passed"]
        for result in results
    )

    required_tool_success = sum(
        result["required_tools_ok"]
        for result in results
    )

    forbidden_tool_success = sum(
        result["forbidden_tools_ok"]
        for result in results
    )

    answer_success = sum(
        result["answer_patterns_ok"]
        for result in results
    )

    validator_success = sum(
        result["validator_ok"]
        for result in results
    )

    efficiency_success = sum(
        result["efficiency_ok"]
        for result in results
    )

    total_tool_calls = sum(
        result["tool_count"]
        for result in results
    )

    return {
        "total_cases":
            total,

        "passed_cases":
            passed,

        "predefined_eval_pass_rate":
            round(
                passed
                / total
                * 100,
                2,
            ),

        "required_tool_success_rate":
            round(
                required_tool_success
                / total
                * 100,
                2,
            ),

        "forbidden_tool_compliance":
            round(
                forbidden_tool_success
                / total
                * 100,
                2,
            ),

        "answer_pattern_success_rate":
            round(
                answer_success
                / total
                * 100,
                2,
            ),

        "validator_or_clarification_success_rate":
            round(
                validator_success
                / total
                * 100,
                2,
            ),

        "tool_efficiency_rate":
            round(
                efficiency_success
                / total
                * 100,
                2,
            ),

        "average_tool_calls":
            round(
                total_tool_calls
                / total,
                2,
            ),
    }


# ============================================================
# Save Results
# ============================================================

def save_results(
    results: list[dict],
    summary: dict,
):

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    json_path = (
        RESULT_DIR
        / f"v2_eval_{timestamp}.json"
    )

    csv_path = (
        RESULT_DIR
        / f"v2_eval_{timestamp}.csv"
    )

    # ========================================================
    # JSON
    # ========================================================

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "summary": summary,
                "results": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # CSV
    # ========================================================

    csv_fields = [
        "id",
        "description",
        "route",
        "tool_count",
        "required_tools_ok",
        "forbidden_tools_ok",
        "answer_patterns_ok",
        "efficiency_ok",
        "validator_ok",
        "validation_passed",
        "needs_clarification",
        "retry_count",
        "rag_rewrite_count",
        "case_passed",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    field:
                        result[field]
                    for field in csv_fields
                }
            )

    return (
        json_path,
        csv_path,
    )


# ============================================================
# Main
# ============================================================

async def main():

    results = []

    for case in EVAL_CASES:

        result = await evaluate_case(
            case
        )

        results.append(
            result
        )

    summary = calculate_summary(
        results
    )

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "V2 LANGGRAPH EVAL SUMMARY"
    )

    print(
        "=" * 70
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    json_path, csv_path = (
        save_results(
            results,
            summary,
        )
    )

    print(
        "\nRESULT FILES"
    )

    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV :",
        csv_path,
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )