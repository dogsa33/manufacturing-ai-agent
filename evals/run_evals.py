import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio

from agent.manufacturing_agent import (
    AGENT_INSTRUCTIONS,
)

from agent.validated_agent import (
    VALIDATOR_INSTRUCTIONS,
    ValidationResult,
    extract_trace,
    validate_result,
)

from evals.eval_cases import (
    EVAL_CASES,
)


# =========================================================
# Project Paths
# =========================================================

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


# =========================================================
# Utility
# =========================================================

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


# =========================================================
# Single Eval
# =========================================================

async def evaluate_case(
    main_agent,
    validator_agent,
    case: dict,
) -> dict:

    print("\n" + "=" * 70)

    print(
        f"EVAL: {case['id']}"
    )

    print(
        case["description"]
    )

    print("=" * 70)

    print("\nQUESTION")
    print(case["question"])

    # -----------------------------------------------------
    # Run Main Agent
    # -----------------------------------------------------

    result = await Runner.run(
        main_agent,
        case["question"],
    )

    # -----------------------------------------------------
    # Extract Tool Trace
    # -----------------------------------------------------

    trace = extract_trace(
        result
    )

    called_tools = [
        item["tool_name"]
        for item in trace["tool_calls"]
    ]

    tool_count = len(
        called_tools
    )

    # -----------------------------------------------------
    # Deterministic Checks
    # -----------------------------------------------------

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
            result.final_output,
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

    # -----------------------------------------------------
    # LLM Validator
    # -----------------------------------------------------

    validation = await validate_result(
        validator_agent=validator_agent,
        user_request=case["question"],
        agent_result=result,
    )

    # -----------------------------------------------------
    # Hard Pass Criteria
    # -----------------------------------------------------

    # Efficiency는 품질지표이지만
    # 전체 pass/fail을 결정하는 hard condition은 아님.
    case_passed = all(
        [
            required_tools_ok,
            forbidden_tools_ok,
            answer_patterns_ok,
            validation.passed,
        ]
    )

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print("\nCALLED TOOLS")

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
            "(No tools called)"
        )

    print("\nCHECKS")

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
        "validator_passed:",
        validation.passed,
    )

    print(
        "case_passed:",
        case_passed,
    )

    print("\nFINAL ANSWER")

    print(
        result.final_output
    )

    # -----------------------------------------------------
    # Result Record
    # -----------------------------------------------------

    return {
        "id": case["id"],
        "description": (
            case["description"]
        ),
        "question": (
            case["question"]
        ),
        "called_tools": (
            called_tools
        ),
        "tool_count": (
            tool_count
        ),
        "required_tools": (
            case["required_tools"]
        ),
        "forbidden_tools": (
            case["forbidden_tools"]
        ),
        "max_tool_calls": (
            case.get(
                "max_tool_calls"
            )
        ),
        "required_tools_ok": (
            required_tools_ok
        ),
        "forbidden_tools_ok": (
            forbidden_tools_ok
        ),
        "answer_patterns_ok": (
            answer_patterns_ok
        ),
        "efficiency_ok": (
            efficiency_ok
        ),
        "validator_passed": (
            validation.passed
        ),
        "validator_grounded": (
            validation.grounded
        ),
        "validator_complete": (
            validation.complete
        ),
        "validator_safe": (
            validation.safe_interpretation
        ),
        "validator_feedback": (
            validation.feedback
        ),
        "case_passed": (
            case_passed
        ),
        "final_answer": (
            result.final_output
        ),
    }


# =========================================================
# Summary
# =========================================================

def calculate_summary(
    results: list[dict],
) -> dict:

    total = len(results)

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
        result["validator_passed"]
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
        "total_cases": total,

        "passed_cases": passed,

        "overall_pass_rate": round(
            passed / total * 100,
            2,
        ),

        "required_tool_accuracy": round(
            required_tool_success
            / total
            * 100,
            2,
        ),

        "forbidden_tool_compliance": round(
            forbidden_tool_success
            / total
            * 100,
            2,
        ),

        "answer_fact_accuracy": round(
            answer_success
            / total
            * 100,
            2,
        ),

        "validator_pass_rate": round(
            validator_success
            / total
            * 100,
            2,
        ),

        "tool_efficiency_rate": round(
            efficiency_success
            / total
            * 100,
            2,
        ),

        "average_tool_calls": round(
            total_tool_calls
            / total,
            2,
        ),
    }


# =========================================================
# Save Results
# =========================================================

def save_results(
    results: list[dict],
    summary: dict,
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        RESULT_DIR
        / f"eval_{timestamp}.json"
    )

    csv_path = (
        RESULT_DIR
        / f"eval_{timestamp}.csv"
    )

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv_fields = [
        "id",
        "description",
        "tool_count",
        "required_tools_ok",
        "forbidden_tools_ok",
        "answer_patterns_ok",
        "efficiency_ok",
        "validator_passed",
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
                    field: result[field]
                    for field in csv_fields
                }
            )

    return (
        json_path,
        csv_path,
    )


# =========================================================
# Main
# =========================================================

async def main():

    python_executable = (
        sys.executable
    )

    async with MCPServerStdio(
        name=(
            "Manufacturing Analysis "
            "MCP Server"
        ),
        params={
            "command": (
                python_executable
            ),
            "args": [
                "-m",
                "mcp_server.server",
            ],
            "cwd": str(
                PROJECT_ROOT
            ),
        },
        cache_tools_list=True,
        use_structured_content=True,
    ) as mcp_server:

        # =================================================
        # Main Agent
        # =================================================

        main_agent = Agent(
            name=(
                "Manufacturing "
                "Analysis Agent"
            ),
            instructions=(
                AGENT_INSTRUCTIONS
            ),
            mcp_servers=[
                mcp_server,
            ],
        )

        # =================================================
        # Validator
        # =================================================

        validator_agent = Agent(
            name=(
                "Manufacturing "
                "Agent Validator"
            ),
            instructions=(
                VALIDATOR_INSTRUCTIONS
            ),
            output_type=(
                ValidationResult
            ),
        )

        # =================================================
        # Run All Eval Cases
        # =================================================

        results = []

        for case in EVAL_CASES:

            result = await evaluate_case(
                main_agent=main_agent,
                validator_agent=validator_agent,
                case=case,
            )

            results.append(
                result
            )

        # =================================================
        # Summary
        # =================================================

        summary = calculate_summary(
            results
        )

        print("\n")
        print("=" * 70)
        print("EVAL SUMMARY")
        print("=" * 70)

        for key, value in summary.items():

            print(
                f"{key}: {value}"
            )

        # =================================================
        # Save
        # =================================================

        json_path, csv_path = (
            save_results(
                results,
                summary,
            )
        )

        print("\nRESULT FILES")

        print(
            "JSON:",
            json_path,
        )

        print(
            "CSV :",
            csv_path,
        )


if __name__ == "__main__":
    asyncio.run(main())