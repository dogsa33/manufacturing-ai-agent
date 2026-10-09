import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path

from agents import Agent, Runner
from agents.items import (
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.mcp import MCPServerStdio


# =========================================================
# Project
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# Agent Instructions
# =========================================================

AGENT_INSTRUCTIONS = """
당신은 제조설비 데이터 분석 AI Agent입니다.

사용자의 제조 데이터 분석, 고장 예측,
고장 원인 분석 요청을 처리합니다.

반드시 다음 규칙을 준수하세요.

1. 제조 데이터와 관련된 수치나 통계를 추측하지 마세요.
   반드시 MCP Tool을 통해 확인하세요.

2. 전체 또는 Product Type별 통계가 필요하면
   process_summary를 사용하세요.

3. 정상과 고장 조건을 비교해야 하면
   compare_process_condition을 사용하세요.

4. 고장 유형별 발생 현황이 필요하면
   failure_type_summary를 사용하세요.

5. 특정 제조조건의 고장 위험을 예측해야 하면
   predict_machine_failure를 사용하세요.

6. 고장 위험의 주요 영향 요인을 설명해야 하면
   explain_machine_failure를 사용하세요.

7. 그래프를 요청하면 적절한 chart Tool을 사용하세요.

8. 고장 예측 입력값은 다음 6개입니다.

   - product_type
   - air_temperature
   - process_temperature
   - rotational_speed
   - torque
   - tool_wear

   입력값이 부족하면 임의로 추정하지 말고
   사용자에게 요청하세요.

9. explain_machine_failure의 결과는
   인과관계가 아니라 local sensitivity입니다.

10. 고장 예측은 Random Forest 모델의 결과이며
    실제 고장을 확정하는 판정처럼 표현하지 마세요.

11. 답변은 기본적으로 한국어로 작성하세요.

12. Tool 결과를 최우선 근거로 사용하세요.
"""


# =========================================================
# Helper
# =========================================================

def get_raw_field(raw_item, field_name):
    """
    raw_item이 dict 또는 SDK 객체인 경우 모두 처리.
    """

    if isinstance(raw_item, Mapping):
        return raw_item.get(field_name)

    return getattr(
        raw_item,
        field_name,
        None,
    )


def print_agent_trace(result):
    """
    Agent가 실제로 어떤 Tool을 호출했고
    어떤 결과를 받았는지 출력한다.
    """

    print("\n" + "-" * 70)
    print("AGENT TRACE")
    print("-" * 70)

    tool_count = 0

    for item in result.new_items:

        # -------------------------------------------------
        # Tool Call
        # -------------------------------------------------

        if isinstance(item, ToolCallItem):

            tool_count += 1

            raw_item = item.raw_item

            tool_name = item.tool_name

            arguments = get_raw_field(
                raw_item,
                "arguments",
            )

            print(
                f"\n[TOOL CALL {tool_count}]"
            )

            print(
                "Tool:",
                tool_name,
            )

            print(
                "Arguments:",
                arguments,
            )

        # -------------------------------------------------
        # Tool Output
        # -------------------------------------------------

        elif isinstance(
            item,
            ToolCallOutputItem,
        ):

            print("\n[TOOL OUTPUT]")

            print(
                item.output
            )

    if tool_count == 0:

        print(
            "\nNo tool calls detected."
        )

    print("\nTotal tool calls:", tool_count)


async def run_test(
    agent,
    question,
    test_name,
):

    print("\n")
    print("=" * 70)
    print(test_name)
    print("=" * 70)

    print("\nUSER")
    print(question)

    result = await Runner.run(
        agent,
        question,
    )

    print_agent_trace(
        result
    )

    print("\n" + "-" * 70)
    print("FINAL ANSWER")
    print("-" * 70)

    print(
        result.final_output
    )


# =========================================================
# Main
# =========================================================

async def main():

    python_executable = sys.executable

    async with MCPServerStdio(
        name="Manufacturing Analysis MCP Server",
        params={
            "command": python_executable,
            "args": [
                "-m",
                "mcp_server.server",
            ],
            "cwd": str(PROJECT_ROOT),
        },
        cache_tools_list=True,
        use_structured_content=True,
    ) as mcp_server:

        agent = Agent(
            name="Manufacturing Analysis Agent",
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[
                mcp_server,
            ],
        )

        # =================================================
        # TEST 1
        # 단일 통계 질문
        # =================================================

        await run_test(
            agent=agent,
            test_name="TEST 1 - PROCESS SUMMARY",
            question=(
                "Product Type L의 전체 샘플 수와 "
                "고장 건수, 고장률을 알려줘."
            ),
        )


        # =================================================
        # TEST 2
        # 고장 예측
        # =================================================

        await run_test(
            agent=agent,
            test_name="TEST 2 - FAILURE PREDICTION",
            question=(
                "다음 제조조건의 고장 위험을 분석해줘. "
                "Product Type은 L, "
                "Air temperature는 301.0, "
                "Process temperature는 310.5, "
                "Rotational speed는 1300, "
                "Torque는 65.0, "
                "Tool wear는 200이야."
            ),
        )


        # =================================================
        # TEST 3
        # 복합 분석
        # =================================================

        await run_test(
            agent=agent,
            test_name="TEST 3 - PREDICTION + EXPLANATION",
            question=(
                "다음 제조조건의 고장 위험을 먼저 예측하고, "
                "위험하다면 왜 위험하게 판단됐는지도 "
                "주요 변수 기준으로 분석해줘. "
                "Product Type은 L, "
                "Air temperature는 301.0, "
                "Process temperature는 310.5, "
                "Rotational speed는 1300, "
                "Torque는 65.0, "
                "Tool wear는 200이야."
            ),
        )


        # =================================================
        # TEST 4
        # 비교 분석
        # =================================================

        await run_test(
            agent=agent,
            test_name="TEST 4 - CONDITION COMPARISON",
            question=(
                "Torque가 정상 제품과 고장 제품에서 "
                "어떻게 다른지 비교해서 설명해줘."
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())