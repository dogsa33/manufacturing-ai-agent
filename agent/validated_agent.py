import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from agents import Agent, Runner
from agents.items import (
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.mcp import MCPServerStdio

from agent.manufacturing_agent import (
    AGENT_INSTRUCTIONS,
)


# =========================================================
# Project
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# Validator Structured Output
# =========================================================

class ValidationResult(BaseModel):
    passed: bool = Field(
        description=(
            "최종 답변이 사용자의 요청을 "
            "충분히 만족하면 True"
        )
    )

    grounded: bool = Field(
        description=(
            "제조 관련 수치와 판단이 Tool 결과에 "
            "근거하고 있으면 True"
        )
    )

    complete: bool = Field(
        description=(
            "사용자가 요청한 분석 항목이 "
            "빠짐없이 처리되었으면 True"
        )
    )

    safe_interpretation: bool = Field(
        description=(
            "모델 예측을 실제 고장 확정이나 "
            "인과관계처럼 과도하게 해석하지 않았으면 True"
        )
    )

    missing_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "부족하거나 잘못된 사항의 목록"
        ),
    )

    feedback: str = Field(
        description=(
            "재실행이 필요한 경우 Main Agent가 "
            "어떻게 수정해야 하는지 설명"
        )
    )


# =========================================================
# Validator Instructions
# =========================================================

VALIDATOR_INSTRUCTIONS = """
당신은 제조 AI Agent의 결과를 검증하는 Validator입니다.

사용자의 원래 요청, Main Agent가 실제 호출한 Tool,
Tool Output, 최종 답변을 검토하세요.

다음 기준을 적용하세요.

1. Grounding
제조 데이터 수치, 통계, 고장확률 등은
Tool 결과와 일치해야 합니다.

LLM이 Tool 결과에 없는 제조 수치를
임의로 만들어냈다면 실패입니다.

2. Completeness
사용자가 요청한 내용을 모두 처리했는지 확인하세요.

예를 들어 사용자가
"고장 위험을 예측하고 왜 그런지도 설명해줘"
라고 했다면 단순 예측만 해서는 충분하지 않습니다.

3. Appropriate tool usage
고장 예측에는 predict_machine_failure,
원인 또는 위험요인 설명에는 explain_machine_failure,
정상/고장 비교에는 compare_process_condition을
사용하는 것이 적절합니다.

다만 필요 이상의 Tool이 하나 더 호출되었다는 이유만으로
무조건 실패시키지는 마세요.
Tool 효율성은 별도 평가 대상입니다.

4. Missing input
고장 예측에 필요한 입력값이 부족한 경우,
Agent가 값을 추측하지 않고 사용자에게
부족한 값을 요청했다면 정상적인 답변으로 판단하세요.

5. Interpretation safety
Random Forest 결과를 실제 설비 고장의 확정 판정처럼
표현하면 안 됩니다.

local sensitivity 결과를 실제 인과관계처럼
표현해서도 안 됩니다.

6. Consistency
최종 답변의 수치와 Tool Output이 서로 일치해야 합니다.

7. passed 판정
grounded, complete, safe_interpretation이 모두 만족되고
중대한 오류가 없다면 passed=True로 판단하세요.

재실행이 필요하면 feedback에
Main Agent가 무엇을 보완해야 하는지
구체적으로 작성하세요.
"""


# =========================================================
# Trace Extraction
# =========================================================

def extract_trace(result) -> dict:
    """
    Runner 결과에서 Tool Call과 Tool Output을
    Validator가 읽기 쉬운 구조로 추출한다.
    """

    tool_calls = []
    tool_outputs = []

    for item in result.new_items:

        # -------------------------------------------------
        # Tool Call
        # -------------------------------------------------

        if isinstance(item, ToolCallItem):

            raw_item = item.raw_item

            arguments = getattr(
                raw_item,
                "arguments",
                None,
            )

            if arguments is None and isinstance(
                raw_item,
                dict,
            ):
                arguments = raw_item.get(
                    "arguments"
                )

            tool_calls.append(
                {
                    "tool_name": item.tool_name,
                    "arguments": arguments,
                }
            )

        # -------------------------------------------------
        # Tool Output
        # -------------------------------------------------

        elif isinstance(
            item,
            ToolCallOutputItem,
        ):

            tool_outputs.append(
                str(item.output)
            )

    return {
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
    }


# =========================================================
# Validator
# =========================================================

async def validate_result(
    validator_agent,
    user_request: str,
    agent_result,
) -> ValidationResult:

    trace = extract_trace(
        agent_result
    )

    validator_input = f"""
==============================
ORIGINAL USER REQUEST
==============================

{user_request}


==============================
TOOL CALLS
==============================

{trace["tool_calls"]}


==============================
TOOL OUTPUTS
==============================

{trace["tool_outputs"]}


==============================
FINAL ANSWER
==============================

{agent_result.final_output}
"""

    validation_run = await Runner.run(
        validator_agent,
        validator_input,
    )

    return validation_run.final_output


# =========================================================
# Harness
# =========================================================

async def run_with_validation(
    main_agent,
    validator_agent,
    user_request: str,
    max_attempts: int = 2,
):

    current_input = user_request

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        print("\n" + "=" * 70)
        print(
            f"AGENT ATTEMPT {attempt}"
        )
        print("=" * 70)

        # -------------------------------------------------
        # Main Agent
        # -------------------------------------------------

        result = await Runner.run(
            main_agent,
            current_input,
        )

        trace = extract_trace(
            result
        )

        print("\nTOOL CALLS")

        for index, tool in enumerate(
            trace["tool_calls"],
            start=1,
        ):
            print(
                f"{index}. "
                f"{tool['tool_name']}"
            )

            print(
                "   Arguments:",
                tool["arguments"],
            )

        print("\nAGENT ANSWER")
        print(result.final_output)

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        validation = await validate_result(
            validator_agent=validator_agent,
            user_request=user_request,
            agent_result=result,
        )

        print("\nVALIDATION")

        print(
            "passed:",
            validation.passed,
        )

        print(
            "grounded:",
            validation.grounded,
        )

        print(
            "complete:",
            validation.complete,
        )

        print(
            "safe_interpretation:",
            validation.safe_interpretation,
        )

        print(
            "missing_requirements:",
            validation.missing_requirements,
        )

        print(
            "feedback:",
            validation.feedback,
        )

        # -------------------------------------------------
        # PASS
        # -------------------------------------------------

        if validation.passed:

            print(
                "\nVALIDATION PASSED"
            )

            return {
                "result": result,
                "validation": validation,
                "attempts": attempt,
                "trace": trace,
            }

        # -------------------------------------------------
        # Final attempt failed
        # -------------------------------------------------

        if attempt == max_attempts:

            print(
                "\nVALIDATION FAILED "
                "AFTER MAX ATTEMPTS"
            )

            return {
                "result": result,
                "validation": validation,
                "attempts": attempt,
                "trace": trace,
            }

        # -------------------------------------------------
        # Re-plan / Retry
        # -------------------------------------------------

        print(
            "\nVALIDATION FAILED "
            "→ RE-PLANNING"
        )

        current_input = f"""
원래 사용자의 요청은 다음과 같습니다.

{user_request}


이전 실행 결과에 대해 Validator가
다음 문제를 발견했습니다.

{validation.feedback}


부족한 항목:

{validation.missing_requirements}


이전 최종 답변:

{result.final_output}


위 문제를 수정하여 처음부터 다시 분석하세요.

중요:
- 이전 답변의 수치를 그대로 신뢰하지 마세요.
- 필요한 제조 수치는 MCP Tool을 다시 호출해 확인하세요.
- 사용자의 원래 요청을 모두 충족하세요.
- 불필요한 추측을 하지 마세요.
- 모델 예측과 인과관계를 구분하세요.
"""


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

        # =================================================
        # Main Manufacturing Agent
        # =================================================

        main_agent = Agent(
            name="Manufacturing Analysis Agent",
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[
                mcp_server,
            ],
        )

        # =================================================
        # Validator Agent
        # =================================================

        validator_agent = Agent(
            name="Manufacturing Agent Validator",
            instructions=VALIDATOR_INSTRUCTIONS,
            output_type=ValidationResult,
        )

        # =================================================
        # Test Question
        # =================================================

        question = (
            "다음 제조조건의 고장 위험을 먼저 예측하고, "
            "위험하다면 왜 위험하게 판단됐는지도 "
            "주요 변수 기준으로 분석해줘. "
            "Product Type은 L, "
            "Air temperature는 301.0, "
            "Process temperature는 310.5, "
            "Rotational speed는 1300, "
            "Torque는 65.0, "
            "Tool wear는 200이야."
        )

        print("=" * 70)
        print("USER REQUEST")
        print("=" * 70)

        print(question)

        harness_result = await run_with_validation(
            main_agent=main_agent,
            validator_agent=validator_agent,
            user_request=question,
            max_attempts=2,
        )

        # =================================================
        # Final
        # =================================================

        print("\n" + "=" * 70)
        print("HARNESS RESULT")
        print("=" * 70)

        print(
            "Attempts:",
            harness_result["attempts"],
        )

        print(
            "Validation passed:",
            harness_result[
                "validation"
            ].passed,
        )

        print("\nFINAL ANSWER")

        print(
            harness_result[
                "result"
            ].final_output
        )


if __name__ == "__main__":
    asyncio.run(main())