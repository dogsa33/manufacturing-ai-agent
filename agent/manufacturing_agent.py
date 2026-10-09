import asyncio
import sys
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


# =========================================================
# Project Path
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

1. 데이터와 관련된 수치나 통계를 추측하지 마세요.
   반드시 제공된 MCP Tool을 사용해 확인하세요.

2. 제조 데이터 전체 또는 Product Type별
   통계가 필요하면 process_summary를 사용하세요.

3. 정상과 고장 조건을 비교해야 하면
   compare_process_condition을 사용하세요.

4. 고장 유형별 발생 현황이 필요하면
   failure_type_summary를 사용하세요.

5. 특정 제조조건의 고장 위험을 예측해야 하면
   predict_machine_failure를 사용하세요.

6. 고장 예측의 주요 영향 요인을 설명해야 하면
   explain_machine_failure를 사용하세요.

7. 사용자가 그래프나 시각화를 요청하면
   적절한 chart Tool을 사용하세요.

8. 고장 예측에 필요한 입력값은 다음 6개입니다.

   - product_type
   - air_temperature
   - process_temperature
   - rotational_speed
   - torque
   - tool_wear

   값이 부족하면 임의로 채우지 말고
   사용자에게 필요한 값을 요청하세요.

9. explain_machine_failure 결과는
   인과관계가 아닙니다.

   이것은 정상 조건의 중앙값으로 변수를
   하나씩 교체했을 때 모델 예측확률이 얼마나
   변하는지를 계산한 local sensitivity 결과입니다.

   따라서
   "Torque 때문에 고장이 발생했다"
   와 같이 인과적으로 단정하지 마세요.

10. Machine Failure Prediction은
    Random Forest 모델의 예측 결과입니다.
    실제 설비 고장의 확정 판정처럼 표현하지 마세요.

11. 답변은 기본적으로 한국어로 작성하세요.

12. 사용자가 별도로 요청하지 않는 한
    지나치게 긴 설명보다 다음 순서로 답하세요.

    - 핵심 결론
    - 근거 수치
    - 필요한 해석 또는 주의사항

13. Tool에서 반환된 값을 최우선 근거로 사용하세요.
"""


# =========================================================
# Main
# =========================================================

async def main():

    # 현재 활성화된 .venv의 python.exe를 그대로 사용
    python_executable = sys.executable

    # -----------------------------------------------------
    # MCP Server 실행
    # -----------------------------------------------------

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

        # -------------------------------------------------
        # Agent 생성
        # -------------------------------------------------

        agent = Agent(
            name="Manufacturing Analysis Agent",
            instructions=AGENT_INSTRUCTIONS,
            mcp_servers=[
                mcp_server,
            ],
        )

        # -------------------------------------------------
        # 첫 번째 자연어 테스트
        # -------------------------------------------------

        question = (
            "Product Type L의 전체 샘플 수와 "
            "고장 건수, 고장률을 알려줘."
        )

        print("=" * 70)
        print("USER")
        print("=" * 70)
        print(question)

        # -------------------------------------------------
        # Agent 실행
        # -------------------------------------------------

        result = await Runner.run(
            agent,
            question,
        )

        # -------------------------------------------------
        # 결과 출력
        # -------------------------------------------------

        print("\n" + "=" * 70)
        print("AGENT")
        print("=" * 70)

        print(result.final_output)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())