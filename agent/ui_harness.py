from agents import Runner

from agent.validated_agent import (
    validate_result,
)


async def run_validated_turn(
    main_agent,
    validator_agent,
    session,
    user_request: str,
    max_attempts: int = 2,
):
    """
    Streamlit용 Agent Harness.

    1. Main Agent 실행
    2. Validator 검증
    3. 실패 시 Validator feedback 기반 재분석
    4. 최종 결과 반환
    """

    current_input = user_request

    last_result = None
    last_validation = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        # =================================================
        # Main Agent
        # =================================================

        result = await Runner.run(
            main_agent,
            current_input,
            session=session,
        )

        # =================================================
        # Validator
        # =================================================

        validation = await validate_result(
            validator_agent=validator_agent,
            user_request=user_request,
            agent_result=result,
        )

        last_result = result
        last_validation = validation

        # =================================================
        # PASS
        # =================================================

        if validation.passed:

            return {
                "result": result,
                "validation": validation,
                "attempts": attempt,
            }

        # =================================================
        # Max attempts
        # =================================================

        if attempt == max_attempts:

            break

        # =================================================
        # Re-plan
        # =================================================

        current_input = f"""
원래 사용자의 요청:

{user_request}

Validator가 이전 분석에서 다음 문제를 발견했습니다.

Feedback:
{validation.feedback}

Missing requirements:
{validation.missing_requirements}

이전 답변:
{result.final_output}

위 문제를 수정해서 다시 분석하세요.

규칙:
- 필요한 제조 수치는 MCP Tool로 다시 확인하세요.
- 수치를 추측하지 마세요.
- 사용자의 원래 요청을 모두 처리하세요.
- Random Forest 예측과 실제 고장 확정을 구분하세요.
- local sensitivity와 인과관계를 구분하세요.
"""

    # =====================================================
    # Failed after retries
    # =====================================================

    return {
        "result": last_result,
        "validation": last_validation,
        "attempts": max_attempts,
    }