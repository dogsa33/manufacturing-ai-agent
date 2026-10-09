import asyncio

import graph.manufacturing_graph as mg


# ============================================================
# Test 1 Validator
# 첫 번째 답변에 수정 요구
# → 재생성된 답변이 실제 수정되었는지 검사
# ============================================================

async def fail_once_validator(
    state,
):
    """
    첫 번째 답변에서는 의도적으로
    '고장 건수도 포함하라'는 feedback을 준다.

    retry 후에는 실제 답변에
    고장 건수 235가 포함됐는지 검사한다.
    """

    retry_count = state.get(
        "retry_count",
        0,
    )

    answer = state.get(
        "answer",
        "",
    )

    print(
        "\n[TEST VALIDATOR]"
    )

    # ========================================================
    # First Attempt
    # 의도적으로 FAIL + 구체적인 수정 Feedback
    # ========================================================

    if retry_count == 0:

        print(
            "FORCED FAIL WITH "
            "CORRECTION FEEDBACK"
        )

        return {
            "validation_passed": False,

            "validation_feedback": (
                "답변에 Product Type L의 "
                "고장 건수도 명시하세요. "
                "Tool 결과에 있는 failures 값을 "
                "사용해야 합니다."
            ),
        }

    # ========================================================
    # Retry
    # Feedback이 실제 답변에 반영됐는지 검사
    # ========================================================

    if "235" in answer:

        print(
            "CORRECTION DETECTED → PASS"
        )

        return {
            "validation_passed": True,

            "validation_feedback": (
                "Validator feedback이 "
                "재생성 답변에 반영되었습니다."
            ),
        }

    # ========================================================
    # 수정되지 않았다면 다시 FAIL
    # ========================================================

    print(
        "CORRECTION NOT DETECTED → FAIL"
    )

    return {
        "validation_passed": False,

        "validation_feedback": (
            "고장 건수 235가 아직 답변에 "
            "포함되지 않았습니다."
        ),
    }


# ============================================================
# Test 2 Validator
# 계속 FAIL → 최대 retry → fallback
# ============================================================

async def always_fail_validator(
    state,
):
    retry_count = state.get(
        "retry_count",
        0,
    )

    print(
        "\n[TEST VALIDATOR]"
    )

    print(
        f"FORCED FAIL "
        f"(retry_count={retry_count})"
    )

    return {
        "validation_passed": False,

        "validation_feedback": (
            "TEST ONLY: "
            "Safe Fallback 경로 검증을 위해 "
            "Validator를 강제로 실패시켰습니다."
        ),
    }


# ============================================================
# Test 1
# FAIL → Feedback 반영 → Replan → 수정된 답변 → PASS
# ============================================================

async def test_replan_then_pass():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TEST 1 - FEEDBACK → REPLAN → CORRECTION → PASS"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # Production Validator 대신
    # Test Validator를 현재 테스트 프로세스에서만 사용
    # ========================================================

    mg.validate_answer = (
        fail_once_validator
    )

    # 변경된 validator로
    # 테스트 전용 graph compile
    test_graph = (
        mg.build_manufacturing_graph()
    )

    config = {
        "configurable": {
            "thread_id":
                "replan-feedback-test",
        }
    }

    result = await test_graph.ainvoke(
        {
            "query": (
                "Product Type L의 "
                "전체 샘플 수와 "
                "고장률을 알려줘."
            )
        },
        config=config,
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        "FINAL RESULT"
    )

    print(
        "-" * 100
    )

    print(
        "validation_passed:",
        result.get(
            "validation_passed"
        ),
    )

    print(
        "retry_count:",
        result.get(
            "retry_count"
        ),
    )

    print(
        "validation_feedback:",
        result.get(
            "validation_feedback"
        ),
    )

    print(
        "\nanswer:"
    )

    print(
        result.get(
            "answer"
        )
    )


# ============================================================
# Test 2
# FAIL → Replan → FAIL → Replan → FAIL → Fallback
# ============================================================

async def test_max_retry_fallback():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TEST 2 - MAX RETRY → FALLBACK"
    )

    print(
        "=" * 100
    )

    mg.validate_answer = (
        always_fail_validator
    )

    test_graph = (
        mg.build_manufacturing_graph()
    )

    config = {
        "configurable": {
            "thread_id":
                "replan-test-fallback",
        }
    }

    result = await test_graph.ainvoke(
        {
            "query": (
                "Product Type L의 "
                "전체 샘플 수와 "
                "고장률을 알려줘."
            )
        },
        config=config,
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        "FINAL RESULT"
    )

    print(
        "-" * 100
    )

    print(
        "validation_passed:",
        result.get(
            "validation_passed"
        ),
    )

    print(
        "retry_count:",
        result.get(
            "retry_count"
        ),
    )

    print(
        "validation_feedback:",
        result.get(
            "validation_feedback"
        ),
    )

    print(
        "\nanswer:"
    )

    print(
        result.get(
            "answer"
        )
    )


# ============================================================
# Main
# ============================================================

async def main():

    await test_replan_then_pass()

    await test_max_retry_fallback()


if __name__ == "__main__":

    asyncio.run(
        main()
    )