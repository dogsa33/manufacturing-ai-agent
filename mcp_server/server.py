from typing import Any

from mcp.server import MCPServer

from src.analysis_tools import (
    compare_conditions,
    get_failure_type_summary,
    get_process_summary,
)
from src.explanation_tool import explain_failure
from src.prediction_tool import predict_failure
from src.visualization_tool import (
    create_failure_rate_by_type,
    create_feature_distribution,
    create_risk_driver_chart,
)


# =========================================================
# MCP Server
# =========================================================

mcp = MCPServer(
    "Manufacturing Analysis Agent Tools"
)


# =========================================================
# 1. Process Summary
# =========================================================

@mcp.tool()
def process_summary(
    product_type: str | None = None,
) -> dict[str, Any]:
    """
    제조 데이터의 전체 또는 Product Type별 요약 정보를 조회한다.

    사용 예:
    - 전체 고장률 조회
    - L/M/H Type별 고장률 조회
    - 평균 Torque, Tool wear 등 기본 공정 통계 조회

    Args:
        product_type:
            None 또는 L, M, H

    Returns:
        제조 데이터 요약 정보
    """

    return get_process_summary(product_type)


# =========================================================
# 2. Normal vs Failure Comparison
# =========================================================

@mcp.tool()
def compare_process_condition(
    feature: str,
) -> dict[str, Any]:
    """
    특정 제조 변수에 대해 정상과 고장 데이터의
    평균, 중앙값, 표준편차를 비교한다.

    허용 feature:
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear

    Args:
        feature:
            비교할 제조 변수명

    Returns:
        정상/고장 그룹 비교 결과
    """

    return compare_conditions(feature)


# =========================================================
# 3. Failure Type Summary
# =========================================================

@mcp.tool()
def failure_type_summary() -> dict[str, Any]:
    """
    세부 고장 유형별 발생 건수를 조회한다.

    고장 유형:
    - TWF
    - HDF
    - PWF
    - OSF
    - RNF

    Returns:
        고장 유형별 발생 건수와 최다 고장 유형
    """

    return get_failure_type_summary()


# =========================================================
# 4. Failure Prediction
# =========================================================

@mcp.tool()
def predict_machine_failure(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict[str, Any]:
    """
    주어진 제조 조건에서 Machine Failure의
    예측 확률과 위험 수준을 반환한다.

    Random Forest 모델을 사용한다.

    Args:
        product_type:
            L, M, H

        air_temperature:
            공기 온도

        process_temperature:
            공정 온도

        rotational_speed:
            회전 속도

        torque:
            Torque

        tool_wear:
            Tool wear

    Returns:
        고장확률, 정상/고장 판정, 위험도
    """

    return predict_failure(
        product_type=product_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )


# =========================================================
# 5. Local Failure Explanation
# =========================================================

@mcp.tool()
def explain_machine_failure(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict[str, Any]:
    """
    특정 제조조건의 고장 예측에 어떤 변수가
    크게 영향을 주는지 설명한다.

    같은 Product Type 정상 데이터의 중앙값으로
    변수 하나씩 교체하여 고장확률 변화를 계산한다.

    이 결과는 인과효과나 SHAP 값이 아니라
    모델의 local sensitivity 결과이다.

    Returns:
        주요 위험요인과 변수별 고장확률 변화
    """

    return explain_failure(
        product_type=product_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )


# =========================================================
# 6. Feature Distribution Chart
# =========================================================

@mcp.tool()
def feature_distribution_chart(
    feature: str,
) -> dict[str, Any]:
    """
    특정 제조 변수의 정상/고장 분포를
    Plotly HTML 그래프로 생성한다.

    허용 feature:
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear

    Returns:
        생성된 HTML 그래프 파일 정보
    """

    return create_feature_distribution(feature)


# =========================================================
# 7. Failure Rate by Type Chart
# =========================================================

@mcp.tool()
def failure_rate_by_type_chart() -> dict[str, Any]:
    """
    L/M/H Product Type별 고장률을
    Plotly Bar Chart로 생성한다.

    Returns:
        생성된 HTML 그래프 파일 정보와 Type별 고장률
    """

    return create_failure_rate_by_type()


# =========================================================
# 8. Risk Driver Chart
# =========================================================

@mcp.tool()
def risk_driver_chart(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict[str, Any]:
    """
    특정 제조조건의 주요 위험 요인을
    local perturbation 결과 기반으로 시각화한다.

    Returns:
        생성된 위험요인 그래프와 주요 위험요인 정보
    """

    return create_risk_driver_chart(
        product_type=product_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    mcp.run()