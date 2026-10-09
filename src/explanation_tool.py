from pathlib import Path

import duckdb

from src.prediction_tool import predict_failure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "database" / "manufacturing.duckdb"


NUMERIC_FEATURES = {
    "air_temperature": "Air temperature",
    "process_temperature": "Process temperature",
    "rotational_speed": "Rotational speed",
    "torque": "Torque",
    "tool_wear": "Tool wear",
}


def get_normal_reference(product_type: str) -> dict:
    """
    같은 Product Type의 정상 샘플에서
    각 제조 변수의 중앙값을 정상 기준값으로 계산한다.
    """

    product_type = product_type.upper()

    if product_type not in {"L", "M", "H"}:
        raise ValueError(
            "product_type must be one of: L, M, H"
        )

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    try:
        query = """
        SELECT
            MEDIAN("Air temperature"),
            MEDIAN("Process temperature"),
            MEDIAN("Rotational speed"),
            MEDIAN("Torque"),
            MEDIAN("Tool wear")
        FROM manufacturing_data
        WHERE Type = ?
          AND "Machine failure" = 0
        """

        result = con.execute(
            query,
            [product_type],
        ).fetchone()

        return {
            "air_temperature": float(result[0]),
            "process_temperature": float(result[1]),
            "rotational_speed": float(result[2]),
            "torque": float(result[3]),
            "tool_wear": float(result[4]),
        }

    finally:
        con.close()


def explain_failure(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    각 입력 변수를 정상 기준값으로 하나씩 교체한 뒤
    고장확률의 변화를 계산한다.

    양수 impact:
        현재 값이 정상 기준값보다
        고장 위험을 높이는 방향으로 작용.

    음수 impact:
        현재 값이 고장 위험을 낮추는 방향으로 작용.

    주의:
        이 값은 SHAP value나 인과효과가 아니다.
        모델의 국소적 sensitivity를 나타낸다.
    """

    product_type = product_type.upper()

    original_input = {
        "product_type": product_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    # -----------------------------------------------------
    # 1. Original prediction
    # -----------------------------------------------------

    original_result = predict_failure(
        **original_input
    )

    original_probability = (
        original_result["failure_probability"]
    )

    # -----------------------------------------------------
    # 2. Normal reference
    # -----------------------------------------------------

    normal_reference = get_normal_reference(
        product_type
    )

    # -----------------------------------------------------
    # 3. Feature perturbation
    # -----------------------------------------------------

    impacts = []

    for feature_key, display_name in NUMERIC_FEATURES.items():

        perturbed_input = original_input.copy()

        # 해당 Feature만 정상 기준값으로 교체
        perturbed_input[feature_key] = (
            normal_reference[feature_key]
        )

        perturbed_result = predict_failure(
            **perturbed_input
        )

        perturbed_probability = (
            perturbed_result["failure_probability"]
        )

        impact = (
            original_probability
            - perturbed_probability
        )

        impacts.append(
            {
                "feature": display_name,
                "input_value": original_input[feature_key],
                "normal_reference": normal_reference[
                    feature_key
                ],
                "original_probability_pct": round(
                    original_probability * 100,
                    2,
                ),
                "perturbed_probability_pct": round(
                    perturbed_probability * 100,
                    2,
                ),
                "risk_impact_pct_point": round(
                    impact * 100,
                    2,
                ),
                "direction": (
                    "RISK_INCREASING"
                    if impact > 0
                    else "RISK_REDUCING"
                    if impact < 0
                    else "NEUTRAL"
                ),
            }
        )

    # 절댓값이 큰 순으로 정렬
    impacts.sort(
        key=lambda x: abs(
            x["risk_impact_pct_point"]
        ),
        reverse=True,
    )

    risk_drivers = [
        item
        for item in impacts
        if item["risk_impact_pct_point"] > 0
    ]

    protective_factors = [
        item
        for item in impacts
        if item["risk_impact_pct_point"] < 0
    ]

    return {
        "prediction": {
            "failure_probability_pct":
                original_result[
                    "failure_probability_pct"
                ],
            "prediction_label":
                original_result[
                    "prediction_label"
                ],
            "risk_level":
                original_result[
                    "risk_level"
                ],
        },
        "normal_reference": normal_reference,
        "feature_impacts": impacts,
        "top_risk_drivers": risk_drivers[:3],
        "protective_factors": protective_factors,
        "method": (
            "one-feature-at-a-time perturbation "
            "against same-product-type normal medians"
        ),
        "interpretation_note": (
            "Risk impact represents model sensitivity "
            "in percentage points. It is not a causal effect "
            "or SHAP value."
        ),
    }