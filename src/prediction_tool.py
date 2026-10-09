from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.pkl"


ALLOWED_TYPES = {"L", "M", "H"}


# 모델은 파일 로딩 비용을 줄이기 위해 모듈 로드 시 한 번만 읽음
model = joblib.load(MODEL_PATH)


def predict_failure(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    threshold: float = 0.5,
) -> dict:
    """
    제조 조건을 입력받아 Machine Failure 위험을 예측한다.

    Args:
        product_type:
            L, M, H

        air_temperature:
            Air temperature

        process_temperature:
            Process temperature

        rotational_speed:
            Rotational speed

        torque:
            Torque

        tool_wear:
            Tool wear

        threshold:
            Failure 판정 기준 확률.
            기본값 0.5.

    Returns:
        구조화된 예측 결과
    """

    product_type = product_type.upper()

    if product_type not in ALLOWED_TYPES:
        raise ValueError(
            f"Invalid product type: {product_type}. "
            f"Allowed values: {sorted(ALLOWED_TYPES)}"
        )

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    input_df = pd.DataFrame(
        [
            {
                "Type": product_type,
                "Air temperature": air_temperature,
                "Process temperature": process_temperature,
                "Rotational speed": rotational_speed,
                "Torque": torque,
                "Tool wear": tool_wear,
            }
        ]
    )

    failure_probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(
        failure_probability >= threshold
    )

    if failure_probability >= 0.7:
        risk_level = "HIGH"
    elif failure_probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "input": {
            "product_type": product_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
        },
        "failure_probability": round(
            failure_probability,
            4,
        ),
        "failure_probability_pct": round(
            failure_probability * 100,
            2,
        ),
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": (
            "FAILURE"
            if prediction == 1
            else "NORMAL"
        ),
        "risk_level": risk_level,
    }