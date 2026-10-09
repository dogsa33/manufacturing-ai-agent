from pathlib import Path

import duckdb
import plotly.express as px
import plotly.graph_objects as go

from src.explanation_tool import explain_failure


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "database" / "manufacturing.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "generated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_FEATURES = {
    "Air temperature": "air_temperature",
    "Process temperature": "process_temperature",
    "Rotational speed": "rotational_speed",
    "Torque": "torque",
    "Tool wear": "tool_wear",
}


def get_connection():
    return duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )


def create_feature_distribution(
    feature: str,
) -> dict:
    """
    정상/고장 그룹의 특정 제조 변수 분포를
    Histogram으로 생성한다.
    """

    if feature not in ALLOWED_FEATURES:
        raise ValueError(
            f"Invalid feature: {feature}. "
            f"Allowed values: {sorted(ALLOWED_FEATURES)}"
        )

    con = get_connection()

    try:
        query = f"""
        SELECT
            "{feature}",
            "Machine failure"
        FROM manufacturing_data
        """

        df = con.execute(query).fetchdf()

    finally:
        con.close()

    df["Status"] = df["Machine failure"].map(
        {
            0: "Normal",
            1: "Failure",
        }
    )

    fig = px.histogram(
        df,
        x=feature,
        color="Status",
        barmode="overlay",
        nbins=50,
        marginal="box",
        title=f"{feature}: Normal vs Failure",
    )

    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
    )

    filename = (
        f"{ALLOWED_FEATURES[feature]}"
        "_distribution.html"
    )

    output_path = OUTPUT_DIR / filename

    fig.write_html(
        output_path,
        include_plotlyjs="inline",
    )

    return {
        "chart_type": "feature_distribution",
        "feature": feature,
        "file_path": str(output_path),
        "samples": len(df),
    }


def create_failure_rate_by_type() -> dict:
    """
    Product Type별 고장률을 Bar Chart로 생성한다.
    """

    con = get_connection()

    try:
        query = """
        SELECT
            Type,
            COUNT(*) AS samples,
            SUM("Machine failure") AS failures,
            ROUND(
                AVG("Machine failure") * 100,
                2
            ) AS failure_rate
        FROM manufacturing_data
        GROUP BY Type
        ORDER BY Type
        """

        df = con.execute(query).fetchdf()

    finally:
        con.close()

    fig = px.bar(
        df,
        x="Type",
        y="failure_rate",
        text="failure_rate",
        title="Machine Failure Rate by Product Type",
    )

    fig.update_layout(
        xaxis_title="Product Type",
        yaxis_title="Failure Rate (%)",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    output_path = (
        OUTPUT_DIR
        / "failure_rate_by_type.html"
    )

    fig.write_html(
        output_path,
        include_plotlyjs="inline",
    )

    return {
        "chart_type": "failure_rate_by_type",
        "file_path": str(output_path),
        "data": df.to_dict(
            orient="records"
        ),
    }


def create_risk_driver_chart(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    Perturbation 기반 Local Explanation 결과를
    Bar Chart로 시각화한다.
    """

    explanation = explain_failure(
        product_type=product_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )

    impacts = explanation["feature_impacts"]

    features = [
        item["feature"]
        for item in impacts
    ]

    impact_values = [
        item["risk_impact_pct_point"]
        for item in impacts
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=impact_values,
            y=features,
            orientation="h",
            text=[
                f"{value:+.2f}%p"
                for value in impact_values
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=(
            "Local Failure Risk Drivers "
            f"(Prediction: "
            f"{explanation['prediction']['failure_probability_pct']:.2f}%)"
        ),
        xaxis_title=(
            "Risk Impact "
            "(percentage points)"
        ),
        yaxis_title="Feature",
    )

    output_path = (
        OUTPUT_DIR
        / "risk_driver_explanation.html"
    )

    fig.write_html(
        output_path,
        include_plotlyjs="inline",
    )

    return {
        "chart_type": "risk_driver_explanation",
        "file_path": str(output_path),
        "failure_probability_pct":
            explanation[
                "prediction"
            ][
                "failure_probability_pct"
            ],
        "top_risk_drivers":
            explanation[
                "top_risk_drivers"
            ],
        "method":
            explanation["method"],
    }