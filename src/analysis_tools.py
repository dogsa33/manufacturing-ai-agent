from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "database" / "manufacturing.duckdb"


# Agent가 임의 컬럼명을 SQL에 집어넣지 못하도록 허용 목록 정의
ALLOWED_FEATURES = {
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
}

ALLOWED_TYPES = {"L", "M", "H"}


def get_connection():
    """DuckDB connection 생성."""
    return duckdb.connect(str(DB_PATH), read_only=True)


def get_process_summary(product_type: str | None = None) -> dict:
    """
    전체 또는 특정 Product Type의 제조 데이터 요약.

    Args:
        product_type:
            None -> 전체
            "L", "M", "H" -> 해당 Type만 조회
    """

    con = get_connection()

    try:
        if product_type is None:
            query = """
            SELECT
                COUNT(*) AS samples,
                SUM("Machine failure") AS failures,
                ROUND(AVG("Machine failure") * 100, 2) AS failure_rate,
                ROUND(AVG("Air temperature"), 2) AS avg_air_temperature,
                ROUND(AVG("Process temperature"), 2) AS avg_process_temperature,
                ROUND(AVG("Rotational speed"), 2) AS avg_rotational_speed,
                ROUND(AVG("Torque"), 2) AS avg_torque,
                ROUND(AVG("Tool wear"), 2) AS avg_tool_wear
            FROM manufacturing_data
            """

            result = con.execute(query).fetchone()

        else:
            product_type = product_type.upper()

            if product_type not in ALLOWED_TYPES:
                raise ValueError(
                    f"Invalid product type: {product_type}. "
                    f"Allowed values: {sorted(ALLOWED_TYPES)}"
                )

            query = """
            SELECT
                COUNT(*) AS samples,
                SUM("Machine failure") AS failures,
                ROUND(AVG("Machine failure") * 100, 2) AS failure_rate,
                ROUND(AVG("Air temperature"), 2) AS avg_air_temperature,
                ROUND(AVG("Process temperature"), 2) AS avg_process_temperature,
                ROUND(AVG("Rotational speed"), 2) AS avg_rotational_speed,
                ROUND(AVG("Torque"), 2) AS avg_torque,
                ROUND(AVG("Tool wear"), 2) AS avg_tool_wear
            FROM manufacturing_data
            WHERE Type = ?
            """

            result = con.execute(query, [product_type]).fetchone()

        return {
            "product_type": product_type or "ALL",
            "samples": int(result[0]),
            "failures": int(result[1]),
            "failure_rate": float(result[2]),
            "avg_air_temperature": float(result[3]),
            "avg_process_temperature": float(result[4]),
            "avg_rotational_speed": float(result[5]),
            "avg_torque": float(result[6]),
            "avg_tool_wear": float(result[7]),
        }

    finally:
        con.close()


def compare_conditions(feature: str) -> dict:
    """
    정상 데이터와 고장 데이터에서 특정 공정 변수 평균을 비교.
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
            "Machine failure",
            COUNT(*) AS samples,
            ROUND(AVG("{feature}"), 2) AS mean_value,
            ROUND(MEDIAN("{feature}"), 2) AS median_value,
            ROUND(STDDEV_SAMP("{feature}"), 2) AS std_value
        FROM manufacturing_data
        GROUP BY "Machine failure"
        ORDER BY "Machine failure"
        """

        rows = con.execute(query).fetchall()

        normal = rows[0]
        failure = rows[1]

        difference = round(float(failure[2]) - float(normal[2]), 2)

        return {
            "feature": feature,
            "normal": {
                "samples": int(normal[1]),
                "mean": float(normal[2]),
                "median": float(normal[3]),
                "std": float(normal[4]),
            },
            "failure": {
                "samples": int(failure[1]),
                "mean": float(failure[2]),
                "median": float(failure[3]),
                "std": float(failure[4]),
            },
            "mean_difference": difference,
        }

    finally:
        con.close()


def get_failure_type_summary() -> dict:
    """
    세부 고장 유형별 발생 건수 조회.
    """

    con = get_connection()

    try:
        query = """
        SELECT
            SUM(TWF) AS TWF,
            SUM(HDF) AS HDF,
            SUM(PWF) AS PWF,
            SUM(OSF) AS OSF,
            SUM(RNF) AS RNF
        FROM manufacturing_data
        """

        result = con.execute(query).fetchone()

        failure_types = {
            "TWF": int(result[0]),
            "HDF": int(result[1]),
            "PWF": int(result[2]),
            "OSF": int(result[3]),
            "RNF": int(result[4]),
        }

        sorted_failure_types = dict(
            sorted(
                failure_types.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return {
            "failure_types": sorted_failure_types,
            "most_common_failure": next(iter(sorted_failure_types)),
        }
    finally:
        con.close()