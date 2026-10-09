from pathlib import Path

import duckdb


CSV_PATH = Path("data/ai4i2020.csv")
DB_PATH = Path("database/manufacturing.duckdb")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


con = duckdb.connect(str(DB_PATH))


# 기존 테이블이 있으면 다시 생성
con.execute("""
DROP TABLE IF EXISTS manufacturing_data
""")


con.execute(f"""
CREATE TABLE manufacturing_data AS
SELECT *
FROM read_csv_auto('{CSV_PATH.as_posix()}')
""")


# 데이터 개수 확인
row_count = con.execute("""
SELECT COUNT(*)
FROM manufacturing_data
""").fetchone()[0]


# 컬럼 확인
columns = con.execute("""
DESCRIBE manufacturing_data
""").fetchdf()


# 고장 분포 확인
failure_summary = con.execute("""
SELECT
    "Machine failure",
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS rate_pct
FROM manufacturing_data
GROUP BY "Machine failure"
ORDER BY "Machine failure"
""").fetchdf()


print("=" * 60)
print("DUCKDB SETUP COMPLETE")
print("=" * 60)

print(f"Database: {DB_PATH}")
print(f"Rows: {row_count}")

print("\nColumns:")
print(columns[["column_name", "column_type"]])

print("\nFailure summary:")
print(failure_summary)


con.close()