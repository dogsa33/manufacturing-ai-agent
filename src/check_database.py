import duckdb


DB_PATH = "database/manufacturing.duckdb"

con = duckdb.connect(DB_PATH)

query = """
SELECT
    Type,
    COUNT(*) AS samples,
    SUM("Machine failure") AS failures,
    ROUND(AVG("Machine failure") * 100, 2) AS failure_rate
FROM manufacturing_data
GROUP BY Type
ORDER BY Type
"""

result = con.execute(query).fetchdf()

print(result)

con.close()