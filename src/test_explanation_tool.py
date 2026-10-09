from pprint import pprint

from src.explanation_tool import explain_failure


print("=" * 60)
print("TEST 1 - Risky condition explanation")
print("=" * 60)

result = explain_failure(
    product_type="L",
    air_temperature=301.0,
    process_temperature=310.5,
    rotational_speed=1300,
    torque=65.0,
    tool_wear=200,
)

pprint(result)


print("\n" + "=" * 60)
print("TOP RISK DRIVERS")
print("=" * 60)

for rank, item in enumerate(
    result["top_risk_drivers"],
    start=1,
):
    print(
        f"{rank}. {item['feature']}: "
        f"{item['risk_impact_pct_point']:+.2f}%p"
    )