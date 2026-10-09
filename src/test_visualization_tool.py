from pprint import pprint

from visualization_tool import (
    create_failure_rate_by_type,
    create_feature_distribution,
    create_risk_driver_chart,
)


print("=" * 60)
print("TEST 1 - Torque Distribution")
print("=" * 60)

result = create_feature_distribution(
    "Torque"
)

pprint(result)


print("\n" + "=" * 60)
print("TEST 2 - Failure Rate by Type")
print("=" * 60)

result = create_failure_rate_by_type()

pprint(result)


print("\n" + "=" * 60)
print("TEST 3 - Risk Driver Chart")
print("=" * 60)

result = create_risk_driver_chart(
    product_type="L",
    air_temperature=301.0,
    process_temperature=310.5,
    rotational_speed=1300,
    torque=65.0,
    tool_wear=200,
)

pprint(result)