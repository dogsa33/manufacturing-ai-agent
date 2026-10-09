from pprint import pprint

from src.prediction_tool import predict_failure


print("=" * 60)
print("TEST 1 - Normal-like condition")
print("=" * 60)

result = predict_failure(
    product_type="L",
    air_temperature=298.2,
    process_temperature=308.7,
    rotational_speed=1500,
    torque=40.0,
    tool_wear=20,
)

pprint(result)


print("\n" + "=" * 60)
print("TEST 2 - Risky condition")
print("=" * 60)

result = predict_failure(
    product_type="L",
    air_temperature=301.0,
    process_temperature=310.5,
    rotational_speed=1300,
    torque=65.0,
    tool_wear=200,
)

pprint(result)