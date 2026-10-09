from pprint import pprint

from analysis_tools import (
    compare_conditions,
    get_failure_type_summary,
    get_process_summary,
)


print("=" * 60)
print("TEST 1 - Overall Process Summary")
print("=" * 60)

pprint(get_process_summary())


print("\n" + "=" * 60)
print("TEST 2 - Product Type L")
print("=" * 60)

pprint(get_process_summary("L"))


print("\n" + "=" * 60)
print("TEST 3 - Torque Comparison")
print("=" * 60)

pprint(compare_conditions("Torque"))


print("\n" + "=" * 60)
print("TEST 4 - Tool Wear Comparison")
print("=" * 60)

pprint(compare_conditions("Tool wear"))


print("\n" + "=" * 60)
print("TEST 5 - Failure Type Summary")
print("=" * 60)

pprint(get_failure_type_summary())