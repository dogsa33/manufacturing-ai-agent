from pathlib import Path

import pandas as pd
import plotly.express as px


DATA_PATH = Path("data/ai4i2020.csv")
FIGURE_DIR = Path("reports/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Load data
# =========================================================
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print(f"Shape: {df.shape}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicated rows:")
print(df.duplicated().sum())


# =========================================================
# 2. Failure distribution
# =========================================================
print("\n" + "=" * 60)
print("2. MACHINE FAILURE DISTRIBUTION")
print("=" * 60)

failure_counts = df["Machine failure"].value_counts().sort_index()
failure_rates = (
    df["Machine failure"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

print("\nCounts:")
print(failure_counts)

print("\nRates (%):")
print(failure_rates.round(2))


# =========================================================
# 3. Product type analysis
# =========================================================
print("\n" + "=" * 60)
print("3. PRODUCT TYPE")
print("=" * 60)

type_summary = (
    df.groupby("Type")
    .agg(
        samples=("Machine failure", "size"),
        failures=("Machine failure", "sum"),
        failure_rate=("Machine failure", "mean"),
    )
)

type_summary["failure_rate"] *= 100

print(type_summary.round(2))


# =========================================================
# 4. Normal vs Failure comparison
# =========================================================
print("\n" + "=" * 60)
print("4. NORMAL VS FAILURE")
print("=" * 60)

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

comparison = (
    df.groupby("Machine failure")[numeric_features]
    .mean()
    .T
)

comparison.columns = ["Normal", "Failure"]
comparison["Difference"] = comparison["Failure"] - comparison["Normal"]

print(comparison.round(2))


# =========================================================
# 5. Failure type counts
# =========================================================
print("\n" + "=" * 60)
print("5. FAILURE TYPES")
print("=" * 60)

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

failure_type_counts = df[failure_types].sum().sort_values(ascending=False)

print(failure_type_counts)


# =========================================================
# 6. Save basic visualizations
# =========================================================

# Machine failure distribution
failure_plot_df = (
    df["Machine failure"]
    .value_counts()
    .rename_axis("Machine failure")
    .reset_index(name="Count")
)

fig = px.bar(
    failure_plot_df,
    x="Machine failure",
    y="Count",
    title="Machine Failure Distribution",
)

fig.write_html(FIGURE_DIR / "failure_distribution.html")


# Product type failure rate
type_plot_df = type_summary.reset_index()

fig = px.bar(
    type_plot_df,
    x="Type",
    y="failure_rate",
    title="Failure Rate by Product Type",
)

fig.write_html(FIGURE_DIR / "failure_rate_by_type.html")


# Torque distribution
fig = px.histogram(
    df,
    x="Torque",
    color="Machine failure",
    barmode="overlay",
    title="Torque Distribution: Normal vs Failure",
)

fig.write_html(FIGURE_DIR / "torque_distribution.html")


# Tool wear distribution
fig = px.histogram(
    df,
    x="Tool wear",
    color="Machine failure",
    barmode="overlay",
    title="Tool Wear Distribution: Normal vs Failure",
)

fig.write_html(FIGURE_DIR / "tool_wear_distribution.html")


print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(f"Figures saved to: {FIGURE_DIR}")