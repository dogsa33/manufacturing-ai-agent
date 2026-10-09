from pathlib import Path
from ucimlrepo import fetch_ucirepo

# UCI AI4I 2020 Predictive Maintenance Dataset
dataset = fetch_ucirepo(id=601)

df = dataset.data.original

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "ai4i2020.csv"
df.to_csv(output_path, index=False)

print("Dataset saved:", output_path)
print("Shape:", df.shape)
print("\nColumns:")
for col in df.columns:
    print("-", col)

print("\nFirst 5 rows:")
print(df.head())