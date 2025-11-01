import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python scripts/csv_to_parquet.py <input_csv> [output_parquet]")
    raise SystemExit(1)

input_csv = Path(sys.argv[1])
output_parquet = Path(sys.argv[2]) if len(sys.argv) > 2 else input_csv.with_suffix(".parquet")

print(f"[INFO] Reading {input_csv} ...")
df = pd.read_csv(input_csv)
print(f"[INFO] Data shape: {df.shape}")

print(f"[INFO] Saving as Parquet → {output_parquet}")
df.to_parquet(output_parquet, index=False)
print("[OK] Conversion complete ✅")
