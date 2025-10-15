"""
Benchmarking Script: CSV Loading and Statistical Function Performance
=====================================================================
This script compares:
1. CSV vs Pandas data loading performance
2. NumPy vs Pure-Python implementations for basic statistical functions
"""

import os
import time
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# ------------------------------------------------------------
# 1. File Setup
# ------------------------------------------------------------
file_path = "tempLog-1.csv"
runs = 3

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# ------------------------------------------------------------
# 2. Data Loading Functions
# ------------------------------------------------------------
def load_data_csv(file_path):
    """Load CSV using Python's built-in csv module."""
    data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                time_stamp = datetime.fromtimestamp(float(row["Timestamp"]))
                air_temp = float(row["Air"])
                cpu_temp = float(row["CPU"])
                data.append({
                    "Timestamp": time_stamp,
                    "Air": air_temp,
                    "CPU": cpu_temp
                })
            except ValueError:
                continue
    return data


def load_data_pandas(file_path):
    """Load CSV using pandas with date parsing."""
    df = pd.read_csv(
        file_path,
        parse_dates=["Timestamp"],
        date_parser=lambda x: pd.to_datetime(float(x), unit="s")
    )
    df[["Air", "CPU"]] = df[["Air", "CPU"]].astype(float)
    df.fillna(df.mean(), inplace=True)
    return df.to_dict("records")


# ------------------------------------------------------------
# 3. Benchmark Utility
# ------------------------------------------------------------
def benchmark(func, *args, runs=3):
    times = []
    for _ in range(runs):
        start = time.time()
        func(*args)
        times.append(time.time() - start)
    return np.mean(times)


# ------------------------------------------------------------
# 4. Benchmark: Data Loading
# ------------------------------------------------------------
csv_time = benchmark(load_data_csv, file_path, runs=runs)
pandas_time = benchmark(load_data_pandas, file_path, runs=runs)

print("\n Data Loading Performance")
print("------------------------------------------------------------")
print(f"CSV (DictReader): {csv_time:.4f} seconds (avg over {runs} runs)")
print(f"Pandas (read_csv): {pandas_time:.4f} seconds (avg over {runs} runs)")


# ------------------------------------------------------------
# 5. Prepare Data for Statistical Tests
# ------------------------------------------------------------
df = pd.read_csv(file_path)
vals = df["CPU"].dropna().tolist()
print(f"\nLoaded {len(vals)} 'CPU' values for computation.")


# ------------------------------------------------------------
# 6. Statistical Functions
# ------------------------------------------------------------
# NumPy implementations
def mean_np(vals): return np.mean(vals)
def median_np(vals): return np.median(vals)
def mode_np(vals):
    counts = Counter(vals)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return modes[0]
def data_range_np(vals): return np.max(vals) - np.min(vals)

# Pure Python implementations
def mean_py(vals): return sum(vals) / len(vals)
def median_py(vals):
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2
    return (vals[mid - 1] + vals[mid]) / 2 if n % 2 == 0 else vals[mid]
def mode_py(vals):
    counts = Counter(vals)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    return modes[0]
def data_range_py(vals): return max(vals) - min(vals)


# ------------------------------------------------------------
# 7. Benchmark: Statistical Functions
# ------------------------------------------------------------
results = {
    "mean (NumPy)": benchmark(mean_np, vals, runs=runs),
    "mean (Python)": benchmark(mean_py, vals, runs=runs),
    "median (NumPy)": benchmark(median_np, vals, runs=runs),
    "median (Python)": benchmark(median_py, vals, runs=runs),
    "range (NumPy)": benchmark(data_range_np, vals, runs=runs),
    "range (Python)": benchmark(data_range_py, vals, runs=runs),
}

# Display as a table
print("\n Statistical Function Performance")
print("------------------------------------------------------------")
print(f"{'Function':<20} {'Avg Runtime (s)':>15}")
print("------------------------------------------------------------")
for name, t in results.items():
    print(f"{name:<20} {t:>15.6f}")
print("------------------------------------------------------------")

# Summary conclusion
faster_load = "CSV" if csv_time < pandas_time else "Pandas"
print(f"\n {faster_load} loader was faster in this test.")
