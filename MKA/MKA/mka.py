import re
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

DATA_PATH = Path("data.json")
variable = "PRECTOTCORR"
# Read file and strip JS-style comments if present
text = DATA_PATH.read_text(encoding="utf-8").strip()
if not text:
    raise ValueError("data.json is empty. Please provide valid JSON with PRECTOTCORR data.")

# Remove // line comments and /* */ block comments
text_no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
text_no_comments = re.sub(r"(^|\s)//.*$", "", text_no_comments, flags=re.M)

try:
    raw = json.loads(text_no_comments)
except json.JSONDecodeError as e:
    raise ValueError(f"data.json is not valid JSON: {e}")

# Helpers to detect precipitation mapping and find it anywhere
def looks_like_prec_map(d: Dict[Any, Any]) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    sample_items = list(d.items())[:5]
    for k, v in sample_items:
        ks = str(k)
        if len(ks) < 6 or not ks[:6].isdigit():
            return False
        if not isinstance(v, (int, float)):
            return False
    return True

def find_prec_map(node: Any) -> Optional[Dict[str, float]]:
    # Direct mapping
    if isinstance(node, dict):
        # If has PRECTOTCORR key and it's a mapping, use it
        for key in node.keys():
            if str(key).upper() == variable and isinstance(node[key], dict) and looks_like_prec_map(node[key]):
                return node[key]
        # If the dict itself looks like the mapping
        if looks_like_prec_map(node):
            return node  # type: ignore
        # Search nested
        for v in node.values():
            found = find_prec_map(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_prec_map(item)
            if found is not None:
                return found
    return None

prec_map = find_prec_map(raw)
if prec_map is None:
    raise KeyError("Could not find PRECTOTCORR mapping anywhere in data.json. Ensure it contains either PRECTOTCORR or a direct YYYYMM:value dict.")

# Convert mapping to rows
rows = []
for k, v in prec_map.items():
    ks = str(k)
    if len(ks) < 6 or not ks[:6].isdigit():
        continue
    year = int(ks[:4])
    month = int(ks[4:6])
    rows.append({"year": year, "month": month, "PRECTOTCORR": float(v)})

if not rows:
    raise ValueError("No valid YYYYMM entries found in PRECTOTCORR mapping.")

df = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)

# Vectorized datetime, invalid months (e.g., 13) become NaT
try:
    df["date"] = pd.to_datetime(
        df.assign(day=1)[["year", "month", "day"]],
        errors="coerce"
    )
except Exception:
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )

precip_df = df[["date", "year", "month", "PRECTOTCORR"]]

# Make date the index for convenience
precip_df = precip_df.set_index("date").sort_index()

import matplotlib.pyplot as plt
import pymannkendall as mk

# Use only valid months (exclude NaT index)
precip_series = precip_df["PRECTOTCORR"].dropna()

# Mann-Kendall trend test
mk_result = mk.original_test(precip_series.values)
print("Mann-Kendall result:", mk_result)

# Build trend line (Sen's slope) over time
idx = precip_series.index
# Numeric time vector (years with fractional months)
time_numeric = idx.year + (idx.month - 1) / 12.0
slope = mk_result.slope if hasattr(mk_result, "slope") else 0.0
# Use positional access with arrays instead of .iloc on Index
t0 = time_numeric[0] if hasattr(time_numeric, "__getitem__") else float(time_numeric)
y0 = precip_series.iloc[0]
intercept = y0 - slope * t0
trend_line = slope * time_numeric + intercept

# Plot precipitation and trend
plt.figure(figsize=(10, 5))
plt.plot(idx, precip_series.to_numpy(), label="PRECTOTCORR", color="tab:blue")
plt.plot(idx, pd.Series(trend_line, index=idx).to_numpy(), label="Sen's trend", color="tab:red")
plt.title("Mann-Kendall Trend for PRECTOTCORR")
plt.xlabel("Date")
plt.ylabel("Precipitation")
plt.legend()
plt.tight_layout()
# plt.show()

# Bar chart of p-values per calendar month
# Group by month across all years and compute MK p-value for each month
month_pvals = []
for m in range(1, 13):
    month_series = precip_df[precip_df["month"] == m]["PRECTOTCORR"].dropna()
    if len(month_series) >= 3:
        res = mk.original_test(month_series.values)
        pval = float(res.p) if hasattr(res, "p") else float("nan")
    else:
        pval = float("nan")
    month_pvals.append({"month": m, "p_value": pval})

pval_df = pd.DataFrame(month_pvals)

import calendar

month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

plt.figure(figsize=(10, 4))
plt.bar(pval_df["month"], pval_df["p_value"], color="tab:green")
plt.xticks(ticks=range(1, 13), labels=month_labels)
plt.ylim(0, 1)
plt.title("Mann-Kendall p-values by Month")
plt.xlabel("Month")
plt.ylabel("p-value")
plt.tight_layout()
plt.show()

print(precip_df.head())
#testing the update