from __future__ import annotations

import re
from pathlib import Path
import argparse

import pandas as pd
from scipy.stats import kendalltau, theilslopes
import pymannkendall as mk

from read_mennkendall_xlsx import load_places_from_excel


def _sanitize_identifier(name: str) -> str:
	safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
	safe = re.sub(r"_+", "_", safe).strip("_")
	if not safe:
		return "place"
	if safe[0].isdigit():
		safe = f"place_{safe}"
	return safe


def _load_merged_places(xlsx_path: Path) -> dict[str, pd.DataFrame]:
	sheet1_defaults = ["Year", "Temperature", "Precipitation", "Vapor Pressure"]
	sheet2_defaults = [
		"Year",
		"Soil Wetness (m³/m³)",
		"Relative Humidity (%)",
		"Wind Speed (m/s)",
	]

	places_sheet1 = load_places_from_excel(
		xlsx_path,
		sheet_name=0,
		default_headers=sheet1_defaults,
	)
	places_sheet2 = load_places_from_excel(
		xlsx_path,
		sheet_name=1,
		default_headers=sheet2_defaults,
	)

	all_places = sorted(set(places_sheet1.keys()) | set(places_sheet2.keys()))
	places: dict[str, pd.DataFrame] = {}
	for place_name in all_places:
		df1 = places_sheet1.get(place_name)
		df2 = places_sheet2.get(place_name)

		if df1 is None:
			merged = df2.copy() if df2 is not None else pd.DataFrame()
		elif df2 is None:
			merged = df1.copy()
		else:
			merged = pd.merge(df1, df2, on="Year", how="outer", sort=True)

		if "Year" in merged.columns:
			merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce").astype("Int64")
			merged = merged.sort_values("Year")

		places[place_name] = merged.reset_index(drop=True)

	return places


def _iter_numeric_parameters(df: pd.DataFrame) -> list[str]:
	cols: list[str] = []
	for c in df.columns:
		if c == "Year":
			continue
		# treat as numeric if it can be coerced
		as_num = pd.to_numeric(df[c], errors="coerce")
		if as_num.notna().any():
			cols.append(c)
	return cols


def _aligned_year_value(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
	if "Year" not in df.columns:
		raise KeyError("Year column missing")
	years = pd.to_numeric(df["Year"], errors="coerce")
	values = pd.to_numeric(df[col], errors="coerce")
	mask = years.notna() & values.notna()
	return years[mask].astype(float), values[mask].astype(float)


def _run_mann_kendall(places: dict[str, pd.DataFrame]) -> None:
	for place, df in places.items():
		params = _iter_numeric_parameters(df)
		rows: list[dict[str, object]] = []
		for col in params:
			years, values = _aligned_year_value(df, col)
			if len(values) < 3:
				continue

			# pymannkendall original test uses time order; we sort by year for safety
			order = years.argsort()
			values_sorted = values.iloc[order].to_numpy()
			years_sorted = years.iloc[order].to_numpy()

			mk_res = mk.original_test(values_sorted)
			# also compute Kendall tau against actual year values
			tau, p_tau = kendalltau(years_sorted, values_sorted)

			rows.append(
				{
					"Parameter": col,
					"N": int(len(values_sorted)),
					"Trend": getattr(mk_res, "trend", None),
					"p": float(getattr(mk_res, "p", float("nan"))),
					"Tau": float(getattr(mk_res, "Tau", float("nan"))),
					"KendallTau(year,value)": float(tau) if tau is not None else float("nan"),
					"p_tau": float(p_tau) if p_tau is not None else float("nan"),
				}
			)

		print(f"\n=== {place} (Mann-Kendall) ===")
		if not rows:
			print("No usable numeric columns found.")
			continue
		out = pd.DataFrame(rows).sort_values(["Parameter"]).reset_index(drop=True)
		print(out.to_string(index=False))


def _run_sens_slope(places: dict[str, pd.DataFrame]) -> None:
	for place, df in places.items():
		params = _iter_numeric_parameters(df)
		rows: list[dict[str, object]] = []
		for col in params:
			years, values = _aligned_year_value(df, col)
			if len(values) < 3:
				continue

			order = years.argsort()
			values_sorted = values.iloc[order].to_numpy()
			years_sorted = years.iloc[order].to_numpy()

			slope, intercept, lo_slope, up_slope = theilslopes(values_sorted, years_sorted)
			tau, p_tau = kendalltau(years_sorted, values_sorted)

			rows.append(
				{
					"Parameter": col,
					"N": int(len(values_sorted)),
					"SenSlope(per year)": float(slope),
					"Intercept": float(intercept),
					"SlopeCI_low": float(lo_slope),
					"SlopeCI_high": float(up_slope),
					"KendallTau": float(tau) if tau is not None else float("nan"),
					"p": float(p_tau) if p_tau is not None else float("nan"),
				}
			)

		print(f"\n=== {place} (Sen's slope) ===")
		if not rows:
			print("No usable numeric columns found.")
			continue
		out = pd.DataFrame(rows).sort_values(["Parameter"]).reset_index(drop=True)
		print(out.to_string(index=False))


def _prompt_analysis_choice() -> str:
	print("\nWhat do you want to calculate?")
	print("  1) Mann-Kendall")
	print("  2) Sen's slope")
	choice = input("Select 1 or 2: ").strip().lower()
	if choice in {"1", "mk", "m", "mann", "mann-kendall", "mann kendall"}:
		return "mk"
	if choice in {"2", "sen", "s", "sens", "sen's", "sens slope", "sens-slope"}:
		return "sens"
	print("Invalid choice; defaulting to Mann-Kendall.")
	return "mk"


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Mann-Kendall / Sen's slope analysis for mennkendall.xlsx")
	parser.add_argument(
		"--analysis",
		choices=["mk", "sens"],
		default=None,
		help="Analysis to run. If omitted, you'll be prompted interactively.",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	here = Path(__file__).resolve().parent
	xlsx_path = here / "mennkendall.xlsx"

	places = _load_merged_places(xlsx_path)

	# Create one DataFrame variable per place name.
	for place_name, df in places.items():
		globals()[_sanitize_identifier(place_name)] = df

	choice = args.analysis or _prompt_analysis_choice()
	print(f"\nLoaded {len(places)} areas from: {xlsx_path}")
	print("Parameters are computed for all numeric columns except 'Year'.")

	if choice == "mk":
		_run_mann_kendall(places)
	else:
		_run_sens_slope(places)


if __name__ == "__main__":
	main()
