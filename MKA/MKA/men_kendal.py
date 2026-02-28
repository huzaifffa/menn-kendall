from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from read_mennkendall_xlsx import load_places_from_excel


def _sanitize_identifier(name: str) -> str:
	safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
	safe = re.sub(r"_+", "_", safe).strip("_")
	if not safe:
		return "place"
	if safe[0].isdigit():
		safe = f"place_{safe}"
	return safe


def main() -> None:
	here = Path(__file__).resolve().parent
	xlsx_path = here / "mennkendall.xlsx"

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

	# Merge by Year so each place gets additional parameters from Sheet 2.
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
			if "Year" not in df1.columns or "Year" not in df2.columns:
				merged = pd.concat([df1, df2], axis=1)
			else:
				merged = pd.merge(df1, df2, on="Year", how="outer", sort=True)

		if "Year" in merged.columns:
			merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce").astype("Int64")
			merged = merged.sort_values("Year")

		places[place_name] = merged.reset_index(drop=True)

	# Create one DataFrame variable per place name.
	for place_name, df in places.items():
		globals()[_sanitize_identifier(place_name)] = df

	# Print all DataFrames (pandas will truncate in console automatically).
	print(f"Loaded {len(places)} place DataFrames from: {xlsx_path}")
	for place_name, df in places.items():
		print(f"\n[{place_name}] -> variable: {_sanitize_identifier(place_name)}")
		print(df)


if __name__ == "__main__":
	main()
