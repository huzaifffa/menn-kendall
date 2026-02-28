from __future__ import annotations

import re
from pathlib import Path

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

	places = load_places_from_excel(xlsx_path, sheet_name=0)

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
