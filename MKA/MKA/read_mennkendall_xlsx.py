from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _sanitize_identifier(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        return "place"
    if safe[0].isdigit():
        safe = f"place_{safe}"
    return safe


def load_places_from_excel(
    xlsx_path: str | Path,
    *,
    sheet_name: str | int = 0,
    data_columns_per_place: int = 4,
) -> Dict[str, pd.DataFrame]:
    """Load Sheet1-style grouped blocks into one DataFrame per place.

    Expected layout (top rows):
    - Row 0: place names, starting each block; typically separated by a blank spacer column
    - Row 1: repeated headers: Year, Temperature, Precipitation, Vapor Pressure
    - Row 2+: numeric data rows

    Returns a dict mapping the original place name (as in Excel) to its DataFrame.
    """

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    raw = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    if raw.shape[0] < 3:
        raise ValueError(
            "Excel sheet is too small; expected at least 3 rows (names, headers, data)."
        )

    place_starts: List[int] = []
    place_names: List[str] = []
    for col_idx, val in enumerate(raw.iloc[0].tolist()):
        if isinstance(val, str) and val.strip():
            place_starts.append(col_idx)
            place_names.append(val.strip())

    if not place_starts:
        raise ValueError(
            "Could not find any place names in the first row. "
            "If they are merged cells, ensure the name appears in the first column of each block."
        )

    results: Dict[str, pd.DataFrame] = {}
    for start_col, place in zip(place_starts, place_names):
        cols = list(range(start_col, start_col + data_columns_per_place))
        headers = raw.iloc[1, cols].tolist()

        # Fall back to expected headers if the row is partially blank.
        default_headers = ["Year", "Temperature", "Precipitation", "Vapor Pressure"]
        cleaned_headers: List[str] = []
        for i, h in enumerate(headers):
            if isinstance(h, str) and h.strip():
                cleaned_headers.append(h.strip())
            else:
                cleaned_headers.append(default_headers[i] if i < len(default_headers) else f"col_{i}")

        df = raw.iloc[2:, cols].copy()
        df.columns = cleaned_headers
        df = df.dropna(how="all")

        # Coerce to numeric where possible.
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if "Year" in df.columns:
            df["Year"] = df["Year"].astype("Int64")

        df = df.reset_index(drop=True)
        results[place] = df

    return results


def main(xlsx_path: Optional[str] = None) -> None:
    here = Path(__file__).resolve().parent
    path = Path(xlsx_path) if xlsx_path else (here / "mennkendall.xlsx")

    places = load_places_from_excel(path, sheet_name=0)

    # Also expose sanitized variable names in globals() for interactive use.
    for place_name, df in places.items():
        globals()[_sanitize_identifier(place_name)] = df

    print(f"Loaded {len(places)} place DataFrames from: {path}")
    print("Names:")
    for name in places.keys():
        print(f"- {name} -> variable: {_sanitize_identifier(name)}")

    print("\n--- DataFrames ---")
    for name, df in places.items():
        print(f"\n[{name}]  shape={df.shape}")
        print(df)


if __name__ == "__main__":
    # Optionally pass a path: python read_mennkendall_xlsx.py path/to/mennkendall.xlsx
    import sys

    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_path)
