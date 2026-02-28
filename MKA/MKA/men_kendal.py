from __future__ import annotations

import re
from pathlib import Path
import argparse

import pandas as pd
from scipy.stats import kendalltau, theilslopes
import pymannkendall as mk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from read_mennkendall_xlsx import load_places_from_excel


def _sanitize_identifier(name: str) -> str:
	safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
	safe = re.sub(r"_+", "_", safe).strip("_")
	if not safe:
		return "place"
	if safe[0].isdigit():
		safe = f"place_{safe}"
	return safe


def _safe_filename(s: str) -> str:
	# Windows-friendly filenames
	safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s).strip())
	safe = re.sub(r"_+", "_", safe).strip("_")
	return safe or "item"


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


def _save_trend_plot(
	*,
	out_path: Path,
	years: pd.Series,
	values: pd.Series,
	trend_values: pd.Series,
	title: str,
	ylabel: str,
	note: str | None = None,
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(10, 5))
	plt.plot(years.to_numpy(), values.to_numpy(), label="Observed", color="tab:blue")
	plt.plot(years.to_numpy(), trend_values.to_numpy(), label="Trend", color="tab:red")
	plt.title(title)
	plt.xlabel("Year")
	plt.ylabel(ylabel)
	plt.grid(True, alpha=0.3)
	plt.legend()
	if note:
		ax = plt.gca()
		ax.text(
			0.02,
			0.98,
			note,
			transform=ax.transAxes,
			va="top",
			ha="left",
			fontsize=9,
			bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "gray"},
		)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def _run_mann_kendall(places: dict[str, pd.DataFrame], *, outdir: Path) -> None:
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

			# Trend line for plotting (prefer MK slope/intercept if available)
			if hasattr(mk_res, "slope") and hasattr(mk_res, "intercept"):
				slope = float(getattr(mk_res, "slope"))
				intercept = float(getattr(mk_res, "intercept"))
			else:
				slope, intercept, _, _ = theilslopes(values_sorted, years_sorted)

			trend = slope * years_sorted + intercept
			pval = float(getattr(mk_res, "p", float("nan")))
			trend_label = getattr(mk_res, "trend", None)
			tau_mk = float(getattr(mk_res, "Tau", float("nan")))
			note = f"trend: {trend_label}\np: {pval:.3g}\nTau: {tau_mk:.3f}\nslope: {slope:.4g} / year"
			plot_path = outdir / "mann_kendall" / _safe_filename(place) / f"{_safe_filename(col)}.png"
			_save_trend_plot(
				out_path=plot_path,
				years=pd.Series(years_sorted),
				values=pd.Series(values_sorted),
				trend_values=pd.Series(trend),
				title=f"{place} - {col} (Mann-Kendall)",
				ylabel=col,
				note=note,
			)

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


def _run_sens_slope(places: dict[str, pd.DataFrame], *, outdir: Path) -> None:
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

			trend = slope * years_sorted + intercept
			plot_path = outdir / "sens_slope" / _safe_filename(place) / f"{_safe_filename(col)}.png"
			_save_trend_plot(
				out_path=plot_path,
				years=pd.Series(years_sorted),
				values=pd.Series(values_sorted),
				trend_values=pd.Series(trend),
				title=f"{place} - {col} (Sen's slope)",
				ylabel=col,
			)

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
	parser.add_argument(
		"--outdir",
		default=None,
		help="Folder to save charts (PNG). Defaults to ./outputs next to this script.",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	here = Path(__file__).resolve().parent
	xlsx_path = here / "mennkendall.xlsx"
	outdir = Path(args.outdir) if args.outdir else (here / "outputs")

	places = _load_merged_places(xlsx_path)

	# Create one DataFrame variable per place name.
	for place_name, df in places.items():
		globals()[_sanitize_identifier(place_name)] = df

	choice = args.analysis or _prompt_analysis_choice()
	print(f"\nLoaded {len(places)} areas from: {xlsx_path}")
	print("Parameters are computed for all numeric columns except 'Year'.")
	print(f"Charts will be saved under: {outdir.resolve()}")

	if choice == "mk":
		_run_mann_kendall(places, outdir=outdir)
	else:
		_run_sens_slope(places, outdir=outdir)


if __name__ == "__main__":
	main()
