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

from read_mennkendall_xlsx import load_places_from_excel, load_climate_sheets_from_excel


MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _prompt_district(places: dict[str, pd.DataFrame]) -> str:
	names = sorted(places.keys())
	if not names:
		raise ValueError("No districts/areas found in the Excel file.")

	print("\nSelect district (sheet name):")
	for i, name in enumerate(names, start=1):
		print(f"  {i}) {name}")
	choice = input(f"Enter 1-{len(names)} (or type name): ").strip()
	if not choice:
		return names[0]
	if choice.isdigit():
		idx = int(choice)
		if 1 <= idx <= len(names):
			return names[idx - 1]

	# match by name (case-insensitive)
	for name in names:
		if name.lower() == choice.lower():
			return name

	print("Invalid district; defaulting to first sheet.")
	return names[0]


def _prompt_parameter(df: pd.DataFrame) -> str:
	params = _list_climate_parameters(df)
	if not params:
		raise ValueError("No parameters found in this district sheet.")

	print("\nSelect parameter:")
	for i, p in enumerate(params, start=1):
		print(f"  {i}) {p}")
	choice = input(f"Enter 1-{len(params)} (or type name): ").strip()
	if not choice:
		return params[0]
	if choice.isdigit():
		idx = int(choice)
		if 1 <= idx <= len(params):
			return params[idx - 1]

	for p in params:
		if p.lower() == choice.lower():
			return p

	print("Invalid parameter; defaulting to first.")
	return params[0]


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


def _load_legacy_mennkendall_places(xlsx_path: Path) -> dict[str, pd.DataFrame]:
	# Legacy format: special block layout across columns with 2 sheets
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


def _detect_input_format(xlsx_path: Path) -> str:
	# Detect based on first sheet columns
	try:
		head = pd.read_excel(xlsx_path, sheet_name=0, nrows=1, engine="openpyxl")
		cols = {str(c).strip().upper() for c in head.columns}
		if "PARAMETER" in cols and "YEAR" in cols:
			return "climate"
	except Exception:
		pass
	return "legacy"


def _load_places(xlsx_path: Path, *, input_format: str) -> dict[str, pd.DataFrame]:
	if input_format == "auto":
		input_format = _detect_input_format(xlsx_path)

	if input_format == "climate":
		# New format: one sheet per area
		return load_climate_sheets_from_excel(xlsx_path)
	if input_format == "legacy":
		return _load_legacy_mennkendall_places(xlsx_path)

	raise ValueError("input_format must be one of: auto, climate, legacy")


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


def _is_climate_wide_format(df: pd.DataFrame) -> bool:
	# New format flattens columns as '<PARAMETER>__<PERIOD>'
	return any("__" in str(c) for c in df.columns)


def _list_climate_parameters(df: pd.DataFrame) -> list[str]:
	params: set[str] = set()
	for c in df.columns:
		s = str(c)
		if "__" in s:
			params.add(s.split("__", 1)[0])
	return sorted(params)


def _monthly_mk_stats(df: pd.DataFrame, base_param: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for m in MONTHS:
		col = f"{base_param}__{m}"
		if col not in df.columns:
			continue
		years, values = _aligned_year_value(df, col)
		if len(values) < 3:
			continue

		order = years.argsort()
		values_sorted = values.iloc[order].to_numpy()
		years_sorted = years.iloc[order].to_numpy()

		res = mk.original_test(values_sorted)
		rows.append(
			{
				"Month": m,
				"N": int(len(values_sorted)),
				# Q is commonly reported as the MK score/statistic; pymannkendall exposes it as 's'
				"Q": int(getattr(res, "s", 0)),
				"Z": float(getattr(res, "z", float("nan"))),
				"p": float(getattr(res, "p", float("nan"))),
				"Trend": getattr(res, "trend", None),
				"Tau": float(getattr(res, "Tau", float("nan"))),
			}
		)

	out = pd.DataFrame(rows)
	if out.empty:
		return out
	# keep month order
	month_order = {m: i for i, m in enumerate(MONTHS)}
	out["_m"] = out["Month"].map(month_order)
	out = out.sort_values(["_m"]).drop(columns=["_m"]).reset_index(drop=True)
	return out


def _monthly_mk_q_only(df: pd.DataFrame, base_param: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for m in MONTHS:
		col = f"{base_param}__{m}"
		if col not in df.columns:
			continue
		years, values = _aligned_year_value(df, col)
		if len(values) < 3:
			continue
		order = years.argsort()
		values_sorted = values.iloc[order].to_numpy()
		res = mk.original_test(values_sorted)
		rows.append(
			{
				"Month": m,
				"Q": int(getattr(res, "s", 0)),
				"Z": float(getattr(res, "z", float("nan"))),
			}
		)

	out = pd.DataFrame(rows)
	if out.empty:
		return out
	month_order = {m: i for i, m in enumerate(MONTHS)}
	out["_m"] = out["Month"].map(month_order)
	out = out.sort_values(["_m"]).drop(columns=["_m"]).reset_index(drop=True)
	return out


def _monthly_means(df: pd.DataFrame, base_param: str) -> pd.Series:
	# Mean across all years for each month (JAN..DEC)
	data: dict[str, float] = {}
	for m in MONTHS:
		col = f"{base_param}__{m}"
		if col not in df.columns:
			data[m] = float("nan")
			continue
		vals = pd.to_numeric(df[col], errors="coerce")
		data[m] = float(vals.mean(skipna=True)) if vals.notna().any() else float("nan")
	return pd.Series(data)


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


def _save_monthly_average_bar_chart(
	*,
	out_path: Path,
	means: pd.Series,
	title: str,
	ylabel: str,
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	vals = [float(means.get(m, float("nan"))) for m in MONTHS]
	plt.figure(figsize=(10, 4))
	plt.bar(MONTHS, vals, color="tab:blue")
	plt.title(title)
	plt.xlabel("Month")
	plt.ylabel(ylabel)
	plt.grid(True, axis="y", alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def _save_monthly_stat_bar_chart(
	*,
	out_path: Path,
	stat_by_month: dict[str, float],
	title: str,
	ylabel: str,
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	vals = [float(stat_by_month.get(m, float("nan"))) for m in MONTHS]
	plt.figure(figsize=(10, 4))
	plt.bar(MONTHS, vals, color="tab:blue")
	plt.title(title)
	plt.xlabel("Month")
	plt.ylabel(ylabel)
	plt.grid(True, axis="y", alpha=0.3)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def _run_mann_kendall(
	places: dict[str, pd.DataFrame],
	*,
	outdir: Path,
	quiet: bool = False,
	include_monthly_tables: bool = True,
) -> None:
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
			z_mk = float(getattr(mk_res, "z", float("nan")))
			note = (
				f"trend: {trend_label}\n"
				f"p: {pval:.3g}\n"
				f"Z: {z_mk:.3f}\n"
				f"Tau: {tau_mk:.3f}\n"
				f"slope: {slope:.4g} / year"
			)
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
					"Z": float(getattr(mk_res, "z", float("nan"))),
					"p": float(getattr(mk_res, "p", float("nan"))),
					"Tau": float(getattr(mk_res, "Tau", float("nan"))),
					"KendallTau(year,value)": float(tau) if tau is not None else float("nan"),
					"p_tau": float(p_tau) if p_tau is not None else float("nan"),
				}
			)

		if not quiet:
			print(f"\n=== {place} (Mann-Kendall) ===")
			if not rows:
				print("No usable numeric columns found.")
				continue
			out = pd.DataFrame(rows).sort_values(["Parameter"]).reset_index(drop=True)
			print(out.to_string(index=False))

		# Monthly MK (Jan..Dec across years) for climate sheet-per-area format
		if (not quiet) and include_monthly_tables and _is_climate_wide_format(df):
			for base_param in _list_climate_parameters(df):
				monthly = _monthly_mk_stats(df, base_param)
				print(f"\n--- {place} : Monthly Mann-Kendall (Q and Z) for '{base_param}' ---")
				if monthly.empty:
					print("No monthly data found.")
				else:
					print(monthly.to_string(index=False))


def _print_selected_district_q_stats(district: str, df: pd.DataFrame) -> None:
	print(f"\n=== {district} : Monthly Q and Z statistics (Mann-Kendall) ===")
	if not _is_climate_wide_format(df):
		print("This input format has no monthly columns (JAN..DEC).")
		print("Q-statistics are not computed monthly for legacy format.")
		return

	params = _list_climate_parameters(df)
	if not params:
		print("No parameters found.")
		return

	for base_param in params:
		qtab = _monthly_mk_q_only(df, base_param)
		print(f"\nParameter: {base_param}")
		if qtab.empty:
			print("No monthly data found.")
		else:
			print(qtab.to_string(index=False))


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
		"--file",
		default=None,
		help="Path to input Excel file. Defaults to ./climate_data.xlsx if present, otherwise ./mennkendall.xlsx.",
	)
	parser.add_argument(
		"--format",
		choices=["auto", "climate", "legacy"],
		default="auto",
		help="Input format. 'climate' = one sheet per area with PARAMETER/YEAR/JAN..ANN; 'legacy' = old block layout.",
	)
	parser.add_argument(
		"--district",
		default=None,
		help="District/sheet name to analyze for Mann-Kendall Q-stats. If omitted, you'll be prompted.",
	)
	parser.add_argument(
		"--parameter",
		default=None,
		help="Parameter to analyze in Mann-Kendall mode (climate format). If omitted, you'll be prompted.",
	)
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
	default_climate = here / "climate_data.xlsx"
	default_legacy = here / "mennkendall.xlsx"
	xlsx_path = Path(args.file) if args.file else (default_climate if default_climate.exists() else default_legacy)
	outdir = Path(args.outdir) if args.outdir else (here / "outputs")

	places = _load_places(xlsx_path, input_format=args.format)

	# Create one DataFrame variable per place name.
	for place_name, df in places.items():
		globals()[_sanitize_identifier(place_name)] = df

	choice = args.analysis or _prompt_analysis_choice()
	print(f"\nLoaded {len(places)} areas from: {xlsx_path}")
	print("Parameters are computed for all numeric columns except 'Year'.")
	print(f"Charts will be saved under: {outdir.resolve()}")

	if choice == "mk":
		district = args.district or _prompt_district(places)
		if district not in places:
			print("District not found; defaulting to first sheet.")
			district = sorted(places.keys())[0]
		df = places[district]

		# Climate-format: one parameter -> one 12-bar monthly-average chart
		if _is_climate_wide_format(df):
			param = args.parameter or _prompt_parameter(df)
			print(f"\n=== {district} : Monthly Q and Z statistics (Mann-Kendall) ===")
			print(f"\nParameter: {param}")
			qtab = _monthly_mk_q_only(df, param)
			if qtab.empty:
				print("No monthly data found.")
			else:
				print(qtab.to_string(index=False))
				# Save Q+Z table alongside the chart
				qz_path = outdir / "mann_kendall" / _safe_filename(district) / f"{_safe_filename(param)}_QZ_stats.csv"
				qz_path.parent.mkdir(parents=True, exist_ok=True)
				qtab.to_csv(qz_path, index=False)
				print(f"\nSaved Q+Z table: {qz_path}")

				# Save Z-stats bar chart alongside the existing chart
				z_by_month = {str(r["Month"]): float(r["Z"]) for r in qtab[["Month", "Z"]].to_dict("records")}
				z_chart_path = outdir / "mann_kendall" / _safe_filename(district) / f"{_safe_filename(param)}_Z_stats.png"
				_save_monthly_stat_bar_chart(
					out_path=z_chart_path,
					stat_by_month=z_by_month,
					title=f"{district} - {param} (Z-stats by month)",
					ylabel="Z",
				)
				print(f"\nSaved Z chart: {z_chart_path}")

			means = _monthly_means(df, param)
			chart_path = outdir / "mann_kendall" / _safe_filename(district) / f"{_safe_filename(param)}_Q_stats.png"
			_save_monthly_average_bar_chart(
				out_path=chart_path,
				means=means,
				title=f"{district} - {param} (Q-stats by month)",
				ylabel=param,
			)
			print(f"\nSaved chart: {chart_path}")
		else:
			# Legacy format fallback
			_print_selected_district_q_stats(district, df)
			_run_mann_kendall({district: df}, outdir=outdir, quiet=True, include_monthly_tables=False)
	else:
		_run_sens_slope(places, outdir=outdir)


if __name__ == "__main__":
	main()
