"""
build_rating_coupon_tables.py
============================

Panel regressions of sovereign credit ratings and bond coupons on macro
controls, following Beirne et al. (2021). Produces SI Tables T5-T8 in
3_output/S_tables/{rating,coupon}/ (.tex, with CSVs in data/).

  rating_table -- T5 (by subset, country + year FE), T6 (ISO-level dummies)
  coupon_table -- T7 (currency + year FE), T8 (US 10y yield + VIX)

Requires the proprietary raw Bloomberg/LSEG bond data and agency rating histories in
0_raw/credit_ratings/ (not shipped; see README), processed in memory by prep.py.

Usage:
  python3 build_rating_coupon_tables.py                       # all four tables
  python3 build_rating_coupon_tables.py --section rating_table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Allow running this script directly: add 2_code/common/ to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "2_code" / "common"))

from prep import (  # noqa: E402
    PATHS,
    load_and_process_all,
    INDEPENDENT_VARS,
    WGI_VARS,
    COUNTRY_GROUPS,
)

# =============================================================================
# Paths
# =============================================================================

OUTPUT_BASE = _PROJECT_ROOT / "3_output" / "S_tables"
TC_CSV      = _PROJECT_ROOT / "0_raw" / "climate_exposures" / "tc2_maxws.csv"

# =============================================================================
# Loaders
# =============================================================================

def load_tc_panel() -> pd.DataFrame:
    """Per-ISO-year TC exposure (avg_ws, totalexposed, ts_v2, cat1_v2, cat3_v2)."""
    tc = pd.read_csv(TC_CSV)
    return tc[["ISO", "year", "avg_ws", "totalexposed",
                "ts_v2", "cat1_v2", "cat3_v2"]].copy()


def load_global_factors() -> pd.DataFrame:
    """Annual global factors used by Beirne et al. (2021): US 10y + VIX."""
    us = pd.read_csv(PATHS["US_10yr"])
    us["observation_date"] = pd.to_datetime(us["observation_date"])
    us["year"] = us["observation_date"].dt.year
    us = (us.rename(columns={"IRLTLT01USA156N": "US_10yr_yield"})
            .groupby("year", as_index=False)["US_10yr_yield"].mean())
    vix = pd.read_csv(_PROJECT_ROOT / "1_input" / "credit_ratings" / "vix.csv")
    vix.columns = [c.strip() for c in vix.columns]
    return us.merge(vix, on="year", how="outer")


def tc_exposed_isos() -> tuple[set[str], set[str]]:
    """ISOs ever exposed (1980+) to cat1+ and to cat3+ winds, excluding
    extratropical countries (whole territory outside latitudes +-23.5)."""
    import geopandas as gpd
    tc = pd.read_csv(PATHS["tc"], usecols=["ISO", "year", "cat1_v2", "cat3_v2"])
    tc_max = tc[tc["year"] >= 1980].groupby("ISO")[["cat1_v2", "cat3_v2"]].max()
    _world = gpd.read_file(PATHS["world_shapefile"])
    _b = (_world[["ADM0_A3", "geometry"]].dropna(subset=["ADM0_A3"])
          .assign(_miny=lambda d: d.geometry.bounds["miny"],
                  _maxy=lambda d: d.geometry.bounds["maxy"])
          .groupby("ADM0_A3").agg(miny=("_miny", "min"), maxy=("_maxy", "max")))
    extratropical = set(_b.index[(_b["miny"] > 23.5) | (_b["maxy"] < -23.5)])
    cat1 = set(tc_max.index[tc_max["cat1_v2"] > 0]) - extratropical
    cat3 = set(tc_max.index[tc_max["cat3_v2"] > 0]) - extratropical
    return cat1, cat3


_BONDS = None


def bond_data() -> pd.DataFrame:
    """Bond-level panel (one row per bond, with the country-year rating and macro data),
    built once from the raw Bloomberg/LSEG files by prep.load_and_process_all."""
    global _BONDS
    if _BONDS is None:
        _BONDS = load_and_process_all()['processed_data']
    return _BONDS.copy()


def load_coupon_panel(min_year: int = 1990) -> tuple[pd.DataFrame, list[str]]:
    """Bond-level coupon panel with TC merged."""
    df = bond_data()
    df = df[df["Cpn"].notna() & (df["Cpn"] > 0) & (df["Cpn"] < 30)].copy()
    df = df[df["maturity"] == 10].copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= min_year].copy()

    # TC merge by ISO + year (= issue year)
    tc = load_tc_panel()
    df = df.merge(tc, on=["ISO", "year"], how="left")
    df["avg_ws"] = df["avg_ws"].fillna(0.0)

    macro_cols = [c for c in INDEPENDENT_VARS + WGI_VARS + ["OECD"] if c in df.columns]
    return df, macro_cols


def load_rating_panel() -> tuple[pd.DataFrame, list[str]]:
    """Rating panel (one row per ISO-year) with macro + TC."""
    df = bond_data()
    macro_cols = [c for c in INDEPENDENT_VARS + WGI_VARS + ["OECD"] if c in df.columns]
    keep = ["ISO", "year", "score_total", "avgscore"] + macro_cols
    rating = (df[keep]
              .dropna(subset=["ISO", "year", "score_total"])
              .groupby(["ISO", "year"], as_index=False).first())
    rating = rating.merge(load_tc_panel(), on=["ISO", "year"], how="left")
    rating["avg_ws"] = rating["avg_ws"].fillna(0.0)
    return rating, macro_cols


# =============================================================================
# Table building (Beirne-style layout)
# =============================================================================

CONTROLS_BEIRNE = [
    "Current account balance", "log(GDP pcap)", "gross debt ratio",
    "deficit", "GDP growth rate", "default_total",
    "US_10yr_yield", "annual_avg_vix",
]
PRETTY = {
    "Current account balance": "Current account/GDP",
    "log(GDP pcap)":           "GDP per capita (log)",
    "gross debt ratio":        "Public debt/GDP",
    "deficit":                 "Primary balance",
    "GDP growth rate":         "GDP growth",
    "default_total":           "Crisis (sovereign default)",
    "US_10yr_yield":           "US 10-yr bond yield",
    "annual_avg_vix":          "VIX",
    "log_reserves_usd":        "log(Reserves USD)",
    "fdi":                     "FDI",
    "inflation":               "Inflation",
    "Unemployment rate":       "Unemployment",
    "govexp_GDP":              "Govt expenditure / GDP",
    "govrev_GDP":              "Govt revenue / GDP",
    "VA":                      "WGI: Voice & Accountability",
    "PS":                      "WGI: Political Stability",
    "GE":                      "WGI: Government Effectiveness",
    "RQ":                      "WGI: Regulatory Quality",
    "RL":                      "WGI: Rule of Law",
    "CC":                      "WGI: Control of Corruption",
    "V20":                     "V20",
    "is_dominant_currency":    "Dominant currency",
    "usd_denominated":         "USD-denominated yield (vs local)",
    "avgscore":                "Sovereign credit rating",
    "score_total":             "Sovereign credit rating",
    "default_total_past5":     "Sovereign default (past 5y)",
    "CurrencyCrisis":          "Currency crisis",
    "BankingCrisis":           "Banking crisis",
    "cat3_v2":                 "Cat-3+ TC exposure (t)",
    "cat3_lag1":               "Cat-3+ TC exposure (t-1)",
    "cat3_lag2":               "Cat-3+ TC exposure (t-2)",
    "cat1_v2":                 "Cat-1+ TC exposure (t)",
    "ts_v2":                   "TS exposure (t)",
    "TC_prone":                "TC-prone country (any cat1+ since 1980)",
}

# Display order for the variables-of-interest section (top of table).
# Anything not listed here falls into a third group rendered above the
# global-factors block. Global factors (US 10y, VIX) are rendered separately
# with a horizontal divider above them.
DISPLAY_ORDER_PRIMARY = [
    "Public debt/GDP",
    "GDP per capita (log)",
    "GDP growth",
    "Primary balance",
    "Current account/GDP",
    "Sovereign credit rating",
    "Dominant currency",
    "V20",
    "Crisis (sovereign default)",
    "Sovereign default (past 5y)",
    "Currency crisis",
    "Banking crisis",
    "Cat-3+ TC exposure (t)",
    "Cat-3+ TC exposure (t-1)",
    "Cat-3+ TC exposure (t-2)",
    "Cat-1+ TC exposure (t)",
    "TS exposure (t)",
    "TC-prone country (any cat1+ since 1980)",
]
DISPLAY_ORDER_GLOBAL = [
    "US 10-yr bond yield",
    "VIX",
]
DOMINANT_CURRENCY_FOOTNOTE = (
    "``Dominant currency'' = bond issued in USD, EUR, GBP, CAD, or JPY.")

# Definitions for the ISO-level dummy variables / subgroups used in the
# rating ISO-dummies and coupon Tables 1-2. Rendered in the tablenotes
# block when the table contains any of these names.
DUMMY_DEFINITIONS = {
    "below_IG":
        r"\textit{below\_IG}: countries with $<$BBB- credit rating in 2019.",
    "SIDS":
        r"\textit{SIDS}: Small Island Developing States.",
    "TC_exposed1":
        r"\textit{TC\_exposed1}: countries exposed to $>$cat1 wind speeds, "
        r"excluding countries that fall outside tropical latitudes "
        r"($\pm$23.5$^\circ$).",
    "TC_exposed2":
        r"\textit{TC\_exposed2}: countries exposed to $>$cat3 wind speeds, "
        r"excluding countries that fall outside tropical latitudes "
        r"($\pm$23.5$^\circ$).",
    "TC_exposed":
        r"\textit{TC\_exposed}: countries exposed to $>$cat1 wind speeds, "
        r"excluding countries that fall outside tropical latitudes "
        r"($\pm$23.5$^\circ$).",
}


def _dummy_notes_for_table(df: pd.DataFrame) -> list[str]:
    """Return tablenotes items defining any ISO-level dummies that appear
    in the table (as column headers or variable names)."""
    cols = set(str(c) for c in df.columns)
    var_names = set(str(v) for v in df["Variable"].tolist()) if "Variable" in df.columns else set()
    text = " ".join(list(cols) + list(var_names))
    notes = []
    # Order: below_IG, SIDS, TC_exposed (or 1/2). Detection by substring so
    # variants like "below_IG_dummy" and "+ below_IG" both match.
    if "below_IG" in text:
        notes.append(DUMMY_DEFINITIONS["below_IG"])
    if "SIDS" in text:
        notes.append(DUMMY_DEFINITIONS["SIDS"])
    if "TC_exposed1" in text:
        notes.append(DUMMY_DEFINITIONS["TC_exposed1"])
    if "TC_exposed2" in text:
        notes.append(DUMMY_DEFINITIONS["TC_exposed2"])
    # Plain "TC_exposed" only when neither of the numbered variants is present
    if "TC_exposed" in text and "TC_exposed1" not in text and "TC_exposed2" not in text:
        notes.append(DUMMY_DEFINITIONS["TC_exposed"])
    return notes


def _fmt_coef(coef: float, se: float) -> str:
    if se is None or not np.isfinite(se) or se == 0:
        return f"{coef:.4f}"
    t = coef / se
    stars = ("***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
              else "*" if abs(t) > 1.645 else "")
    return f"{coef:.4f}{stars} ({se:.4f})"


def _build_table(panel: pd.DataFrame, y_col: str,
                               X_vars: list[str], extra_fe_dummies: pd.DataFrame | None,
                               groups: dict[str, set[str]],
                               year_range: tuple[int, int],
                               country_fe: bool = True,
                               year_fe_label: str = "auto",
                               freq_label: str = "Annual") -> pd.DataFrame:
    """Run country-FE-only OLS for each group on (panel, y_col) and assemble
    the Beirne-Table-1-style wide output."""
    sub_full = panel[(panel["year"] >= year_range[0])
                     & (panel["year"] <= year_range[1])].copy()
    results = {}
    for grp_name, isos in groups.items():
        sub = sub_full[sub_full["ISO"].isin(isos)] if isos else sub_full
        keep = [y_col, "ISO", "year"] + X_vars
        s = sub[[c for c in keep if c in sub.columns]].copy()
        if extra_fe_dummies is not None:
            s = pd.concat([s, extra_fe_dummies.loc[s.index]], axis=1)
        s = s.dropna()
        if len(s) < 25 or s["ISO"].nunique() < 3:
            results[grp_name] = None
            continue
        Xparts = [s[X_vars].astype(float)]
        if extra_fe_dummies is not None:
            extra_cols = [c for c in extra_fe_dummies.columns if c in s.columns]
            if extra_cols:
                Xparts.append(s[extra_cols].astype(float))
        if country_fe:
            Xparts.append(pd.get_dummies(s["ISO"], prefix="iso", drop_first=True).astype(float))
        Xmat = sm.add_constant(pd.concat(Xparts, axis=1))
        try:
            res = sm.OLS(s[y_col].astype(float), Xmat).fit(
                cov_type="cluster", cov_kwds={"groups": s["ISO"].values})
        except Exception as e:
            print(f"  [{grp_name}] fit failed: {e}")
            results[grp_name] = None
            continue
        results[grp_name] = (res, s)

    rows = []
    for var in X_vars:
        row = {"Variable": PRETTY.get(var, var)}
        for grp in groups:
            r = results[grp]
            row[grp] = ("" if r is None or var not in r[0].params.index
                        else _fmt_coef(r[0].params[var], r[0].bse[var]))
        rows.append(row)
    crow = {"Variable": "Constant"}
    for grp in groups:
        r = results[grp]
        crow[grp] = ("" if r is None or "const" not in r[0].params.index
                     else _fmt_coef(r[0].params["const"], r[0].bse["const"]))
    rows.append(crow)
    for label, fn in [
        ("Observations",          lambda r, sub: int(r.nobs)),
        ("R-squared",             lambda r, sub: round(r.rsquared, 3)),
        ("Number of countries",   lambda r, sub: int(sub["ISO"].nunique())),
        ("Country fixed effects", lambda r, sub: "Yes" if country_fe else "No"),
        ("Currency fixed effects",
         lambda r, sub: ("Yes" if extra_fe_dummies is not None
                          and any(c.startswith("cur_")
                                  for c in extra_fe_dummies.columns)
                          else "No")),
        ("Year fixed effects",
         lambda r, sub: ("Yes" if extra_fe_dummies is not None
                          and any(c.startswith("yr_")
                                  for c in extra_fe_dummies.columns)
                          else "No")),
        ("Time period",           lambda r, sub: f"{int(sub['year'].min())}-{int(sub['year'].max())}"),
        ("Frequency",             lambda r, sub: freq_label),
    ]:
        row = {"Variable": label}
        for grp in groups:
            r = results[grp]
            row[grp] = "" if r is None else fn(r[0], r[1])
        rows.append(row)
    return pd.DataFrame(rows)


def _split_coef_se(val) -> tuple[str, str]:
    """Split 'coef*** (SE)' into ('coef***', '(SE)'). Returns (val, '') if no SE.
    NaN / None / empty → ('', '')."""
    # Treat NaN, None, and empty as blank
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "", ""
    except (TypeError, ValueError):
        pass
    if not isinstance(val, str):
        sv = str(val)
        if sv.lower() in ("nan", "none"):
            return "", ""
        return sv, ""
    s = val.strip()
    if not s or s.lower() == "nan":
        return "", ""
    idx = s.find(" (")
    if idx == -1 or not s.endswith(")"):
        return s, ""
    return s[:idx], s[idx + 1:]


_LATEX_SPECIALS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                    ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")]


def _latex_escape(s) -> str:
    # Treat NaN / None as empty string
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
    except (TypeError, ValueError):
        pass
    if not isinstance(s, str):
        s = str(s)
    if s.lower() in ("nan", "none"):
        return ""
    out = s
    for ch, rep in _LATEX_SPECIALS:
        out = out.replace(ch, rep)
    return out


def _render_table_latex(df: pd.DataFrame, out_path: Path, title: str = "",
                         label: str = "", footnote: str | None = None) -> None:
    """Write a standalone-compilable .tex document (pdflatex) containing a Beirne-style
    coefficient table (booktabs + threeparttable).

    Layout:
      1. Variables-of-interest block (DISPLAY_ORDER_PRIMARY first, then any
         remaining variables not in primary or global-factor lists).
      2. \\midrule, then global-factors block (US 10-yr bond yield, VIX).
      3. \\midrule, then summary block (Constant, Observations, R-squared,
         Number of countries, Country fixed effects, Time period, Frequency).
    """
    n_cols = len(df.columns)
    if n_cols == 0:
        return
    col_align = "l" + "c" * (n_cols - 1)
    headers = [_latex_escape(c) for c in df.columns]

    # ISO-dummy tables: Constant goes in the summary block next to Observations.
    constant_in_summary = "iso-dummies" in (label or "")

    summary_labels = {"Observations", "R-squared",
                       "Number of countries", "Country fixed effects",
                       "Currency fixed effects", "Year fixed effects",
                       "Time period", "Frequency"}
    if constant_in_summary:
        summary_labels = summary_labels | {"Constant"}

    # Bucket the variable rows
    var_rows = {}      # variable name -> row dict
    summary_rows = []  # list of (label, row dict)
    for _, row in df.iterrows():
        name = str(row.iloc[0])
        # Drop the "Currency fixed effects" row when it's "No" across all
        # columns (rating tables have no currency FE; the row is uninformative).
        if name == "Currency fixed effects":
            vals = [str(v).strip() for v in row.iloc[1:]]
            if all(v in ("No", "", "nan") for v in vals):
                continue
        if name in summary_labels:
            summary_rows.append((name, row))
        else:
            var_rows[name] = row

    # Write rows in the requested order
    primary_block = [var_rows[n] for n in DISPLAY_ORDER_PRIMARY if n in var_rows]
    global_block  = [var_rows[n] for n in DISPLAY_ORDER_GLOBAL  if n in var_rows]
    listed = set(DISPLAY_ORDER_PRIMARY) | set(DISPLAY_ORDER_GLOBAL)
    other_block   = [r for n, r in var_rows.items() if n not in listed]

    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,portrait,margin=0.35in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{threeparttable}",
        r"\usepackage{caption}",
        r"\usepackage{array}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{document}",
        r"\thispagestyle{empty}",
        r"\begin{table}[h!]",
        r"  \centering",
        r"  \scriptsize",
    ]
    if title:
        lines.append(rf"  \caption{{{_latex_escape(title)}}}")
    if label:
        lines.append(rf"  \label{{{label}}}")
    lines += [
        r"  \begin{threeparttable}",
        rf"  \begin{{tabular}}{{{col_align}}}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]
    def _var_rows_to_tex(row):
        """Two rows per variable: coef on first row, (SE) below."""
        cells = list(row.values)
        var_name = _latex_escape(cells[0])
        splits = [_split_coef_se(c) for c in cells[1:]]
        coefs = [_latex_escape(s[0]) for s in splits]
        ses   = [_latex_escape(s[1]) for s in splits]
        out = []
        out.append("    " + var_name + " & " + " & ".join(coefs) + r" \\")
        if any(se for se in ses):
            out.append("    "        + " & " + " & ".join(ses)   + r" \\[2pt]")
        return out

    def _summary_row_to_tex(row):
        return "    " + " & ".join(_latex_escape(v) for v in row.values) + r" \\"

    # Pull Constant out of other_block so we can render it at the bottom of the
    # variable section (just before the summary midrule).
    constant_row = None
    filtered_other = []
    for r in other_block:
        if str(r.iloc[0]).strip() == "Constant":
            if not constant_in_summary:
                constant_row = r
            # else: already routed into summary_rows by name match above
        else:
            filtered_other.append(r)

    for r in primary_block + filtered_other:
        lines.extend(_var_rows_to_tex(r))

    if global_block:
        lines.append(r"    \midrule")
        for r in global_block:
            lines.extend(_var_rows_to_tex(r))

    if constant_row is not None:
        if not global_block:
            lines.append(r"    \midrule")
        lines.extend(_var_rows_to_tex(constant_row))

    if summary_rows:
        lines.append(r"    \midrule")
        for name, r in summary_rows:
            # Constant in the summary block keeps the two-line coef/(SE)
            # format used for variable rows (so the SE prints below).
            if name == "Constant":
                lines.extend(_var_rows_to_tex(r))
            else:
                lines.append(_summary_row_to_tex(r))

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    # Build footnote: standard SE note + dominant-currency definition
    base_note = footnote or (
        "Standard errors in parentheses, clustered by ISO. "
        "*** $p<0.01$, ** $p<0.05$, * $p<0.1$.")
    needs_dom_note = "Dominant currency" in var_rows
    lines += [
        r"  \begin{tablenotes}",
        r"    \footnotesize",
        rf"    \item {base_note}",
    ]
    if needs_dom_note:
        lines.append(rf"    \item {DOMINANT_CURRENCY_FOOTNOTE}")
    for note in _dummy_notes_for_table(df):
        lines.append(rf"    \item {note}")
    lines += [
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}",
        r"\end{document}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _save_csv_and_latex(df: pd.DataFrame, out_dir: Path, fname_csv: str,
                         title: str = "", label: str = "",
                         footnote: str | None = None) -> None:
    """Save CSV to data/ subdir and a standalone-compilable .tex to out_dir."""
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / fname_csv, index=False)
    tex = out_dir / fname_csv.replace(".csv", ".tex")
    _render_table_latex(df, tex, title=title, label=label, footnote=footnote)


def section_rating_table(verbose: bool = False) -> None:
    """SI Tables T5-T6: credit rating (avgscore_yr) on macro controls, 1990-2019.

    T5 (rating_table_yearFE_avgscore): country + year FE, by subset.
    T6 (rating_table_iso_dummies): year FE, no country FE, + ISO-level dummies.
    """
    out_dir = OUTPUT_BASE / "rating"
    out_dir.mkdir(parents=True, exist_ok=True)

    rp, _ = load_rating_panel()
    # `avgscore` is country-constant (a country-level mean across years). Build a
    # time-varying avgscore = mean(score_sp, score_moodys, score_fitch) per (ISO, year).
    _agency = bond_data()[["ISO","year","score_sp","score_moodys","score_fitch"]]
    _agency = _agency.drop_duplicates(["ISO","year"])
    _agency["avgscore_yr"] = _agency[["score_sp","score_moodys","score_fitch"]].mean(axis=1, skipna=True)
    rp = rp.merge(_agency[["ISO","year","avgscore_yr"]], on=["ISO","year"], how="left")

    # Crisis indicator (not in load_rating_panel; in the bond panel at ISO-year resolution).
    extras = bond_data()[["ISO","year","default_total_past5"]]
    # ISO-year deduped (these vars are constant across bonds within ISO-year)
    extras = extras.drop_duplicates(subset=["ISO","year"])
    rp = rp.merge(extras, on=["ISO","year"], how="left")

    panel_isos = set(rp["ISO"].unique())
    SIDS_set = set(COUNTRY_GROUPS.get("SIDS", []))
    if not SIDS_set:
        from prep import SIDS_ISOS
        SIDS_set = set(SIDS_ISOS)

    rating_med = rp.groupby("ISO")["score_total"].median()
    above_IG = set(rating_med[rating_med >= 12].index)
    below_IG = set(rating_med[rating_med < 12].index)
    rp = rp.sort_values(["ISO", "year"]).reset_index(drop=True)

    tc_exposed1_isos, tc_exposed3_isos = tc_exposed_isos()

    # Rating regressions, 1990-2019: Beirne controls without global factors + extras.
    BEIRNE_NO_GLOBAL = [c for c in CONTROLS_BEIRNE
                        if c not in ("US_10yr_yield", "annual_avg_vix")]
    EXTRAS = ["default_total_past5", "RQ", "RL", "PS",
              "inflation", "Unemployment rate"]

    YEAR_RANGE = (1990, 2019)

    # Build year-FE dummies once (shared across the avgscore + ISO-dummy tables).
    sub_for_yr = rp[(rp["year"] >= YEAR_RANGE[0])
                    & (rp["year"] <= YEAR_RANGE[1])]
    year_dummies = pd.get_dummies(sub_for_yr["year"], prefix="yr",
                                   drop_first=True).astype(float)
    year_dummies_full = pd.DataFrame(0.0, index=rp.index,
                                      columns=year_dummies.columns)
    year_dummies_full.loc[sub_for_yr.index] = year_dummies.values

    # T5: outcome avgscore_yr (mean of S&P, Moody's, Fitch per ISO-year), year FE;
    # TC_exposed1 = cat1+ exposed, TC_exposed2 = cat3+ exposed.
    groups_last = {
        "All":          panel_isos,
        "above_IG":     panel_isos & above_IG,
        "below_IG":     panel_isos & below_IG,
        "SIDS":         panel_isos & SIDS_set,
        "TC_exposed1":  panel_isos & tc_exposed1_isos,
        "TC_exposed2":  panel_isos & tc_exposed3_isos,
    }
    df_yfe_avg = _build_table(
        rp, "avgscore_yr", BEIRNE_NO_GLOBAL + EXTRAS,
        extra_fe_dummies=year_dummies_full,
        groups=groups_last, year_range=YEAR_RANGE, country_fe=True,
    )
    title_yfe_avg = "Table 1"
    _save_csv_and_latex(df_yfe_avg, out_dir, "rating_table_yearFE_avgscore.csv",
                         title=title_yfe_avg, label="rating-table-yearFE-avgscore")
    if verbose:
        print(f"[rating_table:year FE avgscore] -> rating_table_yearFE_avgscore.csv")

    # ---------- T6: same spec as T5 without country FE, so ISO-level dummies
    # are identifiable.
    # Cols: All / +below_IG / +SIDS / +TC_exposed1 / +TC_exposed2 /
    #       +below_IG and TC_exposed1 ----------
    rp["below_IG_dummy"]    = rp["ISO"].isin(below_IG).astype(float)
    rp["SIDS_dummy"]        = rp["ISO"].isin(SIDS_set).astype(float)
    rp["TC_exposed1_dummy"] = rp["ISO"].isin(tc_exposed1_isos).astype(float)
    rp["TC_exposed2_dummy"] = rp["ISO"].isin(tc_exposed3_isos).astype(float)

    spec_rd_base   = list(BEIRNE_NO_GLOBAL) + list(EXTRAS)
    spec_rd_below  = spec_rd_base + ["below_IG_dummy"]
    spec_rd_sids   = spec_rd_base + ["SIDS_dummy"]
    spec_rd_tc1    = spec_rd_base + ["TC_exposed1_dummy"]
    spec_rd_tc2    = spec_rd_base + ["TC_exposed2_dummy"]
    spec_rd_btc2   = spec_rd_base + ["below_IG_dummy", "TC_exposed2_dummy"]

    def _bt_rd(spec, gname):
        return _build_table(
            rp, "avgscore_yr", spec,
            extra_fe_dummies=year_dummies_full,
            groups={gname: panel_isos},
            year_range=YEAR_RANGE, country_fe=False)

    df_rd_all   = _bt_rd(spec_rd_base,  "All_base")
    df_rd_below = _bt_rd(spec_rd_below, "All_below")
    df_rd_sids  = _bt_rd(spec_rd_sids,  "All_sids")
    df_rd_tc1   = _bt_rd(spec_rd_tc1,   "All_tc1")
    df_rd_tc2   = _bt_rd(spec_rd_tc2,   "All_tc2")
    df_rd_btc2  = _bt_rd(spec_rd_btc2,  "All_btc2")

    all_vars_rd = []
    seen_rd = set()
    for src in [df_rd_all, df_rd_below, df_rd_sids, df_rd_tc1, df_rd_tc2,
                df_rd_btc2]:
        for v in src["Variable"].tolist():
            if v not in seen_rd:
                seen_rd.add(v); all_vars_rd.append(v)
    df_rd = pd.DataFrame({"Variable": all_vars_rd})
    for src, internal_key, header in [
        (df_rd_all,   "All_base",  "All"),
        (df_rd_below, "All_below", "+ below_IG"),
        (df_rd_sids,  "All_sids",  "+ SIDS"),
        (df_rd_tc1,   "All_tc1",   "+ TC_exposed1"),
        (df_rd_tc2,   "All_tc2",   "+ TC_exposed2"),
        (df_rd_btc2,  "All_btc2",  "+ below_IG and TC_exposed2"),
    ]:
        m = dict(zip(src["Variable"], src[internal_key]))
        df_rd[header] = df_rd["Variable"].map(m).fillna("")

    title_rd = "Table 2"
    _save_csv_and_latex(df_rd, out_dir, "rating_table_iso_dummies.csv",
                         title=title_rd, label="rating-table-iso-dummies")
    if verbose:
        print(f"[rating_table:iso dummies (no country FE)] "
              f"-> rating_table_iso_dummies.csv")

    with open(out_dir / "group_definitions.txt", "w") as f:
        f.write("Rating panel: outcome = avgscore_yr (mean of S&P, Moody's, Fitch scores; "
                "higher = better).\n\n")
        for grp, isos in groups_last.items():
            f.write(f"{grp} ({len(isos)} ISOs):\n  {sorted(isos)}\n\n")


def section_coupon_style_table(verbose: bool = False) -> None:
    """SI Tables T7-T8: 10y bond coupon at issuance on macro controls + rating, 1990-2019.

    T7 (coupon_table1_iso_dummies): currency + year FE.
    T8 (coupon_table2_iso_dummies_global): currency FE, US 10y yield + VIX instead of year FE.
    """
    out_dir = OUTPUT_BASE / "coupon"
    out_dir.mkdir(parents=True, exist_ok=True)

    cp, _ = load_coupon_panel()
    if "US_10yr_yield" not in cp.columns or "annual_avg_vix" not in cp.columns:
        gf = load_global_factors()
        for col in ["US_10yr_yield", "annual_avg_vix"]:
            if col in cp.columns and col in gf.columns:
                gf = gf.drop(columns=[col])
        cp = cp.merge(gf, on="year", how="left")

    panel_isos = set(cp["ISO"].unique())
    rating_med_panel = pd.read_csv(PATHS["rating_df_v2"]).groupby("ISO")["score_total"].median()
    below_IG = set(rating_med_panel[rating_med_panel < 12].index)

    tc_exposed1_isos, _ = tc_exposed_isos()
    groups = {
        "All":         panel_isos,
        "below_IG":    panel_isos & below_IG,
        "TC_exposed":  panel_isos & tc_exposed1_isos,
    }
    if verbose:
        print(f"  TC_exposed in coupon panel "
              f"({len(groups['TC_exposed'])} ISOs): {sorted(groups['TC_exposed'])}")

    cur_dummies = pd.get_dummies(cp["currency_issued"].astype(str),
                                  prefix="cur", drop_first=True).astype(float)

    # Shared spec building blocks.
    WGI_VARS_3          = ["RQ", "RL", "PS"]
    BEIRNE_NO_GLOBAL_CP = [c for c in CONTROLS_BEIRNE
                            if c not in ("US_10yr_yield", "annual_avg_vix")]
    BEIRNE_PD_NG        = BEIRNE_NO_GLOBAL_CP + ["default_total_past5"]
    YEAR_RANGE          = (1990, 2019)

    # Year-FE dummies (T7).
    sub_for_yr_cp     = cp[(cp["year"] >= YEAR_RANGE[0])
                            & (cp["year"] <= YEAR_RANGE[1])]
    yr_dummies_cp     = pd.get_dummies(sub_for_yr_cp["year"], prefix="yr",
                                        drop_first=True).astype(float)
    yr_dummies_full_cp = pd.DataFrame(0.0, index=cp.index,
                                       columns=yr_dummies_cp.columns)
    yr_dummies_full_cp.loc[sub_for_yr_cp.index] = yr_dummies_cp.values
    fe_yr = pd.concat([cur_dummies, yr_dummies_full_cp], axis=1)

    # ---------- T7: country FE off so the ISO-level dummies are identifiable.
    # Cols: no rating / full / full + country FE / + below_IG / + TC_exposed /
    # + below_IG x TC_exposed. ----------
    cp["below_IG_dummy"]        = cp["ISO"].isin(below_IG).astype(float)
    cp["TC_exposed_dummy"]      = cp["ISO"].isin(tc_exposed1_isos).astype(float)
    cp["below_IG_x_TC_expo"]    = cp["below_IG_dummy"] * cp["TC_exposed_dummy"]

    spec_full_no_rating = BEIRNE_PD_NG + WGI_VARS_3 + ["inflation", "Unemployment rate"]
    spec_full_T6        = spec_full_no_rating + ["score_total"]
    spec_full_below_IG  = spec_full_T6 + ["below_IG_dummy"]
    spec_full_TC_expo   = spec_full_T6 + ["TC_exposed_dummy"]
    spec_full_interact  = spec_full_T6 + ["below_IG_x_TC_expo"]

    df_T6_min     = _build_table(
        cp, "Cpn", spec_full_no_rating, extra_fe_dummies=fe_yr,
        groups={"All_norating": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T6_full    = _build_table(
        cp, "Cpn", spec_full_T6, extra_fe_dummies=fe_yr,
        groups={"All_full": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    # Col 3: full spec with country FE on, for comparison.
    df_T6_full_FE = _build_table(
        cp, "Cpn", spec_full_T6, extra_fe_dummies=fe_yr,
        groups={"All_full_FE": panel_isos},
        year_range=YEAR_RANGE, country_fe=True)
    df_T6_below   = _build_table(
        cp, "Cpn", spec_full_below_IG, extra_fe_dummies=fe_yr,
        groups={"All_below_dum": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T6_tc      = _build_table(
        cp, "Cpn", spec_full_TC_expo, extra_fe_dummies=fe_yr,
        groups={"All_TCexpo_dum": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T6_int     = _build_table(
        cp, "Cpn", spec_full_interact, extra_fe_dummies=fe_yr,
        groups={"All_interact_dum": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)

    all_vars_T6 = []
    seen_T6 = set()
    for src in [df_T6_min, df_T6_full, df_T6_full_FE, df_T6_below, df_T6_tc,
                df_T6_int]:
        for v in src["Variable"].tolist():
            if v not in seen_T6:
                seen_T6.add(v); all_vars_T6.append(v)
    df_T6 = pd.DataFrame({"Variable": all_vars_T6})
    for src, internal_key, header in [
        (df_T6_min,     "All_norating",     "All (no rating)"),
        (df_T6_full,    "All_full",         "All (full)"),
        (df_T6_full_FE, "All_full_FE",      "+ country FE"),
        (df_T6_below,   "All_below_dum",    "+ below_IG"),
        (df_T6_tc,      "All_TCexpo_dum",   "+ TC_exposed"),
        (df_T6_int,     "All_interact_dum", "+ below_IG x TC_exposed"),
    ]:
        m = dict(zip(src["Variable"], src[internal_key]))
        df_T6[header] = df_T6["Variable"].map(m).fillna("")

    title_T6 = "Table 1"
    _save_csv_and_latex(df_T6, out_dir, "coupon_table1_iso_dummies.csv",
                         title=title_T6, label="coupon-table1-iso-dummies")
    if verbose:
        print(f"[coupon_table:Table 1 (ISO dummies, year FE)] "
              f"-> coupon_table1_iso_dummies.csv")

    # ---------- T8: as T7 but US 10yr + VIX (in CONTROLS_BEIRNE) replace year FE. ----------
    BEIRNE_PD_T7 = list(CONTROLS_BEIRNE) + ["default_total_past5"]
    spec_T7_no_rating = BEIRNE_PD_T7 + WGI_VARS_3 + ["inflation", "Unemployment rate"]
    spec_T7_full      = spec_T7_no_rating + ["score_total"]
    spec_T7_below_IG  = spec_T7_full + ["below_IG_dummy"]
    spec_T7_TC_expo   = spec_T7_full + ["TC_exposed_dummy"]
    spec_T7_interact  = spec_T7_full + ["below_IG_x_TC_expo"]

    df_T7_min     = _build_table(
        cp, "Cpn", spec_T7_no_rating, extra_fe_dummies=cur_dummies,
        groups={"All_norating": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T7_full    = _build_table(
        cp, "Cpn", spec_T7_full, extra_fe_dummies=cur_dummies,
        groups={"All_full": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T7_full_FE = _build_table(
        cp, "Cpn", spec_T7_full, extra_fe_dummies=cur_dummies,
        groups={"All_full_FE": panel_isos},
        year_range=YEAR_RANGE, country_fe=True)
    df_T7_below   = _build_table(
        cp, "Cpn", spec_T7_below_IG, extra_fe_dummies=cur_dummies,
        groups={"All_below_dum": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T7_tc      = _build_table(
        cp, "Cpn", spec_T7_TC_expo, extra_fe_dummies=cur_dummies,
        groups={"All_TCexpo_dum": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)
    df_T7_int     = _build_table(
        cp, "Cpn", spec_T7_interact, extra_fe_dummies=cur_dummies,
        groups={"All_interact_dum": panel_isos},
        year_range=YEAR_RANGE, country_fe=False)

    all_vars_T7 = []
    seen_T7 = set()
    for src in [df_T7_min, df_T7_full, df_T7_full_FE, df_T7_below, df_T7_tc,
                df_T7_int]:
        for v in src["Variable"].tolist():
            if v not in seen_T7:
                seen_T7.add(v); all_vars_T7.append(v)
    df_T7 = pd.DataFrame({"Variable": all_vars_T7})
    for src, internal_key, header in [
        (df_T7_min,     "All_norating",     "All (no rating)"),
        (df_T7_full,    "All_full",         "All (full)"),
        (df_T7_full_FE, "All_full_FE",      "+ country FE"),
        (df_T7_below,   "All_below_dum",    "+ below_IG"),
        (df_T7_tc,      "All_TCexpo_dum",   "+ TC_exposed"),
        (df_T7_int,     "All_interact_dum", "+ below_IG x TC_exposed"),
    ]:
        m = dict(zip(src["Variable"], src[internal_key]))
        df_T7[header] = df_T7["Variable"].map(m).fillna("")

    title_T7 = "Table 2"
    _save_csv_and_latex(df_T7, out_dir, "coupon_table2_iso_dummies_global.csv",
                         title=title_T7, label="coupon-table2-iso-dummies-global")
    if verbose:
        print(f"[coupon_table:Table 2 (ISO dummies, US10y + VIX, no year FE)] "
              f"-> coupon_table2_iso_dummies_global.csv")

    with open(out_dir / "group_definitions.txt", "w") as f:
        f.write("Coupon panel: outcome = Cpn (%, bond-level at issuance, "
                "maturity == 10), currency_issued FE.\n\n")
        for grp, isos in groups.items():
            f.write(f"{grp} ({len(isos)} ISOs):\n  {sorted(isos)}\n\n")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", nargs="+", default=["all"],
                        choices=["rating_table", "coupon_table", "all"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sections = ["rating_table", "coupon_table"] if "all" in args.section else args.section
    if "rating_table" in sections:
        print("=== Rating tables (SI T5-T6) ==="); section_rating_table(verbose=args.verbose)
    if "coupon_table" in sections:
        print("=== Coupon tables (SI T7-T8) ==="); section_coupon_style_table(verbose=args.verbose)


if __name__ == "__main__":
    main()
