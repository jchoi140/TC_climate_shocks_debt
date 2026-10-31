# =============================================================================
# SHARED INPUT PATHS + LOADERS
# =============================================================================
# Input-file paths (relative to the project root) and small read_csv loaders
# used by the figure code and the rating-model data prep.
# =============================================================================

import os

import pandas as pd

# Project root = two levels up (2_code/common/ -> repo root).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def p(*parts):
    """Absolute path to *parts* under the project root."""
    return os.path.join(PROJECT_ROOT, *parts)


# =============================================================================
# INPUT FILE PATHS
# =============================================================================

# Base directory holding the per-specification LP coefficient CSVs.
COEF_DF_DIR = p("3_output", "debtratio", "TC+Tshock", "coef_df")


def coef_dir(name):
    """Absolute path to a ``coef_df/<name>`` subdirectory (e.g. ``'LP_coefficients'``)."""
    return os.path.join(COEF_DF_DIR, name)


def coef_file(name, filename):
    """Absolute path to ``coef_df/<name>/<filename>``."""
    return os.path.join(COEF_DF_DIR, name, filename)


PATHS = {
    # --- Climate exposures ---------------------------------------------------
    "temperature":       p("1_input", "climate_exposures", "era5_v4_Tshocks.csv"),
    "tc":                p("0_raw", "climate_exposures", "tc2_maxws.csv"),
    "tc_complete":       p("1_input", "climate_exposures", "tc_maxws_complete.csv"),
    "country_master":    p("0_raw", "climate_exposures", "country_master.csv"),
    "world_key":         p("0_raw", "climate_exposures", "world_key.csv"),

    # --- Economic outcomes ---------------------------------------------------
    "gdp_pcap":          p("0_raw", "economic_outcomes", "gdp_pcap_wdi_constant2015_USD.csv"),
    "gdd_debt":          p("1_input", "economic_outcomes", "gdd_debt_v2.csv"),

    # --- Credit ratings / bonds ---------------------------------------------
    "US_10yr":                p("1_input", "credit_ratings", "US_10yr_yield.csv"),
    "rating_df_v2":           p("1_input", "credit_ratings", "rating_df_v2.csv"),  # score_total only
    # shipped aggregates of the agency ratings (2_code/data_prep/build_proprietary_aggregates.py)
    "rating_grade_2019":      p("1_input", "credit_ratings", "rating_grade_2019.csv"),
    "rated_countries_by_source": p("1_input", "credit_ratings", "rated_countries_by_source.csv"),
    "rating_key":             p("0_raw", "credit_ratings", "ratingscore_key_v2.csv"),
    # proprietary raw data (Bloomberg / LSEG; not shipped, see README "Data")
    "bloomberg":              p("0_raw", "credit_ratings", "bloomberg", "bloomberg_predict_df2.csv"),
    "lseg":                   p("0_raw", "credit_ratings", "lseg", "lseg_predict_df_v2.csv"),
    "bondyield_10yr":         p("0_raw", "credit_ratings", "bondyield_10yr.csv"),
    "sp_ratings":             p("0_raw", "credit_ratings", "sp_ratings_v3.csv"),
    "moodys_ratings":         p("0_raw", "credit_ratings", "moodys_ratings_v3.csv"),
    "fitch_ratings":          p("0_raw", "credit_ratings", "fitch_ratings_v3.csv"),
    "currencies":             p("1_input", "credit_ratings", "currencies.csv"),
    "predicted_cpn_2011_2019": p("1_input", "credit_ratings", "predicted_coupon_2011_2019.csv"),
    "cpn_summary_2011_2019":  p("1_input", "credit_ratings", "cpn_summary_2011_2019_v2.csv"),
    "bond_currency_summary":  p("1_input", "credit_ratings", "bond_currency_summary.csv"),

    # --- Counterfactual ------------------------------------------------------
    "cf_data":            p("3_output", "counterfactual_data", "cf_gdp_debtratio.csv"),
    "iso_to_basin":       p("1_input", "counterfactual_data", "iso_to_basin.csv"),
    "enso_nino34":        p("1_input", "counterfactual_data", "enso_nino34_annual_1970_present.csv"),
    "world_shapefile":    p("0_raw", "shapefiles", "ne_10m_admin_0_map_units", "ne_10m_admin_0_map_units.shp"),

    # Headline LP coefficient specs used by the counterfactual notebook.
    "coef_debt": coef_file("LP_coefficients", "full_public_pp_GDD_1980onwards_gl2_TC10_T02_TC.csv"),
    "coef_gdp":  coef_file("LP_coefficients", "full_gdp_pcap_wdi_constant2015_USD_1970onwards_gl2_TC10_T02_TC.csv"),

    # IBTrACS storm tracks for the M4 panel a overlay (142 MB, downloaded separately;
    # see 2_code/data_prep/DATA.md). Without it the map renders with no tracks.
    "ibtracs_data": p("0_raw", "climate_exposures", "ibtracs", "ibtracs.since1980.list.v04r01.csv"),

    # --- Aliases (same file under a second key name) ---------------------------
    "temp_data": p("1_input", "climate_exposures", "era5_v4_Tshocks.csv"),        # == temperature
    "tc_data":   p("1_input", "climate_exposures", "tc_maxws_complete.csv"),      # == tc_complete
    "gdp_data":  p("0_raw", "economic_outcomes", "gdp_pcap_wdi_constant2015_USD.csv"),  # == gdp_pcap
    "debt_data": p("1_input", "economic_outcomes", "gdd_debt_v2.csv"),        # == gdd_debt
    "tc_isomapping": p("0_raw", "climate_exposures", "world_key.csv"),           # == world_key
}


# =============================================================================
# LOADERS
# =============================================================================

def require_proprietary(key):
    """``PATHS[key]`` for a proprietary Bloomberg/LSEG input (not shipped with the
    package); raise a clear error if the file is missing."""
    path = PATHS[key]
    if not os.path.exists(path):
        raise FileNotFoundError("Please add the required credit rating or bond data.")
    return path


def read(key, **kwargs):
    """``pd.read_csv`` of the file registered under ``PATHS[key]``."""
    return pd.read_csv(PATHS[key], **kwargs)


# One-line loaders for frequently read PATHS entries.
def load_temperature(**kw):        return read("temperature", **kw)
def load_tc(**kw):                 return read("tc", **kw)
def load_tc_complete(**kw):        return read("tc_complete", **kw)
def load_country_master(**kw):     return read("country_master", **kw)
def load_gdp_pcap(**kw):           return read("gdp_pcap", **kw)
def load_gdd_debt(**kw):           return read("gdd_debt", **kw)


def load_coef_dir(name, shock_type="TC"):
    """Concatenate the coefficient CSVs in ``coef_df/<name>``: shock_type 'TEMP'
    reads ``*_TEMP.csv``, 'TC' the other CSVs, 'ALL' every CSV (os.listdir order)."""
    directory = coef_dir(name)
    frames = []
    for filename in os.listdir(directory):
        if not filename.endswith(".csv"):
            continue
        is_temp = filename.endswith("_TEMP.csv")
        if shock_type == "TEMP" and not is_temp:
            continue
        if shock_type == "TC" and is_temp:
            continue
        frames.append(pd.read_csv(os.path.join(directory, filename)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# XGB PREDICTION FILES (split by seed to stay under GitHub's file-size limit)
# =============================================================================
# A run over seeds a..b-1 is tagged "<model>_<scoring>_<a>_<b>" (e.g. the 500-seed run
# XGB_classification_neglogloss_0_500 uses seeds 0-499). Runs larger than 50 MB are stored as
# 100-seed parts named by seed range instead of the run tag:
#     XGB_classification_neglogloss_seeds_000_099.csv ... _seeds_400_499.csv
# ``read_predictions`` takes the path of the single combined file and, if that file is absent,
# concatenates the parts covering seeds a..b-1.
import re as _re
import glob as _glob

_RUN_RE = _re.compile(r"^(?P<base>.+)_(?P<a>\d+)_(?P<b>\d+)$")


def prediction_parts(path):
    """Sorted list of the ``<base>_seeds_NNN_NNN.csv`` parts that make up *path*."""
    stem = path[:-4] if path.endswith(".csv") else path
    m = _RUN_RE.match(stem)
    if not m:
        return []
    lo, hi = int(m["a"]), int(m["b"])          # seeds lo .. hi-1
    parts = []
    for f in sorted(_glob.glob(_glob.escape(m["base"]) + "_seeds_[0-9]*_[0-9]*.csv")):
        pm = _re.search(r"_seeds_(\d+)_(\d+)\.csv$", f)
        if pm and lo <= int(pm[1]) and int(pm[2]) < hi:
            parts.append(f)
    return parts


def predictions_exist(path):
    """True if *path* exists or its seed-split parts do."""
    return os.path.exists(path) or bool(prediction_parts(path))


def read_predictions(path, **kwargs):
    """Read an XGB prediction table, from *path* or from its seed-split parts.

    The parts are re-ordered rated-first (stable sort on ``is_NR``), which
    reproduces the row order of the original combined file exactly.
    """
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    parts = prediction_parts(path)
    if not parts:
        raise FileNotFoundError(f"{path} (and no *_seeds_NNN_NNN.csv parts)")
    df = pd.concat([pd.read_csv(f, **kwargs) for f in parts], ignore_index=True)
    if "is_NR" in df.columns:
        df = df.sort_values("is_NR", kind="stable").reset_index(drop=True)
    return df


def write_predictions(df, path, seeds_per_part=100, max_mb=50):
    """Write *df* to *path*; if the result exceeds *max_mb*, replace it with
    seed-range parts of *seeds_per_part* seeds each (``<base>_seeds_NNN_NNN.csv``)."""
    df.to_csv(path, index=False)
    stem = path[:-4]
    m = _RUN_RE.match(stem)
    if os.path.getsize(path) <= max_mb * 1e6 or "seed" not in df.columns or not m:
        return [path]
    os.remove(path)
    out = []
    lo, hi = int(df["seed"].min()), int(df["seed"].max())
    for start in range(lo - lo % seeds_per_part, hi + 1, seeds_per_part):
        end = start + seeds_per_part - 1
        part = df[(df["seed"] >= start) & (df["seed"] <= end)]
        if len(part):
            f = f"{m['base']}_seeds_{start:03d}_{end:03d}.csv"
            part.to_csv(f, index=False)
            out.append(f)
    return out
