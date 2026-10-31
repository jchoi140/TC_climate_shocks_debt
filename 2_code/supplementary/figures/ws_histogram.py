"""
ws_histogram.png  (SI figure)

Composite wind-speed / land-area-exposure distribution figure.

Top section  ("Average Wind Speed (m/s)"): 3 overlapping histograms of annual
average wind speed (avg_ws), split by Non-SIDS/SIDS, Rated/Non-Rated, and
Investment/Speculative grade.

Bottom section ("% land area exposure"): a 3x3 grid of overlapping histograms of
the % land-area exposed to each tropical-cyclone category (ts_v2 = Tropical
Storm, cat1_v2 = Category 1-3, cat3_v2 = Category >3), with one grouping per
row (Rated/Non-Rated, Non-SIDS/SIDS, Investment/Speculative grade).

All count axes are log-scaled.

Inputs:
  0_raw/climate_exposures/tc2_maxws.csv       (ISO, year, avg_ws, ts_v2, cat1_v2, cat3_v2)
  0_raw/climate_exposures/country_master.csv  (group classification flags)

Output:
  4_figures/supplementary/ws_histogram.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

import sys
sys.path.insert(0, os.path.join(PKG, "2_code", "common"))
from paths import PATHS  # noqa: E402

TC_PATH = PATHS["tc"]
CM_PATH = PATHS["country_master"]
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "ws_histogram.png")

# SIDS = the paper's 41-country list.
sids = [
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "COK", "CUB", "DMA", "DOM",
    "FJI", "GRD", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "FSM",
    "MUS", "NRU", "NIU", "PLW", "PNG", "WSM", "STP", "SGP", "KNA", "LCA",
    "VCT", "SYC", "SLB", "SUR", "TLS", "TON", "TTO", "TUV", "VUT",
    "MDG", "LKA",
]

TC_dist = pd.read_csv(TC_PATH)
country_master = pd.read_csv(CM_PATH)

sovereign_ISOs = country_master[country_master["sovereigncountry"] == 1]["ISO"]
nonrated_ISOs = country_master[country_master["nonrated"] == 1]["ISO"].tolist()
Ainvestmentgrade_ISOs = country_master[country_master["Ainvestmentgrade"] == 1]["ISO"].tolist()
Binvestmentgrade_ISOs = country_master[country_master["Binvestmentgrade"] == 1]["ISO"].tolist()


def subset(iso_list, invert=False):
    m = TC_dist["ISO"].isin(iso_list) & TC_dist["ISO"].isin(sovereign_ISOs)
    if invert:
        m = ~TC_dist["ISO"].isin(iso_list) & TC_dist["ISO"].isin(sovereign_ISOs)
    return TC_dist[m]


# group frames (blue = first label, red = second label)
non_sids_TC = subset(sids, invert=True)
sids_TC = subset(sids)
rated_TC = subset(nonrated_ISOs, invert=True)
nonrated_TC = subset(nonrated_ISOs)
Agrade_TC = subset(Ainvestmentgrade_ISOs)
Bgrade_TC = subset(Binvestmentgrade_ISOs)


def hist_pair(ax, blue_df, red_df, var, blue_label, red_label):
    blue = blue_df[blue_df[var] != 0][var]
    red = red_df[red_df[var] != 0][var]
    max_val = max(blue.max(), red.max())
    bw = max_val / 30.0
    sns.histplot(blue, label=blue_label, kde=False, color="blue", alpha=0.3, ax=ax, binwidth=bw)
    sns.histplot(red, label=red_label, kde=False, color="red", alpha=0.3, ax=ax, binwidth=bw)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_yscale("log")


# --- layout (1024x768 canvas); axes positions are [left, bottom, width, height] fractions
FIG_W, FIG_H = 10.24, 7.68
fig = plt.figure(figsize=(FIG_W, FIG_H))
col_left = [0.092, 0.355, 0.618]
col_width = 0.226

# --- Row 0: Average Wind Speed (avg_ws) -----------------------------------
ws_groups = [
    (non_sids_TC, sids_TC, "Non-SIDS", "SIDS"),
    (rated_TC, nonrated_TC, "Rated", "Non-Rated"),
    (Agrade_TC, Bgrade_TC, "Investment Grade", "Speculative Grade"),
]
for i, (b, r, bl, rl) in enumerate(ws_groups):
    ax = fig.add_axes([col_left[i], 0.792, col_width, 0.124])
    hist_pair(ax, b, r, "avg_ws", bl, rl)
    ax.set_xticks(range(0, 71, 10))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("annual avg windspeed (m/s)", fontsize=7)
    ax.set_ylabel("Count (log scale)" if i == 0 else "", fontsize=7)
    ax.legend(loc="upper right", fontsize=7)

# --- Rows 1-3: % land area exposure grid ----------------------------------
cat_vars = ["ts_v2", "cat1_v2", "cat3_v2"]
cat_labels = [
    "Tropical Storm (18-33 m/s)\n% area exposed",
    "Category 1-3 (33-50 m/s)\n% area exposed",
    "Category >3 (>50 m/s)\n% area exposed",
]
area_rows = [
    (rated_TC, nonrated_TC, "Rated", "Non-Rated"),
    (non_sids_TC, sids_TC, "Non-SIDS", "SIDS"),
    (Agrade_TC, Bgrade_TC, "Investment grade", "Speculative grade"),
]
row_bottom = [0.518, 0.322, 0.115]
for ridx, (b, r, bl, rl) in enumerate(area_rows):
    for cidx, var in enumerate(cat_vars):
        ax = fig.add_axes([col_left[cidx], row_bottom[ridx], col_width, 0.152])
        hist_pair(ax, b, r, var, bl, rl)
        ax.set_xticks(range(0, 101, 20))
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Count (log scale)" if cidx == 0 else "", fontsize=8)
        # x labels only on the bottom-most row
        ax.set_xlabel(cat_labels[cidx] if ridx == 2 else "", fontsize=8, fontweight="bold")
    # one legend per row, to the right of the third column
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

# --- section titles --------------------------------------------------------
fig.text(0.448, 0.952, "Average Wind Speed (m/s)", ha="center", fontsize=8, fontweight="bold")
fig.text(0.493, 0.704, "% land area exposure", ha="center", fontsize=9, fontweight="bold")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200)
print("wrote", OUT_PATH)
