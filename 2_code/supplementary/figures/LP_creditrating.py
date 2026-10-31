"""
LP_creditrating.png  (SI figure)

Local-projection impulse responses of sovereign credit ratings to tropical
cyclone exposure. 3x3 grid: rows = rating source (S&P / Moody's / Fitch),
columns = TC intensity term (TS / Cat1-2 / >Cat3). Each panel shows the
no_trend (solid) and linear (dashed) specifications, with the no_trend
confidence band shaded; horizon is anchored to zero at h = -1.

Inputs:
  3_output/debtratio/TC+Tshock/coef_df/ratings/*_TC.csv

Output:
  4_figures/supplementary/LP_creditrating.png
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
from paths import coef_dir  # noqa: E402

RATINGS_DIR = coef_dir("ratings")
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "LP_creditrating.png")

# --- load credit-rating LP coefficients (all *_TC.csv, excluding *_TEMP.csv)
_dfs_tc = []
for f in os.listdir(RATINGS_DIR):
    if not f.endswith(".csv") or f.endswith("_TEMP.csv"):
        continue
    _dfs_tc.append(pd.read_csv(os.path.join(RATINGS_DIR, f)))
TC_ratings = pd.concat(_dfs_tc, ignore_index=True)

selected_outcomes = ["sp_ratings", "moodys_ratings", "fitch_ratings"]
specification = "1980onwards_gl0_TC05_T02"  # main specification
terms = ["ts_v2", "cat1_v2", "cat3_v2"]
titles = ["TS", "Cat1-2", ">Cat3"]
palette = sns.color_palette("Blues", len(titles))
color_dict = dict(zip(titles, palette))
Tshock_types = ["NATH"]

fig, axes = plt.subplots(len(selected_outcomes), len(titles), figsize=(15, 6))

for k, selected_outcome in enumerate(selected_outcomes):
    for i, (term, title) in enumerate(zip(terms, titles)):
        for time_trend in ["no_trend", "linear"]:
            for Tshock in Tshock_types:
                term_df = TC_ratings[
                    (TC_ratings["term"] == term)
                    & (TC_ratings["model"] == time_trend)
                    & (TC_ratings["outcome"] == selected_outcome)
                    & (TC_ratings["Tshock_type"] == Tshock)
                    & (TC_ratings["specification"] == specification)
                ]

                # Anchor h = -1 at zero (pre-shock baseline)
                term_df_plot = pd.concat([
                    pd.DataFrame({"y_h": [-1], "estimate": [0.0], "conf.low": [0.0], "conf.high": [0.0]}),
                    term_df.sort_values("y_h"),
                ], ignore_index=True)

                sns.lineplot(
                    data=term_df_plot, x="y_h", y="estimate", ax=axes[k, i],
                    label=f"{time_trend}", color=color_dict[title],
                    linestyle="--" if time_trend == "linear" else "-",
                )

                if time_trend == "no_trend":
                    axes[k, i].fill_between(
                        term_df_plot["y_h"], term_df_plot["conf.low"], term_df_plot["conf.high"],
                        color=color_dict[title], alpha=0.2,
                    )

        axes[k, i].axhline(0, color="grey", alpha=0.5)
        axes[k, i].set_xlabel("")
        axes[k, i].set_ylabel("")
        axes[k, i].set_title(f"{title} ({selected_outcome})")
        if k == len(selected_outcomes) - 1:
            axes[k, i].set_ylim(-0.4, 0.4)
        else:
            axes[k, i].set_ylim(-0.1, 0.05)
        axes[k, i].tick_params(axis="y", labelsize=14)
        axes[k, i].legend(loc="lower left", fontsize=8, frameon=True)

fig.suptitle("credit ratings", fontsize=16)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print("wrote", OUT_PATH)
