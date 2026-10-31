"""
temp_histogram.png  (SI figure)

Three-panel kernel-density comparison of mean historical temperature across
country groups:
  Panel 1: Non-SIDS vs SIDS
  Panel 2: Rated vs Non-Rated
  Panel 3: A Investment Grade vs B Investment Grade

Inputs:
  1_input/climate_exposures/era5_v4_Tshocks.csv   (ISO, meantemp_hist)
  0_raw/climate_exposures/country_master.csv      (group classification flags)

Output:
  4_figures/supplementary/temp_histogram.png
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

TEMP_PATH = PATHS["temperature"]
CM_PATH = PATHS["country_master"]
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "temp_histogram.png")

# --- country groupings -----------------------------------------------------
# SIDS = the paper's 41-country list (not the broader country_master SIDS flag, 55 territories).
sids = [
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "COK", "CUB", "DMA", "DOM",
    "FJI", "GRD", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "FSM",
    "MUS", "NRU", "NIU", "PLW", "PNG", "WSM", "STP", "SGP", "KNA", "LCA",
    "VCT", "SYC", "SLB", "SUR", "TLS", "TON", "TTO", "TUV", "VUT",
    "MDG", "LKA",
]

TEMP_dist = pd.read_csv(TEMP_PATH)
country_master = pd.read_csv(CM_PATH)

sovereign_ISOs = country_master[country_master["sovereigncountry"] == 1]["ISO"]

sids_T = TEMP_dist[TEMP_dist["ISO"].isin(sids) & TEMP_dist["ISO"].isin(sovereign_ISOs)]
non_sids_T = TEMP_dist[~TEMP_dist["ISO"].isin(sids) & TEMP_dist["ISO"].isin(sovereign_ISOs)]

rated_T = TEMP_dist[TEMP_dist["ISO"].isin(country_master[country_master["rated"] == 1]["ISO"])]
non_rated_T = TEMP_dist[TEMP_dist["ISO"].isin(country_master[country_master["nonrated"] == 1]["ISO"])]

rated_Agrade_T = TEMP_dist[TEMP_dist["ISO"].isin(country_master[country_master["Ainvestmentgrade"] == 1]["ISO"])]
rated_Bgrade_T = TEMP_dist[TEMP_dist["ISO"].isin(country_master[country_master["Binvestmentgrade"] == 1]["ISO"])]

# --- three-panel KDE -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))

sns.kdeplot(non_sids_T["meantemp_hist"], label="Non-SIDS", fill=True, color="blue", alpha=0.3, ax=axes[0])
sns.kdeplot(sids_T["meantemp_hist"], label="SIDS", fill=True, color="red", alpha=0.3, ax=axes[0])
axes[0].set_xlabel("Mean Temperature")
axes[0].set_ylabel("Density")
axes[0].legend(loc="upper left")

sns.kdeplot(rated_T["meantemp_hist"], label="Rated", fill=True, color="blue", alpha=0.3, ax=axes[1])
sns.kdeplot(non_rated_T["meantemp_hist"], label="Non-Rated", fill=True, color="red", alpha=0.3, ax=axes[1])
axes[1].set_xlabel("Mean Temperature")
axes[1].set_ylabel("")
axes[1].legend(loc="upper left")

sns.kdeplot(rated_Agrade_T["meantemp_hist"], label="A Investment Grade", fill=True, color="blue", alpha=0.3, ax=axes[2])
sns.kdeplot(rated_Bgrade_T["meantemp_hist"], label="B Investment Grade", fill=True, color="red", alpha=0.3, ax=axes[2])
axes[2].set_xlabel("Mean Temperature")
axes[2].set_ylabel("")
axes[2].legend(loc="upper left")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print("wrote", OUT_PATH)
