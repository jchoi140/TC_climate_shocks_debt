"""
GDP_map.png  (SI figure)

World choropleth of the change in GDP per capita in 2019 attributable to
tropical-cyclone and temperature exposure, i.e. the percentage gap between
observed GDP per capita and the combined (TC + temperature) counterfactual.

Inputs:
  3_output/counterfactual_data/cf_gdp_debtratio.csv
      (ISO, year, gdp_pcap_gdp, 1990_cf_combined_gdp)
  0_raw/shapefiles/ne_10m_admin_0_map_units/ne_10m_admin_0_map_units.shp

Output:
  4_figures/supplementary/GDP_map.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

import sys
sys.path.insert(0, os.path.join(PKG, "2_code", "common"))
from paths import PATHS  # noqa: E402

CF_PATH = PATHS["cf_data"]
SHP_PATH = PATHS["world_shapefile"]
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "GDP_map.png")

# --- build the country-level % change --------------------------------------
world = gpd.read_file(SHP_PATH)
d = pd.read_csv(CF_PATH)

df = d[d["year"] == 2019][["ISO", "1990_cf_combined_gdp", "gdp_pcap_gdp"]].copy()
# % change in GDP pcap attributable to exposure = (observed - counterfactual) / observed * 100
df["diff"] = (df["gdp_pcap_gdp"] - df["1990_cf_combined_gdp"]) / df["gdp_pcap_gdp"] * 100
df = df[df["diff"] != 0]

merged = world.merge(df, left_on="ADM0_A3", right_on="ISO", how="left")
merged = merged[merged["ADM0_A3"] != "ATA"]

# --- global Robinson choropleth --------------------------------------------
fig = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_facecolor((0.91, 0.98, 1, 0.5))  # light blue ocean

bins = [-60, -50, -40, -30, -20, -10, 0, 10, 20]
colors = plt.cm.BrBG(np.linspace(0.15, 0.75, len(bins) + 4))
gradient = ListedColormap(colors[[0, 1, 2, 3, 4, 5, 8, 9]])
norm = BoundaryNorm(bins, gradient.N)

sm = plt.cm.ScalarMappable(cmap=gradient, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.02, pad=0.04)
cbar.set_label("GDP pcap (%)", labelpad=-40, y=1.05, fontsize=10)
cbar.ax.set_position([0.5, 0.22, 0.3, 0.02])
cbar.set_ticks([-60, -40, -20, 0, 20])
cbar.set_ticklabels(["-60", "-40", "-20", "0", "20"])

merged.plot(
    column="diff",
    cmap=gradient,
    ax=ax,
    transform=ccrs.PlateCarree(),
    edgecolor="black",
    linewidth=0.5,
    missing_kwds={"color": "lightgrey", "label": "No data", "alpha": 0.9},
    legend=False,
    norm=norm,
)

fig.text(
    0.02, 0.95,
    "Change in GDP per capita in 2019 attributable to tropical cyclone and temperature exposure",
    ha="left", va="top", fontsize=13, fontweight="bold", color="#1F4E79",
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", OUT_PATH)
