"""
rating_coupons.png  (SI figure)

S&P sovereign rating vs. 10-year bond rates. Each point is a country-year 10-yr
bond yield plotted against the country's numeric S&P rating; non-USD bonds are
coloured by country (with one ISO label per country) and USD-denominated bonds
are overlaid in dark green. Two order-2 (quadratic) fits with confidence bands
summarise the non-USD (navy) and USD (green) relationships, and a dashed line
marks the BBB- investment-grade cutoff.

Inputs:
  0_raw/credit_ratings/bondyield_10yr.csv     (wide: Date + "<Country>[ (USD)] 10Y"; proprietary Bloomberg)
  0_raw/credit_ratings/sp_ratings_v3.csv      (S&P rating history; proprietary) + rating_df_v2.csv

Output:
  4_figures/supplementary/rating_coupons.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

import sys
sys.path.insert(0, os.path.join(PKG, "2_code", "common"))
from paths import PATHS, require_proprietary  # noqa: E402
from prep import load_agency_scores  # noqa: E402

YIELD_PATH = require_proprietary("bondyield_10yr")  # Bloomberg, not redistributed
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "rating_coupons.png")

# Ordered S&P / Fitch rating labels for the x-axis (positions 2..21).
text_labels = [
    "CC", "CCC-", "CCC", "CCC+", "B-", "B", "B+", "BB-", "BB", "BB+",
    "BBB-", "BBB", "BBB+", "A-", "A", "A+", "AA-", "AA", "AA+", "AAA",
]

# Bloomberg series country name -> ISO3. "United Kingdom" and "EU" map to None and are dropped.
ISO3 = {
    "Brazil": "BRA", "Colombia": "COL", "Argentina": "ARG", "Chile": "CHL",
    "Peru": "PER", "Mexico": "MEX", "Canada": "CAN", "Costa Rica": "CRI",
    "Panama": "PAN", "United States": "USA", "Venezuela": "VEN", "Turkey": "TUR",
    "Czech Republic": "CZE", "United Kingdom": None, "Hungary": "HUN",
    "Iceland": "ISL", "Israel": "ISR", "Lebanon": "LBN", "Nigeria": "NGA",
    "Poland": "POL", "Romania": "ROU", "South Africa": "ZAF", "Saudi Arabia": "SAU",
    "Ukraine": "UKR", "Russia": "RUS", "Austria": "AUT", "Belgium": "BEL",
    "Cyprus": "CYP", "Denmark": "DNK", "EU": None, "Finland": "FIN",
    "France": "FRA", "Germany": "DEU", "Greece": "GRC", "Croatia": "HRV",
    "Ireland": "IRL", "Italy": "ITA", "Latvia": "LVA", "Lithuania": "LTU",
    "Netherlands": "NLD", "Norway": "NOR", "Portugal": "PRT", "Slovakia": "SVK",
    "Spain": "ESP", "Slovenia": "SVN", "Sweden": "SWE", "Switzerland": "CHE",
    "Philippines": "PHL", "Indonesia": "IDN", "Australia": "AUS", "China": "CHN",
    "India": "IND", "Japan": "JPN", "Malaysia": "MYS", "New Zealand": "NZL",
    "Pakistan": "PAK", "Singapore": "SGP", "South Korea": "KOR", "Taiwan": "TWN",
    "Thailand": "THA",
}

# --- build the long-format yield panel -------------------------------------
yield_10yr = pd.read_csv(YIELD_PATH)
yield_10yr["Date"] = pd.to_datetime(yield_10yr["Date"])
yield_10yr["Year"] = yield_10yr["Date"].dt.year
yield_10yr.loc[(yield_10yr["Year"] >= 2050) & (yield_10yr["Year"] <= 2099), "Year"] -= 1000
yield_10yr = yield_10yr.drop(columns=["Date"])
yield_10yr = yield_10yr.groupby("Year").mean(numeric_only=True).reset_index()
yield_10yr = yield_10yr[yield_10yr["Year"] >= 1990]
yield_10yr = yield_10yr.melt(id_vars=["Year"], var_name="Country", value_name="Yield")
yield_10yr["Countryname"] = yield_10yr["Country"].str.replace(r"\(USD\)|10Y", "", regex=True).str.strip()
yield_10yr["USD_flag"] = yield_10yr["Country"].str.contains(r"\(USD\)").astype(int)
yield_10yr["ISO"] = yield_10yr["Countryname"].map(ISO3)

# per-agency scores (proprietary raw histories) on the rows the public panel rates
rating_data = pd.read_csv(PATHS["rating_df_v2"])[["ISO", "year", "score_total"]].merge(
    load_agency_scores(), on=["ISO", "year"], how="left")
_agency = ["score_sp", "score_moodys", "score_fitch"]
rating_data[_agency] = rating_data[_agency].where(rating_data["score_total"].notna())
yield_10yr = yield_10yr.merge(rating_data.rename(columns={"year": "Year"}), on=["ISO", "Year"], how="left")

# --- plot ("S&P Rating and 10-yr Bond rates") ------------------------------
plot_df = yield_10yr[(yield_10yr["USD_flag"] == 0) & (yield_10yr["Year"] >= 2000)
                     & (yield_10yr["Yield"] > 0) & ~yield_10yr["ISO"].isin(["LBN"])]
plot_df_USD = yield_10yr[(yield_10yr["USD_flag"] == 1) & (yield_10yr["Year"] >= 2000)
                         & (yield_10yr["Yield"] > 0) & ~yield_10yr["ISO"].isin(["LBN", "VEN"])]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=plot_df["score_sp"], y=plot_df["Yield"], hue=plot_df["ISO"], data=plot_df,
                alpha=0.7, edgecolor="white", palette="tab20", legend=False)

for iso, group in plot_df.groupby("ISO"):
    row = group.iloc[0]
    plt.text(row["score_sp"], row["Yield"], row["ISO"], fontsize=8, alpha=0.7)

sns.scatterplot(x=plot_df_USD["score_sp"], y=plot_df_USD["Yield"], color="darkgreen",
                data=plot_df_USD, alpha=0.3, edgecolor="white")

sns.regplot(x=plot_df["score_sp"], y=plot_df["Yield"], scatter=False, color="navy",
            label="quadratic fit (non-USD bonds)", order=2, seed=0)
sns.regplot(x=plot_df_USD["score_sp"], y=plot_df_USD["Yield"], scatter=False, color="darkgreen",
            label="quadratic fit (USD bonds)", order=2, line_kws={"alpha": 0.5}, seed=0)

plt.xticks(ticks=list(range(2, 22)), labels=text_labels, rotation=45, ha="right")
plt.axvline(x=text_labels.index("BBB-") + 2, color="grey", linestyle="--", alpha=0.7)
plt.gca().invert_xaxis()
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

plt.xlabel("S&P Rating")
plt.ylabel("annual coupon (%)")
plt.title("S&P Rating and 10-yr Bond rates")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print("wrote", OUT_PATH)
