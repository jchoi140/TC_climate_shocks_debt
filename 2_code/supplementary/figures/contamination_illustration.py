"""
contamination_illustration.png  (SI figure)

Multi-country contamination-bias illustration for the LP-DiD design.

Countries are split into Cat3+-dominant (small islands, negative within-country
Cat1-2 / Cat3+ correlation) and Cat1-2-dominant (larger countries, positive
correlation). Panel A shows the within-country demeaned exposures with each
country's fitted slope (dashed) and the pooled OLS slope (black). Panel B shows
the residuals from that pooled regression; nonzero within-country slopes signal
nonzero contamination weights (Goldsmith-Pinkham et al. 2024).

Inputs:
  0_raw/climate_exposures/tc2_maxws.csv        (ISO, year, cat1_v2, cat3_v2)
  0_raw/climate_exposures/country_master.csv   (sovereigncountry flag)
  1_input/economic_outcomes/gdd_debt_v2.csv (debt-sample ISOs)

Output:
  4_figures/supplementary/contamination_illustration.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

import sys
sys.path.insert(0, os.path.join(PKG, "2_code", "common"))
from paths import PATHS  # noqa: E402

TC_CSV = PATHS["tc"]
MASTER_CSV = PATHS["country_master"]
DEBT_CSV = PATHS["gdd_debt"]
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "contamination_illustration.png")

# Country groups: countries with sufficient TC events AND Cat3 variation
C3_DOM_ISOS = ["KNA", "DMA", "ATG", "TON", "JAM", "VUT"]           # negative within-corr
C1_DOM_ISOS = ["DOM", "HTI", "PHL", "FJI", "BLZ", "BGD", "HND"]    # positive within-corr


def main():
    tc = pd.read_csv(TC_CSV)
    master = pd.read_csv(MASTER_CSV)
    debt = pd.read_csv(DEBT_CSV)
    sovereign = master[master["sovereigncountry"] == 1]["ISO"].unique()
    debt_isos = set(debt["ISO"].unique())

    panel = tc[(tc["year"] >= 1980) & (tc["year"] < 2020)].copy()
    panel = panel[panel["ISO"].isin(sovereign) & panel["ISO"].isin(debt_isos)]
    panel[["cat1_v2", "cat3_v2"]] = panel[["cat1_v2", "cat3_v2"]].fillna(0)

    all_isos = C3_DOM_ISOS + C1_DOM_ISOS
    sub = panel[panel["ISO"].isin(all_isos)].copy()
    tc_exp = sub[(sub["cat1_v2"] > 0) | (sub["cat3_v2"] > 0)].copy()

    # Demean within country
    for iso in all_isos:
        mask = tc_exp["ISO"] == iso
        if mask.sum() < 2:
            continue
        tc_exp.loc[mask, "c1_demean"] = tc_exp.loc[mask, "cat1_v2"] - tc_exp.loc[mask, "cat1_v2"].mean()
        tc_exp.loc[mask, "c3_demean"] = tc_exp.loc[mask, "cat3_v2"] - tc_exp.loc[mask, "cat3_v2"].mean()

    tc_exp = tc_exp.dropna(subset=["c1_demean", "c3_demean"])

    # Seeded jitter to reduce overlap (reproducible)
    rng = np.random.default_rng(42)
    tc_exp["c3_jitter"] = tc_exp["c3_demean"] + rng.uniform(-1.5, 1.5, len(tc_exp))
    tc_exp["c1_jitter"] = tc_exp["c1_demean"] + rng.uniform(-1.5, 1.5, len(tc_exp))

    # Pooled regression of demeaned C1 on demeaned C3
    lr_pool = LinearRegression()
    lr_pool.fit(tc_exp[["c3_demean"]].values, tc_exp["c1_demean"].values)
    tc_exp["c1_fwl_resid"] = tc_exp["c1_demean"].values - lr_pool.predict(tc_exp[["c3_demean"]].values)

    # Distinct shades per country within each group
    c3_cmap = cm.get_cmap("Reds", len(C3_DOM_ISOS) + 3)
    c1_cmap = cm.get_cmap("Blues", len(C1_DOM_ISOS) + 3)
    c3_colors = {iso: c3_cmap(i / (len(C3_DOM_ISOS) + 1) + 0.3) for i, iso in enumerate(C3_DOM_ISOS)}
    c1_colors = {iso: c1_cmap(i / (len(C1_DOM_ISOS) + 1) + 0.25) for i, iso in enumerate(C1_DOM_ISOS)}
    country_colors = {**c3_colors, **c1_colors}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ── Panel A ──
    ax = axes[0]
    for iso in all_isos:
        d = tc_exp[tc_exp["ISO"] == iso]
        if len(d) < 2:
            continue
        color = country_colors[iso]
        ax.scatter(d["c3_jitter"], d["c1_jitter"], color=color, s=45, alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=3)
        if len(d) > 2:
            lr_w = LinearRegression().fit(d[["c3_demean"]].values, d["c1_demean"].values)
            x_range = np.linspace(d["c3_demean"].min() - 3, d["c3_demean"].max() + 3, 50)
            ax.plot(x_range, lr_w.predict(x_range.reshape(-1, 1)),
                    color=color, ls="--", lw=1.0, alpha=0.5)

    x_all = np.linspace(tc_exp["c3_demean"].min() - 5, tc_exp["c3_demean"].max() + 5, 50)
    ax.plot(x_all, lr_pool.predict(x_all.reshape(-1, 1)), color="black", lw=2.5, ls="-", zorder=4)

    ax.scatter([], [], color=c3_cmap(0.6), s=45, label=f'Cat3+ dominant: {", ".join(C3_DOM_ISOS)}')
    ax.scatter([], [], color=c1_cmap(0.6), s=45, label=f'Cat1-2 dominant: {", ".join(C1_DOM_ISOS)}')
    ax.plot([], [], color="black", lw=2.5, label="Pooled OLS line")

    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"Demeaned Cat3+ ($\tilde{X}_{i2}$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Demeaned Cat1-2 ($\tilde{X}_{i1}$)", fontsize=12, fontweight="bold")
    ax.set_title("A. Within-country demeaned exposures", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.15)

    # ── Panel B ──
    ax = axes[1]
    tc_exp["resid_jitter"] = tc_exp["c1_fwl_resid"] + rng.uniform(-1.5, 1.5, len(tc_exp))
    for iso in all_isos:
        d = tc_exp[tc_exp["ISO"] == iso]
        if len(d) < 2:
            continue
        color = country_colors[iso]
        ax.scatter(d["c3_jitter"], d["resid_jitter"], color=color, s=45, alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=3)
        if len(d) > 2:
            lr_r = LinearRegression().fit(d[["c3_demean"]].values, d["c1_fwl_resid"].values)
            x_range = np.linspace(d["c3_demean"].min() - 3, d["c3_demean"].max() + 3, 50)
            ax.plot(x_range, lr_r.predict(x_range.reshape(-1, 1)),
                    color=color, ls="--", lw=1.0, alpha=0.5)

    ax.axhline(0, color="black", lw=1.5, ls="-", alpha=0.7)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"Demeaned Cat3+ ($\tilde{X}_{i2}$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"FWL residual ($\overset{\approx}{X}_{i1}$)", fontsize=12, fontweight="bold")
    ax.set_title("B. Residuals from pooled regression", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)

    fig.suptitle("Within-Country Correlation in Treatment Bins",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
