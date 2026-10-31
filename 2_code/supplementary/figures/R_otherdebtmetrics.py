"""
R_otherdebtmetrics.png  (SI robustness figure)

Local-projection impulse responses of FOUR alternative debt metrics to tropical
cyclone exposure, one stacked section each:
  1. Public sector debt as share of GDP   (public_pp_GDD)
  2. Private sector debt as share of GDP  (private_pp_GDD)
  3. External debt as a share of GNI      (ed_pp_GNI)
  4. External debt as a share of exports  (ed_pp_exports)

Each section is a 1x3 grid over TC intensity terms (TS / Cat1-3 / >Cat3) with one
line per country subset (full / rated / non-rated / non-oecd / sids) and the
"full" confidence band shaded. The four sections are rendered separately and
stacked vertically.

Inputs:
  3_output/debtratio/TC+Tshock/coef_df/LP_coefficients/*_TC.csv
  3_output/debtratio/TC+Tshock/coef_df/alternativeTC/*_TC.csv

Output:
  4_figures/supplementary/R_otherdebtmetrics.png
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
COEF = os.path.join(PKG, "3_output", "debtratio", "TC+Tshock", "coef_df")
OUT_PATH = os.path.join(PKG, "4_figures", "supplementary", "R_otherdebtmetrics.png")

KEY = ["term", "y_h", "Tshock_type", "outcome", "specification", "model", "subset"]


def load_coef():
    frames = []
    for sub in ("LP_coefficients",):
        for f in glob.glob(os.path.join(COEF, sub, "*.csv")):
            if f.endswith("_TEMP.csv"):
                continue
            frames.append(pd.read_csv(f))
    TC = pd.concat(frames, ignore_index=True)
    # one row per (panel line, horizon): drop duplicated ed_* files across dirs
    return TC.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)


def sensitivity_irfs(df, outcome="public_pp_GDD",
                     terms=("ts_v2", "cat1_v2", "cat3_v2"), subset="full",
                     time_trend="linear", tshock_type="NATH",
                     specification="1980onwards_gl2_TC10_T02", titles=None,
                     labels=None, colors=None, linestyles=None, ci_index=0,
                     figsize=(16, 2), legend_width_ratio=0.25, sharey=True,
                     suptitle="", ylabel="Estimate", ylim=None, yticks=None,
                     y_padding=0.1, pct_format=False, ci_alpha=0.2,
                     legend_title=None, legend_fontsize=10, title_fontsize=10,
                     ylabel_fontsize=10, xlabel="Years since impact", wspace=0.1,
                     xticks=(0, 2, 4, 6, 8, 10, 12, 14)):
    """Plot IRFs for multiple terms, overlaying whichever argument is a list."""
    params = {"subset": subset, "time_trend": time_trend, "tshock_type": tshock_type,
              "specification": specification, "outcome": outcome}
    col_map = {"subset": "subset", "time_trend": "model", "tshock_type": "Tshock_type",
               "specification": "specification", "outcome": "outcome"}

    varying_param = varying_values = None
    for name, value in params.items():
        if isinstance(value, list):
            if varying_param is not None:
                raise ValueError("Only one parameter can be a list")
            varying_param, varying_values = name, value
    if varying_param is None:
        varying_param, varying_values = "subset", [subset]

    fixed_params = {col_map[n]: v for n, v in params.items() if n != varying_param}

    if titles is None:
        titles = ["Tropical storm (18-33 m/s)", "Category 1-3 (33-50 m/s)", "Category >3 (>50 m/s)"]
    if labels is None:
        labels = {v: str(v) for v in varying_values}
    if colors is None:
        default = ["steelblue", "coral", "seagreen", "mediumpurple", "goldenrod", "crimson", "teal"]
        colors = {v: default[i % len(default)] for i, v in enumerate(varying_values)}
    if linestyles is None:
        linestyles = {v: "-" if i == 0 else "--" for i, v in enumerate(varying_values)}
    ci_value = varying_values[ci_index] if isinstance(ci_index, int) and ci_index < len(varying_values) else None
    if legend_title is None:
        legend_title = {"subset": "Subset", "time_trend": "Time Trend", "tshock_type": "Shock Type",
                        "specification": "Specification", "outcome": "Outcome"}.get(varying_param, varying_param)

    n_terms = len(terms)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_terms + 1, width_ratios=[1] * n_terms + [legend_width_ratio], wspace=wspace)
    axes = []
    for i in range(n_terms):
        ax = fig.add_subplot(gs[0, i]) if (i == 0 or not sharey) else fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)
    legend_ax = fig.add_subplot(gs[0, n_terms])
    legend_ax.axis("off")

    if suptitle:
        fig.subplots_adjust(top=0.72)
        fig.suptitle(suptitle, fontsize=14, y=0.99)

    legend_handles, legend_labels = [], []
    for idx, (ax, term, title) in enumerate(zip(axes, terms, titles)):
        for val in varying_values:
            mask = (df["term"] == term)
            for col, fv in {col_map[varying_param]: val, **fixed_params}.items():
                mask &= (df[col] == fv)
            data = df[mask].copy()
            if len(data) == 0:
                continue
            data = data[data["y_h"] != -1]
            anchor = pd.DataFrame({"y_h": [-1], "estimate": [0.0], "conf.low": [0.0], "conf.high": [0.0]})
            data = pd.concat([anchor, data], ignore_index=True).sort_values("y_h").reset_index(drop=True)
            color, linestyle, label = colors.get(val, "steelblue"), linestyles.get(val, "-"), labels.get(val, str(val))
            line, = ax.plot(data["y_h"], data["estimate"], color=color, linestyle=linestyle,
                            linewidth=1.5 if val == varying_values[0] else 1, label=label)
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)
            if val == ci_value:
                ax.fill_between(data["y_h"], data["conf.low"], data["conf.high"], alpha=ci_alpha, color=color)

        ax.axhline(0, color="grey", linestyle="-", alpha=0.6)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel)
        if xticks is not None:
            ax.set_xticks(list(xticks))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, left=False)
        if pct_format:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
        if ylim is not None:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)

    if legend_handles:
        legend_ax.legend(legend_handles, legend_labels, title=legend_title,
                         loc="center left", fontsize=legend_fontsize, frameon=False)
    return fig


# (outcome, title, y-limits, y-ticks): each row has its own scale, ticks at 0.5% steps.
SECTIONS = [
    ("public_pp_GDD", "public sector debt as share of GDP",
     (-0.006, 0.011), [-0.005, 0.0, 0.005, 0.01]),
    ("private_pp_GDD", "Private sector debt as share of GDP",
     (-0.0065, 0.0065), [-0.005, 0.0, 0.005]),
    ("ed_pp_GNI", "External debt as a share of GNI",
     (-0.0115, 0.0115), [-0.01, -0.005, 0.0, 0.005, 0.01]),
    ("ed_pp_exports", "External debt as a share of exports",
     (-0.0125, 0.0115), [-0.01, -0.005, 0.0, 0.005, 0.01]),
]
SUBSETS = ["full", "rated", "non-rated", "non-oecd", "sids"]
SUBSET_COLORS = {"full": "C0", "rated": "C1", "non-rated": "C2", "non-oecd": "C3", "sids": "C4"}


def main():
    TC = load_coef()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tmp_paths = []
    for outcome, suptitle, ylim, yticks in SECTIONS:
        fig = sensitivity_irfs(
            df=TC, terms=["ts_v2", "cat1_v2", "cat3_v2"], subset=SUBSETS,
            outcome=outcome, specification="1980onwards_gl2_TC10_T02",
            time_trend="linear", tshock_type="NATH", suptitle=suptitle,
            ylabel="Δ debt ratio / area exposed", ylim=ylim, yticks=yticks,
            pct_format=True, colors=SUBSET_COLORS,
        )
        p = os.path.join(os.path.dirname(OUT_PATH), f"_tmp_R_{outcome}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        tmp_paths.append(p)

    # stack the four sections vertically
    imgs = [Image.open(p).convert("RGB") for p in tmp_paths]
    width = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)
    canvas = Image.new("RGB", (width, total_h), "white")
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(OUT_PATH)
    for p in tmp_paths:
        os.remove(p)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
