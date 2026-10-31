"""
M5_basispoint_plot.py

Draws Fig. 5 and the related SI figures from the per-seed XGB predictions written by
generate_XGB_predictions.py (panel primitives are in M5_basispoint.py):

  --metric bps --all-isos:
    4_figures/main/M5_basispoint/M5_partA_grade.png, M5_partB_nonrated.png   (Fig. 5)
    4_figures/supplementary/XGB/M5_bp_currency_{N}seeds.png                  (Fig. S33)
  --metric notches --all-isos:
    4_figures/supplementary/XGB/M5_rating_notch_full_{N}seeds.png            (Fig. S31)

Inputs (read-only):
  3_output/XGB/predictions/{TAG}.csv   (rated + non-rated combined)
where TAG = "{model}_classification_{scoring}_0_{N}" for XGB,
            "XGBO_{scoring}_0_{N}" for XGBO.

Usage:
    python M5_basispoint_plot.py [--seeds 50] [--model XGB]
                                  [--out-dir <path>]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "2_code", "common"))

from M5_basispoint import (  # noqa: E402
    plot_two_panel, plot_single_panel, plot_three_panel,
)
from paths import read_predictions, predictions_exist  # noqa: E402
from prep import (  # noqa: E402
    load_all_data,
    get_currency_iso_groups,
    PATHS, REGIONS, NONRATED_ISOS,
)

PRED_DIR = os.path.join(_PROJECT_ROOT, "3_output", "XGB", "predictions")
SI_OUTDIR          = os.path.join(_PROJECT_ROOT, "4_figures", "supplementary", "XGB")
# Main-text Fig. 5, written as two files (part A: IG + SG panels, part B: non-rated).
M5_MAIN_OUTDIR     = os.path.join(_PROJECT_ROOT, "4_figures", "main", "M5_basispoint")

# X-axis label for main-text Fig. 5 (the SI currency figure keeps "Risk premium estimate").
M5_MAIN_XLABEL = 'Credit risk premium\n(basis points)'

# Notch axis: symlog with a linear core near zero (most countries within ±0.1 notch).
NOTCH_AXIS_STYLE = dict(
    xscale='symlog',
    xscale_kwargs=dict(linthresh=0.05, linscale=0.5),
    xticks=[-1.0, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0],
    xlim=(-1.2, 2.0),
    xlabel='estimated rating change (notch)\n>0 means relative downgrade',
    gridlines=[-1.0, -0.1, 0.1, 0.5, 1.0, 2.0],
    tick_fontsize=6,
    label_fontsize=7,
)
# Non-rated panel uses a tighter axis (no country reaches the +2 territory).
NOTCH_AXIS_STYLE_NR = {
    **NOTCH_AXIS_STYLE,
    'xticks':    [-1.0, -0.1, 0.0, 0.1, 0.5, 1.0],
    'xlim':      (-1.2, 1.2),
    'gridlines': [-1.0, -0.1, 0.1, 0.5, 1.0],
}


def _hide_notch_full_xticklabels(fig):
    """Blank the IG '1.0' and SG/NR '-1.0' xtick labels on the notch three-panel plots."""
    panel_axes = [a for a in fig.axes if not getattr(a, '_label', '')
                                          .startswith('<colorbar')]
    panel_axes = fig.axes[:3]
    hide_map = {0: 1.0, 1: -1.0, 2: -1.0}
    for idx, hide_val in hide_map.items():
        if idx >= len(panel_axes):
            continue
        ax = panel_axes[idx]
        ticks = list(ax.get_xticks())
        labels = [t.get_text() for t in ax.get_xticklabels()]
        new_labels = []
        for t, lab in zip(ticks, labels):
            if abs(t - hide_val) < 1e-9:
                new_labels.append('')
            else:
                new_labels.append(lab)
        ax.set_xticklabels(new_labels)


def _expected_notch(probs_strs):
    """Convert space-separated probability-vector strings to expected class index.

    E[notch] = sum_i i * P(class=i), from strings like '0.01 0.04 ...'.
    """
    arr = np.vstack([np.fromstring(s, sep=' ') for s in probs_strs])
    n_classes = arr.shape[1]
    class_idx = np.arange(n_classes, dtype=float)
    return arr @ class_idx


def _load_metadata():
    """Load all metadata needed to merge onto saved predictions."""
    all_data = load_all_data(include_bonds=False)
    # Per-country currency shares of bond issuance (all years and maturities; aggregate of the
    # proprietary bond data, written by 2_code/data_prep/build_proprietary_aggregates.py).
    currency_summary = pd.read_csv(PATHS['bond_currency_summary'])
    iso_groups = get_currency_iso_groups(currency_summary)

    cpn_summary = pd.read_csv(PATHS['cpn_summary_2011_2019'])
    country_names = (all_data['country_master'][['ISO', 'country']]
                     .drop_duplicates())

    tc_path = os.path.join(_PROJECT_ROOT, "0_raw", "climate_exposures", "tc2_maxws.csv")
    tc_df = pd.read_csv(tc_path)
    never_exposed = set(
        tc_df.groupby('ISO')['totalexposed']
             .max().pipe(lambda s: s[s <= 0].index)
    )

    # ISOs with valid CF inputs in attributable_shares.csv; others are dropped from the plot.
    attr_path = os.path.join(_PROJECT_ROOT, "3_output",
                              "counterfactual_data", "attributable_shares.csv")
    attr_isos = set(pd.read_csv(attr_path, usecols=['ISO'])['ISO'])

    # Non-rated sovereigns with no bond in currency_summary: drawn as open markers.
    nr_bond_issuers   = set(currency_summary['ISO']) & set(NONRATED_ISOS)
    never_issued_bond = set(NONRATED_ISOS) - nr_bond_issuers
    # Any ISO (rated or not) with a bond in currency_summary; the full plots draw the rest open.
    ever_issued_bond = set(currency_summary['ISO'])
    # Eurobond issuers missing from the Bloomberg/LSEG data: MOZ (USD 2031), COG (USD 2007).
    ever_issued_bond |= {'MOZ', 'COG'}
    # Myanmar has domestic instruments only, no international sovereign bond.
    ever_issued_bond -= {'MMR'}

    # 2019 grade: investment grade if ANY agency rates the country BBB- or above
    # (e.g. ZAF, Moody's Baa3); countries absent from the file are unrated.
    grade_2019 = pd.read_csv(PATHS['rating_grade_2019']).set_index('ISO')['grade_2019']

    # Extratropical sovereigns: country bounds entirely poleward of 23.5° (NOR forced in,
    # since Bouvet Island pulls its bounds south).
    try:
        import geopandas as gpd
        shp = os.path.join(_PROJECT_ROOT, "0_raw", "shapefiles", "ne_10m_admin_0_map_units", "ne_10m_admin_0_map_units.shp")
        world = gpd.read_file(shp)
        b = (world[['ADM0_A3', 'geometry']].dropna(subset=['ADM0_A3'])
             .assign(_miny=lambda d: d.geometry.bounds['miny'],
                     _maxy=lambda d: d.geometry.bounds['maxy'])
             .groupby('ADM0_A3').agg(miny=('_miny', 'min'),
                                       maxy=('_maxy', 'max')))
        extratropical_isos = set(b.index[(b['miny'] > 23.5) |
                                          (b['maxy'] < -23.5)]) | {'NOR'}
    except Exception:
        extratropical_isos = set()

    return {
        'Agrade_ISOs': all_data['Agrade_ISOs'],
        'Bgrade_ISOs': all_data['Bgrade_ISOs'],
        'avg_warming': all_data['avg_warming'],
        'avg_ws_by_iso': all_data['avg_ws_by_iso'],
        'country_names': country_names,
        'cpn_summary': cpn_summary,
        'dominant_bond_isos': iso_groups['dominant_currency'],
        'other_bond_isos': iso_groups['other_currency'],
        'never_exposed': never_exposed,
        'never_issued_bond': never_issued_bond,
        'ever_issued_bond': ever_issued_bond,
        'attr_isos': attr_isos,
        'grade_2019': grade_2019,
        'extratropical_isos': extratropical_isos,
    }


def _build_plot_dfs(preds: pd.DataFrame, meta: dict, cf: str = 'combined',
                    metric: str = 'bps', include_all_isos: bool = False):
    """Merge metadata onto saved predictions and split into plot DataFrames.

    Parameters
    ----------
    cf : str
        CF variant. One of {'combined', 'debt', 'TC', 'temp', 'GDPcomb'}.
    metric : str
        'bps' (default): expected coupon change (basis points).
        'notches': expected rating-class change, E[class|CF] - E[class|obs]
        (>0 = climate downgrade).
    """
    # Drop ISOs missing from attributable_shares.csv (NaN CF inputs give spurious predictions).
    attr_isos = meta.get('attr_isos')
    if attr_isos is not None:
        n_before = preds['ISO'].nunique()
        dropped = sorted(set(preds['ISO']) - attr_isos)
        preds = preds[preds['ISO'].isin(attr_isos)].copy()
        if dropped:
            print(f"  [filter] dropped {len(dropped)} ISO(s) missing from "
                  f"attributable_shares.csv: {dropped}")

    rated = preds[preds['is_NR'] == 0].copy()
    nr    = preds[preds['is_NR'] == 1].copy()

    cd_col = f'change_dominant_{cf}'
    co_col = f'change_other_{cf}'

    # Pre-compute notch column on raw (rated, nr) frames before merging.
    if metric == 'notches':
        for _df in (rated, nr):
            obs_notch = _expected_notch(_df['y_probs'].values)
            cf_notch  = _expected_notch(_df[f'y_probs_cf_{cf}'].values)
            _df['notch'] = cf_notch - obs_notch  # >0 = climate downgrade

    def _merge_and_compute(df):
        df = df.drop(columns=['warming', 'avg_ws_period', 'meantemp_hist',
                              'country', 'cpn_dominant', 'cpn_other'],
                     errors='ignore')
        df = df.merge(meta['avg_warming'], on='ISO', how='left')
        df = df.merge(meta['avg_ws_by_iso'], on='ISO', how='left')
        df = df.merge(meta['country_names'], on='ISO', how='left')
        df = df.merge(meta['cpn_summary'][['ISO', 'cpn_dominant', 'cpn_other']],
                      on='ISO', how='left')
        if cd_col in df.columns:
            df['cpn_dominant_cf'] = df['cpn_dominant'] - df[cd_col]
            df['climate_share']   = (df[cd_col] / df['cpn_dominant']) * 100
            df['bps']             = (df[cd_col] * 100).round(1)
        if co_col in df.columns:
            df['cpn_other_cf'] = df['cpn_other'] - df[co_col]
        return df

    cf_all_merged    = _merge_and_compute(rated)
    cf_all_nr_merged = _merge_and_compute(nr)

    dom_isos = meta['dominant_bond_isos']
    oth_isos = meta['other_bond_isos']
    Agrade   = meta['Agrade_ISOs']
    Bgrade   = meta['Bgrade_ISOs']

    dom_mask = (cf_all_merged['ISO'].isin(dom_isos + ['SYC', 'JPN'])
                & cf_all_merged['cpn_dominant'].notna())
    plot_dominant = cf_all_merged[dom_mask].copy()

    other_mask = (cf_all_merged['ISO'].isin(oth_isos)
                  & cf_all_merged['cpn_other'].notna())
    plot_other = cf_all_merged[other_mask].copy()
    plot_other['bps']           = (plot_other[co_col] * 100).round(1)
    plot_other['climate_share'] = (plot_other[co_col] / plot_other['cpn_other']) * 100

    plot_df_Agrade = pd.concat([
        plot_dominant[plot_dominant['ISO'].isin(Agrade)],
        plot_other[plot_other['ISO'].isin(Agrade)],
    ], ignore_index=True)
    plot_df_Bgrade = pd.concat([
        plot_dominant[plot_dominant['ISO'].isin(Bgrade)],
        plot_other[plot_other['ISO'].isin(Bgrade)],
    ], ignore_index=True)

    plot_df_NR = cf_all_nr_merged.copy()
    plot_df_NR['bps'] = (plot_df_NR[co_col] * 100).round(1)
    plot_df_NR_filtered = plot_df_NR[plot_df_NR['ISO'].isin(NONRATED_ISOS)].copy()

    out = {
        'Agrade':   plot_df_Agrade,
        'Bgrade':   plot_df_Bgrade,
        'dominant': plot_dominant,
        'other':    plot_other,
        'NR':       plot_df_NR_filtered,
    }

    if include_all_isos:
        # "Full" panels: every ISO, split by 2019 grade into IG / SG / NR (unrated).
        rated_full = cf_all_merged.copy()
        nr_full    = cf_all_nr_merged.copy()
        is_dom = rated_full['ISO'].isin(set(dom_isos) | {'SYC', 'JPN'})
        rated_full['bps'] = np.where(
            is_dom,
            (rated_full[cd_col] * 100).round(1),
            (rated_full[co_col] * 100).round(1),
        )
        nr_full['bps'] = (nr_full[co_col] * 100).round(1)
        plot_df_full = pd.concat([rated_full, nr_full], ignore_index=True)
        out['full'] = plot_df_full

        grade_2019 = meta['grade_2019']          # ISO -> 'IG' / 'SG' (unrated: absent)
        IG_isos = set(grade_2019.index[grade_2019 == 'IG'])
        SG_isos = set(grade_2019.index[grade_2019 == 'SG'])
        NR_isos = set(plot_df_full['ISO']) - IG_isos - SG_isos

        out['full_IG'] = plot_df_full[plot_df_full['ISO'].isin(IG_isos)].copy()
        out['full_SG'] = plot_df_full[plot_df_full['ISO'].isin(SG_isos)].copy()
        out['full_NR_obs'] = plot_df_full[plot_df_full['ISO'].isin(NR_isos)].copy()

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=50)
    p.add_argument('--model', type=str, default='XGB',
                   choices=['XGB', 'XGBO'])
    p.add_argument('--scoring', type=str, default='neglogloss')
    p.add_argument('--out-dir', type=str, default=None,
                   help='Override the SI output directory (default 4_figures/supplementary/XGB).')
    p.add_argument('--preds-path', type=str, default=None,
                   help='Override path to predictions CSV (default uses tag)')
    p.add_argument('--cf', type=str, default='combined',
                   choices=['combined', 'debt', 'TC', 'temp', 'GDPcomb'],
                   help='CF variant to plot (default: combined). Selects '
                        'change_dominant_{cf}/change_other_{cf} columns.')
    p.add_argument('--metric', type=str, default='bps',
                   choices=['bps', 'notches'],
                   help="Plot metric. 'bps' (default) = basis-point change in "
                        "expected coupon. 'notches' = probability-weighted "
                        "rating-class shift (>0 = climate downgrade).")
    p.add_argument('--all-isos', action='store_true',
                   help='Also produce an M5_full single-panel plot showing every '
                        'ISO in the prediction set (including non-issuers); '
                        'non-issuers render as open markers per the existing '
                        'never_issued_bond convention.')
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = SI_OUTDIR

    tag = (f'{args.model}_classification_{args.scoring}_0_{args.seeds}'
           if args.model == 'XGB'
           else f'XGBO_{args.scoring}_0_{args.seeds}')
    preds_path = args.preds_path or os.path.join(PRED_DIR, f'{tag}.csv')

    print(f"Reading {preds_path}")
    if not predictions_exist(preds_path):
        sys.exit(f"  [error] not found: {preds_path}")
    preds = read_predictions(preds_path)
    print(f"  {len(preds)} rows, {preds['ISO'].nunique()} ISOs, "
          f"seeds={preds['seed'].nunique()}, "
          f"non-rated rows={(preds['is_NR']==1).sum()}")

    print("Loading metadata...")
    meta = _load_metadata()
    print(f"  never-exposed ISOs (totalexposed never >0): "
          f"{len(meta['never_exposed'])}")

    plot_dfs = _build_plot_dfs(preds, meta, cf=args.cf, metric=args.metric,
                                include_all_isos=args.all_isos)
    for k, v in plot_dfs.items():
        print(f"  {k}: {v['ISO'].nunique()} ISOs, {len(v)} rows")

    os.makedirs(args.out_dir, exist_ok=True)
    # Match generate_XGB_predictions.py inline naming when cf=='combined'; suffix otherwise
    sfx = (f'_{args.seeds}seeds' if args.cf == 'combined'
           else f'_{args.seeds}seeds_{args.cf}')

    if args.metric == 'notches':
        plot_var   = 'notch'
        prefix     = 'M5_rating_notch'
        axis_style = NOTCH_AXIS_STYLE
    else:
        plot_var   = 'bps'
        prefix     = 'M5_bp'
        axis_style = None  # default bps styling

    # SI Fig. S33 (bps): basis-point change by issuer currency.
    if args.metric == 'bps':
        fig2 = plot_two_panel(
            plot_dfs['dominant'], plot_dfs['other'], plot_var, REGIONS,
            figsize=(6, 7), iso_label_fontsize=4.5, region_label_fontsize=8,
            label_threshold='none',
            left_title='Dominant Currency issuers',
            right_title='Local Currency issuers',
            title='', cbar_show=True,
            never_exposed=meta['never_exposed'],
            never_issued_bond=meta['never_issued_bond'],
            extratropical=meta['extratropical_isos'],
            axis_style=axis_style,
        )
        fig2.savefig(os.path.join(args.out_dir, f'{prefix}_currency{sfx}.png'),
                     dpi=300, bbox_inches='tight')
        print(f"  Saved {prefix}_currency{sfx}.png")

    # Every ISO, classified by 2019 sovereign credit rating: SI Fig. S31 (notches)
    # and main-text Fig. 5 (bps).
    if args.all_isos and 'full' in plot_dfs:
        full_df = plot_dfs['full']
        # Open markers: ISOs that never issued a bond.
        no_obs_isos = set(full_df['ISO']) - set(meta['ever_issued_bond'])
        n_ig = plot_dfs['full_IG']['ISO'].nunique()
        n_sg = plot_dfs['full_SG']['ISO'].nunique()
        n_nr = plot_dfs['full_NR_obs']['ISO'].nunique()

        if args.metric == 'notches':
            fig_full = plot_three_panel(
                plot_dfs['full_IG'], plot_dfs['full_SG'], plot_dfs['full_NR_obs'],
                plot_var, REGIONS,
                figsize=(8, max(0.12 * max(n_ig, n_sg), 7.6)),
                iso_label_fontsize=4.5, region_label_fontsize=8,
                label_threshold='none',
                left_title='Investment-grade',
                middle_title='Speculative-grade',
                right_title='Non-rated countries',
                title='', cbar_show=True,
                never_exposed=meta['never_exposed'],
                never_issued_bond=no_obs_isos,
                extratropical=meta['extratropical_isos'],
                axis_style=axis_style,
                axis_style_left=NOTCH_AXIS_STYLE_NR,
                axis_style_right=NOTCH_AXIS_STYLE_NR,
            )
            _hide_notch_full_xticklabels(fig_full)
            fig_full.savefig(os.path.join(args.out_dir, f'{prefix}_full{sfx}.png'),
                             dpi=300, bbox_inches='tight')
            print(f"  Saved {prefix}_full{sfx}.png "
                  f"(IG={n_ig}, SG={n_sg}, NR={n_nr})")

        # --- Main-text Fig. 5 (all ISOs, by 2019 grade) --------------------
        #   M5_partA_grade.png    -> investment- and speculative-grade panels
        #   M5_partB_nonrated.png -> non-rated panel
        if args.metric == 'bps':
            os.makedirs(M5_MAIN_OUTDIR, exist_ok=True)
            m5_ig_style = {'xlim': (-14, 20),
                           'xticks': [-10, -5, -1, 0, 1, 5, 10, 20],
                           'gridlines': [-5, -1, 1, 5, 10],
                           'xlabel': M5_MAIN_XLABEL}
            m5_sg_style = {'xlim': (-14, 60),
                           'xticks': [-10, -5, -1, 0, 1, 5, 10, 50],
                           'gridlines': [-5, -1, 1, 5, 10],
                           'xlabel': M5_MAIN_XLABEL}
            m5_nr_style = {'xlim': (-5, 55),
                           'xticks': [-1, 0, 1, 5, 10, 50],
                           'gridlines': [-1, 1, 5, 10],
                           'xlabel': M5_MAIN_XLABEL}

            fig_a = plot_two_panel(
                plot_dfs['full_IG'], plot_dfs['full_SG'], plot_var, REGIONS,
                figsize=(6.5, max(0.12 * max(n_ig, n_sg), 7.6)),
                iso_label_fontsize=4.5, region_label_fontsize=8,
                label_threshold='none',
                left_title='Investment-grade', right_title='Speculative-grade',
                title='', cbar_show=True,
                never_exposed=meta['never_exposed'],
                never_issued_bond=no_obs_isos,
                extratropical=meta['extratropical_isos'],
                axis_style=m5_sg_style,
                legend_no_bond=True,
            )
            # Re-style the left (investment-grade) axis with the tighter range.
            _style_axis_left = fig_a.axes[0]
            _style_axis_left.set_xlim(*m5_ig_style['xlim'])
            _style_axis_left.set_xticks(m5_ig_style['xticks'])
            _style_axis_left.set_xticklabels(
                ['' if abs(t - 20) < 1e-9 else str(t).replace('.0', '')
                 for t in m5_ig_style['xticks']],
                fontsize=8, fontweight='bold')
            fig_a.savefig(os.path.join(M5_MAIN_OUTDIR, 'M5_partA_grade.png'),
                          dpi=300, bbox_inches='tight')
            print(f"  Saved M5_partA_grade.png -> {M5_MAIN_OUTDIR} "
                  f"(IG={n_ig}, SG={n_sg})")

            # Part B has no colorbar / legend / region labels (part A carries them).
            fig_b = plot_single_panel(
                plot_dfs['full_NR_obs'], plot_var, REGIONS,
                figsize=(2.8, 0.13 * n_nr + 1.6),
                title='Non-rated countries', cbar_show=False,
                label_threshold='none', iso_label_fontsize=4.5,
                region_label_fontsize=8, title_fontsize=9,
                never_exposed=meta['never_exposed'],
                never_issued_bond=no_obs_isos,
                extratropical=meta['extratropical_isos'],
                axis_style=m5_nr_style,
                marker_legend=False, region_labels=False,
            )
            fig_b.savefig(os.path.join(M5_MAIN_OUTDIR, 'M5_partB_nonrated.png'),
                          dpi=300, bbox_inches='tight')
            print(f"  Saved M5_partB_nonrated.png -> {M5_MAIN_OUTDIR} (NR={n_nr})")

    plt.close('all')


if __name__ == '__main__':
    main()
