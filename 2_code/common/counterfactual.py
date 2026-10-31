"""
counterfactual.py - builds counterfactual debt/GDP paths from LP coefficients,
plus the plots drawn from them.

Drawing
    plot_heatmap                     Fig. 2 panels b, d
    plot_counterfactual_comparison   Fig. 3
    plot_exposure_timeseries
    _draw_panel, _draw_paired_panel  Figs. S21, S22
    _make_two_row_plot               Fig. S23

Everything else builds the counterfactual paths and per-country impact frames.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import (LinearSegmentedColormap, TwoSlopeNorm,
                               ListedColormap, BoundaryNorm)
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.ndimage import gaussian_filter

from coef_loader import load_coefficients, extract_model_coefficients

__all__ = ['EXTRA_DARK', 'EXTRA_LIGHT', 'GREY_END_YEAR', 'PLOT_START_YEAR', 'SALMON_DARK', 'SALMON_LIGHT', 'TC_DARK', 'TC_LIGHT', 'analysis_years', 'vector_length'] + ['generate_iso_yearly_vectors_onlyTC', 'generate_iso_yearly_vectors_onlyTEMP_delta', 'process_results', 'calculate_country_counterfactuals', 'calculate_combined_counterfactuals', 'truncate_colormap', 'plot_heatmap', 'create_heatmap_grid', 'plot_counterfactual_comparison', 'prepare_map_data', 'plot_exposure_timeseries', '_draw_panel', '_draw_paired_panel', 'yearly_vectors_to_long', 'calculate_debt_ratio_impact_vector', 'build_tc_vectors_merged', 'prepare_alternative_tc_exposure', 'sum_vectors', 'calculate_cumulative_debt_ratio_impact', 'build_alternative_counterfactual_paths', 'build_alternative_counterfactual', 'impact_for_base', 'impact_for_alt', 'impact_for_paths', '_make_two_row_plot', 'generate_iso_yearly_vectors_avgws']


analysis_years = list(range(1990, 2020))
vector_length  = len(analysis_years)
PLOT_START_YEAR = 1970     # first year plotted; set to 1990 for a 1990-start series
GREY_END_YEAR   = 1980     # storm-year fills before this stay lightgrey
SALMON_LIGHT = '#F4B8A6'
SALMON_DARK  = '#D67862'
TC_LIGHT     = 'lightsteelblue'
TC_DARK      = 'steelblue'
EXTRA_LIGHT  = '#D9D9D9'
EXTRA_DARK   = '#7F7F7F'


def generate_iso_yearly_vectors_onlyTC(df, a, b, c, start_year=1990, end_year=2022):
    """Counterfactual vectors for TC shocks only: one vector per storm year in
    [start_year, end_year], covering calendar years [start_year, end_year-1]."""
    all_years = np.arange(start_year, end_year + 1, 1)
    vec_len = end_year - start_year
    iso_yearly_vectors = {}
    for iso in df['ISO'].unique():
        df_iso = df[df['ISO'] == iso]
        yearly_vectors = {}
        for yr in all_years:
            yearly_values = np.zeros(vec_len)
            max_i = min(len(a), vec_len - (yr - start_year))
            for i in range(max_i):
                index = yr - start_year + i
                if index < 0:
                    continue
                yearly_values[index] = np.sum(np.where(
                    df_iso['year'] == yr,
                    df_iso['ts_v2'] * a[i] + df_iso['cat1_v2'] * b[i] + df_iso['cat3_v2'] * c[i], 0), axis=0)
            yearly_vectors[str(yr)] = yearly_values
        iso_yearly_vectors[str(iso)] = yearly_vectors
    return iso_yearly_vectors


def generate_iso_yearly_vectors_onlyTEMP_delta(df, T, TT, start_year=1990, end_year=2022):
    """Generate counterfactual vectors for temperature delta shocks only."""
    all_years = np.arange(start_year, end_year + 1, 1)
    vec_len = end_year - start_year
    iso_yearly_vectors = {}
    for iso in df['ISO'].unique():
        df_iso = df[df['ISO'] == iso]
        yearly_vectors = {}
        for yr in all_years:
            yearly_values = np.zeros(vec_len)
            max_i = min(16, vec_len - (yr - start_year))
            for i in range(max_i):
                index = yr - start_year + i
                if index < 0:
                    continue
                yearly_values[index] = np.sum(np.where(
                    df_iso['year'] == yr,
                    df_iso['delta']*T[i] + df_iso['delta']*df_iso['meantemp_hist']*TT[i], 0), axis=0)
            yearly_vectors[str(yr)] = yearly_values
        iso_yearly_vectors[str(iso)] = yearly_vectors
    return iso_yearly_vectors


def process_results(results, start_year=1990, end_year=2022):
    """Counterfactual results as a tidy DataFrame (calendar years start_year..end_year-1)."""
    vec_len = end_year - start_year
    combined_df_list = []
    for iso, yearly_vectors in results.items():
        yearly_df_list = []
        for i, (year, values) in enumerate(yearly_vectors.items()):
            year_df = pd.DataFrame({f"{year}_cf": values})
            if i == 0:
                year_df['ISO'] = iso
            yearly_df_list.append(year_df)
        iso_df = pd.concat(yearly_df_list, axis=1)
        combined_df_list.append(iso_df)
    results_df = pd.concat(combined_df_list, ignore_index=True)
    years = np.tile(np.arange(start_year, end_year), len(results_df) // vec_len)
    results_df['year'] = years
    for year in range(start_year, end_year):
        cols = [f'{y}_cf' for y in range(year, end_year)]
        results_df[f'{year}_cf'] = results_df[cols].sum(axis=1)
    return results_df


def calculate_country_counterfactuals(countries, observed_df, model_results, outcome_col, start_year=1990, end_year=2023):
    """Calculate counterfactual outcomes for a list of countries."""
    combined_results = []
    for country in countries:
        outcome = observed_df[(observed_df['year'] >= start_year) & (observed_df['ISO'] == country)][['year', outcome_col]]
        country_cf = model_results[model_results['ISO'] == country].merge(outcome, on='year', how='left')
        country_cf['ISO'] = country
        for year in range(start_year, end_year):
            cf_col = f'{year}_cf'
            if cf_col in country_cf.columns:
                country_cf[f'debt_{year}'] = country_cf[outcome_col] / (1 + country_cf[cf_col])
        combined_results.append(country_cf)
    return pd.concat(combined_results, ignore_index=True)


def calculate_combined_counterfactuals(countries, observed_df, tc_results, temp_results, outcome_col, start_year=1990):
    """Calculate combined TC + temperature counterfactuals."""
    cf_col = f'{start_year}_cf'
    cf_temp = temp_results[[cf_col, 'ISO', 'year']].rename(columns={cf_col: f'{start_year}_cf_temp_rate'})
    cf_tc = tc_results[[cf_col, 'ISO', 'year']].rename(columns={cf_col: f'{start_year}_cf_TC_rate'})
    cf_combined = cf_tc.merge(cf_temp, on=['ISO', 'year'], how='left')
    cf_combined[f'{start_year}_cf_combined_rate'] = cf_combined[f'{start_year}_cf_temp_rate'] + cf_combined[f'{start_year}_cf_TC_rate']
    combined_results = []
    for country in countries:
        outcome = observed_df[(observed_df['year'] >= start_year) & (observed_df['ISO'] == country)][['year', outcome_col]]
        country_cf = cf_combined[cf_combined['ISO'] == country].merge(outcome, on='year', how='left')
        country_cf['ISO'] = country
        country_cf[f'{start_year}_cf_temp'] = country_cf[outcome_col] / (1 + country_cf[f'{start_year}_cf_temp_rate'])
        country_cf[f'{start_year}_cf_TC'] = country_cf[outcome_col] / (1 + country_cf[f'{start_year}_cf_TC_rate'])
        country_cf[f'{start_year}_cf_combined'] = country_cf[outcome_col] / (1 + country_cf[f'{start_year}_cf_combined_rate'])
        combined_results.append(country_cf)
    return pd.concat(combined_results, ignore_index=True)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate colormap range to focus on specific color range."""
    return LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', cmap(np.linspace(minval, maxval, n)))


def plot_heatmap(ax, data, cmap, tick_positions, tick_labels):
    """Plot a single TC impact heatmap panel."""
    hm = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        cbar_kws={'orientation': 'vertical', 'shrink': 0.9},
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 15.5)
    ax.set_xlabel("", fontsize=12)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([0, 25, 50, 75, 100], fontsize=12)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=13)
    ax.set_xlabel("Years since TC impact", fontsize=12, fontweight='bold')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(0.5)
    return hm


def create_heatmap_grid(coef_df, model_type='linear', shock_type='NATH', sign=1):
    """Create grid data for TC impact heatmaps."""
    coefs = extract_model_coefficients(coef_df, model_type, shock_type)
    area = np.arange(0, 100)
    years = np.arange(0, 16)
    grid = pd.DataFrame([(year, a) for year in years for a in area], columns=['years', 'area'])
    grid['value_ts'] = grid.apply(lambda row: row['area'] * coefs['ts'][int(row['years'])] * 100 * sign, axis=1)
    grid['value_cat1'] = grid.apply(lambda row: row['area'] * coefs['cat1'][int(row['years'])] * 100 * sign, axis=1)
    grid['value_cat3'] = grid.apply(lambda row: row['area'] * coefs['cat3'][int(row['years'])] * 100 * sign, axis=1)
    return grid


def plot_counterfactual_comparison(df, countries, outcome_col, cf_col, titles=None, cf_label='Counterfactual', ylabel=None, figsize=(15, 2.5)):
    """Plot observed vs counterfactual time series for multiple countries."""
    ncols = len(countries)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    if ncols == 1: axes = [axes]
    if titles is None: titles = countries
    if ylabel is None: ylabel = outcome_col.replace('_', ' ').title()
    for i, (iso, title) in enumerate(zip(countries, titles)):
        ax = axes[i]
        plot_data = df[df['ISO'] == iso]
        ax.plot(plot_data['year'], plot_data[outcome_col], 'o-', color='black', linewidth=2, label='Observed', markersize=3)
        ax.plot(plot_data['year'], plot_data[cf_col], 's--', color='#4169E1', linewidth=2, label=cf_label, markersize=3, alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        if i == 0: ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, axes


def prepare_map_data(world_gdf, cf_df, year=2019, observed_col='public_pp_gdd', cf_col='debt_1990'):
    """Prepare data for map visualization."""
    df = cf_df[cf_df['year'] == year][['ISO', cf_col, observed_col]].copy()
    df['diff'] = (df[observed_col] - df[cf_col]) / df[observed_col] * 100
    df = df[df['diff'] != 0]
    return world_gdf.merge(df, left_on='ADM0_A3', right_on='ISO', how='left')


def plot_exposure_timeseries(df, isos, exposure_cols, colors, ylabel):
    """Plot exposure time series for multiple countries."""
    fig, axes = plt.subplots(1, len(isos), figsize=(15, 3))
    if len(isos) == 1: axes = [axes]
    for i, iso in enumerate(isos):
        ax = axes[i]
        filtered_df = df[(df['ISO'] == iso) & (df['year'] >= 1990) & (df['year'] <= 2020)]
        if len(exposure_cols) == 1:
            ax.plot(filtered_df['year'], filtered_df[exposure_cols[0]], 'o-', color=colors[0], linewidth=2, markersize=4)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        else:
            bottom = np.zeros(len(filtered_df))
            for col, color in zip(exposure_cols, colors):
                values = filtered_df[col].values * 100
                ax.bar(filtered_df['year'], values, bottom=bottom, color=color, label=col.replace('_v2', '').upper(), alpha=0.8)
                bottom += values
            ax.legend(fontsize=8)
        ax.set_title(iso, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        if i == 0: ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, axes


def _draw_panel(ax, stats, title, xlabel, line_c, mk_c,
                differentiate_extra=True, extratropical_isos=(),
                line_c_extra=EXTRA_LIGHT, mk_c_extra=EXTRA_DARK,
                tick_color_extra=EXTRA_DARK):
    """Draw IQR + median markers, one row per country."""
    y = np.arange(len(stats))
    is_extra = stats['ISO'].isin(set(extratropical_isos)).values if differentiate_extra \
                 else np.zeros(len(stats), dtype=bool)
    for mask, lc, mc in [(~is_extra, line_c, mk_c),
                          ( is_extra, line_c_extra, mk_c_extra)]:
        if not mask.any():
            continue
        sub = stats[mask]
        ys  = y[mask]
        ax.hlines(ys, sub['q25'], sub['q75'], color=lc, linewidth=2.5, alpha=0.9)
        ax.plot(sub['median'], ys, 'o', color=mc, markersize=3.5)
        ax.plot(sub['q25'],    ys, '|', color=mc, markersize=5)
        ax.plot(sub['q75'],    ys, '|', color=mc, markersize=5)
    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(stats['ISO'], fontsize=6)
    if differentiate_extra:
        for tick_label, extra in zip(ax.get_yticklabels(), is_extra):
            if extra:
                tick_label.set_color(tick_color_extra)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.2)


def _draw_paired_panel(ax, df_tc, df_temp, is_extra_arr, xlabel, offset=0.18):
    """Draw TWO IQR+median entries per row (TC above, temp below). df_tc/df_temp
    must share the same ISO order. Rows where is_extra_arr is True use grey.
    Rows whose TC or temp median is NaN have that marker skipped."""
    isos = df_tc['ISO'].tolist()
    y = np.arange(len(isos))
    is_extra = np.asarray(is_extra_arr, dtype=bool)

    def _draw_one(df, y_offset, color_main, color_extra):
        for mask, color in [(~is_extra, color_main), (is_extra, color_extra)]:
            if not mask.any():
                continue
            sub = df[mask]
            valid = sub['median'].notna().values
            if not valid.any():
                continue
            sub_v = sub[valid]
            ys = y[mask][valid] + y_offset
            line_c, mk_c = color
            ax.hlines(ys, sub_v['q25'], sub_v['q75'], color=line_c,
                      linewidth=2.0, alpha=0.9)
            ax.plot(sub_v['median'], ys, 'o', color=mk_c, markersize=3.0)
            ax.plot(sub_v['q25'],    ys, '|', color=mk_c, markersize=4)
            ax.plot(sub_v['q75'],    ys, '|', color=mk_c, markersize=4)

    # TC: grey for extratropical (no exposure); colored for tropical TC>0.
    _draw_one(df_tc,  +offset, (TC_LIGHT,     TC_DARK),     (EXTRA_LIGHT,  EXTRA_DARK))
    # Temp: keep salmon for both tropical and extratropical.
    _draw_one(df_temp,-offset, (SALMON_LIGHT, SALMON_DARK), (SALMON_LIGHT, SALMON_DARK))

    ax.axvline(0.0, color='grey', linestyle=':', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(isos, fontsize=6)
    for tick_label, extra in zip(ax.get_yticklabels(), is_extra):
        if extra:
            tick_label.set_color(EXTRA_DARK)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.2)


def yearly_vectors_to_long(results_dict):
    tc_vector_rows = []
    for iso, yearly_vectors in results_dict.items():
        for year_str, values in yearly_vectors.items():
            year = int(year_str)
            if 1990 <= year <= 2019:
                tc_vector_rows.append({
                    'ISO': iso,
                    'year': year,
                    'vector': values[:vector_length].tolist()
                })
    return pd.DataFrame(tc_vector_rows).sort_values(['ISO', 'year']).reset_index(drop=True)


def calculate_debt_ratio_impact_vector(row, debt_lookup):
    observed_vector = debt_lookup.get(row['ISO'])
    if observed_vector is None:
        return [np.nan] * vector_length
    tc_rate_vector = np.array(row['vector'], dtype=float)
    debt_counterfactual_vector = observed_vector / (1 + tc_rate_vector)
    debt_ratio_impact_vector = observed_vector - debt_counterfactual_vector
    return debt_ratio_impact_vector.tolist()


def build_tc_vectors_merged(results_dict, exposure_df, debt_lookup):
    results_vectors = yearly_vectors_to_long(results_dict)
    merged = results_vectors.merge(exposure_df, on=['ISO', 'year'], how='left')
    merged = merged.drop(columns=['total', 'country', 'country2', 'Unnamed: 0'], errors='ignore')
    merged['debt_ratio_impact_vector'] = merged.apply(
        calculate_debt_ratio_impact_vector, axis=1, debt_lookup=debt_lookup)
    merged = merged[['ISO', 'year', 'vector'] + [
        col for col in merged.columns if col not in ['ISO', 'year', 'vector']
    ]]
    return merged


def prepare_alternative_tc_exposure(alt_tc_raw_df, country_master):
    alt_tc_df = alt_tc_raw_df
    alt_tc_df = alt_tc_df[(alt_tc_df['year'] <= 2019)].copy()
    country_iso_map = country_master[['country', 'ISO']].dropna().drop_duplicates()
    unmatched_countries = sorted(alt_tc_df.loc[~alt_tc_df['country'].isin(country_iso_map['country']), 'country'].drop_duplicates().tolist())
    alt_tc_df = alt_tc_df.merge(country_iso_map, on='country', how='left')
    alt_tc_df = alt_tc_df.dropna(subset=['ISO']).copy()
    alt_tc_df = alt_tc_df.rename(columns={
        'exposed_pp_total': 'totalexposed',
        'exposed_pp_ts': 'ts_v2',
        'exposed_pp_cat1': 'cat1_v2',
        'exposed_pp_cat3': 'cat3_v2'
    })
    return alt_tc_df, unmatched_countries


def sum_vectors(series):
    vectors = [np.array(v, dtype=float) for v in series if isinstance(v, list)]
    if not vectors:
        return [np.nan] * vector_length
    return np.sum(np.vstack(vectors), axis=0).tolist()


def calculate_cumulative_debt_ratio_impact(row, debt_lookup):
    observed_vector = debt_lookup.get(row['ISO'])
    if observed_vector is None:
        return [np.nan] * vector_length
    cumulative_tc_rate_vector = np.array(row['cumulative_tc_rate_vector'], dtype=float)
    debt_counterfactual_vector = observed_vector / (1 + cumulative_tc_rate_vector)
    return (observed_vector - debt_counterfactual_vector).tolist()


def build_alternative_counterfactual_paths(filtered_df, debt_lookup):
    path_frames = []
    for iso in sorted(filtered_df['ISO'].unique()):
        observed_vector = debt_lookup.get(iso)
        if observed_vector is None:
            continue

        iso_filtered = filtered_df[filtered_df['ISO'] == iso].sort_values('year')
        retained_vectors = {
            int(row['year']): np.array(row['vector'], dtype=float)
            for _, row in iso_filtered.iterrows()
        }

        country_df = pd.DataFrame({
            'ISO': iso,
            'year': analysis_years,
            'public_pp_gdd': observed_vector
        })

        for start_year in analysis_years:
            vectors_to_keep = [
                vector for year, vector in retained_vectors.items() if year >= start_year
            ]
            if vectors_to_keep:
                cumulative_tc_rate_vector = np.sum(np.vstack(vectors_to_keep), axis=0)
            else:
                cumulative_tc_rate_vector = np.zeros(vector_length, dtype=float)
            country_df[f'debt_{start_year}'] = observed_vector / (1 + cumulative_tc_rate_vector)

        path_frames.append(country_df)

    return pd.concat(path_frames, ignore_index=True)


def build_alternative_counterfactual(base_df, severity_col, top_n, debt_lookup):
    filtered = base_df.copy()
    rank_col = f'{severity_col}_rank'
    filtered[rank_col] = filtered.groupby('ISO')[severity_col].rank(method='first', ascending=False)
    filtered = filtered[filtered[rank_col] > top_n].copy()

    cumulative = (
        filtered.groupby('ISO', as_index=False)['vector']
        .agg(sum_vectors)
        .rename(columns={'vector': 'cumulative_tc_rate_vector'})
    )
    cumulative['cumulative_debt_ratio_impact_vector'] = cumulative.apply(
        calculate_cumulative_debt_ratio_impact, axis=1, debt_lookup=debt_lookup
    )
    paths = build_alternative_counterfactual_paths(filtered, debt_lookup)
    return filtered, cumulative, paths


def impact_for_base(alt_select_countries, observed_debt, results_df_onlyTC):
    cf = calculate_country_counterfactuals(alt_select_countries, observed_debt, results_df_onlyTC, 'public_pp_gdd')
    out = cf[cf['year'] == 2019][['ISO', 'public_pp_gdd', 'debt_1990']].copy()
    out['impact_2019'] = out['public_pp_gdd'] - out['debt_1990']
    return out


def impact_for_alt(scenario_name, alt_select_countries, observed_debt, alternative_results_df_onlyTC):
    cf = calculate_country_counterfactuals(alt_select_countries, observed_debt, alternative_results_df_onlyTC[scenario_name], 'public_pp_gdd')
    out = cf[cf['year'] == 2019][['ISO', 'public_pp_gdd', 'debt_1990']].copy()
    out['impact_2019'] = out['public_pp_gdd'] - out['debt_1990']
    return out


def impact_for_paths(paths_df, alt_select_countries, observed_debt):
    obs = observed_debt[(observed_debt['year'] == 2019) & (observed_debt['ISO'].isin(alt_select_countries))][['ISO', 'public_pp_gdd']]
    out = paths_df[paths_df['year'] == 2019][['ISO', 'debt_1990']].merge(obs, on='ISO', how='inner')
    out['impact_2019'] = out['public_pp_gdd'] - out['debt_1990']
    return out


def _make_two_row_plot(panels, title, bar_subset_order, subset_colors,
                        ylim_row2=(0, 100), save_path=None):
    """panels: list of (panel_name, country_subset_df). Plots 2 rows × len(panels) cols."""
    fig, axes = plt.subplots(2, len(panels),
                              figsize=(2.2 * len(panels) + 1, 6.5),
                              sharey='row')
    if len(panels) == 1:
        axes = axes.reshape(2, 1)
    scenarios = [str(s) for s in bar_subset_order]
    for col, (panel_name, country_sub) in enumerate(panels):
        n_countries = country_sub['ISO'].nunique()

        # ROW 1: hybrid rate (× 100 → % uplift over counterfactual)
        ax_rate = axes[0, col]
        rate_data = [country_sub.loc[country_sub['scenario'] == s, 'plot_rate']
                       .dropna().values * 100
                     for s in scenarios]
        bp_rate = ax_rate.boxplot(rate_data, patch_artist=True, showfliers=True,
                                   medianprops={'color': 'black', 'linewidth': 1.2},
                                   flierprops={'marker': '.', 'markersize': 4,
                                                'markerfacecolor': 'gray',
                                                'markeredgecolor': 'gray'},
                                   widths=0.6)
        for patch, color in zip(bp_rate['boxes'], subset_colors):
            patch.set_facecolor(color); patch.set_alpha(0.85); patch.set_edgecolor('white')
        ax_rate.set_xticks(range(1, len(scenarios) + 1))
        ax_rate.set_xticklabels(scenarios, fontsize=7)
        ax_rate.axhline(0, color='gray', linestyle='--', linewidth=0.6)
        ax_rate.set_title(f'{panel_name}\n(n={n_countries})',
                           fontsize=10, fontweight='bold', loc='left')
        ax_rate.grid(axis='y', alpha=0.2)
        ax_rate.spines[['top', 'right']].set_visible(False)
        if col == 0:
            ax_rate.set_ylabel('% of debt ratio\nattributable to TCs',
                                fontsize=9)

        # ROW 2: hybrid pct (skip 'all TCs' which is 100% by definition)
        ax_box = axes[1, col]
        pct_data = [country_sub.loc[country_sub['scenario'] == s, 'plot_pct']
                      .dropna().values
                    for s in scenarios[1:]]
        positions = list(range(2, len(scenarios) + 1))
        bp = ax_box.boxplot(pct_data, positions=positions, patch_artist=True,
                             showfliers=True,
                             medianprops={'color': 'black', 'linewidth': 1.2},
                             flierprops={'marker': '.', 'markersize': 4,
                                          'markerfacecolor': 'gray',
                                          'markeredgecolor': 'gray'},
                             widths=0.6)
        for patch, color in zip(bp['boxes'], subset_colors[1:]):
            patch.set_facecolor(color); patch.set_alpha(0.85); patch.set_edgecolor('white')
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(list(scenarios[1:]), fontsize=7)
        ax_box.set_xlim(0.5, len(scenarios) + 0.5)
        ax_box.axhline(0, color='gray', linestyle='--', linewidth=0.6)
        ax_box.set_ylim(*ylim_row2)
        ax_box.grid(axis='y', alpha=0.2)
        ax_box.spines[['top', 'right']].set_visible(False)
        if col == 0:
            ax_box.set_ylabel('% of total impact', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.suptitle(title, fontsize=13, fontweight='bold', x=0.02, y=0.99, ha='left')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_iso_yearly_vectors_avgws(df, beta, exposure_col="avg_ws",
                                       start_year=None, end_year=2022):
    """Single-variable analog of generate_iso_yearly_vectors_onlyTC: applies
    one coefficient vector beta[h] to the contemporaneous exposure column,
    rolling each storm-year shock forward up to len(beta) horizons.
    start_year defaults to PLOT_START_YEAR."""
    if start_year is None:
        start_year = PLOT_START_YEAR
    all_years = np.arange(start_year, end_year + 1, 1)
    vec_len = end_year - start_year
    iso_yearly_vectors = {}
    for iso in df["ISO"].unique():
        df_iso = df[df["ISO"] == iso]
        yearly_vectors = {}
        for yr in all_years:
            yearly_values = np.zeros(vec_len)
            max_i = min(len(beta), vec_len - (yr - start_year))
            for i in range(max_i):
                idx = yr - start_year + i
                if idx < 0:
                    continue
                yearly_values[idx] = np.sum(np.where(
                    df_iso["year"] == yr,
                    df_iso[exposure_col].fillna(0) * beta[i], 0), axis=0)
            yearly_vectors[str(yr)] = yearly_values
        iso_yearly_vectors[str(iso)] = yearly_vectors
    return iso_yearly_vectors


