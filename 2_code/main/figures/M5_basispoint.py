# =============================================================================
# M5 BASIS POINT PLOTS 
# =============================================================================
# Panel-drawing functions (plot_two_panel, plot_single_panel, plot_three_panel)
# used by M5_basispoint_plot.py, which writes Fig. 5 and SI Figs. S31 and S33.
# =============================================================================

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


class _HandlerStackedMarkers(HandlerBase):
    """Legend handle that stacks several Line2D markers vertically (top marker first)."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                        width, height, fontsize, trans):
        x = width / 2.0 - xdescent
        n = len(orig_handle)
        if n == 2:
            ys = [0.85 * height - ydescent, 0.15 * height - ydescent]
        else:
            ys = [((n - 0.5 - i) / n) * height - ydescent for i in range(n)]
        artists = []
        for m, y in zip(orig_handle, ys):
            a = Line2D([x], [y],
                       marker=m.get_marker(),
                       markersize=m.get_markersize(),
                       markerfacecolor=m.get_markerfacecolor(),
                       markeredgecolor=m.get_markeredgecolor(),
                       markeredgewidth=m.get_markeredgewidth(),
                       linestyle='None',
                       transform=trans)
            artists.append(a)
        return artists


def _add_marker_legend(fig, include_no_bond=False, color='gray',
                        bbox_to_anchor=(0.96, 0.005), loc='lower right',
                        ncol=None):
    """Marker legend: TC exposed (circle), not TC exposed (square), optionally no bond
    issuance (open). Bottom-right row by default; pass loc/bbox_to_anchor/ncol=1 to stack."""
    handles = [
        Line2D([0], [0], marker='o', color='w', markersize=6,
               markerfacecolor=color, markeredgecolor='white',
               markeredgewidth=0.4, label='TC exposed'),
        Line2D([0], [0], marker='s', color='w', markersize=6,
               markerfacecolor=color, markeredgecolor='white',
               markeredgewidth=0.4, label='Not TC exposed'),
    ]
    handler_map = None
    labels = [h.get_label() for h in handles]
    if include_no_bond:
        # Open circle + open square stacked in one legend handle.
        stacked_ms = 4
        open_circle = Line2D([0], [0], marker='o', color='w', markersize=stacked_ms,
                              markerfacecolor='none', markeredgecolor=color,
                              markeredgewidth=0.6, linestyle='None')
        open_square = Line2D([0], [0], marker='s', color='w', markersize=stacked_ms,
                              markerfacecolor='none', markeredgecolor=color,
                              markeredgewidth=0.6, linestyle='None')
        handles.append((open_circle, open_square))
        labels.append('No sovereign bond issuance \non global market')
        handler_map = {tuple: _HandlerStackedMarkers()}
    if ncol is None:
        ncol = len(handles)
    fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor,
               fontsize=6, frameon=False,
               handletextpad=0.4, labelspacing=0.8, ncol=ncol,
               columnspacing=1.5,
               handleheight=1.6,                # taller box so two markers fit
               handler_map=handler_map)

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "2_code", "common"))

# Shared M5 drawing helpers
from m5_plot_common import (
    prepare_percentiles as _prepare_percentiles,
    compute_layout,
    style_axis as _style_axis,
    add_temp_colorbar as _add_temp_colorbar,
    add_region_hlines as _add_region_hlines,
    BIN_COLORS, TEMP_BIN_LABELS, COUNTRY_LABEL_COLOR,
)




# =============================================================================
# PLOTTING
# =============================================================================

REGION_ORDER = ['Oceania', 'Europe', 'Asia', 'Americas', 'Africa', 'Small Island States']


DEFAULT_HEIGHT_PER_POINT = 0.43
# Per-region row-height overrides (keep ISO labels readable).
HEIGHT_PER_POINT_BY_REGION = {
    'Europe':              0.42,
    'Africa':              0.45,
    'Asia':                0.50,
    'Small Island States': 0.50,
}


def _hp(region):
    return HEIGHT_PER_POINT_BY_REGION.get(region, DEFAULT_HEIGHT_PER_POINT)


def _compute_layout(region_order, *dfs):
    """Y-positions and row heights with M5's per-region height map `_hp`."""
    return compute_layout(region_order, *dfs, hp=_hp)


def _plot_panel(ax, data, variable, y_positions, row_heights, region_order,
                 label_threshold='none', never_exposed=None,
                 never_issued_bond=None, extratropical=None,
                 marker_size_scale=1.0):
    """Plot data points with error bars for one panel. Returns country positions.

    never_exposed: ISOs never TC-exposed (square marker; effect is from temperature only).
    never_issued_bond: ISOs with no bond issuance in the sample (open marker).
    """
    never_exposed     = set(never_exposed)     if never_exposed     else set()
    never_issued_bond = set(never_issued_bond) if never_issued_bond else set()
    extratropical     = set(extratropical)     if extratropical     else set()

    country_positions = []
    for region in region_order:
        rdata = data[data['region'] == region]
        if len(rdata) == 0:
            continue
        y_base = y_positions[region]
        n = len(rdata)
        rb, rt = row_heights[(region, 'bounds')]
        rh = rt - rb
        # Cap the jitter span so sparse panels are not spread across a tall row.
        height_per_point = _hp(region)
        natural_span = (n - 1) * height_per_point
        max_span     = max(rh - 0.1, 0.0)
        span         = min(natural_span, max_span)
        jitter = np.linspace(-span / 2, span / 2, n) if n > 1 else [0]
        rdata_sorted = rdata.sort_values(f'{variable}_50')

        for i, (_, row) in enumerate(rdata_sorted.iterrows()):
            yp = y_base + jitter[i]
            ne  = row['ISO'] in never_exposed
            nib = row['ISO'] in never_issued_bond
            ext = row['ISO'] in extratropical
            mk = 's' if ne else 'o'
            mk_size_eb = (4.5 if ne else 6) * marker_size_scale
            mk_size_sc = (22  if ne else 35) * marker_size_scale
            # Marker face:
            #   - open if no observed bond
            #   - grey for extratropical TC-exposed COUNTRIES (circles only)
            #   - else region color
            if nib:
                face = 'none'
            elif ext and not ne:
                face = '#7F7F7F'           # grey median for extratropical circles
            else:
                face = row['color']
            # Marker edge: row color for open markers, white for filled markers.
            edge      = row['color'] if nib else 'white'
            edge_w_eb = 0.7 if nib else 0.25
            edge_w_sc = 0.7 if nib else 0.4
            ax.errorbar(row[f'{variable}_50'], yp,
                        xerr=[[row[f'{variable}_50'] - row[f'{variable}_25']],
                              [row[f'{variable}_75'] - row[f'{variable}_50']]],
                        fmt=mk, color=row['color'], capsize=3, markersize=mk_size_eb,
                        linewidth=1.4, alpha=0.7, zorder=3,
                        markerfacecolor=face,
                        markeredgecolor=edge, markeredgewidth=edge_w_eb,
                        ecolor=row['color'])
            ax.scatter(row[f'{variable}_50'], yp, s=mk_size_sc, alpha=0.9,
                       zorder=4, marker=mk,
                       facecolors=face, edgecolors=edge, linewidth=edge_w_sc)
            if label_threshold == 'none' or abs(row[f'{variable}_50']) > label_threshold:
                country_positions.append((yp, row['country']))
    return country_positions


def plot_two_panel(df_left, df_right, variable, regions, figsize=(6, 6.5),
                   left_title='Investment-grade', right_title='Speculative-grade',
                   title='', cbar_show=True, label_threshold='none',
                   iso_label_fontsize=4.5, region_label_fontsize=8,
                   never_exposed=None, never_issued_bond=None, extratropical=None, axis_style=None,
                   legend_no_bond=False):
    """Two-panel basis point plot.

    legend_no_bond: add the "no sovereign bond issuance" (open marker) legend entry.
    never_exposed: ISOs never TC-exposed, drawn as squares.
    """
    pct_left = _prepare_percentiles(df_left, variable, regions)
    pct_right = _prepare_percentiles(df_right, variable, regions)
    y_pos, rh, total_h = _compute_layout(REGION_ORDER, pct_left, pct_right)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.subplots_adjust(wspace=0.15)

    cp_l = _plot_panel(ax_l, pct_left, variable, y_pos, rh, REGION_ORDER,
                       label_threshold, never_exposed=never_exposed,
                       never_issued_bond=never_issued_bond, extratropical=extratropical)
    cp_r = _plot_panel(ax_r, pct_right, variable, y_pos, rh, REGION_ORDER,
                       label_threshold, never_exposed=never_exposed,
                       never_issued_bond=never_issued_bond, extratropical=extratropical)

    for ax in [ax_l, ax_r]:
        _style_axis(ax, total_h, axis_style=axis_style)
        _add_region_hlines(ax, REGION_ORDER, rh)
    ax_l.set_title(left_title, fontsize=9, fontweight='bold', pad=10)
    ax_r.set_title(right_title, fontsize=9, fontweight='bold', pad=10)
    ax_l.spines['right'].set_visible(False)
    ax_r.spines['left'].set_visible(False)
    ax_l.set_yticks([]); ax_r.set_yticks([])

    for yp, country in cp_l:
        ax_l.annotate(country, xy=(-0.02, yp), xycoords=('axes fraction', 'data'),
                      fontsize=iso_label_fontsize, color=COUNTRY_LABEL_COLOR, ha='right', va='center')
    for yp, country in cp_r:
        ax_r.annotate(country, xy=(1.02, yp), xycoords=('axes fraction', 'data'),
                      fontsize=iso_label_fontsize, color=COUNTRY_LABEL_COLOR, ha='left', va='center')

    if cbar_show:
        fig.subplots_adjust(bottom=0.18, left=0.08, right=0.92)
    else:
        fig.subplots_adjust(bottom=0.08, left=0.08, right=0.92)
    fig.canvas.draw()

    ax_l_right = ax_l.get_position().x1
    ax_r_left = ax_r.get_position().x0
    center_x = (ax_l_right + ax_r_left) / 2
    for region in REGION_ORDER:
        yp = y_pos[region]
        y_min, y_max = ax_l.get_ylim()
        y_frac = (yp - y_min) / (y_max - y_min)
        bbox = ax_l.get_position()
        y_fig = bbox.y0 + y_frac * bbox.height
        fig.text(center_x, y_fig, region, ha='center', va='center',
                 fontsize=region_label_fontsize, fontweight='bold', color='#333333')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    if cbar_show:
        _add_temp_colorbar(fig, BIN_COLORS, TEMP_BIN_LABELS)

    _add_marker_legend(fig, include_no_bond=legend_no_bond)

    return fig


def plot_single_panel(df, variable, regions, figsize=(2.5, 5),
                      title='Non-rated countries', cbar_show=False, label_threshold='none',
                      iso_label_fontsize=4.5, region_label_fontsize=8, title_fontsize=9,
                      never_exposed=None, never_issued_bond=None, extratropical=None,
                      axis_style=None, marker_legend=False, region_labels=True):
    """Single-panel basis point plot for non-rated countries (square = never TC-exposed,
    open = no bond issuance). region_labels=False drops the rotated region names."""
    region_order_nr = ['Asia', 'Africa', 'Small Island States']
    pct = _prepare_percentiles(df, variable, regions)
    y_pos, rh, total_h = _compute_layout(region_order_nr, pct)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cp = _plot_panel(ax, pct, variable, y_pos, rh, region_order_nr,
                     label_threshold, never_exposed=never_exposed,
                     never_issued_bond=never_issued_bond, extratropical=extratropical)
    _style_axis(ax, total_h, axis_style=axis_style)
    _add_region_hlines(ax, region_order_nr, rh)

    ax.spines['right'].set_visible(False)
    # Shift region labels toward the top of each row for better alignment with data
    label_y = []
    for r in region_order_nr:
        rb, rt = rh[(r, 'bounds')]
        label_y.append(rt - (rt - rb) * 0.15)  # 15% down from top of row
    if region_labels:
        ax.set_yticks(label_y)
        ax.set_yticklabels(region_order_nr, fontsize=region_label_fontsize, fontweight='bold',
                           rotation=90, color='#333333')
    else:
        ax.set_yticks([])
    ax.tick_params(axis='y', length=0)

    for yp, country in cp:
        ax.annotate(country, xy=(1.02, yp), xycoords=('axes fraction', 'data'),
                    fontsize=iso_label_fontsize, color=COUNTRY_LABEL_COLOR, ha='left', va='center')

    if cbar_show:
        fig.subplots_adjust(bottom=0.18, left=0.15, right=0.85)
    else:
        fig.subplots_adjust(bottom=0.08, left=0.15, right=0.85)

    # Title sits directly above the panel (no large fig.suptitle gap).
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=6)
    if cbar_show:
        _add_temp_colorbar(fig, BIN_COLORS, TEMP_BIN_LABELS)
    if marker_legend:
        _add_marker_legend(fig, include_no_bond=True, ncol=1,
                           bbox_to_anchor=(1.02, 1.0), loc='upper left')

    return fig


def plot_three_panel(df_left, df_middle, df_right, variable, regions,
                      figsize=(8, 6.5),
                      left_title='Investment-grade',
                      middle_title='Speculative-grade',
                      right_title='Non-rated countries',
                      title='', cbar_show=True, label_threshold='none',
                      iso_label_fontsize=4.5, region_label_fontsize=8,
                      never_exposed=None, never_issued_bond=None, extratropical=None,
                      axis_style=None, axis_style_right=None,
                      axis_style_left=None):
    """Three-panel layout: investment-grade | speculative-grade | non-rated.

    IG and SG share region rows; NR has its own (Asia, Africa, SIDS). Square = never
    TC-exposed, open = no bond issuance.
    """
    pct_left   = _prepare_percentiles(df_left,   variable, regions)
    pct_middle = _prepare_percentiles(df_middle, variable, regions)
    pct_right  = _prepare_percentiles(df_right,  variable, regions)
    y_pos_lm, rh_lm, total_h_lm = _compute_layout(REGION_ORDER, pct_left, pct_middle)

    region_order_nr = ['Asia', 'Africa', 'Small Island States']
    y_pos_r, rh_r, total_h_r = _compute_layout(region_order_nr, pct_right)

    fig = plt.figure(figsize=figsize)
    # Columns: IG | spacer | SG | wider spacer | NR (narrower).
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.10, 1, 0.10, 0.9], wspace=0.05)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 2], sharey=ax_l)
    ax_r = fig.add_subplot(gs[0, 4])

    cp_l = _plot_panel(ax_l, pct_left,   variable, y_pos_lm, rh_lm, REGION_ORDER,
                        label_threshold, never_exposed=never_exposed,
                        never_issued_bond=never_issued_bond, extratropical=extratropical)
    cp_m = _plot_panel(ax_m, pct_middle, variable, y_pos_lm, rh_lm, REGION_ORDER,
                        label_threshold, never_exposed=never_exposed,
                        never_issued_bond=never_issued_bond, extratropical=extratropical)
    cp_r = _plot_panel(ax_r, pct_right,  variable, y_pos_r,  rh_r,  region_order_nr,
                        label_threshold, never_exposed=never_exposed,
                        never_issued_bond=never_issued_bond, extratropical=extratropical,
                        marker_size_scale=0.75)

    _style_axis(ax_l, total_h_lm, axis_style=axis_style_left or axis_style)
    _style_axis(ax_m, total_h_lm, axis_style=axis_style)
    for ax in (ax_l, ax_m):
        _add_region_hlines(ax, REGION_ORDER, rh_lm)
    _style_axis(ax_r, total_h_r, axis_style=axis_style_right or axis_style)
    _add_region_hlines(ax_r, region_order_nr, rh_r)

    ax_l.set_title(left_title,   fontsize=9, fontweight='bold', pad=10)
    ax_m.set_title(middle_title, fontsize=9, fontweight='bold', pad=10)
    ax_r.set_title(right_title,  fontsize=9, fontweight='bold', pad=10)
    for ax in (ax_l, ax_m, ax_r):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax_l.set_yticks([]); ax_m.set_yticks([]); ax_r.set_yticks([])
    ax_l.tick_params(axis='y', length=0)
    ax_r.tick_params(axis='y', length=0)

    # Country labels: A on left of A, B on right of B, NR on right of NR.
    for yp, country in cp_l:
        ax_l.annotate(country, xy=(-0.02, yp), xycoords=('axes fraction', 'data'),
                      fontsize=iso_label_fontsize, color=COUNTRY_LABEL_COLOR,
                      ha='right', va='center')
    for yp, country in cp_m:
        ax_m.annotate(country, xy=(1.02, yp), xycoords=('axes fraction', 'data'),
                      fontsize=iso_label_fontsize, color=COUNTRY_LABEL_COLOR,
                      ha='left', va='center')
    for yp, country in cp_r:
        ax_r.annotate(country, xy=(1.02, yp), xycoords=('axes fraction', 'data'),
                      fontsize=iso_label_fontsize, color=COUNTRY_LABEL_COLOR,
                      ha='left', va='center')

    # NR panel: no region tick labels.
    ax_r.set_yticks([])
    ax_r.tick_params(axis='y', length=0)

    if cbar_show:
        fig.subplots_adjust(bottom=0.18, left=0.06, right=0.96)
    else:
        fig.subplots_adjust(bottom=0.08, left=0.06, right=0.96)
    fig.canvas.draw()

    # IG/SG region labels sit in the gap between the two panels.
    ax_l_right = ax_l.get_position().x1
    ax_m_left  = ax_m.get_position().x0
    center_x = (ax_l_right + ax_m_left) / 2
    for region in REGION_ORDER:
        # SIDS label sits low in its row to clear Africa above.
        if region == 'Small Island States':
            rb_, rt_ = rh_lm[(region, 'bounds')]
            yp = rb_ + (rt_ - rb_) * 0.20      # 20% above the bottom
        else:
            yp = y_pos_lm[region]
        y_min, y_max = ax_l.get_ylim()
        y_frac = (yp - y_min) / (y_max - y_min)
        bbox = ax_l.get_position()
        y_fig = bbox.y0 + y_frac * bbox.height
        fig.text(center_x, y_fig, region, ha='center', va='center',
                 fontsize=region_label_fontsize, fontweight='bold', color='#333333')

    # NR panel: shorter, bottom-aligned, shifted toward SG, smaller markers.
    _pos_m = ax_m.get_position()
    _pos_r = ax_r.get_position()
    _new_h = _pos_r.height * 0.7
    _new_x = _pos_m.x1 + 0.12      # ~12% figure-width gap from SG to NR
    ax_r.set_position([_new_x,
                       _pos_r.y0,
                       _pos_r.width,
                       _new_h])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    if cbar_show:
        _add_temp_colorbar(fig, BIN_COLORS, TEMP_BIN_LABELS)

    # Marker legend stays at bottom-right but labels stack vertically (ncol=1).
    _add_marker_legend(fig, include_no_bond=True, ncol=1)

    return fig


# =============================================================================
# MAIN
# =============================================================================
