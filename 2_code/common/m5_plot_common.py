# =============================================================================
# M5 BASIS-POINT PLOT — SHARED HELPERS
# =============================================================================
# Drawing helpers for the M5 basis-point panels (2_code/main/figures/M5_basispoint.py).
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Polygon


# -----------------------------------------------------------------------------
# Temperature colour palette / bins
# -----------------------------------------------------------------------------
CASSATT1 = ["#b1615c", "#d88782", "#e3aba7", "#edd7d9", "#c9c9dd", "#9d9dc7", "#8282aa", "#5a5a83"]
BIN_COLORS = {
    '<5': CASSATT1[7], '5-10': CASSATT1[6], '10-15': CASSATT1[5],
    '15-20': CASSATT1[2], '20-25': CASSATT1[1], '>25': CASSATT1[0],
}
TEMP_BINS = [-np.inf, 5, 10, 15, 20, 25, np.inf]
TEMP_BIN_LABELS = ['<5', '5-10', '10-15', '15-20', '20-25', '>25']
COUNTRY_LABEL_COLOR = '#57595B'


def convert_to_array(item):
    """Convert various input types to numpy array."""
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, str):
        return np.fromstring(item.strip('[]'), sep=' ')
    else:
        return np.array(item)


def prepare_percentiles(df, variable, regions):
    """Compute per-ISO 25/50/75 percentiles and assign region/temp-colour."""
    pct = df.groupby('ISO')[variable].quantile([0.25, 0.5, 0.75]).unstack()
    pct.columns = [f'{variable}_25', f'{variable}_50', f'{variable}_75']
    pct = pct.reset_index()
    pct = pct.merge(df.groupby('ISO')['meantemp_hist'].mean().reset_index(), on='ISO')
    pct = pct.merge(df.groupby('ISO')['country'].first().reset_index(), on='ISO')
    pct['region'] = pct['ISO'].apply(
        lambda iso: next((r for r, isos in regions.items() if iso in isos), None))
    pct = pct[pct['region'].notna()]
    pct['temp_bin'] = pd.cut(pct['meantemp_hist'], bins=TEMP_BINS, labels=TEMP_BIN_LABELS, right=False)
    pct['color'] = pct['temp_bin'].map(BIN_COLORS)
    return pct


def compute_layout(region_order, *dfs, hp, region_spacing=0.5):
    """Compute y-positions and row heights for a region layout.
    `hp` maps region -> height per data point."""
    max_counts = {}
    for region in region_order:
        counts = [len(df[df['region'] == region]) for df in dfs]
        max_counts[region] = max(counts)

    row_heights = {}
    y_positions = {}
    current_y = 0

    for region in region_order:
        row_h = max(0.3, max_counts[region] * hp(region))
        row_heights[region] = row_h
        y_positions[region] = current_y + row_h / 2
        row_heights[(region, 'bounds')] = (current_y, current_y + row_h)
        current_y += row_h + region_spacing

    total_height = current_y - region_spacing
    return y_positions, row_heights, total_height


def style_axis(ax, total_height, axis_style=None):
    """Apply the basis-point x-axis styling. `axis_style` (dict) can override
    'xscale', 'xscale_kwargs', 'xticks', 'xlim', 'xlabel', 'gridlines'."""
    if axis_style is None:
        axis_style = {}
    xscale = axis_style.get('xscale', 'symlog')
    xscale_kwargs = axis_style.get('xscale_kwargs', dict(linthresh=3, linscale=0.5))
    xticks = axis_style.get('xticks', [-10, -5, -1, 0, 1, 5, 10, 50])
    xlim = axis_style.get('xlim', (-14, 55))
    xlabel = axis_style.get('xlabel', 'Risk premium estimate \n (basis points)')
    gridlines = axis_style.get('gridlines', [10, 5, 1, -1, -5])

    tick_fontsize = axis_style.get('tick_fontsize', 8)
    label_fontsize = axis_style.get('label_fontsize', 8)
    if xscale == 'symlog':
        ax.set_xscale('symlog', **xscale_kwargs)
    else:
        ax.set_xscale(xscale)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=tick_fontsize, fontweight='bold')
    for xval in gridlines:
        ax.axvline(xval, color='grey', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
    ax.axvline(0, color='grey', linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.2, total_height + 0.3)
    ax.spines['top'].set_visible(False)


def add_temp_colorbar(fig, bin_colors, bin_labels,
                      position=(0.3, 0.05, 0.4, 0.025),
                      title_fontsize=10, tick_fontsize=7):
    """Add temperature colorbar legend. Default placement is the bottom-center
    of the figure; pass `position=(x, y, w, h)` (figure-fraction units) to
    move/resize, e.g. upper-right small variant: (0.74, 0.93, 0.20, 0.018)."""
    legend_ax = fig.add_axes(position)
    arrow_width = 0.3
    legend_ax.set_xlim(-arrow_width - 0.2, len(bin_labels) + arrow_width + 0.2)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')

    left_pts = np.array([[0, 0.2], [0, 0.6], [-arrow_width, 0.4]])
    legend_ax.add_patch(Polygon(left_pts, facecolor=bin_colors['<5'], edgecolor='none'))

    for i, label in enumerate(bin_labels):
        legend_ax.add_patch(Rectangle((i, 0.2), 1, 0.4, facecolor=bin_colors[label], edgecolor='none'))

    right_pts = np.array([[len(bin_labels), 0.2], [len(bin_labels), 0.6],
                           [len(bin_labels) + arrow_width, 0.4]])
    legend_ax.add_patch(Polygon(right_pts, facecolor=bin_colors['>25'], edgecolor='none'))

    for i in range(1, len(bin_labels)):
        legend_ax.plot([i, i], [0.15, 0.65], 'k-', linewidth=1.5)
    for pos, lbl in zip(range(1, len(bin_labels)), ['5', '10', '15', '20', '25']):
        legend_ax.text(pos, 0.0, f'{lbl}°C', ha='center', va='top', fontsize=tick_fontsize)
    legend_ax.text(len(bin_labels) / 2, 0.85, 'Mean Historical Temperature',
                   ha='center', va='bottom', fontsize=title_fontsize)


def add_region_hlines(ax, region_order, row_heights, region_spacing=0.5):
    """Add horizontal separators between regions."""
    for region in region_order[:-1]:
        _, rt = row_heights[(region, 'bounds')]
        ax.axhline(rt + region_spacing / 2, color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.7)
