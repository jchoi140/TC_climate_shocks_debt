"""
lp_plots.py — local-projection IRF plotting.

Each function takes a coefficient table (see coef_loader.py) and returns a figure.

    plot_TC_debt                  Fig. 2  panel a
    plot_TC_gdp                   Fig. 2  panel c
    plot_temp_irfs                Fig. S7 top panel
    sensitivity_irfs              Fig. S5 rows 2-6
    sensitivity_irfs_SEs          Fig. S5 row 1
    sensitivity_temp_irfs         Fig. S6 rows 1-2, Fig. S7 bottom row
    sensitivity_temp_irfs_subset  Fig. S6 row 3
    plot_irf_avg_ws               Fig. S20 panel a
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

__all__ = ['plot_TC_debt', 'plot_TC_gdp', 'plot_temp_irfs', 'sensitivity_irfs',
           'sensitivity_irfs_SEs', 'sensitivity_temp_irfs',
           'sensitivity_temp_irfs_subset', 'plot_irf_avg_ws']


def plot_TC_debt(df, outcome, specification, subset='full', model='linear',
                 tshock_type='NATH', titles=None, palette=None,
                 figsize=(20, 2), wspace=0.3,
                 ylabel='Δ debt ratio / \n % area exposed',
                 xticks=[0, 5, 10, 15], yticks=[0, 0.005],
                 ytick_labels=['0%', '0.5%'], ylim=(-0.001, 0.01),
                 linewidth=2.5, title_fontsize=14, title_fontweight='bold',
                 ylabel_fontsize=13, tick_fontsize=14):
    """
    Plot tropical cyclone impulse response functions for debt outcomes.

    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    
    terms = ['ts_v2', 'cat1_v2', 'cat3_v2']  

    if titles is None:
        titles = ['Tropical storm (18-33 m/s)', 'Category 1-3 (33-50 m/s)', 'Category >3 (>50 m/s)']
    
    if palette is None:
        palette = ["#B5A8D5", "#7A73D1", "#4D55CC"][:len(terms)]
    
    color_dict = dict(zip(titles, palette))
    
    fig, axes = plt.subplots(1, len(terms), figsize=figsize, gridspec_kw={'wspace': wspace})
    
    if len(terms) == 1:
        axes = [axes]
    
    for idx, (ax, term, title) in enumerate(zip(axes, terms, titles)):
        data = df[
            (df['term'] == term) & 
            (df['subset'] == subset) & 
            (df['outcome'] == outcome) & 
            (df['specification'] == specification) & 
            (df['model'] == model) &
            (df['Tshock_type'] == tshock_type)
        ].sort_values('y_h')
        
        if len(data) > 0:
            sns.lineplot(
                data=data,
                x='y_h',
                y='estimate',
                linewidth=linewidth,
                ax=ax,
                color=color_dict[title]
            )
            
            ax.fill_between(
                data['y_h'],
                data['conf.low'],
                data['conf.high'],
                color=color_dict[title],
                alpha=0.2
            )
        
        ax.axhline(0, color='grey', alpha=0.5)
        
        ax.set_xlabel('')
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=tick_fontsize)
        
        ax.set_ylabel(ylabel if idx == 0 else '', fontsize=ylabel_fontsize)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if yticks is not None:
            ax.set_yticks(yticks)
        if ytick_labels is not None:
            ax.set_yticklabels(ytick_labels, fontsize=tick_fontsize)
        
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def plot_TC_gdp(df, subset, outcome, specification, model,
              tshock_type='NATH', titles=None, palette=None,
              yticks=[0, -0.001, -0.002],
              ytick_labels=['0%', '-0.1%', '-0.2%'], 
              ylim=(-0.002, 0.0005),
              figsize=(20, 2), sharey=True, wspace=0.3,
              ylabel='Δ GDP pcap / \n % area exposed',
              pct_format=True, linewidth=2.5,
              title_fontsize=14, title_fontweight='bold',
              ylabel_fontsize=13, tick_fontsize=14,
              remove_box=True):
    """
    Plot impulse response functions for multiple terms.
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    terms = ['ts_v2', 'cat1_v2', 'cat3_v2']  

    
    if titles is None:
        titles = ['Tropical storm (18-33 m/s)', 'Category 1-3 (33-50 m/s)', 'Category >3 (>50 m/s)']
    
    if palette is None:
        palette = ["#B3D9F2", "#4682B4", "#1C4E80"][:len(terms)]
    
    color_dict = dict(zip(titles, palette))
    
    fig, axes = plt.subplots(1, len(terms), figsize=figsize, sharey=sharey,
                             gridspec_kw={'wspace': wspace})
    
    if len(terms) == 1:
        axes = [axes]
    
    for idx, (ax, term, title) in enumerate(zip(axes, terms, titles)):
        data = df[
            (df['term'] == term) & 
            (df['subset'] == subset) & 
            (df['outcome'] == outcome) & 
            (df['specification'] == specification) & 
            (df['model'] == model) &
            (df['Tshock_type'] == tshock_type)
        ].sort_values('y_h')
        
        if len(data) > 0:
            sns.lineplot(
                data=data,
                x='y_h',
                y='estimate',
                linewidth=linewidth,
                ax=ax,
                color=color_dict[title]
            )
            
            ax.fill_between(
                data['y_h'],
                data['conf.low'],
                data['conf.high'],
                color=color_dict[title],
                alpha=0.2
            )
        
        ax.axhline(0, color='grey', alpha=0.5)
        
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
        
        xticks=[0, 5, 10, 15]
        yticks=None
        ylim=None

        ax.set_xlabel('')
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=tick_fontsize)
        
        ax.set_ylabel(ylabel if idx == 0 else '', fontsize=ylabel_fontsize)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if yticks is not None:
            ax.set_yticks(yticks)
        
        ax.tick_params(axis='y', labelsize=tick_fontsize, labelleft=True)


        if pct_format:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        if remove_box:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def plot_temp_irfs(df, outcome, specification, subset='full', model='linear',
                   tshock_type='NATH', 
                   figsize=(5, 2), title="Temperature shock",
                   ylabel='Δ debt ratio /\n 1C shock',
                   pct_format=True, legend_loc='upper right'):
    """
    Plot impulse response functions for temperature shocks at different mean temperatures.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    meantemps=[5, 15, 25]
    palette=['#FFB200', '#EB5B00', '#D91656']

    for meantemp, color in zip(meantemps, palette):
        data = df[
            (df['meantemp'] == meantemp) &
            (df['subset'] == subset) &
            (df['model'] == model) &
            (df['outcome'] == outcome) &
            (df['Tshock_type'] == tshock_type) &
            (df['specification'] == specification)
        ].sort_values('y_h')
        
        if len(data) > 0:
            sns.lineplot(
                data=data,
                x='y_h',
                y='est',
                ax=ax,
                label=f"{meantemp}°C",
                color=color
            )
            
            ax.fill_between(
                data['y_h'],
                data['conf.low'],
                data['conf.high'],
                alpha=0.12,
                color=color
            )
    
    ax.axhline(0, color='grey', linestyle='--', alpha=0.7)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    xticks=[0, 5, 10, 15]
    yticks=[-0.025, 0, 0.025]
    
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=10)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    
    if pct_format:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    
    ax.legend(fontsize=8, loc=legend_loc)
    
    plt.tight_layout()
    return fig, ax


def sensitivity_irfs(df, 
                     outcome='public_pp_GDD',
                     terms=['ts_v2', 'cat1_v2', 'cat3_v2'],
                     subset='full',
                     time_trend='linear',
                     tshock_type='NATH',
                     specification='1970onwards_gl2_TC10_T02',
                     titles=None,
                     labels=None,
                     colors=None,
                     linestyles=None,
                     ci_index=0,
                     figsize=(16, 2),
                     legend_width_ratio=0.25,
                     sharey=True,
                     suptitle="",
                     ylabel='Estimate',
                     ylim=None,
                     yticks=None,
                     y_padding=0.1,
                     pct_format=False,
                     ci_alpha=0.2,
                     legend_title=None,
                     legend_fontsize=10,
                     title_fontsize=10,
                     ylabel_fontsize=10,
                     xlabel='Years since impact',
                     wspace=0.1):
    """
    Plot impulse response functions for multiple terms with flexible overlay parameter.
    
    Automatically detects which parameter has multiple values (list) and uses that
    for the overlay lines. Only one of subset, time_trend, tshock_type, specification,
    or outcome should be a list.
    """
    
    # --- Detect which parameter has multiple values ---
    params = {
        'subset': subset,
        'time_trend': time_trend,
        'tshock_type': tshock_type,
        'specification': specification,
        'outcome': outcome
    }
    
    col_map = {
        'subset': 'subset',
        'time_trend': 'model',
        'tshock_type': 'Tshock_type',
        'specification': 'specification',
        'outcome': 'outcome'
    }
    
    # Find which parameter is a list (the varying one)
    varying_param = None
    varying_values = None
    
    for param_name, param_value in params.items():
        if isinstance(param_value, list):
            if varying_param is not None:
                raise ValueError("Only one of subset, time_trend, tshock_type, specification, or outcome can be a list")
            varying_param = param_name
            varying_values = param_value
    
    # If none are lists, default to subset as single value
    if varying_param is None:
        varying_param = 'subset'
        varying_values = [subset]
    
    # Get fixed values for non-varying parameters
    fixed_params = {}
    for param_name, param_value in params.items():
        if param_name != varying_param:
            fixed_params[col_map[param_name]] = param_value
    
    if titles is None:
        titles = ['Tropical storm (18-33 m/s)', 'Category 1-3 (33-50 m/s)', 'Category >3 (>50 m/s)']
    
    if labels is None:
        labels = {val: str(val) for val in varying_values}
    
    if colors is None:
        default_color_list = ['steelblue', 'coral', 'seagreen', 'mediumpurple', 'goldenrod', 'crimson', 'teal']
        colors = {val: default_color_list[i % len(default_color_list)] 
                  for i, val in enumerate(varying_values)}
    
    if linestyles is None:
        linestyles = {val: '-' if i == 0 else '--' for i, val in enumerate(varying_values)}
    
    if isinstance(ci_index, int):
        ci_value = varying_values[ci_index] if ci_index < len(varying_values) else None
    else:
        ci_value = ci_index if ci_index in varying_values else None
    
    if legend_title is None:
        legend_title_map = {
            'subset': 'Subset',
            'time_trend': 'Time Trend',
            'tshock_type': 'Shock Type',
            'specification': 'Specification',
            'outcome': 'Outcome'
        }
        legend_title = legend_title_map.get(varying_param, varying_param)
    
    # --- Calculate ylim if not provided ---
    if ylim is None:
        all_data = []
        for term in terms:
            for val in varying_values:
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['term'] == term)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                data = df[mask]
                if len(data) > 0:
                    all_data.append(data)
        
        if len(all_data) > 0:
            combined_data = pd.concat(all_data)
            y_min = combined_data['conf.low'].min()
            y_max = combined_data['conf.high'].max()
            y_range = y_max - y_min
            y_min = y_min - y_padding * y_range
            y_max = y_max + y_padding * y_range
            ylim = (y_min, y_max)
    
    # --- Create figure with GridSpec ---
    n_terms = len(terms)
    width_ratios = [1] * n_terms + [legend_width_ratio]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_terms + 1, width_ratios=width_ratios, wspace=wspace)
    
    axes = []
    for i in range(n_terms):
        if i == 0 or not sharey:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)
    
    legend_ax = fig.add_subplot(gs[0, n_terms])
    legend_ax.axis('off')
    
    if suptitle:
        # Leave the top 28% of the figure for the suptitle and panel titles
        fig.subplots_adjust(top=0.72)
        fig.suptitle(suptitle, fontsize=14, y=0.99)
    
    # --- Plot data ---
    legend_handles = []
    legend_labels = []
    
    for idx, (ax, term, title) in enumerate(zip(axes, terms, titles)):
        has_data = False
        
        for val in varying_values:
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['term'] == term)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            
            data = df[mask].copy()
            
            if len(data) > 0:
                # Anchor h=-1 at zero (pre-shock baseline)
                data = data[data['y_h'] != -1]
                anchor = pd.DataFrame({
                    'y_h':       [-1],
                    'estimate':  [0.0],
                    'conf.low':  [0.0],
                    'conf.high': [0.0],
                })
                data = pd.concat([anchor, data], ignore_index=True)
                data = data.sort_values('y_h').reset_index(drop=True)
                has_data = True
                color = colors.get(val, 'steelblue')
                linestyle = linestyles.get(val, '-')
                label = labels.get(val, str(val))
                
                line, = ax.plot(
                    data['y_h'],
                    data['estimate'],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5 if val == varying_values[0] else 1,
                    label=label
                )
                
                # Collect handles/labels from first subplot only
                if idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)
                
                if val == ci_value:
                    ax.fill_between(
                        data['y_h'],
                        data['conf.low'],
                        data['conf.high'],
                        alpha=ci_alpha,
                        color=color
                    )
        
        ax.axhline(0, color='grey', linestyle='-', alpha=0.6)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False, left=False)
        
        if pct_format:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if yticks is not None:
            ax.set_yticks(yticks)
    
    if legend_handles:
        legend_ax.legend(
            legend_handles, 
            legend_labels,
            title=legend_title,
            loc='center left',
            fontsize=legend_fontsize,
            frameon=False
        )
    
    return fig, axes


def sensitivity_irfs_SEs(df, outcome,
                         terms=['ts_v2', 'cat1_v2', 'cat3_v2'],
                         subset='full',
                         time_trend='linear',
                         tshock_type='NATH',
                         specification='1970onwards_gl2_TC10_T02',
                         titles=None,
                         labels=None,
                         colors=None,
                         linestyles=None,
                         ci_index=0,
                         show_all_ci=False,
                         figsize=(15, 2),
                         legend_width_ratio=0.25,
                         sharey=True,
                         suptitle="",
                         ylabel='Estimate',
                         ylim=None,
                         yticks=None,
                         y_padding=0.1,
                         pct_format=False,
                         ci_alpha=0.15,
                         ci_linestyle='-.',
                         ci_linewidth=1,
                         legend_title=None,
                         legend_fontsize=10,
                         title_fontsize=10,
                         ylabel_fontsize=8,
                         wspace=0.1):
    """
    Plot impulse response functions with fixed subplot sizes and flexible legend.
    
    When multiple specifications are provided, shows confidence intervals as light
    shading for the first specification and boundary lines for the rest.
    
    Parameters:
    -----------
    df : DataFrame
        Data containing IRF estimates
    outcome : str
        Outcome variable to plot
    terms : list
        List of terms (columns)
    subset : str or list
        Subset(s) to plot
    time_trend : str or list
        Time trend model(s) to plot
    tshock_type : str or list
        Shock type(s) to plot
    specification : str or list
        Specification(s) to plot
    titles : list, optional
        Titles for each term subplot
    labels : dict, optional
        Dictionary mapping parameter values to display labels
    colors : dict, optional
        Dictionary mapping parameter values to colors
    linestyles : dict, optional
        Dictionary mapping parameter values to linestyles
    ci_index : int or str, optional
        Index or value of the line to show filled CI for (default: 0, first line)
    show_all_ci : bool, optional
        If True, show CI for all specifications (default: False, auto-enabled for specifications)
    figsize : tuple, optional
        Figure size (default: (16, 2.5))
    legend_width_ratio : float, optional
        Width ratio for legend column (default: 0.3)
    sharey : bool, optional
        Share y-axis across subplots (default: True)
    ci_alpha : float, optional
        Alpha for filled confidence interval shading (default: 0.15)
    ci_linestyle : str, optional
        Linestyle for CI boundary lines (default: '--')
    ci_linewidth : float, optional
        Linewidth for CI boundary lines (default: 1)
    legend_title : str, optional
        Title for legend (auto-detected if not provided)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    
    # --- Detect which parameter has multiple values ---
    params = {
        'subset': subset,
        'time_trend': time_trend,
        'tshock_type': tshock_type,
        'specification': specification
    }
    
    col_map = {
        'subset': 'subset',
        'time_trend': 'model',
        'tshock_type': 'Tshock_type',
        'specification': 'specification'
    }
    
    varying_param = None
    varying_values = None
    
    for param_name, param_value in params.items():
        if isinstance(param_value, list):
            if varying_param is not None:
                raise ValueError("Only one of subset, time_trend, tshock_type, or specification can be a list")
            varying_param = param_name
            varying_values = param_value
    
    if varying_param is None:
        varying_param = 'subset'
        varying_values = [subset]
    
    if varying_param == 'specification' and isinstance(specification, list):
        show_all_ci = True
    
    fixed_params = {}
    for param_name, param_value in params.items():
        if param_name != varying_param:
            fixed_params[col_map[param_name]] = param_value
    
    if titles is None:
        titles = ['Tropical storm (18-33 m/s)', 'Category 1-3 (33-50 m/s)', 'Category >3 (>50 m/s)']
    
    if labels is None:
        labels = {val: str(val) for val in varying_values}
    
    if colors is None:
        default_color_list = ['steelblue', 'coral', 'seagreen', 'mediumpurple', 'goldenrod', 'crimson', 'teal']
        colors = {val: default_color_list[i % len(default_color_list)] 
                  for i, val in enumerate(varying_values)}
    
    if linestyles is None:
        linestyles = {val: '-' for val in varying_values}
    
    if isinstance(ci_index, int):
        ci_value = varying_values[ci_index] if ci_index < len(varying_values) else None
    else:
        ci_value = ci_index if ci_index in varying_values else None
    
    if legend_title is None:
        legend_title_map = {
            'subset': 'Subset',
            'time_trend': 'Time Trend',
            'tshock_type': 'Shock Type',
            'specification': 'Specification'
        }
        legend_title = legend_title_map.get(varying_param, varying_param)
    
    # --- Calculate ylim ---
    if ylim is None:
        all_data = []
        for term in terms:
            for val in varying_values:
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['term'] == term) & (df['outcome'] == outcome)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                data = df[mask]
                if len(data) > 0:
                    all_data.append(data)
        
        if len(all_data) > 0:
            combined_data = pd.concat(all_data)
            y_min = combined_data['conf.low'].min()
            y_max = combined_data['conf.high'].max()
            y_range = y_max - y_min
            y_min = y_min - y_padding * y_range
            y_max = y_max + y_padding * y_range
            ylim = (y_min, y_max)
    
    # --- Create figure with GridSpec ---
    n_terms = len(terms)
    width_ratios = [1] * n_terms + [legend_width_ratio]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_terms + 1, width_ratios=width_ratios, wspace=wspace)
    
    axes = []
    for i in range(n_terms):
        if i == 0 or not sharey:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)
    
    legend_ax = fig.add_subplot(gs[0, n_terms])
    legend_ax.axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    
    # --- Plot data ---
    legend_handles = []
    legend_labels = []
    
    for idx, (ax, term, title) in enumerate(zip(axes, terms, titles)):
        has_data = False
        
        # First pass: draw CI
        for i, val in enumerate(varying_values):
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['term'] == term) & (df['outcome'] == outcome)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            
            data = df[mask].sort_values('y_h')
            
            if len(data) > 0:
                color = colors.get(val, 'steelblue')
                
                if show_all_ci:
                    if i == 0:
                        ax.fill_between(
                            data['y_h'],
                            data['conf.low'],
                            data['conf.high'],
                            alpha=ci_alpha,
                            color=color
                        )
                    else:
                        ax.plot(
                            data['y_h'],
                            data['conf.low'],
                            color=color,
                            linestyle=ci_linestyle,
                            linewidth=ci_linewidth,
                            alpha=0.7
                        )
                        ax.plot(
                            data['y_h'],
                            data['conf.high'],
                            color=color,
                            linestyle=ci_linestyle,
                            linewidth=ci_linewidth,
                            alpha=0.7
                        )
                else:
                    if val == ci_value:
                        ax.fill_between(
                            data['y_h'],
                            data['conf.low'],
                            data['conf.high'],
                            alpha=ci_alpha,
                            color=color
                        )
        
        # Second pass: draw estimate lines (only for first value)
        for i, val in enumerate(varying_values):
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['term'] == term) & (df['outcome'] == outcome)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            
            data = df[mask].sort_values('y_h')
            
            if len(data) > 0:
                has_data = True
                color = colors.get(val, 'steelblue')
                linestyle = linestyles.get(val, '-')
                label = labels.get(val, str(val))
                
                # Only plot estimate line for the first value
                if i == 0:
                    line, = ax.plot(
                        data['y_h'],
                        data['estimate'],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.5,
                        label=label
                    )
                    
                    # Collect handles/labels from first subplot only
                    if idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(label)
                else:
                    # For non-first values, just add to legend without plotting estimate
                    if idx == 0:
                        # Create a dummy line for the legend
                        dummy_line = plt.Line2D([0], [0], color=color, linestyle=ci_linestyle, 
                                                linewidth=ci_linewidth, alpha=0.7)
                        legend_handles.append(dummy_line)
                        legend_labels.append(label)
        
        ax.axhline(0, color='grey', linestyle='-', alpha=0.6)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel('Years since impact')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False, left=False)
        
        if pct_format:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        if ylim is not None:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
    
    if legend_handles:
        legend_ax.legend(
            legend_handles, 
            legend_labels,
            title=legend_title,
            loc='center left',
            fontsize=legend_fontsize,
            frameon=False
        )
    
    return fig, axes


def sensitivity_temp_irfs(df, 
                          outcome='public_pp_GDD',
                          meantemps=[5, 15, 25],
                          subset='full',
                          time_trend='linear',
                          tshock_type='NATH',
                          specification='1970onwards_gl2_TC10_T02',
                          titles=None,
                          labels=None,
                          colors=None,
                          linestyles=None,
                          ci_colors=None,
                          ci_index=0,
                          figsize=(15, 2),
                          legend_width_ratio=0.25,
                          sharey=True,
                          suptitle="",
                          ylabel='Δ debt ratio / 1°C shock',
                          ylim=None,
                          yticks=None,
                          y_padding=0.1,
                          pct_format=True,
                          ci_alpha=0.2,
                          ci_alpha_secondary=0.15,
                          show_all_ci=True,
                          legend_title=None,
                          legend_fontsize=10,
                          title_fontsize=10,
                          ylabel_fontsize=8,
                          wspace=0.1,
                          estimate_col='est'):
    """
    Plot impulse response functions for temperature shocks at different mean temperatures
    with flexible overlay parameter.
    
    Automatically detects which parameter has multiple values (list) and uses that
    for the overlay lines. Only one of subset, time_trend, tshock_type, specification,
    or outcome should be a list.
    
    Parameters:
    -----------
    df : DataFrame
        Data containing IRF estimates
    outcome : str or list
        Outcome variable(s) to plot
    meantemps : list
        List of mean temperatures for subplots (default: [5, 15, 25])
    subset : str or list
        Subset(s) to plot
    time_trend : str or list
        Time trend model(s) to plot
    tshock_type : str or list
        Shock type(s) to plot
    specification : str or list
        Specification(s) to plot
    titles : list, optional
        Titles for each meantemp subplot
    labels : dict, optional
        Dictionary mapping parameter values to display labels
    colors : dict, optional
        Dictionary mapping parameter values to line colors
    linestyles : dict, optional
        Dictionary mapping parameter values to linestyles
    ci_colors : dict, optional
        Dictionary mapping parameter values to CI fill colors
        e.g., {'NATH': '#FEC288', 'BILAL': 'green', 'Tshock0': 'blue'}
        If None, uses the same colors as lines
    ci_index : int or str, optional
        Index or value of the line to show filled CI for (default: 0, first line)
        Only used if show_all_ci=False
    show_all_ci : bool, optional
        If True, show CI for all lines (default: True)
    ci_alpha : float, optional
        Alpha for primary CI shading (default: 0.2)
    ci_alpha_secondary : float, optional
        Alpha for secondary CI shading (default: 0.15)
    estimate_col : str, optional
        Column name for point estimates (default: 'est')
    
    Returns:
    --------
    fig, axes : matplotlib figure and axis objects
    """
    
    # Auto-detect estimate column
    if estimate_col not in df.columns:
        if 'orig_estimate' in df.columns:
            estimate_col = 'orig_estimate'
        elif 'est' in df.columns:
            estimate_col = 'est'
        else:
            raise ValueError(f"Could not find estimate column. Available columns: {df.columns.tolist()}")
    
    # --- Detect which parameter has multiple values ---
    params = {
        'subset': subset,
        'time_trend': time_trend,
        'tshock_type': tshock_type,
        'specification': specification,
        'outcome': outcome
    }
    
    col_map = {
        'subset': 'subset',
        'time_trend': 'model',
        'tshock_type': 'Tshock_type',
        'specification': 'specification',
        'outcome': 'outcome'
    }
    
    # Find which parameter is a list (the varying one)
    varying_param = None
    varying_values = None
    
    for param_name, param_value in params.items():
        if isinstance(param_value, list):
            if varying_param is not None:
                raise ValueError("Only one of subset, time_trend, tshock_type, specification, or outcome can be a list")
            varying_param = param_name
            varying_values = param_value
    
    # If none are lists, default to subset as single value
    if varying_param is None:
        varying_param = 'subset'
        varying_values = [subset]
    
    # Get fixed values for non-varying parameters
    fixed_params = {}
    for param_name, param_value in params.items():
        if param_name != varying_param:
            fixed_params[col_map[param_name]] = param_value
    
    if titles is None:
        titles = [f'{t}°C mean temp' for t in meantemps]
    
    if labels is None:
        labels = {val: str(val) for val in varying_values}
    
    if colors is None:
        default_color_list = ['#FEC288', 'steelblue', 'seagreen', 'mediumpurple', 'goldenrod', 'crimson', 'teal']
        colors = {val: default_color_list[i % len(default_color_list)] 
                  for i, val in enumerate(varying_values)}
    
    # --- Default CI colors: use line colors if not specified ---
    if ci_colors is None:
        ci_colors = colors.copy()
    else:
        # Fill in any missing values with line colors
        for val in varying_values:
            if val not in ci_colors:
                ci_colors[val] = colors.get(val, 'grey')
    
    if linestyles is None:
        default_linestyle_list = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2))]
        linestyles = {val: default_linestyle_list[i % len(default_linestyle_list)] 
                      for i, val in enumerate(varying_values)}
    
    if isinstance(ci_index, int):
        ci_value = varying_values[ci_index] if ci_index < len(varying_values) else None
    else:
        ci_value = ci_index if ci_index in varying_values else None
    
    if legend_title is None:
        legend_title_map = {
            'subset': 'Subset',
            'time_trend': 'Time Trend',
            'tshock_type': 'Shock Type',
            'specification': 'Specification',
            'outcome': 'Outcome'
        }
        legend_title = legend_title_map.get(varying_param, varying_param)
    
    # --- Calculate ylim if not provided ---
    if ylim is None:
        all_data = []
        for meantemp in meantemps:
            for val in varying_values:
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['meantemp'] == meantemp)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                data = df[mask]
                if len(data) > 0:
                    all_data.append(data)
        
        if len(all_data) > 0:
            combined_data = pd.concat(all_data)
            y_min = combined_data['conf.low'].min()
            y_max = combined_data['conf.high'].max()
            y_range = y_max - y_min
            y_min = y_min - y_padding * y_range
            y_max = y_max + y_padding * y_range
            ylim = (y_min, y_max)
    
    # --- Create figure with GridSpec ---
    n_temps = len(meantemps)
    width_ratios = [1] * n_temps + [legend_width_ratio]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_temps + 1, width_ratios=width_ratios, wspace=wspace)
    
    axes = []
    for i in range(n_temps):
        if i == 0 or not sharey:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)
    
    legend_ax = fig.add_subplot(gs[0, n_temps])
    legend_ax.axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    
    # --- Plot data ---
    legend_handles = []
    legend_labels = []
    
    for idx, (ax, meantemp, title) in enumerate(zip(axes, meantemps, titles)):
        has_data = False
        
        # First pass: draw CI shading
        if show_all_ci:
            # Draw secondary CIs first (so primary is on top)
            for i, val in enumerate(varying_values):
                if val == ci_value:
                    continue  # Skip primary CI, will draw last
                
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['meantemp'] == meantemp)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                
                data = df[mask].sort_values('y_h')
                
                if len(data) > 0:
                    ci_color = ci_colors.get(val, 'grey')
                    ax.fill_between(
                        data['y_h'],
                        data['conf.low'],
                        data['conf.high'],
                        alpha=ci_alpha_secondary,
                        color=ci_color
                    )
            
            # Draw primary CI on top
            for val in varying_values:
                if val != ci_value:
                    continue
                
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['meantemp'] == meantemp)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                
                data = df[mask].sort_values('y_h')
                
                if len(data) > 0:
                    ci_color = ci_colors.get(val, colors.get(val, 'steelblue'))
                    ax.fill_between(
                        data['y_h'],
                        data['conf.low'],
                        data['conf.high'],
                        alpha=ci_alpha,
                        color=ci_color
                    )
        else:
            # Only show CI for the primary (ci_value)
            for val in varying_values:
                if val != ci_value:
                    continue
                
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['meantemp'] == meantemp)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                
                data = df[mask].sort_values('y_h')
                
                if len(data) > 0:
                    ci_color = ci_colors.get(val, colors.get(val, 'steelblue'))
                    ax.fill_between(
                        data['y_h'],
                        data['conf.low'],
                        data['conf.high'],
                        alpha=ci_alpha,
                        color=ci_color
                    )
        
        # Second pass: draw all estimate lines
        for i, val in enumerate(varying_values):
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['meantemp'] == meantemp)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            
            data = df[mask].sort_values('y_h')
            
            if len(data) > 0:
                has_data = True
                color = colors.get(val, 'steelblue')
                linestyle = linestyles.get(val, '-')
                label = labels.get(val, str(val))
                
                line, = ax.plot(
                    data['y_h'],
                    data[estimate_col],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5 if val == ci_value else 1,
                    label=label
                )
                
                # Collect handles/labels from first subplot only
                if idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)
        
        ax.axhline(0, color='grey', linestyle='-', alpha=0.6)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel('Years since impact')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False, left=False)
        
        if pct_format:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if yticks is not None:
            ax.set_yticks(yticks)
    
    if legend_handles:
        legend_ax.legend(
            legend_handles, 
            legend_labels,
            title=legend_title,
            loc='center left',
            fontsize=legend_fontsize,
            frameon=False
        )
    
    return fig, axes


def sensitivity_temp_irfs_subset(df, 
                          outcome='public_pp_GDD',
                          meantemps=[5, 15, 25],
                          subset='full',
                          time_trend='linear',
                          tshock_type='NATH',
                          specification='1970onwards_gl2_TC10_T02',
                          titles=None,
                          labels=None,
                          colors=None,
                          linestyles=None,
                          ci_index=0,
                          figsize=(16, 2),
                          legend_width_ratio=0.25,
                          sharey=True,
                          suptitle="",
                          ylabel='Δ debt ratio / 1°C shock',
                          ylim=None,
                          yticks=None,
                          y_padding=0.1,
                          pct_format=True,
                          ci_alpha=0.2,
                          ci_alpha_secondary=0.15,
                          ci_color_secondary='grey',
                          line_color_secondary='dimgrey',
                          legend_title=None,
                          legend_fontsize=10,
                          title_fontsize=10,
                          ylabel_fontsize=10,
                          wspace=0.1,
                          last_panel_only_subsets=None,
                          lines_only=False):
    """
    Plot impulse response functions for temperature shocks at different mean temperatures
    with flexible overlay parameter.
    
    Automatically detects which parameter has multiple values (list) and uses that
    for the overlay lines. Only one of subset, time_trend, tshock_type, specification,
    or outcome should be a list.
    
    First line is solid with colored CI, others are dashed/dotted grey lines with grey CI.
    
    If subset is a list and last_panel_only_subsets is provided (or defaults to ['non_rated', 'sids']),
    those subsets will only appear in the last panel, not in earlier panels.
    
    Parameters:
    -----------
    lines_only : bool, optional
        If True, only show CI for the first value; other values show lines only (default: False)
    """
    
    # --- Detect which parameter has multiple values ---
    params = {
        'subset': subset,
        'time_trend': time_trend,
        'tshock_type': tshock_type,
        'specification': specification,
        'outcome': outcome
    }
    
    col_map = {
        'subset': 'subset',
        'time_trend': 'model',
        'tshock_type': 'Tshock_type',
        'specification': 'specification',
        'outcome': 'outcome'
    }
    
    # Find which parameter is a list (the varying one)
    varying_param = None
    varying_values = None
    
    for param_name, param_value in params.items():
        if isinstance(param_value, list):
            if varying_param is not None:
                raise ValueError("Only one of subset, time_trend, tshock_type, specification, or outcome can be a list")
            varying_param = param_name
            varying_values = param_value
    
    # If none are lists, default to subset as single value
    if varying_param is None:
        varying_param = 'subset'
        varying_values = [subset]
    
    # Set default last_panel_only_subsets if subset is the varying parameter
    if varying_param == 'subset' and last_panel_only_subsets is None:
        last_panel_only_subsets = ['non_rated', 'sids']
    
    # Get fixed values for non-varying parameters
    fixed_params = {}
    for param_name, param_value in params.items():
        if param_name != varying_param:
            fixed_params[col_map[param_name]] = param_value
    
    if titles is None:
        titles = [f'{t}°C mean temp' for t in meantemps]
    
    if labels is None:
        labels = {val: str(val) for val in varying_values}
    
    # --- Default colors: first is colored, rest cycle through secondary colors ---
    if colors is None:
        secondary_colors = ['dimgrey', 'darkgrey', 'slategrey', 'grey', 'darkslategrey']
        colors = {}
        secondary_idx = 0
        for i, val in enumerate(varying_values):
            if i == 0:
                colors[val] = '#FEC288'
            else:
                colors[val] = secondary_colors[secondary_idx % len(secondary_colors)]
                secondary_idx += 1
    
    # --- Default linestyles: solid for first, then dashed, dotted, dashdot, etc. ---
    if linestyles is None:
        default_linestyle_list = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2))]
        linestyles = {val: default_linestyle_list[i % len(default_linestyle_list)] 
                      for i, val in enumerate(varying_values)}
    
    if isinstance(ci_index, int):
        ci_value = varying_values[ci_index] if ci_index < len(varying_values) else None
    else:
        ci_value = ci_index if ci_index in varying_values else None
    
    if legend_title is None:
        legend_title_map = {
            'subset': 'Subset',
            'time_trend': 'Time Trend',
            'tshock_type': 'Shock Type',
            'specification': 'Specification',
            'outcome': 'Outcome'
        }
        legend_title = legend_title_map.get(varying_param, varying_param)
    
    # --- Calculate ylim if not provided ---
    if ylim is None:
        all_data = []
        for meantemp in meantemps:
            for val in varying_values:
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['meantemp'] == meantemp)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                data = df[mask]
                if len(data) > 0:
                    all_data.append(data)
        
        if len(all_data) > 0:
            combined_data = pd.concat(all_data)
            y_min = combined_data['conf.low'].min()
            y_max = combined_data['conf.high'].max()
            y_range = y_max - y_min
            y_min = y_min - y_padding * y_range
            y_max = y_max + y_padding * y_range
            ylim = (y_min, y_max)
    
    # --- Create figure with GridSpec ---
    n_temps = len(meantemps)
    width_ratios = [1] * n_temps + [legend_width_ratio]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_temps + 1, width_ratios=width_ratios, wspace=wspace)
    
    axes = []
    for i in range(n_temps):
        if i == 0 or not sharey:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)
    
    legend_ax = fig.add_subplot(gs[0, n_temps])
    legend_ax.axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    
    # --- Build legend handles for all values upfront ---
    legend_handles = []
    legend_labels_list = []
    for i, val in enumerate(varying_values):
        color = colors.get(val, 'steelblue')
        linestyle = linestyles.get(val, '-')
        label = labels.get(val, str(val))
        
        dummy_line = plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1)
        legend_handles.append(dummy_line)
        legend_labels_list.append(label)
    
    # --- Plot data ---
    for idx, (ax, meantemp, title) in enumerate(zip(axes, meantemps, titles)):
        has_data = False
        is_last_panel = (idx == len(meantemps) - 1)
        
        # Determine which values to plot for this panel
        if is_last_panel or varying_param != 'subset' or last_panel_only_subsets is None:
            # Last panel: show all values
            panel_values = varying_values
        else:
            # Earlier panels: exclude last_panel_only_subsets
            panel_values = [val for val in varying_values if val not in last_panel_only_subsets]
        
        # First pass: draw all CI shading (secondary ones first, then primary on top)
        if not lines_only:
            for i, val in enumerate(varying_values):
                if val not in panel_values:
                    continue
                if val == ci_value:
                    continue  # Skip primary CI, will draw last
                
                filter_dict = {col_map[varying_param]: val, **fixed_params}
                mask = (df['meantemp'] == meantemp)
                for col, filter_val in filter_dict.items():
                    mask &= (df[col] == filter_val)
                
                data = df[mask].sort_values('y_h')
                
                if len(data) > 0:
                    ax.fill_between(
                        data['y_h'],
                        data['conf.low'],
                        data['conf.high'],
                        alpha=ci_alpha_secondary,
                        color=ci_color_secondary
                    )
        
        # Draw primary CI on top (always shown)
        for val in varying_values:
            if val not in panel_values:
                continue
            if val != ci_value:
                continue
            
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['meantemp'] == meantemp)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            
            data = df[mask].sort_values('y_h')
            
            if len(data) > 0:
                color = colors.get(val, 'steelblue')
                ax.fill_between(
                    data['y_h'],
                    data['conf.low'],
                    data['conf.high'],
                    alpha=ci_alpha,
                    color=color
                )
        
        # Second pass: draw all estimate lines
        for i, val in enumerate(varying_values):
            if val not in panel_values:
                continue
            
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['meantemp'] == meantemp)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            
            data = df[mask].sort_values('y_h')
            
            if len(data) > 0:
                has_data = True
                color = colors.get(val, 'steelblue')
                linestyle = linestyles.get(val, '-')
                label = labels.get(val, str(val))
                
                ax.plot(
                    data['y_h'],
                    data['est'],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1,
                    label=label
                )
        
        ax.axhline(0, color='grey', linestyle='-', alpha=0.6)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel('Years since impact')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False, left=False)
        
        if pct_format:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if yticks is not None:
            ax.set_yticks(yticks)
    
    if legend_handles:
        legend_ax.legend(
            legend_handles, 
            legend_labels_list,
            title=legend_title,
            loc='center left',
            fontsize=legend_fontsize,
            frameon=False
        )
    
    return fig, axes


def plot_irf_avg_ws(df, 
                    outcome,
                    specification,
                    subset='full',
                    time_trend='linear',
                    tshock_type='NATH',
                    figsize=(8, 3),
                    ylabel='Δ debt ratio / windspeed (m/s)',
                    ylim=None,
                    y_padding=0.1,
                    pct_format=True,
                    ci_alpha=0.2,
                    ci_index=0,
                    colors=None,
                    linestyles=None,
                    labels=None,
                    title='Average windspeed (m/s)',
                    title_fontsize=10,
                    ylabel_fontsize=8,
                    legend_fontsize=8,
                    legend_title=None):
    """
    Plot IRF for avg_ws term with flexible overlay parameter.
    
    Automatically detects which parameter has multiple values (list) and uses that
    for the overlay lines. Only one of subset, time_trend, or specification should be a list.
    
    Parameters:
    -----------
    df : DataFrame
        Data containing IRF estimates
    outcome : str
        Outcome variable to plot
    specification : str or list
        Specification(s) to plot
    subset : str or list
        Subset(s) to plot
    time_trend : str or list
        Time trend model(s) to plot
    tshock_type : str
        Shock type to filter on (default: 'NATH')
    ci_index : int or str
        Index or value of the line to show confidence interval for (default: 0, first line)
    colors : dict, optional
        Dictionary mapping parameter values to colors
    linestyles : dict, optional
        Dictionary mapping parameter values to linestyles
    labels : dict, optional
        Dictionary mapping parameter values to display labels
    title : str, optional
        Plot title (default: 'Average windspeed (m/s)')
    legend_title : str, optional
        Title for legend (auto-detected if not provided)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # --- Detect which parameter has multiple values ---
    params = {
        'subset': subset,
        'time_trend': time_trend,
        'specification': specification
    }
    
    col_map = {
        'subset': 'subset',
        'time_trend': 'model',
        'specification': 'specification'
    }
    
    varying_param = None
    varying_values = None
    
    for param_name, param_value in params.items():
        if isinstance(param_value, list):
            if varying_param is not None:
                raise ValueError("Only one of subset, time_trend, or specification can be a list")
            varying_param = param_name
            varying_values = param_value
    
    # If none are lists, default to time_trend as single value
    if varying_param is None:
        varying_param = 'time_trend'
        varying_values = [time_trend]
    
    # Get fixed values for non-varying parameters
    fixed_params = {}
    for param_name, param_value in params.items():
        if param_name != varying_param:
            fixed_params[col_map[param_name]] = param_value
    
    if colors is None:
        default_color_list = ['steelblue', 'coral', 'seagreen', 'mediumpurple', 'goldenrod', 'crimson', 'teal']
        colors = {val: default_color_list[i % len(default_color_list)] 
                  for i, val in enumerate(varying_values)}
    
    if linestyles is None:
        if varying_param == 'time_trend':
            linestyles = {
                'linear': '-',
                'no_trend': '--',
                'quadratic': ':'
            }
            for val in varying_values:
                if val not in linestyles:
                    linestyles[val] = '-'
        else:
            linestyles = {val: '-' if i == 0 else '--' for i, val in enumerate(varying_values)}
    
    if labels is None:
        labels = {val: str(val) for val in varying_values}
    
    if isinstance(ci_index, int):
        ci_value = varying_values[ci_index] if ci_index < len(varying_values) else None
    else:
        ci_value = ci_index if ci_index in varying_values else None
    
    if legend_title is None:
        legend_title_map = {
            'subset': 'Subset',
            'time_trend': 'Time Trend',
            'specification': 'Specification'
        }
        legend_title = legend_title_map.get(varying_param, varying_param)
    
    # --- Calculate ylim if not provided ---
    if ylim is None:
        all_data = []
        for val in varying_values:
            filter_dict = {col_map[varying_param]: val, **fixed_params}
            mask = (df['term'] == 'avg_ws') & (df['outcome'] == outcome) & (df['Tshock_type'] == tshock_type)
            for col, filter_val in filter_dict.items():
                mask &= (df[col] == filter_val)
            data = df[mask]
            if len(data) > 0:
                all_data.append(data)
        
        if len(all_data) > 0:
            combined_data = pd.concat(all_data)
            y_min = combined_data['conf.low'].min()
            y_max = combined_data['conf.high'].max()
            y_range = y_max - y_min
            y_min = y_min - y_padding * y_range
            y_max = y_max + y_padding * y_range
            ylim = (y_min, y_max)
    
    # --- Create figure with GridSpec for legend space ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.25], wspace=0.05)
    
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis('off')
    
    # --- Plot each line ---
    legend_handles = []
    legend_labels = []
    
    for i, val in enumerate(varying_values):
        filter_dict = {col_map[varying_param]: val, **fixed_params}
        mask = (df['term'] == 'avg_ws') & (df['outcome'] == outcome) & (df['Tshock_type'] == tshock_type)
        for col, filter_val in filter_dict.items():
            mask &= (df[col] == filter_val)
        
        term_df = df[mask].sort_values('y_h')
        
        if len(term_df) > 0:
            color = colors.get(val, 'steelblue')
            linestyle = linestyles.get(val, '-')
            label = labels.get(val, str(val))
            
            line, = ax.plot(
                term_df['y_h'],
                term_df['estimate'],
                linestyle=linestyle,
                color=color,
                linewidth=1.5 if val == ci_value else 1
            )
            
            legend_handles.append(line)
            legend_labels.append(label)
            
            if val == ci_value:
                ax.fill_between(
                    term_df['y_h'],
                    term_df['conf.low'],
                    term_df['conf.high'],
                    alpha=ci_alpha,
                    color=color
                )
    
    # --- Formatting ---
    ax.axhline(0, color='grey', linestyle='-', alpha=0.6)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Years since impact')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    
    if pct_format:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # --- Add legend to dedicated axis ---
    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc='center left',
            fontsize=legend_fontsize,
            frameon=False,
            title=legend_title
        )
    
    return fig, ax
