"""
XGB_supplementary_plots.py

Reads the per-seed XGB output CSVs saved by generate_XGB_predictions.py / XGB_model.py:
  - 3_output/XGB/predictions/{TAG}.csv         (rated + non-rated combined)
  - 3_output/XGB/features/feature_importances_{TAG}.csv
  - 3_output/XGB/metrics/                      (Figs. S27, S29)
where TAG = "XGB_classification_neglogloss_0_{N}".

Generates the XGB supplementary figures: S28 feature importance (default), and
S27 / S29 / S30 / S32 with the --s27 / --s29 / --s30 / --s32 flags.

Outputs go to 4_figures/supplementary/XGB/.

Usage:
    python XGB_supplementary_plots.py [--seeds 50] [--isos USA,PHL,BRB]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.ticker import MaxNLocator, MultipleLocator
from sklearn.metrics import confusion_matrix


# Resolve project root (2_code/supplementary/figures/ -> project root)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "2_code", "common"))
from prep import RATING_LABELS  # noqa: E402
from paths import read_predictions, predictions_exist  # noqa: E402

PRED_DIR = os.path.join(_PROJECT_ROOT, "3_output", "XGB", "predictions")
FEAT_DIR = os.path.join(_PROJECT_ROOT, "3_output", "XGB", "features")
OUT_DIR  = os.path.join(_PROJECT_ROOT, "4_figures", "supplementary", "XGB")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# (1) Feature-importance plot (mean ± std across seeds)
# =============================================================================

def plot_feature_importance(fi_path: str, out_path: str) -> None:
    fi = pd.read_csv(fi_path)
    importance_types = ['weight', 'gain', 'cover']
    feature_names = (fi[fi['importance_type'] == 'weight']
                       .groupby('feature')['importance']
                       .mean()
                       .index.tolist())
    wrapped_names = [textwrap.fill(n, width=21) for n in feature_names]

    importance_dfs = {}
    for imp_type in importance_types:
        sub = fi[fi['importance_type'] == imp_type]
        agg = sub.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        agg = agg.set_index('feature').loc[feature_names].reset_index()
        agg['feature_wrapped'] = wrapped_names
        agg = agg.sort_values('mean', ascending=True)
        importance_dfs[imp_type] = agg

    fig, axes = plt.subplots(1, 3, figsize=(12, max(4, len(feature_names) * 0.3)))
    for ax, imp_type in zip(axes, importance_types):
        df = importance_dfs[imp_type]
        ax.barh(df['feature_wrapped'], df['mean'],
                xerr=df['std'].fillna(0), capsize=2, color='steelblue', alpha=0.85)
        ax.set_xlabel(f'{imp_type.capitalize()} ± Std')
        ax.set_title(f'Feature Importance ({imp_type})')
        ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# =============================================================================
# (0b) Fig. S29 - accuracy across seeds, holding out 20% of countries
# =============================================================================

def plot_heldout_accuracy(metrics_paths: dict, out_path: str) -> None:
    """Fig. S29 (`XGBoost_heldout_test.png`) - one panel per sample.

    `metrics_paths` maps a panel heading to a `metrics_*.csv` written by
    XGB_model.py. Each panel shows Exact / +-1 / +-2 for the test sample and
    for the full sample, as a mean dot with a min-max whisker and an
    interquartile box across seeds.
    """
    cols   = ['acc_test', 'acc1_test', 'acc2_test', 'acc', 'acc1', 'acc2']
    labels = ['Exact', '\u00b11', '\u00b12', 'Exact', '\u00b11', '\u00b12']
    n = len(metrics_paths)

    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), squeeze=False)
    for ax, (heading, path) in zip(axes[0], metrics_paths.items()):
        m = pd.read_csv(path)
        for i, c in enumerate(cols):
            v = m[c]
            ax.plot([i, i], [v.min(), v.max()], color='gray', linewidth=2)
            ax.fill_between([i - 0.2, i + 0.2], v.quantile(0.25), v.quantile(0.75),
                            color='lightblue', alpha=0.5)
            ax.scatter(i, v.mean(), color='blue', zorder=3)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylabel('Accuracy')
        # Same 0.1 tick grid on both panels.
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_title(f'Accuracy holding out 20% of countries '
                     f'({m["seed"].nunique()} random seeds)', fontsize=11, pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # Group labels for the two triplets, below the tick labels.
        for x, grp in ((1.0, 'Test sample'), (4.0, 'Full sample')):
            ax.annotate(grp, xy=(x, 0), xytext=(0, -32), textcoords='offset points',
                        xycoords=('data', 'axes fraction'), ha='center', fontsize=10)
        # Panel heading above the axes, left-aligned to it.
        ax.annotate(heading, xy=(0, 1), xytext=(0, 34), textcoords='offset points',
                    xycoords='axes fraction', ha='left', va='bottom', fontsize=14)

    handles = [plt.Line2D([], [], marker='o', color='blue', linestyle='None', label='Mean'),
               plt.Line2D([], [], color='gray', linewidth=2, label='Min-Max Range'),
               Patch(facecolor='lightblue', alpha=0.5, label='25%-75% Quartile Range')]
    axes[0][-1].legend(handles=handles, fontsize=8, loc='lower right')
    fig.subplots_adjust(top=0.80, bottom=0.16)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# =============================================================================
# (0c) Fig. S32 panel a - rating PDFs by rating source
# =============================================================================

# Fig. S32 panel a - probability-axis ceiling per country row.
PANEL_A_YMAX = {'USA': 0.45, 'PHL': 0.3, 'BRB': 0.2}
PANEL_A_YMAX_DEFAULT = 0.3


def plot_prediction_models_a(sources: dict, isos: list, out_path: str,
                             cf: str = 'combined', row_ymax: dict = None) -> None:
    """Fig. S32 panel a - observed vs counterfactual rating PDFs.

    `sources` maps a column heading ('S&P', 'Fitch', "Moody's", 'Average
    rating') to a predictions CSV from `XGB_model.py --rating-source ...`.
    Rows are `isos`.

    Every column uses the same x scale (the 19 classes CCC- to AAA). The
    probability axis is fixed per country row (`row_ymax`, default
    `PANEL_A_YMAX`) so the four sources are comparable within a row.
    """
    row_ymax = {**PANEL_A_YMAX, **(row_ymax or {})}
    # Shared axis 'CCC-' ... 'AAA'; the average-rating model starts at CCC, so its CCC- bin is empty.
    axis_labels = RATING_LABELS['full'][1:20]
    bbb = axis_labels.index('BBB-')
    x = np.arange(len(axis_labels))

    def _to_axis(pdf):
        """Put one model's class vector on the shared CCC- ... AAA axis."""
        lo = 20 - len(pdf)              # index of its lowest class in 'full'
        out = np.zeros(len(axis_labels))
        if lo <= 1:                     # reaches CCC- or below: fold the tail in
            out[:] = pdf[1 - lo:]
            out[0] += pdf[:1 - lo].sum()
        else:                           # floors above CCC-: pad empty bins below
            out[lo - 1:] = pdf
        return out

    nrow, ncol = len(isos), len(sources)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 1.25 * nrow),
                             squeeze=False)

    for j, (heading, path) in enumerate(sources.items()):
        preds = read_predictions(path)
        if 'is_NR' in preds.columns:
            preds = preds[preds['is_NR'] == 0]
        cf_col = f'y_probs_cf_{cf}' if f'y_probs_cf_{cf}' in preds.columns else 'y_probs_cf'
        _vec = lambda v: np.fromstring(str(v).strip().strip('[]'), sep=' ')

        for i, iso in enumerate(isos):
            ax = axes[i][j]
            iso_df = preds[preds['ISO'] == iso]
            if iso_df.empty:
                ax.text(0.5, 0.5, f'no rows for {iso}', transform=ax.transAxes,
                        ha='center', fontsize=7, color='grey')
                ax.set_axis_off()
                continue
            pdf_obs = _to_axis(np.vstack([_vec(v) for v in iso_df['y_probs']]).mean(axis=0))
            pdf_cf  = _to_axis(np.vstack([_vec(v) for v in iso_df[cf_col]]).mean(axis=0))

            ax.fill_between(x, pdf_obs, color='#4E6688', alpha=0.4)
            ax.fill_between(x, pdf_cf,  color='#F49BAB', alpha=0.5)
            ax.plot(x, pdf_obs, color='#4E6688', alpha=0.6, linewidth=1, label='observed')
            ax.plot(x, pdf_cf,  color='#F49BAB', alpha=0.9, linewidth=1.5,
                    linestyle='--', label='counterfactual')
            ax.axvline(bbb, color='grey', linestyle='--', linewidth=1, alpha=0.7)
            ax.invert_xaxis()                      # AAA on the left
            for sp in ax.spines.values():
                sp.set_visible(False)
            _ymax = row_ymax.get(iso, PANEL_A_YMAX_DEFAULT)
            ax.set_ylim(0, _ymax)
            ax.set_yticks(np.round(np.arange(0, _ymax + 1e-9, 0.1), 2))
            ax.tick_params(axis='y', labelsize=7)

            if j == 0:
                ax.set_ylabel(iso, fontsize=9)
            if i == nrow - 1:
                ax.set_xticks([axis_labels.index('AAA'), bbb, 0])
                ax.set_xticklabels(['AAA', 'BBB-', 'CCC-'], fontsize=8)
            else:
                ax.set_xticks([])
            if i == 0:
                ax.set_title(heading, fontsize=11, fontweight='bold', pad=26)
                # Investment-grade annotations, straddling the BBB- line. The
                # x-axis is inverted (AAA on the left), so a POSITIVE point
                # offset moves toward the speculative-grade end.
                ax.annotate('Above\ninvestment-grade', xy=(bbb, 1.02), xytext=(-6, 2),
                            textcoords='offset points', xycoords=('data', 'axes fraction'),
                            ha='right', va='bottom', fontsize=7.5, color='#444444')
                ax.annotate('Below\ninvestment-grade', xy=(bbb, 1.02), xytext=(6, 2),
                            textcoords='offset points', xycoords=('data', 'axes fraction'),
                            ha='left', va='bottom', fontsize=7.5, color='#444444')
                ax.legend(fontsize=7, frameon=False, loc='upper right')

    fig.text(0.005, 0.99, 'a', ha='left', va='top', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=(0.01, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# =============================================================================
# (0a) Fig. S32 panel b - change-in-probability histograms per model variant
# =============================================================================

def prob_change_by_iso(preds_path: str, cf: str = 'combined') -> pd.DataFrame:
    """Per-country change in P(speculative grade) and P(rating upgrade), in pp.

    From the saved predictions:

      prob_below_threshold_diff   = (sum P(class < BBB-) obs - counterfactual) * 100
      higher_rating_pp_increase1  = (sum P(class > obs) cf  - observed)        * 100

    then averaged over seeds per ISO. One row per ISO.
    """
    preds = read_predictions(preds_path)
    if 'is_NR' in preds.columns:
        preds = preds[preds['is_NR'] == 0]
    cf_col = f'y_probs_cf_{cf}' if f'y_probs_cf_{cf}' in preds.columns else 'y_probs_cf'

    # y_probs may be stored as '[0.02 0.03 ...]' or bare '0.02 0.03 ...'; strip brackets first.
    _vec = lambda v: np.fromstring(str(v).strip().strip('[]'), sep=' ')
    p0 = np.vstack([_vec(v) for v in preds['y_probs']])
    p1 = np.vstack([_vec(v) for v in preds[cf_col]])
    n_classes = p0.shape[1]
    threshold = 9 - (19 - n_classes)          # BBB- on the plotted class scale

    obs = preds['obs'].round().astype(int).values
    idx = np.arange(n_classes)[None, :]
    below = idx < threshold                   # classes strictly below BBB-
    above = idx >= (obs[:, None] + 1)         # classes strictly above the observed rating

    out = pd.DataFrame({
        'ISO': preds['ISO'].values,
        'prob_below_threshold_diff':  ((p0 * below).sum(1) - (p1 * below).sum(1)) * 100,
        'higher_rating_pp_increase1': ((p1 * above).sum(1) - (p0 * above).sum(1)) * 100,
    })
    return out.groupby('ISO', as_index=False).mean().round(2)


def plot_probability_histograms(changes: pd.DataFrame, modeltype: str, out_path: str,
                                bin_size_left: float = 0.5, bin_size_right: float = 0.5,
                                print_stats: bool = True,
                                preds_path: str = None, isos: list = None,
                                cf: str = 'combined', row_height: float = 1.85) -> None:
    """Fig. S32 panel b, one model row: the two change-in-probability histograms.

    Pass `preds_path` and `isos` to add the row's third element, the per-country
    observed-vs-counterfactual rating PDFs.

    `row_height` is the height in inches of one panel-b row; it sets how tall
    the histograms are drawn and so how tall the stacked panel b ends up.
    """
    def _bins(data, step):
        if len(data) == 0:
            return np.array([-step, 0.0, step])
        lo = np.floor(data.min() / step) * step
        hi = np.ceil(data.max() / step) * step
        return np.unique(np.concatenate([np.arange(0, lo - step, -step)[::-1],
                                         np.arange(0, hi + step, step)]))

    want_pdfs = preds_path is not None and isos
    if want_pdfs:
        fig = plt.figure(figsize=(14, row_height))
        gs = fig.add_gridspec(len(isos), 3, width_ratios=[1, 1, 0.75], wspace=0.28)
        axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
        pdf_axes = [fig.add_subplot(gs[i, 2]) for i in range(len(isos))]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, row_height))
        pdf_axes = []

    left = changes.loc[changes['prob_below_threshold_diff'] != 0, 'prob_below_threshold_diff']
    axes[0].hist(left, bins=_bins(left, bin_size_left), edgecolor='white',
                 color='darkred', alpha=0.7)
    axes[0].set_title('probability of being assigned speculative grade', fontsize=10)

    # Negated so the axis reads as the change in likelihood of an upgrade.
    right = -changes.loc[changes['higher_rating_pp_increase1'] != 0, 'higher_rating_pp_increase1']
    axes[1].hist(right, bins=_bins(right, bin_size_right), edgecolor='white',
                 color='darkgreen', alpha=0.7)
    axes[1].set_title('probability of a rating upgrade', fontsize=10)

    for ax in axes:
        ax.vlines(0, 0, ax.get_ylim()[1], colors='grey', linestyles='dashed', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)
        ax.set_xlabel('Change in Probability (percentage points)')
    axes[0].set_ylabel('countries')
    axes[1].set_ylabel('Frequency')

    if want_pdfs:
        preds = read_predictions(preds_path)
        if 'is_NR' in preds.columns:
            preds = preds[preds['is_NR'] == 0]
        cf_col = f'y_probs_cf_{cf}' if f'y_probs_cf_{cf}' in preds.columns else 'y_probs_cf'
        _vec = lambda v: np.fromstring(str(v).strip().strip('[]'), sep=' ')
        n_classes = len(_vec(preds['y_probs'].iloc[0]))
        labels = RATING_LABELS['full'][20 - n_classes:20]
        bbb = labels.index('BBB-') if 'BBB-' in labels else n_classes // 2
        for k, iso in enumerate(isos):
            ax = pdf_axes[k]
            d = preds[preds['ISO'] == iso]
            if d.empty:
                ax.text(0.5, 0.5, f'no rows for {iso}', transform=ax.transAxes,
                        ha='center', fontsize=6, color='grey'); ax.set_axis_off(); continue
            po = np.vstack([_vec(v) for v in d['y_probs']]).mean(axis=0)
            pc = np.vstack([_vec(v) for v in d[cf_col]]).mean(axis=0)
            x = np.arange(n_classes)
            ax.fill_between(x, po, color='#4E6688', alpha=0.4)
            ax.fill_between(x, pc, color='#F49BAB', alpha=0.5)
            ax.plot(x, po, color='#4E6688', alpha=0.6, linewidth=1, label='observed')
            ax.plot(x, pc, color='#F49BAB', alpha=0.9, linewidth=1.3, linestyle='--',
                    label='counterfactual')
            ax.axvline(bbb, color='grey', linestyle='--', linewidth=1, alpha=0.7)
            ax.invert_xaxis()
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_ylabel(iso, fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
            ax.tick_params(axis='y', labelsize=6)
            if k == len(isos) - 1:
                ax.set_xticks([labels.index('AAA'), bbb, 0])
                ax.set_xticklabels(['AAA', 'BBB-', labels[0]], fontsize=7)
            else:
                ax.set_xticks([])
            if k == 0:
                ax.legend(fontsize=6, frameon=False, loc='upper right')

    plt.suptitle(modeltype, y=1.0 + 0.32 / row_height, fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")
    if print_stats:
        b = changes['prob_below_threshold_diff']; h = changes['higher_rating_pp_increase1']
        print(f"    speculative grade: {(b > 0).sum()} up / {(b < 0).sum()} down / {(b > 5).sum()} by >5pp")
        print(f"    rating upgrade:    {(h > 0).sum()} up / {(h < 0).sum()} down / {(h > 5).sum()} by >5pp")


def draw_panel_letter(img, letter: str, fontsize_pt: int = 20, dpi: int = 200):
    """Stamp a bold panel letter into an image's top-left corner.

    Used for panel b, which is stacked from separately rendered rows.
    """
    from PIL import ImageDraw, ImageFont
    from matplotlib import font_manager
    path = font_manager.findfont(
        font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
    font = ImageFont.truetype(path, size=int(round(fontsize_pt * dpi / 72)))
    ImageDraw.Draw(img).text((14, 2), letter, fill='black', font=font)
    return img


# =============================================================================
# (0) Fig. S30 - a single decision tree from the base XGB model
# =============================================================================

def _svg_to_png(svg_path: str, png_path: str, width: int = 2515) -> bool:
    """Rasterise an SVG with `rsvg-convert`. Returns False if it is not on PATH.

    dtreeviz can only save SVG on some platforms, so the figure is rasterised separately.
    """
    import shutil, subprocess
    exe = shutil.which('rsvg-convert')
    if exe is None:
        print('  [warn] rsvg-convert not found; left the SVG at ' + svg_path)
        return False
    subprocess.run([exe, '-w', str(width), svg_path, '-o', png_path], check=True)
    return True


def plot_decision_tree(out_path: str, class_n: int = 6, boost_n: int = 8,
                       seed: int = 0, collapse_low_ratings: bool = True,
                       scoring_method: str = 'neglogloss') -> None:
    """Fig. S30 (`decisiontree.png`) - one booster tree, drawn with dtreeviz.

    A dtreeviz classifier view (class histogram at each split, pie chart at each
    leaf) of the model `XGB_model.run_model` fits for `--model XGB_classification`
    (same 20% country holdout, CV and parameter grid). The tree for class
    `class_n` at boosting round `boost_n` is index ``class_n + boost_n * n_classes``;
    the published figure uses class 6, round 8.

    Requires `dtreeviz`, the `dot` binary, and `rsvg-convert` (see
    `_svg_to_png`).
    """
    import warnings
    import numpy as np
    import xgboost as xgb
    import dtreeviz
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, '2_code', 'main', 'analysis', 'rating_model'))
    from XGB_model import prepare_model_data, SCORING_METHODS

    md = prepare_model_data(collapse_low_ratings=collapse_low_ratings)
    X = md['X_xgb_total']
    y = md['y_xgb_multiclass_total']
    ISO_index = md['ISO_index_total']

    y_enc = LabelEncoder().fit_transform(y)
    n_classes = len(np.unique(y_enc))

    train_ISOs, test_ISOs = train_test_split(ISO_index.unique(), test_size=0.2,
                                             random_state=seed)
    test_mask = ISO_index.isin(test_ISOs)
    X_train, y_train = X[~test_mask], y_enc[~test_mask]
    X_test, y_test = X[test_mask], y_enc[test_mask]
    groups = ISO_index[~test_mask]

    cv = GroupKFold(n_splits=min(3, groups.nunique()))
    base = xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes,
                             early_stopping_rounds=5, random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid={
            'n_estimators': [10, 20],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_lambda': [1.0, 5.0],
            'gamma': [0, 1, 5],
        },
        scoring=SCORING_METHODS[scoring_method], cv=cv, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train, groups=groups,
             eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    best = grid.best_estimator_
    print(f'  best params: {grid.best_params_}')

    tree_index = class_n + boost_n * n_classes
    n_trees = len(best.get_booster().get_dump())
    if tree_index >= n_trees:
        raise ValueError(
            f'tree index {tree_index} (class {class_n}, round {boost_n}) is past '
            f'the {n_trees} trees this fit produced '
            f'({best.n_estimators} rounds x {n_classes} classes)')

    # dtreeviz walks the training frame to size the node histograms and cannot
    # take NaN; XGBoost itself routes missing values, so this only affects the
    # drawing, not the splits.
    X_viz = X_train.fillna(X_train.median())

    svg_path = os.path.splitext(out_path)[0] + '.svg'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        viz = dtreeviz.model(best, X_train=X_viz, y_train=pd.Series(y_train),
                             feature_names=list(X_train.columns),
                             target_name='default', tree_index=tree_index)
        viz.view().save(svg_path)

    if _svg_to_png(svg_path, out_path):
        os.remove(svg_path)
        # dtreeviz also leaves its graphviz source next to the SVG
        if os.path.exists(os.path.splitext(svg_path)[0]):
            os.remove(os.path.splitext(svg_path)[0])
        print(f'  Saved {out_path}  (tree {tree_index} = class {class_n}, '
              f'round {boost_n}; {n_classes} classes)')


# =============================================================================
# (2a) Fig. S27 - single-seed confusion matrices (Full Sample + Holdout Test)
# =============================================================================

def plot_confusion_s27(pred_train_path: str, pred_test_path: str, out_path: str,
                       seed: int = 31, n_classes: int = 20,
                       cbar_max: float = 100) -> None:
    """Fig. S27 (`XGBoost_accuracy.png`) - one fit, two panels.

    Left  "Full Sample"  : train + held-out test pooled, as the published panel.
    Right "Holdout Test" : the held-out 20% of countries only.

    A single fit (seed 31; the caption's accuracy numbers come from it) on the full
    set of observations with 20% of countries held out, as in Fig. S29.
    `n_classes=20` is the full CC...AAA scale and needs predictions from a
    `--rating-scale full` run. `cbar_max` caps the colour scale (cells above it
    saturate) so the skewed counts do not wash out the off-diagonal cells.
    """
    tr = pd.read_csv(pred_train_path)
    te = pd.read_csv(pred_test_path)
    tr, te = tr[tr['seed'] == seed], te[te['seed'] == seed]
    if len(tr) == 0 or len(te) == 0:
        print(f"  [skip] seed {seed} not present in the prediction files")
        return
    full = pd.concat([tr, te], ignore_index=True)

    labels = RATING_LABELS["full"][20 - n_classes:20]
    ig_cut = 10 - (20 - n_classes)        # BBB- boundary on the plotted scale

    def _acc(d, k):
        return float((np.abs(d['y_true'] - d['y_pred']) <= k).mean())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, d, title in ((axes[0], full, 'Full Sample'),
                         (axes[1], te,   'Holdout Test')):
        cm = confusion_matrix(d['y_true'], d['y_pred'].round(),
                              labels=range(n_classes))
        sns.heatmap(cm, annot=np.where(cm == 0, "", cm.astype(str)), fmt="",
                    cmap='Blues', xticklabels=labels, yticklabels=labels,
                    annot_kws={'fontsize': 6}, ax=ax, cbar=True,
                    vmin=0, vmax=cbar_max)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        # Red dashed investment-grade / speculative-grade divider at BBB-
        ax.axhline(y=ig_cut, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(x=ig_cut, color='red', linestyle='--', linewidth=1.2)
        ax.text(0.5, -0.30, f'Accuracy \u00b12 classes: {_acc(d, 2):.0%}',
                transform=ax.transAxes, ha='center', fontsize=15,
                fontweight='bold', color='#444444')
        ax.text(0.5, -0.36, f'(exact {_acc(d, 0):.0%}, \u00b11 class {_acc(d, 1):.0%}; '
                            f'n = {len(d):,})',
                transform=ax.transAxes, ha='center', fontsize=10, color='#666666')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}  (seed {seed}, {n_classes} classes)")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, default=50,
                    help='N_SEEDS used in generate_XGB_predictions.py (default 50)')
    p.add_argument('--model', type=str, default='XGB',
                    choices=['XGB', 'XGBO'])
    p.add_argument('--scoring', type=str, default='neglogloss')
    p.add_argument('--isos', type=str, default='USA,PHL,BRB',
                    help='Comma-separated ISOs for the Fig. S32 rating PDFs')
    p.add_argument('--s27', action='store_true',
                    help="Also render Fig. S27 (single-seed Full Sample / Holdout "
                         "Test confusion matrices) from a --rating-scale full run.")
    p.add_argument('--s27-tag', type=str, default='XGB_classification_neglogloss_10_40_fullscale',
                    help='Prediction-file tag for the Fig. S27 run.')
    p.add_argument('--s29-source', default='sp',
                    choices=['total', 'sp', 'fitch', 'moodys'],
                    help="Rating label behind Fig. S29. Default 'sp': the published "
                         "figure's Full sample is n=3128, which is exactly the "
                         "score_sp sample (score_total gives 3572).")
    p.add_argument('--s29-seeds', type=int, default=50,
                    help='Seed count for the Fig. S29 runs (default 50, as published).')
    p.add_argument('--s29', action='store_true',
                    help='Also render Fig. S29 (accuracy across seeds, country holdout).')
    p.add_argument('--s32a-seeds', type=int, default=10,
                    help='Seed count of the per-rating-source runs behind Fig. S32 '
                         'panel a (default 10). Each seed holds out a different '
                         '20%% of countries, so the PDFs are averaged over seeds; '
                         'too few and a single draw dominates the shape.')
    p.add_argument('--s32', action='store_true',
                    help='Also render Fig. S32 panel b rows (one per model variant found).')
    p.add_argument('--s30', action='store_true',
                    help='Also render Fig. S30 (one decision tree from the base model).')
    p.add_argument('--s30-class', type=int, default=6,
                   help='Fig. S30: which class the drawn booster tree belongs to')
    p.add_argument('--s30-round', type=int, default=8,
                   help='Fig. S30: which boosting round the drawn tree comes from')
    p.add_argument('--s30-seed', type=int, default=0,
                   help='Fig. S30: seed for the ISO holdout split used to fit')
    p.add_argument('--s27-seed', type=int, default=31,
                    help="Seed to draw. The published panel reports ~76%% holdout "
                         "accuracy; seed 8 lands on 0.760, mid-upper in the "
                         "40-seed distribution (0.61-0.82).")
    args = p.parse_args()

    tag = (f'{args.model}_classification_{args.scoring}_0_{args.seeds}'
           if args.model == 'XGB'
           else f'XGBO_{args.scoring}_0_{args.seeds}')

    fi_path = os.path.join(FEAT_DIR, f'feature_importances_{tag}.csv')

    print(f"Reading from tag '{tag}'")
    if os.path.exists(fi_path):
        plot_feature_importance(
            fi_path, os.path.join(OUT_DIR, f'feature_importance_{args.model}_{args.seeds}seeds.png'))
    else:
        print(f"  [skip] {fi_path} missing")

    if args.s29:
        mdir = os.path.join(_PROJECT_ROOT, '3_output', 'XGB', 'metrics')
        _panels = {}
        _src = '' if args.s29_source == 'total' else f'_{args.s29_source}'
        for _title, _sub, _tag in [
            ('Restricted sample', '(excluding samples with missing feature data)',
             f'{args.model}_classification_{args.scoring}_0_{args.s29_seeds}_restricted{_src}'),
            ('Full sample', '(includes samples with missing feature data)',
             f'{args.model}_classification_{args.scoring}_0_{args.s29_seeds}{_src}'),
        ]:
            _m = os.path.join(mdir, f'metrics_{_tag}.csv')
            _tr = os.path.join(mdir, f'predictions_train_{_tag}.csv')
            _te = os.path.join(mdir, f'predictions_test_{_tag}.csv')
            if not os.path.exists(_m):
                print(f'  [skip] S29 panel "{_title}": metrics_{_tag}.csv not found')
                continue
            _n = ''
            if os.path.exists(_tr) and os.path.exists(_te):
                # Per-seed total (train/test sizes differ by seed).
                _a, _b = pd.read_csv(_tr), pd.read_csv(_te)
                _per = _a.groupby('seed').size().add(_b.groupby('seed').size(),
                                                     fill_value=0)
                _n = f' (n={int(_per.max())})'
            _panels[f'{_title}{_n}\n{_sub}'] = _m
        if _panels:
            plot_heldout_accuracy(_panels, os.path.join(OUT_DIR, 'XGBoost_heldout_test.png'))

    if args.s32:
        # --- panel a: one column per rating source -------------------------
        _a_sources, _a_seeds = {}, args.s32a_seeds
        for _label, _suffix in [('S&P', '_sp'), ('Fitch', '_fitch'),
                                ("Moody's", '_moodys'), ('Average rating', '')]:
            _p = os.path.join(PRED_DIR,
                              f'XGB_classification_{args.scoring}_0_{_a_seeds}{_suffix}.csv')
            if os.path.exists(_p):
                _a_sources[_label] = _p
            else:
                print(f'  [skip] S32 panel a column "{_label}": '
                      f'{os.path.basename(_p)} not found')
        if _a_sources:
            plot_prediction_models_a(
                _a_sources, [s.strip() for s in args.isos.split(',')],
                os.path.join(OUT_DIR, 'supp-predictionmodels_a.png'))

        # --- panel b: one row per model variant ----------------------------
        _B_ROWS = tempfile.mkdtemp()   # one PNG per model row, stacked below
        _b_rows = []
        for _tag, _label in [
            (f'XGB_classification_neglogloss_0_{args.seeds}', 'XGB (neg log loss)'),
            ('XGB_classification_accuracy_0_10',              'XGB (accuracy)'),
            ('XGBO_neglogloss_0_10',                          'XGB Ordinal'),
        ]:
            _p = os.path.join(PRED_DIR, f'{_tag}.csv')
            if not predictions_exist(_p):
                print(f'  [skip] {_label}: {_tag}.csv not found')
                continue
            _row = os.path.join(_B_ROWS, f'{_tag}.png')
            plot_probability_histograms(
                prob_change_by_iso(_p), _label, _row,
                preds_path=_p, isos=[i.strip() for i in args.isos.split(',')])
            _b_rows.append(_row)

        if _b_rows:
            from PIL import Image as _Image
            _ims = [_Image.open(r).convert('RGB') for r in _b_rows]
            _w = max(i.width for i in _ims)
            def _pad(im):
                if im.width == _w:
                    return im
                out = _Image.new('RGB', (_w, im.height), 'white')
                out.paste(im, (0, 0)); return out
            _ims = [_pad(i) for i in _ims]
            _out = _Image.new('RGB', (_w, sum(i.height for i in _ims)), 'white')
            _y = 0
            for i in _ims:
                _out.paste(i, (0, _y)); _y += i.height
            draw_panel_letter(_out, 'b')
            _bp = os.path.join(OUT_DIR, 'supp-predictionmodels_b.png')
            _out.save(_bp)
            print(f'  Saved {_bp}  ({len(_ims)} model rows)')
        shutil.rmtree(_B_ROWS)

    if args.s30:
        plot_decision_tree(os.path.join(OUT_DIR, 'decisiontree.png'),
                           class_n=args.s30_class, boost_n=args.s30_round,
                           seed=args.s30_seed)

    if args.s27:
        mdir = os.path.join(_PROJECT_ROOT, '3_output', 'XGB', 'metrics')
        plot_confusion_s27(
            os.path.join(mdir, f'predictions_train_{args.s27_tag}.csv'),
            os.path.join(mdir, f'predictions_test_{args.s27_tag}.csv'),
            os.path.join(OUT_DIR, 'XGBoost_accuracy.png'),
            seed=args.s27_seed)


if __name__ == '__main__':
    main()
