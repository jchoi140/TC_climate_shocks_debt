# =============================================================================
# ALTERNATIVE RATING PREDICTION MODELS (SI: "Types of prediction models", Table T9)
# =============================================================================
# Fits the two non-XGBoost models the SI compares the XGB classifier with, on the
# same features, labels (average rating across S&P / Moody's / Fitch) and 20%
# country hold-out as the XGB runs:
#   - Ordered Probit (statsmodels OrderedModel)
#   - Ordered Forest (orf package; optional, skipped with a message if missing)
# Writes test-set metrics and diagnostic plots to 3_output/rating_alternatives/
# {oprobit,oforest}/ plus comparison_summary.csv. No figure or table reads these
# outputs (Table T9 is a qualitative pros/cons table); they document the comparison.
#
# Usage:
#   python3 alternatives_rating.py                      # both models, seeds 0-9
#   python3 alternatives_rating.py --method oprobit --seeds 0 25
# =============================================================================

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ImportError:
    OrderedModel = None

try:
    import orf as orf_pkg
except ImportError:
    orf_pkg = None

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common")))

from prep import (  # noqa: E402
    load_all_data, prepare_training_data, RATING_LABELS, INDEPENDENT_VARS, WGI_VARS,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUTPUT_BASE = os.path.join(_PROJECT_ROOT, "3_output", "rating_alternatives")

TEXT_LABELS_FULL = RATING_LABELS['full']


# =============================================================================
# DATA
# =============================================================================

def load_data():
    """Training panel (same features and average-rating label as the XGB model)."""
    TRAIN_DATA = prepare_training_data(load_all_data(include_bonds=False))
    X_vars = INDEPENDENT_VARS + WGI_VARS + ['OECD']
    df = TRAIN_DATA[['ISO', 'year', 'score_total'] + X_vars].copy()
    # Collapse low-frequency classes for score_total
    df['score_total_adj'] = df['score_total'].apply(lambda x: 4 if x in [1, 2, 3] else x)
    df = df.dropna(subset=['score_total'] + X_vars)
    return df, X_vars


def split_data(df, X_vars, score_col='score_total_adj', seed=42):
    """Split by ISO (20% held out)."""
    unique_ISOs = df['ISO'].unique()
    train_ISOs, test_ISOs = train_test_split(unique_ISOs, test_size=0.2, random_state=seed)

    train_mask = df['ISO'].isin(train_ISOs)
    df_train = df[train_mask]
    df_test = df[~train_mask]

    X_train = df_train[X_vars].values
    X_test = df_test[X_vars].values
    y_train = df_train[score_col].values
    y_test = df_test[score_col].values

    return X_train, X_test, y_train, y_test, df_train, df_test


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def compute_metrics(y_true, y_pred, prefix=""):
    """Compute standard regression/classification metrics."""
    y_pred_int = np.rint(y_pred).astype(int)
    mae = mean_absolute_error(y_true, y_pred_int)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_int))
    r2 = r2_score(y_true, y_pred)
    acc = (y_pred_int == y_true).mean()
    acc1 = (np.abs(y_pred_int - y_true) <= 1).mean()
    acc2 = (np.abs(y_pred_int - y_true) <= 2).mean()

    return {
        f'{prefix}MAE': round(mae, 3),
        f'{prefix}RMSE': round(rmse, 3),
        f'{prefix}R2': round(r2, 3),
        f'{prefix}Accuracy': round(acc, 3),
        f'{prefix}Acc_pm1': round(acc1, 3),
        f'{prefix}Acc_pm2': round(acc2, 3),
    }


def plot_confusion_matrix(y_true, y_pred, n_classes, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    y_pred_int = np.rint(y_pred).astype(int)
    labels = list(range(int(min(y_true.min(), y_pred_int.min())),
                        int(max(y_true.max(), y_pred_int.max())) + 1))

    cm = confusion_matrix(y_true, y_pred_int, labels=labels)
    annot = np.where(cm == 0, "", cm.astype(str))

    start_idx = len(TEXT_LABELS_FULL) - len(labels)
    tick_labels = TEXT_LABELS_FULL[max(0, start_idx):]
    if len(tick_labels) > len(labels):
        tick_labels = tick_labels[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt="", cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels,
                annot_kws={'fontsize': 6}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if 'BBB-' in tick_labels:
        idx = tick_labels.index('BBB-')
        ax.axhline(y=idx, color='grey', linestyle='--', linewidth=1)
        ax.axvline(x=idx, color='grey', linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    """Scatter plot of actual vs predicted ratings."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=10, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7, label='Perfect prediction')
    ax.set_xlabel('Actual Rating Score')
    ax.set_ylabel('Predicted Rating Score')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# ORDERED PROBIT
# =============================================================================

def run_ordered_probit(df, X_vars, output_dir, seed_start=0, seed_end=10):
    """Ordered probit model on rating scores."""
    if OrderedModel is None:
        print("statsmodels OrderedModel not available. Skipping ordered probit.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("ORDERED PROBIT")
    print("=" * 50)

    all_metrics = []

    for seed in range(seed_start, seed_end):
        _, _, _, _, df_train, df_test = split_data(df, X_vars, seed=seed)

        y_train = pd.Categorical(df_train['score_total_adj'], ordered=True)
        y_test_raw = df_test['score_total_adj'].values
        X_train = df_train[X_vars]
        X_test = df_test[X_vars]

        model = OrderedModel(y_train, X_train, distr='probit')
        result = model.fit(method='bfgs', disp=False)

        # Predictions (argmax of predicted probabilities)
        probs_train = result.predict(X_train, which='prob')
        probs_test = result.predict(X_test, which='prob')

        # Map back to rating scores
        categories = np.array(sorted(y_train.categories))
        y_pred_train = categories[probs_train.values.argmax(axis=1)]
        y_pred_test = categories[probs_test.values.argmax(axis=1)]

        train_m = compute_metrics(df_train['score_total_adj'].values, y_pred_train, prefix="train_")
        test_m = compute_metrics(y_test_raw, y_pred_test, prefix="test_")

        print(f"  Seed {seed}: Train Acc±1={train_m['train_Acc_pm1']:.2f}, "
              f"Test Acc±1={test_m['test_Acc_pm1']:.2f}, Test MAE={test_m['test_MAE']:.2f}")

        all_metrics.append({'seed': seed, **train_m, **test_m})

        if seed == seed_start:
            plot_confusion_matrix(y_test_raw, y_pred_test, len(categories),
                                  f"Ordered Probit Confusion Matrix (seed {seed})",
                                  os.path.join(output_dir, "confusion_matrix_test.png"))
            plot_actual_vs_predicted(y_test_raw, y_pred_test,
                                     f"Ordered Probit: Actual vs Predicted (seed {seed})",
                                     os.path.join(output_dir, "actual_vs_predicted.png"))

            # Coefficient plot (excluding cutpoints)
            n_cutpoints = len(categories) - 1
            coefs = result.params[:-n_cutpoints]
            ses = result.bse[:-n_cutpoints]
            fig, ax = plt.subplots(figsize=(8, max(4, len(coefs) * 0.3)))
            ax.errorbar(coefs, range(len(coefs)), xerr=ses,
                        fmt='o', color='black', ecolor='gray', elinewidth=2, capsize=0,
                        markersize=4, alpha=0.6)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax.set_yticks(range(len(coefs)))
            ax.set_yticklabels(coefs.index, fontsize=7)
            ax.set_xlabel('Coefficient')
            ax.set_title('Ordered Probit Coefficients')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "coefficients.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {os.path.join(output_dir, 'coefficients.png')}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"\n  Mean Test Acc±1: {metrics_df['test_Acc_pm1'].mean():.3f}")
    print(f"  Mean Test MAE:   {metrics_df['test_MAE'].mean():.3f}")
    return metrics_df


# =============================================================================
# ORDERED FOREST
# =============================================================================

def run_ordered_forest(df, X_vars, output_dir, seed_start=0, seed_end=10):
    """Ordered Forest (orf package) on rating scores."""
    if orf_pkg is None:
        print("orf package not installed. Install with: pip install orf")
        return None

    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("ORDERED FOREST")
    print("=" * 50)

    all_metrics = []

    for seed in range(seed_start, seed_end):
        _, _, _, _, df_train, df_test = split_data(df, X_vars, seed=seed)

        le = LabelEncoder()
        y_train = le.fit_transform(df_train['score_total_adj'].values)
        y_test_raw = df_test['score_total_adj'].values
        y_test = le.transform(y_test_raw)
        categories = le.classes_

        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[X_vars].values)
        X_test = scaler.transform(df_test[X_vars].values)

        oforest = orf_pkg.OrderedForest(n_estimators=500, min_samples_leaf=10, random_state=seed)
        oforest.fit(X=X_train, y=y_train.astype(int))

        pred_dict = oforest.predict(X=X_test, prob=False)
        y_pred_test = pred_dict['predictions']
        # orf predictions are 1-indexed class labels; subtract 1 for array indexing
        y_pred_test_scores = categories[np.clip(y_pred_test.astype(int) - 1, 0, len(categories) - 1)]

        pred_train_dict = oforest.predict(X=X_train, prob=False)
        y_pred_train = pred_train_dict['predictions']
        y_pred_train_scores = categories[np.clip(y_pred_train.astype(int) - 1, 0, len(categories) - 1)]

        train_m = compute_metrics(df_train['score_total_adj'].values, y_pred_train_scores, prefix="train_")
        test_m = compute_metrics(y_test_raw, y_pred_test_scores, prefix="test_")

        print(f"  Seed {seed}: Train Acc±1={train_m['train_Acc_pm1']:.2f}, "
              f"Test Acc±1={test_m['test_Acc_pm1']:.2f}, Test MAE={test_m['test_MAE']:.2f}")

        all_metrics.append({'seed': seed, **train_m, **test_m})

        if seed == seed_start:
            plot_confusion_matrix(y_test_raw, y_pred_test_scores, len(categories),
                                  f"Ordered Forest Confusion Matrix (seed {seed})",
                                  os.path.join(output_dir, "confusion_matrix_test.png"))
            plot_actual_vs_predicted(y_test_raw, y_pred_test_scores,
                                     f"Ordered Forest: Actual vs Predicted (seed {seed})",
                                     os.path.join(output_dir, "actual_vs_predicted.png"))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"\n  Mean Test Acc±1: {metrics_df['test_Acc_pm1'].mean():.3f}")
    print(f"  Mean Test MAE:   {metrics_df['test_MAE'].mean():.3f}")
    return metrics_df


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

def compare_methods(method_results, output_dir):
    """Test-set metrics averaged over seeds, one row per model."""
    rows = [{'Method': m, 'MAE': d['test_MAE'].mean(), 'Acc±1': d['test_Acc_pm1'].mean(),
             'R²': d['test_R2'].mean()}
            for m, d in method_results.items() if d is not None]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
    print("\n" + "=" * 50)
    print("METHOD COMPARISON (test set)")
    print("=" * 50)
    print(summary.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Alternative Rating Prediction Models")
    parser.add_argument("--method", nargs="+", default=["oprobit", "oforest"],
                        choices=["oprobit", "oforest"],
                        help="Models to run (default: both)")
    parser.add_argument("--seeds", nargs=2, type=int, default=[0, 10],
                        metavar=("START", "END"),
                        help="Seed range [start, end) (default: 0 10)")
    args = parser.parse_args()
    seed_start, seed_end = args.seeds

    print("Loading data...")
    df, X_vars = load_data()
    print(f"  {len(df)} observations, {df['ISO'].nunique()} countries, {len(X_vars)} features")

    method_results = {}
    if "oprobit" in args.method:
        method_results['Ordered Probit'] = run_ordered_probit(
            df, X_vars, os.path.join(OUTPUT_BASE, "oprobit"), seed_start, seed_end)
    if "oforest" in args.method:
        method_results['Ordered Forest'] = run_ordered_forest(
            df, X_vars, os.path.join(OUTPUT_BASE, "oforest"), seed_start, seed_end)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    compare_methods(method_results, OUTPUT_BASE)
    print("\nDone!")


if __name__ == "__main__":
    main()
