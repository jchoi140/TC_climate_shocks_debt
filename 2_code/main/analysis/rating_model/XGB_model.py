# =============================================================================
# XGB CREDIT RATING MODEL
# =============================================================================
# Standalone script for training XGB/XGBO ordinal model with random ISO seeds.
# Outputs predicted ratings and counterfactual coupon changes.
#
# Usage:
#   python XGB_model.py
#   python XGB_model.py --model XGBO --scoring neglogloss --seeds 0 100
# =============================================================================

import os
import sys
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from xgbordinal import XGBOrdinal as _XGBOrdinal
    class _xgbo_ns:
        XGBOrdinal = _XGBOrdinal
    xgbo = _xgbo_ns()
except ImportError:
    xgbo = None

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Add 2_code/common/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common")))

from paths import predictions_exist  # noqa: E402
from prep import (
    load_all_data, prepare_training_data,
    PATHS, COUNTRY_GROUPS, RATING_LABELS,
    X_VARS, INDEPENDENT_VARS, WGI_VARS,
    NONRATED_ISOS,
)

# Resolve project root (2_code/main/analysis/rating_model/ -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "3_output", "XGB")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_model_data(collapse_low_ratings=True, restrict_features=False,
                       rating_source='total'):
    """Load all data and prepare training/counterfactual matrices.

    collapse_low_ratings: True (default, headline results) merges notches 1-3 of `score_total`
        into 4 (18 classes); False keeps the raw 20-notch scale (Fig. S27 confusion matrix).
    restrict_features: True drops rows with any missing feature ("Restricted sample", Fig. S29).
    rating_source: 'total' (default, average of the three agencies) or 'sp' / 'fitch' /
        'moodys' (Fig. S32 panel a).
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    all_data = load_all_data(verbose=True, include_bonds=False)

    # Per-agency scores are proprietary; only the S&P / Fitch / Moody's variants load them.
    TRAIN_DATA = prepare_training_data(all_data, agencies=(rating_source != 'total'))
    cf_data = pd.read_csv(PATHS['cf_data'])
    cf_data = cf_data[cf_data['year'] == 2019][[
        'ISO', '1990_cf_combined_debtratio', '1990_cf_TC_debtratio', '1990_cf_temp_debtratio',
        '1990_cf_combined_gdp', '1990_cf_TC_gdp', '1990_cf_temp_gdp'
    ]]

    # Model variables
    X_vars = X_VARS

    # Build training splits per CRA
    scores = [c for c in ['score_sp', 'score_fitch', 'score_moodys', 'score_total']
              if c in TRAIN_DATA.columns]
    df = TRAIN_DATA[['ISO', 'year', 'Region.Name', 'Sub.region.Name'] + scores + X_vars]

    # Agency scores below CCC- (3 on the 1-21 scale) fold into CCC- (19 classes, Fig. S32a scale).
    _floor_at_cccminus = lambda x: 3 if x <= 2 else x

    # Total (average of 3 CRAs)
    df4 = df.copy()
    df4['score_total_adj'] = (df4['score_total'].apply(lambda x: 4 if x in [1, 2, 3] else x)
                              if collapse_low_ratings else df4['score_total'])
    df4 = df4.dropna(subset=['score_total'])

    # Prediction set: every country rated by at least one agency, whatever the training source.
    df_predict = df4

    # Training labels: the average (df4) or one agency.
    if rating_source not in ('total', 'sp', 'fitch', 'moodys'):
        raise ValueError("rating_source must be one of ['fitch', 'moodys', 'sp', 'total']")
    if rating_source == 'total':
        _score_adj, _score_raw = 'score_total_adj', 'score_total'
    else:
        _score_raw = f'score_{rating_source}'
        _score_adj = f'{_score_raw}_adj'
        df4 = df.copy()
        df4.loc[:, _score_adj] = df4[_score_raw].apply(_floor_at_cccminus)
        df4 = df4.dropna(subset=[_score_raw])
    print(f"  rating source: {rating_source} ({_score_raw}), {len(df4)} training observations")

    if restrict_features:
        # Fig. S29 "Restricted sample": feature-complete rows only.
        _before = len(df4)
        df4 = df4.dropna(subset=X_vars)
        # Keep the prediction set on the same footing as training.
        df_predict = df_predict.dropna(subset=X_vars)
        print(f"  restricted sample: {len(df4)} of {_before} observations "
              f"have complete features")

    ISO_index_total = df4['ISO']
    X_xgb_total = df4[X_vars]
    y_xgb_multiclass_total = df4[_score_adj]

    # Counterfactual data - rated countries (common 2019 cross-section)
    X_df0 = df_predict[df_predict['year'] == 2019][X_vars + ['ISO']]
    X_df1 = X_df0.copy()
    X_df1 = X_df1.merge(cf_data, on='ISO', how='left')
    X_df1['gross debt ratio'] = X_df1['1990_cf_TC_debtratio']
    X_df1['log(GDP pcap)'] = np.log(X_df1['1990_cf_combined_gdp'])
    cf_cols = [c for c in X_df1.columns if c.startswith('1990_cf_')]
    X_df1 = X_df1.drop(columns=['ISO'] + cf_cols)

    # Observed rating under this source (NaN where the agency does not rate the country).
    X_df_obs_rating_total = df_predict[df_predict['year'] == 2019][_score_raw]

    # Counterfactual data - non-rated countries
    X_df0_nr = TRAIN_DATA[['ISO', 'year', 'Region.Name', 'Sub.region.Name'] + X_vars]
    X_df0_nr = X_df0_nr[X_df0_nr['ISO'].isin(NONRATED_ISOS)]
    X_df0_nr = X_df0_nr[X_df0_nr['year'] == 2019][X_vars + ['ISO']]
    X_df1_nr = X_df0_nr.copy()
    X_df1_nr = X_df1_nr.merge(cf_data, on='ISO', how='left')
    X_df1_nr['gross debt ratio'] = X_df1_nr['1990_cf_TC_debtratio']
    X_df1_nr['log(GDP pcap)'] = np.log(X_df1_nr['1990_cf_combined_gdp'])
    cf_cols_nr = [c for c in X_df1_nr.columns if c.startswith('1990_cf_')]
    X_df1_nr = X_df1_nr.drop(columns=['ISO'] + cf_cols_nr)

    # Observation matrices (no ISO column)
    X_df_obs = X_df0.drop(columns=['ISO'])
    X_df_obs_nr = X_df0_nr.drop(columns=['ISO'])

    return {
        'X_xgb_total': X_xgb_total,
        'y_xgb_multiclass_total': y_xgb_multiclass_total,
        'ISO_index_total': ISO_index_total,
        'X_df_obs': X_df_obs,
        'X_df1': X_df1,
        'X_df_obs_nr': X_df_obs_nr,
        'X_df1_nr': X_df1_nr,
        'X_df0': X_df0,
        'X_df0_nr': X_df0_nr,
        'X_df_obs_rating_total': X_df_obs_rating_total,
        'X_vars': X_vars,
    }


# =============================================================================
# MODEL TRAINING
# =============================================================================

# Scoring methods (GridSearchCV maximizes, so losses use sklearn's 'neg_' names)
SCORING_METHODS = {
    'neglogloss': 'neg_log_loss',       # classification only
    'mae': 'neg_mean_absolute_error',   # regression/ordinal
    'accuracy': 'accuracy',             # classification only
    'f1_macro': 'f1_macro',             # classification only
}


def run_model(model_data, training_model='XGBO', scoring_method='neglogloss',
              seed_start=0, seed_end=10):
    """
    Train XGB/XGBO model across multiple random ISO seeds.

    Parameters:
        model_data (dict): Output from prepare_model_data()
        training_model (str): 'XGB' or 'XGBO'
        scoring_method (str): Key from SCORING_METHODS
        seed_start (int): First seed
        seed_end (int): Last seed (exclusive)

    Returns:
        dict: results_list, feature_importances_list, cf_df_all, cf_df_all_NR, metrics_df
    """
    if training_model == 'XGBO' and xgbo is None:
        raise ImportError("xgbordinal not installed. Install with: pip install xgbordinal")

    X = model_data['X_xgb_total']
    y = model_data['y_xgb_multiclass_total']
    ISO_index = model_data['ISO_index_total']
    X_df_obs = model_data['X_df_obs']
    X_df1 = model_data['X_df1']
    X_df_obs_nr = model_data['X_df_obs_nr']
    X_df1_nr = model_data['X_df1_nr']
    X_df0 = model_data['X_df0']
    X_df0_nr = model_data['X_df0_nr']
    X_df_obs_rating_total = model_data['X_df_obs_rating_total']

    unique_ISOs = ISO_index.unique()

    results_list = []
    feature_importances_list = []

    print("\n" + "=" * 60)
    print(f"TRAINING {training_model} | scoring: {scoring_method}")
    print(f"Seeds: {seed_start} to {seed_end - 1}")
    print("=" * 60)

    skipped_seeds = []
    for seed in range(seed_start, seed_end):
        np.random.seed(seed)

        # Label-encode y to 0..K-1
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Hold out 20% of ISOs for test set
        train_ISOs, test_ISOs = train_test_split(unique_ISOs, test_size=0.2, random_state=seed)
        test_mask = ISO_index.isin(test_ISOs)

        X_train = X[~test_mask]
        X_test = X[test_mask]
        y_train = y_encoded[~test_mask]
        y_test = y_encoded[test_mask]

        print(f"\n--- Seed {seed} ---")
        print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        print(f"Held-out ISOs ({len(test_ISOs)}): {sorted(test_ISOs)[:5]}...")
        print(f"Class distribution - Train: {dict(Counter(y_train))}")

        # Skip seeds whose country holdout leaves a rating class out of training
        # (XGBoost needs contiguous labels; happens on the Fig. S29 "Restricted sample").
        _missing = set(range(len(np.unique(y_encoded)))) - set(np.unique(y_train))
        if _missing:
            print(f"  [skip seed {seed}] training split is missing rating "
                  f"class(es) {sorted(_missing)}")
            skipped_seeds.append(seed)
            continue

        # CV setup (by ISO groups)
        train_groups = ISO_index[~test_mask]
        n_splits = min(3, train_groups.nunique())
        cv = GroupKFold(n_splits=n_splits)

        # Grid search parameters (model-dependent)
        if training_model == 'XGB_regression':
            param_grid = {
                "n_estimators": [20, 30],
                "max_depth": [5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 5.0],
                "gamma": [0, 1, 5],
            }
            base = xgb.XGBRegressor(
                objective="reg:squarederror",
                early_stopping_rounds=5,
                random_state=42,
            )
        elif training_model == 'XGB_classification':
            param_grid = {
                "n_estimators": [10, 20],
                "max_depth": [5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 5.0],
                "gamma": [0, 1, 5],
            }
            base = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=len(np.unique(y_encoded)),
                early_stopping_rounds=5,
                random_state=42,
            )
        elif training_model == 'XGBO':
            param_grid = {
                "n_estimators": [10, 15],
                "max_depth": [5, 7],
                "learning_rate": [0.05],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 5.0],
                "gamma": [0, 1, 5],
            }
            base = xgbo.XGBOrdinal()

        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring=SCORING_METHODS[scoring_method],
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(
            X_train, y_train,
            groups=train_groups,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        print(f"Best params: {best_params}")

        # Evaluation
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Round regression predictions for classification metrics
        if training_model == 'XGB_regression':
            y_pred_train = np.rint(y_pred_train).astype(int)
            y_pred_test = np.rint(y_pred_test).astype(int)

        mae = mean_absolute_error(y_train, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        R2 = r2_score(y_train, y_pred_train)
        acc = (y_pred_train == y_train).mean()
        acc_1 = (np.abs(y_pred_train - y_train) <= 1).mean()
        acc_2 = (np.abs(y_pred_train - y_train) <= 2).mean()

        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        R2_test = r2_score(y_test, y_pred_test)
        acc_test = (y_pred_test == y_test).mean()
        acc_1_test = (np.abs(y_pred_test - y_test) <= 1).mean()
        acc_2_test = (np.abs(y_pred_test - y_test) <= 2).mean()

        print(f"Train - MAE: {mae:.3f}, Acc: {acc:.2f}, Acc±1: {acc_1:.2f}")
        print(f"Test  - MAE: {mae_test:.3f}, Acc: {acc_test:.2f}, Acc±1: {acc_1_test:.2f}")

        # Predictions - rated countries
        y_pred_obs = best_model.predict(X_df_obs)
        y_pred_cf = best_model.predict(X_df1)

        # For regression, round predictions to nearest integer for rating classes
        if training_model == 'XGB_regression':
            y_pred_obs = np.rint(y_pred_obs).astype(int)
            y_pred_cf = np.rint(y_pred_cf).astype(int)

        has_proba = hasattr(best_model, 'predict_proba')
        if has_proba:
            y_probs_obs = best_model.predict_proba(X_df_obs)
            y_probs_cf = best_model.predict_proba(X_df1)
        else:
            y_probs_obs = [None] * len(y_pred_obs)
            y_probs_cf = [None] * len(y_pred_cf)

        cf_df = pd.concat([
            pd.DataFrame({
                'obs': X_df_obs_rating_total - 4,
                'y_pred': y_pred_obs,
                'y_pred_cf': y_pred_cf,
                'pred_change': y_pred_obs - y_pred_cf,
                'y_probs': list(y_probs_obs),
                'y_probs_cf': list(y_probs_cf),
                'seed': seed,
            }).reset_index(drop=True),
            X_df0[['ISO']].reset_index(drop=True),
        ], axis=1)

        # Predictions - non-rated countries
        y_pred_obs_NR = best_model.predict(X_df_obs_nr)
        y_pred_cf_NR = best_model.predict(X_df1_nr)

        if training_model == 'XGB_regression':
            y_pred_obs_NR = np.rint(y_pred_obs_NR).astype(int)
            y_pred_cf_NR = np.rint(y_pred_cf_NR).astype(int)

        if has_proba:
            y_probs_obs_NR = best_model.predict_proba(X_df_obs_nr)
            y_probs_cf_NR = best_model.predict_proba(X_df1_nr)
        else:
            y_probs_obs_NR = [None] * len(y_pred_obs_NR)
            y_probs_cf_NR = [None] * len(y_pred_cf_NR)

        cf_df_NR = pd.concat([
            pd.DataFrame({
                'y_pred': y_pred_obs_NR,
                'y_pred_cf': y_pred_cf_NR,
                'pred_change': y_pred_obs_NR - y_pred_cf_NR,
                'y_probs': list(y_probs_obs_NR),
                'y_probs_cf': list(y_probs_cf_NR),
                'seed': seed,
            }).reset_index(drop=True),
            X_df0_nr[['ISO']].reset_index(drop=True),
        ], axis=1)

        # Feature importances (weight, gain, cover)
        feature_names = X.columns.tolist()
        for imp_type in ['weight', 'gain', 'cover']:
            if training_model == 'XGBO':
                scores = best_model.feature_importance(importance_type=imp_type)
            else:  # XGB_classification or XGB_regression
                booster = best_model.get_booster()
                scores = booster.get_score(importance_type=imp_type)
            for i, fname in enumerate(feature_names):
                key = f'f{i}'
                feature_importances_list.append({
                    'seed': seed,
                    'feature': fname,
                    'importance_type': imp_type,
                    'importance': scores.get(key, scores.get(fname, 0)),
                })

        results_list.append({
            'seed': seed,
            'MAE': mae, 'RMSE': rmse, 'R2': R2,
            'acc': round(acc, 2), 'acc1': round(acc_1, 2), 'acc2': round(acc_2, 2),
            'MAE_test': mae_test, 'RMSE_test': rmse_test, 'R2_test': R2_test,
            'acc_test': round(acc_test, 2), 'acc1_test': round(acc_1_test, 2),
            'acc2_test': round(acc_2_test, 2),
            'cf_df': cf_df, 'cf_df_NR': cf_df_NR,
            'best_params': best_params,
            'y_train': y_train, 'y_pred_train': y_pred_train,
            'y_test': y_test, 'y_pred_test': y_pred_test,
            'n_classes': len(np.unique(y_encoded)),
        })

    # Combine results across seeds
    if skipped_seeds:
        print(f"\nSkipped {len(skipped_seeds)} of {seed_end - seed_start} seeds "
              f"(training split missing a rating class): {skipped_seeds}")
    if not results_list:
        raise RuntimeError("every seed was skipped - no usable training splits")

    cf_df_all = pd.concat([r['cf_df'] for r in results_list], ignore_index=True)
    cf_df_all_NR = pd.concat([r['cf_df_NR'] for r in results_list], ignore_index=True)

    metrics_df = pd.DataFrame([{
        'seed': r['seed'],
        'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2'],
        'acc': r['acc'], 'acc1': r['acc1'], 'acc2': r['acc2'],
        'MAE_test': r['MAE_test'], 'RMSE_test': r['RMSE_test'], 'R2_test': r['R2_test'],
        'acc_test': r['acc_test'], 'acc1_test': r['acc1_test'], 'acc2_test': r['acc2_test'],
        'best_params': r['best_params'],
    } for r in results_list])

    importance_df = pd.DataFrame(feature_importances_list)

    # Train/test predictions for confusion matrices
    predictions_train = pd.concat([
        pd.DataFrame({'seed': r['seed'], 'y_true': r['y_train'], 'y_pred': r['y_pred_train']})
        for r in results_list
    ], ignore_index=True)
    predictions_test = pd.concat([
        pd.DataFrame({'seed': r['seed'], 'y_true': r['y_test'], 'y_pred': r['y_pred_test']})
        for r in results_list
    ], ignore_index=True)

    n_classes = results_list[0]['n_classes']

    return {
        'results_list': results_list,
        'cf_df_all': cf_df_all,
        'cf_df_all_NR': cf_df_all_NR,
        'metrics_df': metrics_df,
        'importance_df': importance_df,
        'predictions_train': predictions_train,
        'predictions_test': predictions_test,
        'n_classes': n_classes,
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(model_results, training_model, scoring_method, seed_start, seed_end,
                 tag_suffix=""):
    """Save model outputs to 3_output/XGBruns/.

    `tag_suffix` keeps runs on another rating scale from overwriting the 18-class outputs.
    """
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    tag = f"{training_model}_{scoring_method}_{seed_start}_{seed_end}{tag_suffix}"

    # Counterfactual predictions
    path_rated = os.path.join(PREDICTIONS_DIR, f"{tag}.csv")
    path_nr = os.path.join(PREDICTIONS_DIR, f"{tag}_NR.csv")
    # Rated {tag}.csv + {tag}_NR.csv. Never overwrite the Fig. 5 500-seed parts written by
    # generate_XGB_predictions.py under the same tag.
    if predictions_exist(path_rated) and not os.path.exists(path_rated):
        sys.exit(f"{tag}: seed-range parts from generate_XGB_predictions.py already exist in {PREDICTIONS_DIR}; "
                 "use generate_XGB_predictions.py for the Fig. 5 run, or choose a different seed range.")
    model_results['cf_df_all'].to_csv(path_rated, index=False)
    model_results['cf_df_all_NR'].to_csv(path_nr, index=False)
    print(f"\nSaved: {path_rated}")
    print(f"Saved: {path_nr}")

    # Feature importances
    path_feat = os.path.join(FEATURES_DIR, f"feature_importances_{tag}.csv")
    model_results['importance_df'].to_csv(path_feat, index=False)
    print(f"Saved: {path_feat}")

    # Metrics
    path_metrics = os.path.join(METRICS_DIR, f"metrics_{tag}.csv")
    model_results['metrics_df'].to_csv(path_metrics, index=False)
    print(f"Saved: {path_metrics}")

    # Train/test predictions (for confusion matrices)
    path_pred_train = os.path.join(METRICS_DIR, f"predictions_train_{tag}.csv")
    path_pred_test = os.path.join(METRICS_DIR, f"predictions_test_{tag}.csv")
    model_results['predictions_train'].to_csv(path_pred_train, index=False)
    model_results['predictions_test'].to_csv(path_pred_test, index=False)
    print(f"Saved: {path_pred_train}")
    print(f"Saved: {path_pred_test}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="XGB Credit Rating Model")
    parser.add_argument("--model", default="XGB_classification",
                        choices=["XGB_classification", "XGB_regression", "XGBO"],
                        help="Model type (default: XGB_classification)")
    parser.add_argument("--scoring", default=None,
                        choices=list(SCORING_METHODS.keys()),
                        help="Scoring method (default: neglogloss for classification/XGBO, mae for regression)")
    parser.add_argument("--seeds", nargs=2, type=int, default=[0, 10],
                        metavar=("START", "END"),
                        help="Seed range [start, end) (default: 0 10)")
    parser.add_argument("--rating-source", default="total",
                        choices=["total", "sp", "fitch", "moodys"],
                        help="Rating label to train on. 'total' (default) is the "
                             "average across agencies used for every headline "
                             "result; the others drive Fig. S32 panel a.")
    parser.add_argument("--restrict-features", action="store_true",
                        help="Drop rows with any missing feature (Fig. S29's "
                             "'Restricted sample'). Default keeps them.")
    parser.add_argument("--rating-scale", default="collapsed",
                        choices=["collapsed", "full"],
                        help="'collapsed' (default) merges the three lowest notches "
                             "-> 18 classes, as used for every headline result. "
                             "'full' keeps all 20 notches (CC...AAA), the label set "
                             "behind Fig. S27's confusion matrix.")
    args = parser.parse_args()

    # Default scoring based on model type
    if args.scoring is None:
        if args.model == 'XGB_regression':
            args.scoring = 'mae'
        else:
            args.scoring = 'neglogloss'

    # Validate compatibility
    classification_only = {'neglogloss', 'accuracy', 'f1_macro'}
    if args.model == 'XGB_regression' and args.scoring in classification_only:
        parser.error(f"--scoring {args.scoring} is not compatible with XGB_regression. Use 'mae' instead.")

    training_model = args.model
    scoring_method = args.scoring
    seed_start, seed_end = args.seeds

    model_data = prepare_model_data(
        collapse_low_ratings=(args.rating_scale == 'collapsed'),
        restrict_features=args.restrict_features,
        rating_source=args.rating_source)

    model_results = run_model(
        model_data,
        training_model=training_model,
        scoring_method=scoring_method,
        seed_start=seed_start,
        seed_end=seed_end,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(model_results['metrics_df'][['seed', 'MAE', 'acc', 'acc1', 'MAE_test', 'acc_test', 'acc1_test']].to_string(index=False))
    print(f"\nMean test accuracy: {model_results['metrics_df']['acc_test'].mean():.3f}")
    print(f"Mean test acc±1:    {model_results['metrics_df']['acc1_test'].mean():.3f}")

    # A single agency's observed ratings are proprietary: keep them out of the saved
    # predictions (no figure reads `obs` from the per-agency runs).
    if args.rating_source != 'total':
        model_results['cf_df_all'] = model_results['cf_df_all'].drop(columns='obs')

    save_results(model_results, training_model, scoring_method, seed_start, seed_end,
                 tag_suffix=(('' if args.rating_scale == 'collapsed' else '_fullscale')
                             + ('_restricted' if args.restrict_features else '')
                             + ('' if args.rating_source == 'total'
                                else f'_{args.rating_source}')))

    print("\nDone!")


if __name__ == "__main__":
    main()
