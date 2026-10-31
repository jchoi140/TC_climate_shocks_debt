"""
generate_XGB_predictions.py

XGBoost rating model: 
trains the 18-class XGB classifier (neg-log-loss grid search, GroupKFold by ISO) on N random
country-level train/test splits (seeds 0..N-1) and, for each seed, predicts 2019 ratings for
rated and non-rated sovereigns under observed conditions and five counterfactuals built from
3_output/counterfactual_data/attributable_shares.csv (written by main_figures.ipynb):

    combined : debt = cf,       GDP = cf (TC + temperature)   <- headline Fig. 5
    debt     : debt = cf,       GDP = observed
    TC       : debt = observed, GDP = cf (TC only)
    temp     : debt = observed, GDP = cf (temperature only)
    GDPcomb  : debt = observed, GDP = cf (TC + temperature)

Expected coupons use the rating -> coupon curve in 1_input/credit_ratings/predicted_coupon_2011_2019.csv.

Outputs (3_output/XGB/ by default), tag = XGB_classification_neglogloss_0_<N>:
    predictions/<tag>.csv                 rated + non-rated rows (is_NR), one row per ISO x seed;
                                          runs > 50 MB are written as seed-range parts
                                          (XGB_classification_neglogloss_seeds_000_099.csv, ...)
    features/feature_importances_<tag>.csv

The published 500-seed file is `--seeds 500` (seeds 0-499). Plotting is done separately by
2_code/main/figures/M5_basispoint_plot.py.

Usage:
    python3 2_code/main/analysis/rating_model/generate_XGB_predictions.py --seeds 500
    python3 2_code/main/analysis/rating_model/generate_XGB_predictions.py --seeds 5 --out-dir /tmp/xgb_test
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
try:
    import xgbordinal as xgbo
except ImportError:
    xgbo = None  # XGBO model unavailable; only the XGB classifier is supported

# Shared package modules (2_code/common/).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "2_code", "common"))
from prep import (  # noqa: E402
    load_all_data, prepare_training_data, prepare_model_splits,
    prepare_counterfactual_data, PATHS,
)
from paths import write_predictions  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, default=10,
                    help='Number of random seeds, run as seeds 0..N-1 (500 for the published run)')
parser.add_argument('--model', type=str, default='XGB', choices=['XGB', 'XGBO'])
parser.add_argument('--cf-path', type=str,
                    default=os.path.join(_PROJECT_ROOT, '3_output', 'counterfactual_data',
                                         'attributable_shares.csv'),
                    help='attributable_shares.csv with debt_cf_2019 and gdp_pcap_cf_*_2019 columns.')
parser.add_argument('--out-dir', type=str, default=os.path.join(_PROJECT_ROOT, '3_output', 'XGB'),
                    help='Output root; predictions/ and features/ are created inside it.')
parser.add_argument('--checkpoint-every', type=int, default=100,
                    help='Save partial CSVs every N seeds (default 100; 0 disables).')
args = parser.parse_args()

N_SEEDS = args.seeds
CHECKPOINT_EVERY = args.checkpoint_every
MODEL_TYPE = args.model
CF_PATH = args.cf_path
if MODEL_TYPE == 'XGBO' and xgbo is None:
    sys.exit("XGBO requires the xgbordinal package (see README).")
print(f"  CF source: {CF_PATH}")

# Respect SLURM CPU allocation to avoid oversubscription
N_CPUS = int(os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 4))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

param_grid = {
    "n_estimators": [10, 15],
    "max_depth": [5, 7],
    "learning_rate": [0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1.0, 5.0],
    "gamma": [0, 1, 5],
}

# =============================================================================
# LOAD DATA
# =============================================================================
t0 = time.time()
print(f"=== M5 Figure Generation ({MODEL_TYPE}, {N_SEEDS} seeds) ===\n")

print("Loading data...")
all_data = load_all_data(include_bonds=False)

print("Preparing training data...")
TRAIN_DATA = prepare_training_data(all_data)

print("Preparing model splits...")
splits = prepare_model_splits(TRAIN_DATA, score_col='score_total')
X, y = splits['X'], splits['y']
ISO_index = splits['ISO_index']

print("Preparing counterfactual data...")
cf_inputs = prepare_counterfactual_data(splits, train_data=TRAIN_DATA, cf_path=CF_PATH)
X_df_obs    = cf_inputs['X_obs'].drop(columns=['ISO'])
X_df_obs_nr = cf_inputs['X_obs_nr'].drop(columns=['ISO'])

# -----------------------------------------------------------------------------
# Build 5 counterfactual feature matrices per ISO group (rated, non-rated): each swaps
# gross debt ratio and/or log(GDP pcap) to counterfactual values; all else observed.
#   combined : debt=cf,  GDP=cf_combined  (full effect, headline)
#   debt     : debt=cf,  GDP=observed     (debt channel only)
#   TC       : debt=obs, GDP=cf_TC        (TC-only GDP channel)
#   temp     : debt=obs, GDP=cf_temp      (temperature-only GDP channel)
#   GDPcomb  : debt=obs, GDP=cf_combined  (full GDP channel, debt held)
# -----------------------------------------------------------------------------
CF_VARIANTS = ['combined', 'debt', 'TC', 'temp', 'GDPcomb']

_attr_path = CF_PATH
_attr = pd.read_csv(_attr_path)
_required_cols = ['ISO', 'debt_cf_2019',
                   'gdp_pcap_cf_combined_2019',
                   'gdp_pcap_cf_TC_2019',
                   'gdp_pcap_cf_temp_2019']
for c in _required_cols:
    if c not in _attr.columns:
        raise KeyError(f"Required column '{c}' not in {_attr_path}. "
                       "Multi-CF inference requires the joint TC+temp "
                       "attributable_shares.csv format.")
_attr = _attr[_required_cols]


def _build_cf_dict(X_obs_with_iso):
    """Return {variant_name: X_cf_no_iso} for 5 CF specs."""
    base = X_obs_with_iso.merge(_attr, on='ISO', how='left')
    cf_meta_cols = [c for c in _required_cols if c != 'ISO']
    out = {}

    def _make(label, swap_debt, gdp_source):
        X = base.copy()
        if swap_debt:
            X['gross debt ratio'] = X['debt_cf_2019']
        if gdp_source != 'observed':
            X['log(GDP pcap)']    = np.log(X[gdp_source])
        out[label] = X.drop(columns=['ISO'] + cf_meta_cols)

    _make('combined', True,  'gdp_pcap_cf_combined_2019')
    _make('debt',     True,  'observed')
    _make('TC',       False, 'gdp_pcap_cf_TC_2019')
    _make('temp',     False, 'gdp_pcap_cf_temp_2019')
    _make('GDPcomb',  False, 'gdp_pcap_cf_combined_2019')
    return out


X_cf_dict    = _build_cf_dict(cf_inputs['X_obs'])
X_cf_dict_nr = _build_cf_dict(cf_inputs['X_obs_nr'])
print(f"  Built {len(CF_VARIANTS)} CF feature matrices: {CF_VARIANTS}")

# Rating score for 2019 (adjusted to match 18-class encoding: score_total - 4)
obs_rating_total = splits['df'][splits['df']['year'] == 2019]['score_total']

# Coupon curve: maps rating class -> expected coupon by currency type
predicted_cpn = pd.read_csv(PATHS['predicted_cpn_2011_2019'])
le_dummy = LabelEncoder()
y_enc_dummy = le_dummy.fit_transform(y)
n_classes = len(np.unique(y_enc_dummy))
# Align coupon values to the n_classes used in training (trim from top of 20-class scale)
cpn_values_dominant = predicted_cpn['dominant_currency'].values[(20 - n_classes):20]
cpn_values_other = predicted_cpn['local_currency'].values[(20 - n_classes):20]

print(f"  n_classes = {n_classes}, coupon curve aligned")
print(f"  X: {X.shape}, rated 2019: {len(X_df_obs)}, non-rated 2019: {len(X_df_obs_nr)}")

# =============================================================================
# OUTPUT PATHS + CHECKPOINT HELPER
# =============================================================================
PROJECT_OUT = args.out_dir
PRED_DIR = os.path.join(PROJECT_OUT, 'predictions')
FEAT_DIR = os.path.join(PROJECT_OUT, 'features')
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)
SCORING = 'neglogloss'
TAG = (f'{MODEL_TYPE}_classification_{SCORING}_0_{N_SEEDS}'
       if MODEL_TYPE == 'XGB'
       else f'XGBO_{SCORING}_0_{N_SEEDS}')


def _save_checkpoint(rlist, label='checkpoint'):
    """Concat partial results and overwrite the output CSVs (checkpoint)."""
    if not rlist:
        return
    rated = pd.concat([r['cf_df']    for r in rlist], ignore_index=True).copy()
    nr    = pd.concat([r['cf_df_nr'] for r in rlist], ignore_index=True).copy()
    rated['is_NR'] = 0
    nr['is_NR']    = 1
    nr['obs']      = pd.NA
    preds = pd.concat([rated, nr], ignore_index=True, sort=False)
    p_preds = os.path.join(PRED_DIR, f'{TAG}.csv')
    write_predictions(preds, p_preds)   # >50 MB runs are written as seed-range parts

    fi_frames = [r['fi_df'] for r in rlist if r['fi_df'] is not None]
    if fi_frames:
        fi = pd.concat(fi_frames, ignore_index=True)
        p_fi = os.path.join(FEAT_DIR, f'feature_importances_{TAG}.csv')
        fi.to_csv(p_fi, index=False)
    n_done = len(rlist)
    print(f"  [{label}] saved {n_done} seeds → {TAG}.csv "
          f"({len(preds)} rows), "
          f"features ({len(fi) if fi_frames else 0} rows)")


# =============================================================================
# MULTI-SEED TRAINING LOOP
# =============================================================================
print(f"\nRunning {N_SEEDS}-seed training loop "
      f"(checkpoint every {CHECKPOINT_EVERY if CHECKPOINT_EVERY > 0 else 'never'})...")
results_list = []

for seed in range(N_SEEDS):
    np.random.seed(seed)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    train_ISOs, test_ISOs = train_test_split(ISO_index.unique(), test_size=0.2, random_state=seed)
    test_mask = ISO_index.isin(test_ISOs)
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y_encoded[~test_mask], y_encoded[test_mask]
    train_groups = ISO_index[~test_mask]
    cv = GroupKFold(n_splits=min(3, train_groups.nunique()))

    if MODEL_TYPE == 'XGBO':
        base = xgbo.XGBOrdinal(nthread=1)
    else:
        base = xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            n_estimators=50, early_stopping_rounds=5, random_state=42, nthread=1)

    grid = GridSearchCV(base, param_grid=param_grid, scoring='neg_log_loss',
                        cv=cv, n_jobs=N_CPUS, verbose=0)
    grid.fit(X_train, y_train, groups=train_groups,
             eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    best = grid.best_estimator_

    # Test metrics
    yp = best.predict(X_test)
    acc = (yp == y_test).mean()
    mae = mean_absolute_error(y_test, yp)

    # Observed predictions — shared across all CF variants
    y_pred_obs    = best.predict(X_df_obs)
    y_probs_obs   = best.predict_proba(X_df_obs)
    y_pred_obs_nr = best.predict(X_df_obs_nr)
    y_probs_obs_nr = best.predict_proba(X_df_obs_nr)

    cf_df = pd.DataFrame({
        'ISO':    cf_inputs['X_obs']['ISO'].values,
        'obs':    obs_rating_total.values - 4,  # adjust to n_classes encoding
        'y_pred': y_pred_obs,
        'seed':   seed,
    })
    cf_df['y_probs'] = [' '.join(f'{x:.6f}' for x in row) for row in y_probs_obs]
    cf_df['expected_cpn0_dominant'] = y_probs_obs @ cpn_values_dominant
    cf_df['expected_cpn0_other']    = y_probs_obs @ cpn_values_other

    cf_df_nr = pd.DataFrame({
        'ISO':    cf_inputs['X_obs_nr']['ISO'].values,
        'y_pred': y_pred_obs_nr,
        'seed':   seed,
    })
    cf_df_nr['y_probs'] = [' '.join(f'{x:.6f}' for x in row) for row in y_probs_obs_nr]
    cf_df_nr['expected_cpn0_other'] = y_probs_obs_nr @ cpn_values_other

    # Run inference for each of the 5 CF variants and store results with suffix
    for variant in CF_VARIANTS:
        # Rated
        y_pred_cf  = best.predict(X_cf_dict[variant])
        y_probs_cf = best.predict_proba(X_cf_dict[variant])
        cf_df[f'y_pred_cf_{variant}']   = y_pred_cf
        cf_df[f'pred_change_{variant}'] = y_pred_obs - y_pred_cf
        cf_df[f'y_probs_cf_{variant}']  = [' '.join(f'{x:.6f}' for x in row)
                                            for row in y_probs_cf]
        exp_dom = y_probs_cf @ cpn_values_dominant
        exp_oth = y_probs_cf @ cpn_values_other
        cf_df[f'expected_cpn1_dominant_{variant}'] = exp_dom
        cf_df[f'change_dominant_{variant}']        = cf_df['expected_cpn0_dominant'] - exp_dom
        cf_df[f'expected_cpn1_other_{variant}']    = exp_oth
        cf_df[f'change_other_{variant}']           = cf_df['expected_cpn0_other'] - exp_oth

        # Non-rated
        y_pred_cf_nr  = best.predict(X_cf_dict_nr[variant])
        y_probs_cf_nr = best.predict_proba(X_cf_dict_nr[variant])
        cf_df_nr[f'y_pred_cf_{variant}']   = y_pred_cf_nr
        cf_df_nr[f'pred_change_{variant}'] = y_pred_obs_nr - y_pred_cf_nr
        cf_df_nr[f'y_probs_cf_{variant}']  = [' '.join(f'{x:.6f}' for x in row)
                                                for row in y_probs_cf_nr]
        exp_oth_nr = y_probs_cf_nr @ cpn_values_other
        cf_df_nr[f'expected_cpn1_other_{variant}'] = exp_oth_nr
        cf_df_nr[f'change_other_{variant}']        = cf_df_nr['expected_cpn0_other'] - exp_oth_nr

    # Per-seed feature importances (weight, gain, cover); keys are column names or f{i}.
    fi_records = []
    if MODEL_TYPE == 'XGB':
        booster = best.get_booster()
        for imp_type in ['weight', 'gain', 'cover']:
            scores = booster.get_score(importance_type=imp_type)
            for i, fname in enumerate(X.columns):
                fi_records.append({
                    'seed': seed, 'feature': fname,
                    'importance_type': imp_type,
                    'importance': scores.get(fname, scores.get(f'f{i}', 0.0)),
                })
    fi_df = pd.DataFrame(fi_records) if fi_records else None

    results_list.append({'seed': seed, 'acc': acc, 'mae': mae,
                          'cf_df': cf_df, 'cf_df_nr': cf_df_nr,
                          'fi_df': fi_df})

    if (seed + 1) % 50 == 0 or seed == N_SEEDS - 1:
        print(f"  Seed {seed + 1}/{N_SEEDS} — acc: {acc:.3f}, MAE: {mae:.3f}")

    # Checkpoint: write partial CSVs every CHECKPOINT_EVERY seeds.
    if (CHECKPOINT_EVERY > 0
        and (seed + 1) % CHECKPOINT_EVERY == 0
        and (seed + 1) < N_SEEDS):
        _save_checkpoint(results_list, label=f'checkpoint @ seed {seed + 1}/{N_SEEDS}')


# =============================================================================
# FINAL PERSIST + AGGREGATION
# =============================================================================
_save_checkpoint(results_list, label='final')


print(f"\nDone in {time.time() - t0:.0f}s.")
