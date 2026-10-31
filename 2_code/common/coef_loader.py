"""
coef_loader.py — loading and assembling local-projection coefficient tables.

    load_coefficients            read coefficient CSVs (single file or directory)
    extract_model_coefficients   pull the ts / cat1 / cat3 / temp vectors out
    add_zero_horizon             insert the y_h = -1 zero row every IRF anchors on
    _append                      append bootstrap rows onto a coefficient table
"""

import os

import numpy as np
import pandas as pd

__all__ = ['load_coefficients', 'extract_model_coefficients',
           'add_zero_horizon', '_append']


def load_coefficients(target, shock_type=None):
    """Read LP coefficients from a CSV file or a directory of CSVs (in sorted order);
    for a directory, shock_type 'TC' or 'TEMP' picks which files to read."""
    if os.path.isdir(target):
        if shock_type not in ('TC', 'TEMP'):
            raise ValueError("shock_type must be 'TC' or 'TEMP' for a directory")
        want_temp = shock_type == 'TEMP'
        files = [f for f in sorted(os.listdir(target))
                 if f.endswith('.csv') and f.endswith('_TEMP.csv') == want_temp]
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(os.path.join(target, f)) for f in files],
                         ignore_index=True)
    if not os.path.exists(target):
        print(f'Warning: file not found: {target}')
        return pd.DataFrame()
    return pd.read_csv(target)


def extract_model_coefficients(coef_df, model_type='linear', shock_type='NATH'):
    """Extract TC and temperature coefficients; defaults to the linear NATH specification."""
    subset = coef_df[(coef_df['model'] == model_type) & (coef_df['Tshock_type'] == shock_type)]
    return {
        'ts': subset[subset['term'] == "ts_v2"]['estimate'].values,
        'cat1': subset[subset['term'] == "cat1_v2"]['estimate'].values,
        'cat3': subset[subset['term'] == "cat3_v2"]['estimate'].values,
        'temp': subset[subset['term'] == "temp"]['estimate'].values,
        'temp_interaction': subset[subset['term'] == "temp:meantemp"]['estimate'].values
    }


def add_zero_horizon(tc_df, temp_df, terms):
    """Append the y_h = -1 zero row each IRF starts from, for every model/subset/outcome/spec."""
    models         = tc_df['model'].unique()
    subsets        = tc_df['subset'].unique()
    outcomes       = tc_df['outcome'].unique()
    tshocks        = tc_df['Tshock_type'].unique()
    specifications = tc_df['specification'].unique()
    meantemps      = temp_df['meantemp'].unique()

    base = dict(**{'std.error': 0.0, 'statistic': 0.0, 'p.value': 0.0,
                   'conf.low': 0.000001, 'conf.high': 0.000001, 'y_h': -1})

    rows_tc, rows_temp = [], []
    for model in models:
        for subset in subsets:
            for outcome in outcomes:
                for tshock in tshocks:
                    for spec in specifications:
                        common = dict(base, outcome=outcome, specification=spec,
                                      Tshock_type=tshock, model=model, subset=subset)
                        for term in terms:
                            rows_tc.append(dict(common, term=term, estimate=0.0))
                        for meantemp in meantemps:
                            rows_temp.append(dict(common, meantemp=meantemp, est=0.0))

    tc_out   = pd.concat([tc_df,   pd.DataFrame(rows_tc)],   ignore_index=True)
    temp_out = pd.concat([temp_df, pd.DataFrame(rows_temp)], ignore_index=True)
    return tc_out, temp_out


def _append(base_df, extra_df, rename_map=None):
    extra = extra_df.copy()
    if rename_map:
        extra.rename(columns=rename_map, inplace=True)
    extra = extra.reindex(columns=base_df.columns)
    return pd.concat([base_df, extra], ignore_index=True)
