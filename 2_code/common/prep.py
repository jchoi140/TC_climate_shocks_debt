# =============================================================================
# RATING MODEL DATA PREPARATION
# =============================================================================
# Data loading, processing, and coupon analysis utilities for the credit rating
# prediction model (XGB scripts, SI rating figures and tables).
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API.")
os.environ["PYTHONWARNINGS"] = "ignore:pkg_resources is deprecated:UserWarning"

# Project root (2_code/common/ -> repo root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(_PROJECT_ROOT)

# =============================================================================
# FILE PATHS
# =============================================================================
# Input-file paths live in paths.py.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PATHS, require_proprietary  # noqa: E402,F401

# =============================================================================
# COUNTRY GROUPINGS
# =============================================================================

# Small Island Developing States
SIDS_ISOS = [
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "COK", "CUB", "DMA", "DOM",
    "FJI", "GRD", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "FSM",
    "MUS", "NRU", "NIU", "PLW", "PNG", "WSM", "STP", "KNA", "LCA",
    "VCT", "SYC", "SLB", "SUR", "TLS", "TON", "TTO", "TUV", "VUT"
]

# Regional groupings
REGIONS = {
    'Africa': [
        'ZAF', 'GHA', 'CIV', 'GAB', 'NGA', 'NAM', 'KEN', 'AGO', 'CMR', 'ZMB', 
        'TUN', 'SEN', 'RWA', 'MAR', 'EGY', 'ETH', 'DZA', 'LBY', 'MRT', 'MLI', 
        'NER', 'TCD', 'SDN', 'SSD', 'ERI', 'DJI', 'SOM', 'UGA', 'TZA', 'MOZ', 
        'MWI', 'ZWE', 'BWA', 'LSO', 'SWZ', 'MDG', 'COD', 'COG', 'CAF', 'GNQ', 
        'BEN', 'TGO', 'BFA', 'GIN', 'SLE', 'LBR'
    ],
    'Americas': [
        'CHL', 'USA', 'CAN', 'CRI', 'BRA', 'PRY', 'PAN', 'HND', 'ECU', 'URY', 
        'GTM', 'PER', 'BOL', 'MEX', 'ARG', 'COL', 'VEN', 'NIC', 'SLV'
    ],
    'Asia': [
        'OMN', 'MYS', 'KOR', 'BHR', 'SAU', 'SGP', 'QAT', 'ISR', 'CYP', 'CHN', 
        'ARE', 'KWT', 'THA', 'BGD', 'PHL', 'VNM', 'LBN', 'LKA', 'KAZ', 'UZB', 
        'TUR', 'TJK', 'PAK', 'MNG', 'IND', 'IDN', 'GEO', 'AZE', 'ARM', 'JPN', 
        'TWN', 'HKG', 'MAC', 'PRK', 'MMR', 'LAO', 'KHM', 'NPL', 'BTN', 'AFG', 
        'IRN', 'IRQ', 'SYR', 'JOR', 'PSE', 'YEM', 'KGZ', 'TKM'
    ],
    'Europe': [
        'ISL', 'PRT', 'ITA', 'SWE', 'SVN', 'NOR', 'NLD', 'LUX', 'LTU', 'IRL', 
        'HUN', 'FRA', 'FIN', 'DEU', 'CZE', 'CHE', 'BEL', 'AUT', 'SVK', 'ESP', 
        'POL', 'LVA', 'GRC', 'UKR', 'SRB', 'ROU', 'HRV', 'BGR', 'ALB', 'RUS', 
        'GBR', 'DNK', 'EST', 'MLT', 'BIH', 'MNE', 'MKD', 'BLR', 'MDA'
    ],
    'Oceania': ['NZL', 'AUS'],
    'Small Island States': SIDS_ISOS
}

# Economic/Political groupings
COUNTRY_GROUPS = {
    'OECD': [
        "AUS", "AUT", "BEL", "CAN", "CHL", "CZE", "DNK", "EST", "FIN", "FRA", 
        "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", 
        "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", 
        "ESP", "SWE", "CHE", "TUR", "GBR", "USA"
    ],
    'OPEC': [
        "DZA", "AGO", "GNQ", "GAB", "IRN", "IRQ", "KWT", "LBY", "NGA", "SAU", 
        "ARE", "VEN"
    ],
    'ASEAN': [
        "BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"
    ],
    'Eurozone': [
        "AUT", "BEL", "HRV", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL", 
        "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP", 
        "AND", "MCO", "SMR", "VAT"
    ],
    'Low Income': [
        "AFG", "BDI", "BFA", "CAF", "TCD", "COD", "ERI", "ETH", "GMB", "GNB", 
        "HTI", "KEN", "KGZ", "LBY", "LBR", "MDG", "MLI", "MRT", "MOZ", "MWI", 
        "NER", "NGA", "NPL", "RWA", "SDN", "SLE", "SOH", "SSD", "SYR", "TGO", 
        "TZA", "UGA", "YEM", "ZMB", "ZWE", "AGO", "ARM", "BEN", "BTN", "CAM", 
        "CPV", "COG", "COM", "DJI", "DZA", "EGY", "SLV", "FJI", "GEO", "GHA", 
        "GTM", "HND", "IND", "IDN", "JOR", "KHM", "KIR", "LAO", "LSO", "MAR", 
        "MDA", "MMR", "MNG", "NAM", "NIC", "PAK", "PHL", "PNG", "PSE", "STP", 
        "SWZ", "TLS", "TON", "TUN", "UKR", "UZB", "VUT", "VNM", "WSM", "XKX"
    ],
    'Non-Rated': [
        "AFG", "DZA", "ATG", "BTN", "BRN", "BDI", "CAF", "TCD", "COM", "DJI", 
        "DMA", "GNQ", "ERI", "GRD", "GIN", "GNB", "GUY", "HTI", "KIR", "LBR", 
        "MKD", "MWI", "MHL", "MRT", "FSM", "MCO", "MMR", "NRU", "NPL", "PRK", 
        "PLW", "WSM", "STP", "SLE", "SOM", "SSD", "KNA", "LCA", "SDN", "SYR", 
        "GMB", "TLS", "TON", "TUV", "VUT", "YEM", "ZWE"
    ]
}

# Currency blocs
CURRENCY_BLOCS = {
    'USD_Primary': [
        "BES", "ECU", "SLV", "PLW", "MHL", "FSM", "PAN", "VGB", "TCA", "TLS", "ZWE"
    ],
    'USD_Pegged': [
        "ABW", "BHS", "BRB", "BLZ", "BMU", "KHM", "CAN", "CYM", "CRI", "CUW", 
        "GTM", "HND", "JAM", "LBN", "LBR", "MEX", "MMR", "NIC", "KNA", "SXM", "VNM"
    ],
    'EUR_Bloc': [
        "AUT", "BEL", "HRV", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL", 
        "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP",
        "AND", "XKX", "MCO", "MNE", "SMR", "VAT"
    ]
}

# =============================================================================
# RATING LABELS
# =============================================================================

RATING_LABELS = {
    'full': [  # 20 classes (0-19)
        'CC', 'CCC-', 'CCC', 'CCC+', 'B-', 'B', 'B+', 'BB-', 'BB', 'BB+',
        'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA'
    ],
    'adj': [  # 19 classes (0-18)
        'CCC-', 'CCC', 'CCC+', 'B-', 'B', 'B+', 'BB-', 'BB', 'BB+',
        'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA'
    ],
    'adj_v2': [  # 18 classes (0-17)
        'CCC', 'CCC+', 'B-', 'B', 'B+', 'BB-', 'BB', 'BB+', 'BBB-',
        'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA'
    ]
}

# Dominant currencies for coupon analysis
DOMINANT_CURRENCIES = ['USD', 'EUR', 'GBP', 'CHF', 'CAD']

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_country_master():
    """Load and process country master data."""
    country_master = pd.read_csv(PATHS['country_master'])
    
    # Extract sovereign countries
    sovereign_ISOs = country_master[country_master['sovereigncountry'] == 1]['ISO']
    
    # Extract investment grade categories
    Agrade_ISOs = country_master[country_master['Ainvestmentgrade'] == 1]['ISO'].unique().tolist()
    Bgrade_ISOs = country_master[country_master['Binvestmentgrade'] == 1]['ISO'].unique().tolist()
    
    # Currency mapping
    currency_map = country_master[country_master['sovereigncountry'] == 1][['ISO', 'currency_code']]
    
    # ISO master with development indicators
    isomaster = country_master[['ISO', 'Region.Name', 'Sub.region.Name', 
                                'Least.Developed.Countries..LDC.', 
                                'Land.Locked.Developing.Countries..LLDC.', 
                                'Small.Island.Developing.States..SIDS.']].drop_duplicates()
    isomaster = isomaster.rename(columns={
        'Small.Island.Developing.States..SIDS.': 'SIDS',
        'Least.Developed.Countries..LDC.': 'LDC',
        'Land.Locked.Developing.Countries..LLDC.': 'LLDC'
    })
    isomaster[['LDC', 'LLDC', 'SIDS']] = isomaster[['LDC', 'LLDC', 'SIDS']].fillna(0)
    
    return {
        'country_master': country_master,
        'sovereign_ISOs': sovereign_ISOs,
        'Agrade_ISOs': Agrade_ISOs,
        'Bgrade_ISOs': Bgrade_ISOs,
        'currency_map': currency_map,
        'isomaster': isomaster
    }


def load_climate_data():
    """Load and process climate exposure data."""
    TEMP_dist = pd.read_csv(PATHS['temperature'])
    TC_dist = pd.read_csv(PATHS['tc'])
    
    # 2019 snapshot
    TEMP_2019 = TEMP_dist[TEMP_dist['year'] == 2019][['ISO', 'temp', 'temp_noANTH', 'meantemp_hist']]
    TC_2019 = TC_dist[TC_dist['year'] == 2019][['ISO', 'avg_ws']]
    
    # Calculate warming (anthropogenic contribution)
    avg_warming = TEMP_2019[['ISO', 'meantemp_hist']].copy()
    avg_warming['warming'] = TEMP_2019['temp'] - TEMP_2019['temp_noANTH']
    
    # Calculate average TC exposure (1980+)
    avg_ws_by_iso = TC_dist[TC_dist['year'] >= 1980].groupby('ISO')['avg_ws'].mean().reset_index()
    avg_ws_by_iso.rename(columns={'avg_ws': 'avg_ws_period'}, inplace=True)
    
    # TC exposure classification
    TC_exposed_ISOs = avg_ws_by_iso[avg_ws_by_iso['avg_ws_period'] > 0]['ISO'].tolist()
    TC_notexposed_ISOs = avg_ws_by_iso[avg_ws_by_iso['avg_ws_period'] == 0]['ISO'].tolist()
    
    return {
        'TEMP_dist': TEMP_dist,
        'TC_dist': TC_dist,
        'TEMP_2019': TEMP_2019,
        'TC_2019': TC_2019,
        'avg_warming': avg_warming,
        'avg_ws_by_iso': avg_ws_by_iso,
        'TC_exposed_ISOs': TC_exposed_ISOs,
        'TC_notexposed_ISOs': TC_notexposed_ISOs
    }


def load_economic_data():
    """Load and process economic data."""
    # GDP per capita (log transformed)
    observed_gdp = pd.read_csv(PATHS['gdp_pcap']).rename(columns={"value": "log(GDP pcap)"})
    observed_gdp = observed_gdp.drop(columns=['Country.Name'])
    observed_gdp['log(GDP pcap)'] = np.log(observed_gdp['log(GDP pcap)'])

    # US 10-year yield
    US_10yr = pd.read_csv(PATHS['US_10yr'])
    US_10yr['observation_date'] = pd.to_datetime(US_10yr['observation_date'])
    US_10yr['year'] = US_10yr['observation_date'].dt.year
    US_10yr = US_10yr.rename(columns={'IRLTLT01USA156N': 'US_10yr_yield'})
    US_10yr = US_10yr[['year', 'US_10yr_yield']]

    return {
        'observed_gdp': observed_gdp,
        'US_10yr': US_10yr
    }


AGENCY_SCORES = ['score_sp', 'score_moodys', 'score_fitch']


def load_agency_scores():
    """Per-agency S&P / Moody's / Fitch scores (ISO, year, score_sp, score_moodys,
    score_fitch) from the raw rating histories in 0_raw/credit_ratings/ (proprietary,
    not shipped)."""
    out = None
    for col, key in [('score_sp', 'sp_ratings'), ('score_moodys', 'moodys_ratings'),
                     ('score_fitch', 'fitch_ratings')]:
        s = pd.read_csv(require_proprietary(key)).rename(columns={'value': col})
        out = s if out is None else out.merge(s, on=['ISO', 'year'], how='outer')
    return out


def load_rating_panel(agencies=False):
    """Rating panel (`rating_df_v2.csv`: average rating `score_total` + macro features),
    scores rounded to whole notches. agencies=True adds the per-agency scores
    (load_agency_scores, proprietary) and `avgscore`, each country's mean over the panel
    years of the three-agency average."""
    data = pd.read_csv(PATHS['rating_df_v2'])
    if agencies:
        data = data.merge(load_agency_scores(), on=['ISO', 'year'], how='left')
        # Agency scores only where the panel has a rating: countries the paper treats as
        # non-rated (NONRATED_ISOS, e.g. MKD) keep no agency scores, as in the published panel.
        data[AGENCY_SCORES] = data[AGENCY_SCORES].where(data['score_total'].notna())
        data['avgscore'] = (data[AGENCY_SCORES].mean(axis=1)
                            .groupby(data['ISO']).transform('mean'))
    scores = [c for c in AGENCY_SCORES + ['score_total'] if c in data.columns]
    data[scores] = data[scores].round()
    return data


def load_rating_data():
    """Load rating data and key."""
    rating_data = pd.read_csv(PATHS['rating_df_v2'])[['ISO', 'year', 'score_total']]
    rating_key = pd.read_csv(PATHS['rating_key'])[['S&P', 'sp_score']]
    
    return {
        'rating_data': rating_data,
        'rating_key': rating_key
    }


def load_and_merge_bond_data():
    """Load and merge Bloomberg and LSEG bond data."""
    bloomberg = pd.read_csv(require_proprietary('bloomberg'))
    lseg = pd.read_csv(require_proprietary('lseg'))
    
    # Filter Bloomberg (exclude private placements, require coupon)
    bloomberg_filtered = bloomberg[bloomberg['Market Type'] != 'PRIV PLACEMENT'].dropna(subset=['Cpn'])
    
    # Filter LSEG (require coupon)
    lseg_filtered = lseg.dropna(subset=['Cpn'])
    
    # Count bonds by source and ISO
    bloomberg_counts = bloomberg_filtered.groupby('ISO').size().reset_index(name='b_counts')
    lseg_counts = lseg_filtered.groupby('ISO').size().reset_index(name='l_counts')
    
    # Determine which source to use for each ISO (use source with more data)
    merged_counts = pd.merge(bloomberg_counts, lseg_counts, on='ISO', how='outer').fillna(0)
    use_bloomberg_isos = merged_counts[merged_counts['b_counts'] > merged_counts['l_counts']]['ISO'].unique()
    
    # Prepare Bloomberg data
    bloomberg_final = bloomberg_filtered[bloomberg_filtered['ISO'].isin(use_bloomberg_isos)].copy()
    bloomberg_final = bloomberg_final.drop(
        columns=['score_sp', 'score_moodys', 'score_fitch', 'score_total', 'currency_code'], 
        errors='ignore'
    )
    bloomberg_final = bloomberg_final.rename(columns={'issue_yr': 'year', 'currency': 'currency_issued'})
    
    # Prepare LSEG data (for ISOs not covered by Bloomberg)
    lseg_final = lseg_filtered[~lseg_filtered['ISO'].isin(use_bloomberg_isos)].copy()
    lseg_final = lseg_final.reindex(columns=bloomberg_final.columns)
    
    combined_df = pd.concat([bloomberg_final, lseg_final], ignore_index=True)
    
    return {
        'bloomberg_raw': bloomberg,
        'lseg_raw': lseg,
        'combined_bonds': combined_df,
        'use_bloomberg_isos': use_bloomberg_isos
    }


# =============================================================================
# MAIN DATA LOADING
# =============================================================================

def load_all_data(verbose=False, include_bonds=True):
    """Load all data and return as dictionary.

    include_bonds=False skips the proprietary Bloomberg/LSEG bond files; the rating
    model (training and counterfactual predictions) does not need them.
    """
    if verbose: print("Loading country master data...")
    country_data = load_country_master()
    if verbose: print("Loading climate data...")
    climate_data = load_climate_data()
    if verbose: print("Loading economic data...")
    economic_data = load_economic_data()
    if verbose: print("Loading rating data...")
    rating_data = load_rating_data()
    bond_data = {}
    if include_bonds:
        if verbose: print("Loading and merging bond data...")
        bond_data = load_and_merge_bond_data()
    if verbose: print("Data loading complete!")
    
    return {
        **country_data,
        **climate_data,
        **economic_data,
        **rating_data,
        **bond_data
    }


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_rating_data(all_data):
    """
    Process rating data and merge with bond, economic, and country data.
    
    Parameters:
        all_data (dict): Output from load_all_data()
        
    Returns:
        DataFrame: Processed dataset ready for modeling
    """
    combined_df = all_data['combined_bonds']
    observed_gdp = all_data['observed_gdp']
    US_10yr = all_data['US_10yr']
    currency_map = all_data['currency_map']
    isomaster = all_data['isomaster']
    
    # Rating panel with the per-agency scores (proprietary)
    data = load_rating_panel(agencies=True)
    
    data.columns.name = None
    data = data.rename(columns={
        'Gross domestic product, constant prices': 'GDP growth rate',
        'General government gross debt': 'gross debt ratio',
        'Inflation, average consumer prices': 'inflation',
    })
    
    # Filter to sovereign countries only
    data = data[data['sovereign'] == 1]
    
    # Binary classification dummies (investment grade = score >= 12)
    for col in ['total', 'sp', 'moodys', 'fitch']:
        score_col = f'score_{col}'
        if score_col in data.columns:
            data[f'binary_{col}'] = data[score_col].apply(lambda x: 1 if x >= 12 else 0).astype(int)
    
    # Merge with bond data
    data = data.merge(combined_df, on=['ISO', 'year'], how='left')
    
    # Merge with economic data
    data = data.merge(observed_gdp, on=['ISO', 'year'], how='left')
    data = data.merge(US_10yr, on='year', how='left')
    
    # Merge with country data
    data = data.merge(currency_map, on='ISO', how='left')
    data = data.merge(isomaster, on='ISO', how='left')
    
    # Create region dummies
    for region in ['Asia', 'Africa', 'Americas', 'Oceania', 'Europe']:
        data[f'Region_{region}'] = (data['Region.Name'] == region).astype(int)
    
    # Fill missing currency values for OECD and Eurozone countries
    # For DOMESTIC market type in OECD (excluding Japan)
    domestic_oecd_mask = (
        (data['Market Type'] == 'DOMESTIC') & 
        (data['ISO'].isin(COUNTRY_GROUPS['OECD'])) & 
        (data['ISO'] != 'JPN')
    )
    data.loc[domestic_oecd_mask, 'currency_issued'] = \
        data.loc[domestic_oecd_mask, 'currency_issued'].fillna(data.loc[domestic_oecd_mask, 'currency_code'])
    
    # For EURO market type in EUR bloc
    euro_bloc_mask = (
        (data['Market Type'] == 'EURO') & 
        (data['ISO'].isin(CURRENCY_BLOCS['EUR_Bloc']))
    )
    data.loc[euro_bloc_mask, 'currency_issued'] = \
        data.loc[euro_bloc_mask, 'currency_issued'].fillna(data.loc[euro_bloc_mask, 'currency_code'])
    
    # Foreign currency flag
    data['FC_flag'] = np.where(
        data['currency_issued'].isna(),
        np.nan,
        np.where(data['currency_issued'] == data['currency_code'], 0, 1)
    )
    
    # Country/group dummy variables
    data['issuer_USA'] = (data['ISO'] == 'USA').astype(int)
    data['issuer_CHN'] = (data['ISO'] == 'CHN').astype(int)
    data['OECD'] = data['ISO'].isin(COUNTRY_GROUPS['OECD']).astype(int)
    data['ASEAN'] = data['ISO'].isin(COUNTRY_GROUPS['ASEAN']).astype(int)
    data['OPEC'] = data['ISO'].isin(COUNTRY_GROUPS['OPEC']).astype(int)
    data['USD_bloc_primary'] = data['ISO'].isin(CURRENCY_BLOCS['USD_Primary']).astype(int)
    data['USD_bloc_pegged'] = data['ISO'].isin(CURRENCY_BLOCS['USD_Pegged']).astype(int)
    data['EUR_bloc'] = data['ISO'].isin(CURRENCY_BLOCS['EUR_Bloc']).astype(int)
    
    # Fill governance indicators within ISO-year groups
    governance_cols = ['VA', 'PS', 'GE', 'RQ', 'RL', 'CC']
    for col in governance_cols:
        if col in data.columns:
            data[col] = data.groupby(['ISO', 'year'])[col].transform(
                lambda x: x.fillna(x.dropna().iloc[0]) if x.notna().any() else x
            )
    
    return data


def analyze_coupon_by_currency(data, min_year=2011):
    """
    Analyze coupon rates by currency type (FC vs LC).
    
    Parameters:
        data (DataFrame): Processed rating data with bonds
        min_year (int): Minimum year to include
        
    Returns:
        tuple: (coupon_pivot DataFrame, dominant_fc_isos list)
    """
    valid_coupon_data = data[(~data['Cpn'].isna()) & (data['year'] >= min_year)]
    
    # Summary by ISO and FC_flag
    coupon_summary = valid_coupon_data.groupby(['ISO', 'FC_flag'], dropna=False).agg(
        avg_coupon=('Cpn', lambda x: round(x.mean())),
        count_bonds=('Cpn', 'size')
    ).reset_index()
    
    # Pivot to get FC_0, FC_1, and FC_missing columns
    coupon_pivot = coupon_summary.pivot(index='ISO', columns='FC_flag', values=['avg_coupon', 'count_bonds'])
    coupon_pivot.columns = [
        f"{stat}_FC_{int(flag) if pd.notna(flag) else 'missing'}" 
        for stat, flag in coupon_pivot.columns
    ]
    coupon_pivot = coupon_pivot.reset_index()
    
    # Fill missing values and calculate shares
    bond_cols = [col for col in coupon_pivot.columns if 'count_bonds' in col]
    coupon_pivot[bond_cols] = coupon_pivot[bond_cols].fillna(0).astype(int)
    coupon_pivot['total_bonds'] = coupon_pivot[bond_cols].sum(axis=1)
    
    for col in bond_cols:
        share_col = col.replace('count_bonds', 'share')
        coupon_pivot[share_col] = ((coupon_pivot[col] / coupon_pivot['total_bonds']) * 100).round(0)
    
    # Identify ISOs where FC bonds are dominant
    if 'share_FC_1' in coupon_pivot.columns:
        dominant_fc_isos = coupon_pivot[coupon_pivot['share_FC_1'] > 50]['ISO'].tolist()
        num_fc_dominant = len(dominant_fc_isos)
        print(f"Number of ISOs where FC bonds are dominant (>50%): {num_fc_dominant}")
    else:
        dominant_fc_isos = []
    
    return coupon_pivot, dominant_fc_isos


def create_currency_summary(data, min_year=2000):
    """
    Create comprehensive currency issuance summary by ISO.
    
    Parameters:
        data (DataFrame): Processed rating data with bonds
        min_year (int): Minimum year to include
        
    Returns:
        DataFrame: Currency summary with bond counts and percentages
    """
    filtered_data = data[(~data['Cpn'].isna()) & (data['year'] > min_year)]
    
    currency_summary = filtered_data.groupby('ISO').agg({
        'currency_issued': [
            lambda x: ', '.join(sorted(x.dropna().unique())),
            lambda x: sum(x == 'USD'),
            lambda x: sum(x == 'EUR'),
            lambda x: sum(x == 'GBP'),
            lambda x: sum((x != 'USD') & (x != 'EUR') & (x != 'GBP') & x.notna()),
            lambda x: sum(x.isna())
        ],
        'currency_code': lambda x: ', '.join(sorted(x.dropna().unique())),
        'year': ['min', 'max', 'nunique'],
        'maturity': lambda x: round(x.mean(), 0) if x.notna().any() else np.nan,
        'Amt Issued_USD': lambda x: round(x.mean(), 2) if x.notna().any() else np.nan,
        'Cpn': 'count',
        'FC_flag': lambda x: round(x.sum(), 2) if x.notna().any() else 0
    }).reset_index()
    
    currency_summary.columns = [
        'ISO', 'currency_issuance', 'USD_bonds', 'EUR_bonds', 'GBP_bonds',
        'other_bonds', 'na_bonds', 'currency_code_issuance', 'first_year',
        'last_year', 'years_active', 'avg_maturity', 'avg_amt_issued',
        'total_bonds', 'FC_bonds'
    ]
    
    currency_summary = currency_summary[[
        'ISO', 'currency_issuance', 'currency_code_issuance', 'avg_maturity',
        'avg_amt_issued', 'first_year', 'last_year', 'years_active',
        'total_bonds', 'FC_bonds', 'USD_bonds', 'EUR_bonds', 'GBP_bonds',
        'other_bonds', 'na_bonds'
    ]]
    
    currency_summary['num_currencies'] = currency_summary['currency_issuance'].apply(
        lambda x: len([c.strip() for c in x.split(',') if c.strip()]) if pd.notna(x) and x else 0
    )
    
    for col, pct_col in [('USD_bonds', 'USD_percentage'), ('EUR_bonds', 'EUR_percentage'),
                          ('GBP_bonds', 'GBP_percentage'), ('other_bonds', 'other_percentage'),
                          ('FC_bonds', 'FC_percentage'), ('na_bonds', 'na_percentage')]:
        currency_summary[pct_col] = (currency_summary[col] / currency_summary['total_bonds'] * 100).round(1)
    
    # Dominant currency percentage (USD + EUR + GBP)
    currency_summary['dominant_percentage'] = (
        currency_summary[['USD_bonds', 'EUR_bonds', 'GBP_bonds']].sum(axis=1) / 
        currency_summary['total_bonds'] * 100
    ).round(1)
    
    return currency_summary


def get_currency_iso_groups(currency_summary):
    """
    Get ISO groupings based on currency dominance.
    
    Parameters:
        currency_summary (DataFrame): Output from create_currency_summary
        
    Returns:
        dict: Dictionary with ISO lists for each currency group
    """
    return {
        'FC_dominant': currency_summary[currency_summary['FC_percentage'] >= 50]['ISO'].tolist(),
        'USD_dominant': currency_summary[currency_summary['USD_percentage'] >= 50]['ISO'].tolist(),
        'dominant_currency': currency_summary[currency_summary['dominant_percentage'] >= 50]['ISO'].tolist(),
        'other_currency': currency_summary[currency_summary['other_percentage'] >= 50]['ISO'].tolist()
    }


def calculate_coupon_summary_2011_2019(data, exclude_isos=None):
    """
    Calculate coupon averages by currency type for 2011-2019 period.
    
    Parameters:
        data (DataFrame): Processed rating data
        exclude_isos (list): ISOs to exclude (e.g., ['VEN', 'ARG'])
        
    Returns:
        DataFrame: Coupon summary by ISO and currency type
    """
    if exclude_isos is None:
        exclude_isos = ['VEN', 'ARG']
    
    cpn_avg = data[
        (data['year'] >= 2011) & 
        (data['year'] <= 2019) & 
        (~data['ISO'].isin(exclude_isos)) & 
        (data['maturity'] == 10)
    ].copy()
    
    # Fix Seychelles blue bond currency
    cpn_avg.loc[cpn_avg['ISO'] == 'SYC', 'currency_issued'] = 'USD'
    
    cpn_summary = cpn_avg.groupby('ISO', group_keys=False).apply(
        lambda g: pd.Series({
            'cpn_FC': g.loc[g['FC_flag'] == 1, 'Cpn'].mean(),
            'cpn_LC': g.loc[g['FC_flag'] == 0, 'Cpn'].mean(),
            'cpn_USD': g.loc[g['currency_issued'].eq('USD'), 'Cpn'].mean(),
            'cpn_dominant': g.loc[g['currency_issued'].isin(DOMINANT_CURRENCIES), 'Cpn'].mean(),
            'cpn_other': g.loc[~g['currency_issued'].isin(DOMINANT_CURRENCIES) & g['currency_issued'].notna(), 'Cpn'].mean(),
            'cpn_missing': g.loc[g['currency_issued'].isna(), 'Cpn'].mean(),
        })
    ).reset_index()
    
    return cpn_summary


def load_and_process_all(verbose=False):
    """
    Load all data and process for modeling.

    Parameters:
        verbose (bool): If True, print progress messages

    Returns:
        dict: Dictionary containing all processed data and summaries
    """
    all_data = load_all_data(verbose=verbose)
    processed_data = process_rating_data(all_data)

    if verbose:
        print(f"\nProcessed data summary:")
        print(f"  - Total records: {len(processed_data)}")
        print(f"  - Unique countries: {processed_data['ISO'].nunique()}")
        print(f"  - Year range: {processed_data['year'].min()} - {processed_data['year'].max()}")

    coupon_pivot, dominant_fc_isos = analyze_coupon_by_currency(processed_data, min_year=2011)
    currency_summary = create_currency_summary(processed_data, min_year=2000)
    currency_iso_groups = get_currency_iso_groups(currency_summary)
    cpn_summary_2011_2019 = calculate_coupon_summary_2011_2019(processed_data)

    if verbose:
        print(f"\nCurrency analysis summary:")
        print(f"  - FC-dominant ISOs: {len(currency_iso_groups['FC_dominant'])}")
        print(f"  - USD-dominant ISOs: {len(currency_iso_groups['USD_dominant'])}")
        print(f"  - Dominant currency ISOs: {len(currency_iso_groups['dominant_currency'])}")
        print(f"  - Other currency ISOs: {len(currency_iso_groups['other_currency'])}")

    return {
        'all_data': all_data,
        'processed_data': processed_data,
        'coupon_pivot': coupon_pivot,
        'dominant_fc_isos': dominant_fc_isos,
        'currency_summary': currency_summary,
        'currency_iso_groups': currency_iso_groups,
        'cpn_summary_2011_2019': cpn_summary_2011_2019
    }


# =============================================================================
# MODEL TRAINING DATA PREPARATION
# =============================================================================

# Non-rated sovereign ISOs (countries without credit ratings from any major agency)
NONRATED_ISOS = [
    "AFG", "DZA", "ATG", "BTN", "BRN", "BDI", "CAF", "TCD", "COM", "DJI", "DMA", "GNQ",
    "ERI", "GRD", "GIN", "GNB", "GUY", "HTI", "KIR", "LBR", "MKD", "MWI", "MHL", "MRT",
    "FSM", "MCO", "MMR", "NRU", "NPL", "PRK", "PLW", "WSM", "STP", "SLE", "SOM", "SSD",
    "KNA", "LCA", "SDN", "SYR", "GMB", "TLS", "TON", "TUV", "VUT", "YEM", "ZWE"
]

# Model feature specification
INDEPENDENT_VARS = [
    'Current account balance',
    'General government net lending/borrowing',
    'gross debt ratio',
    'log(GDP pcap)',
    'GDP growth rate',
    'deficit',
    'log_reserves_usd',
    'fdi',
    'inflation',
    'Unemployment rate',
    'default_total',
    'govexp_GDP',
    'govrev_GDP'
]

WGI_VARS = ['VA', 'PS', 'RQ', 'GE', 'RL', 'CC']
X_VARS = INDEPENDENT_VARS + WGI_VARS + ['OECD']


def prepare_training_data(all_data, agencies=False):
    """
    Prepare data for XGB model training: load processed bond+rating data,
    merge with GDP/debt observations, create region/group dummies, filter.

    Parameters:
        all_data (dict): Output from load_all_data()
        agencies (bool): also load the per-agency scores (proprietary; only the
            S&P / Fitch / Moody's model variants need them)

    Returns:
        DataFrame: Training-ready panel data
    """
    observed_gdp = pd.read_csv(PATHS['gdp_pcap']).rename(columns={"value": "log(GDP pcap)"}).drop(columns=['Country.Name'])
    observed_gdp['log(GDP pcap)'] = np.log(observed_gdp['log(GDP pcap)'])
    observed_debt = pd.read_csv(PATHS['gdd_debt'])[['ISO', 'year', 'public_debt_pp']].rename(
        columns={'public_debt_pp': 'gross debt ratio'})
    observed = pd.merge(observed_gdp, observed_debt, on=['ISO', 'year'], how='left')

    isomaster = all_data['isomaster']

    data = load_rating_panel(agencies=agencies)
    data.columns.name = None
    data = data.rename(columns={
        'Gross domestic product, constant prices': 'GDP growth rate',
        'Inflation, average consumer prices': 'inflation',
    })

    data = data.merge(observed, on=['ISO', 'year'], how='left')
    data = data.merge(isomaster, on='ISO', how='left')

    for region in ['Asia', 'Africa', 'Americas', 'Oceania']:
        data[f'Region_{region}'] = (data['Region.Name'] == region).astype(int)

    data['OECD'] = data['ISO'].isin(COUNTRY_GROUPS['OECD']).astype(int)
    data['OPEC'] = data['ISO'].isin(COUNTRY_GROUPS['OPEC']).astype(int)

    data = data[data['inflation'] <= 100]
    data = data[data['sovereign'] == 1]
    data = data[data['year'] <= 2022]

    return data


def prepare_model_splits(data, score_col='score_total', x_vars=None):
    """
    Prepare X, y, and index arrays for a given rating score column.
    Collapses low-frequency rating classes (CCC- and below) into one bin.

    Parameters:
        data (DataFrame): Output from prepare_training_data()
        score_col (str): Which rating score to use ('score_total', 'score_sp', etc.)
        x_vars (list): Feature columns. Defaults to X_VARS.

    Returns:
        dict with keys: 'df', 'X', 'y', 'ISO_index', 'year_index'
    """
    if x_vars is None:
        x_vars = X_VARS

    scores = [c for c in ['score_sp', 'score_fitch', 'score_moodys', 'score_total'] if c in data.columns]
    df = data[['ISO', 'year', 'Region.Name', 'Sub.region.Name'] + scores + x_vars].copy()

    if score_col == 'score_total':
        df['score_adj'] = df[score_col].apply(lambda x: 4 if x in [1, 2, 3] else x)
    else:
        df['score_adj'] = df[score_col].apply(lambda x: max(x, 3))

    df = df.dropna(subset=[score_col])

    return {
        'df': df,
        'X': df[x_vars],
        'y': df['score_adj'],
        'ISO_index': df['ISO'],
        'year_index': df['year'],
    }


def prepare_counterfactual_data(model_split, cf_path=None, nonrated_ISOs=None, train_data=None, x_vars=None):
    """
    Build observed 2019 and counterfactual feature matrices for rated and non-rated countries.

    Parameters:
        model_split (dict): Output from prepare_model_splits()
        cf_path (str): Path to counterfactual CSV. Defaults to PATHS['cf_data'].
        nonrated_ISOs (list): List of non-rated ISO codes. Defaults to NONRATED_ISOS.
        train_data (DataFrame): Full training data (for non-rated countries).
                                If None, uses model_split['df'].
        x_vars (list): Feature columns. Defaults to X_VARS.

    Returns:
        dict with keys: 'X_obs', 'X_cf', 'X_obs_nr', 'X_cf_nr',
                        'obs_rating', 'iso_index', 'iso_index_nr'
    """
    if cf_path is None:
        cf_path = PATHS['cf_data']
    if nonrated_ISOs is None:
        nonrated_ISOs = NONRATED_ISOS
    if x_vars is None:
        x_vars = X_VARS
    if train_data is None:
        train_data = model_split['df']

    cf_data_raw = pd.read_csv(cf_path)
    # Two accepted inputs:
    #   attributable_shares.csv: 'debt_cf_2019', 'gdp_pcap_cf_2019' (combined TC+temp counterfactual);
    #   cf_gdp_debtratio.csv: '1990_cf_*' columns (TC-only debt, combined GDP).
    if 'debt_cf_2019' in cf_data_raw.columns and 'gdp_pcap_cf_2019' in cf_data_raw.columns:
        cf_data = cf_data_raw[['ISO', 'debt_cf_2019', 'gdp_pcap_cf_2019']].copy()
        cf_format = 'attributable_shares'
    else:
        cf_data = cf_data_raw[cf_data_raw['year'] == 2019][[
            'ISO', '1990_cf_combined_debtratio', '1990_cf_TC_debtratio', '1990_cf_temp_debtratio',
            '1990_cf_combined_gdp', '1990_cf_TC_gdp', '1990_cf_temp_gdp'
        ]]
        cf_format = 'probit_legacy'

    df = model_split['df']

    def _apply_cf(X_df0):
        X_df1 = X_df0.copy().merge(cf_data, on='ISO', how='left')
        if cf_format == 'attributable_shares':
            X_df1['gross debt ratio'] = X_df1['debt_cf_2019']
            X_df1['log(GDP pcap)']    = np.log(X_df1['gdp_pcap_cf_2019'])
            drop_cols = ['debt_cf_2019', 'gdp_pcap_cf_2019']
        else:
            X_df1['gross debt ratio'] = X_df1['1990_cf_TC_debtratio']
            X_df1['log(GDP pcap)']    = np.log(X_df1['1990_cf_combined_gdp'])
            drop_cols = [c for c in X_df1.columns if c.startswith('1990_cf_')]
        return X_df1.drop(columns=['ISO'] + drop_cols)

    # Rated countries (2019 slice)
    X_df0 = df[df['year'] == 2019][x_vars + ['ISO']]
    X_df1 = _apply_cf(X_df0)

    obs_rating = df[df['year'] == 2019][model_split['y'].name if hasattr(model_split['y'], 'name') else 'score_total']

    # Non-rated countries (2019 slice)
    X_df0_nr = train_data[train_data['ISO'].isin(nonrated_ISOs)]
    X_df0_nr = X_df0_nr[X_df0_nr['year'] == 2019][x_vars + ['ISO']]
    X_df1_nr = _apply_cf(X_df0_nr)

    return {
        'X_obs': X_df0,
        'X_cf': X_df1,
        'X_obs_nr': X_df0_nr,
        'X_cf_nr': X_df1_nr,
        'obs_rating': obs_rating,
        'iso_index': X_df0['ISO'],
        'iso_index_nr': X_df0_nr['ISO'],
    }




# =============================================================================
# QUADRATIC FIT FUNCTIONS (coupon-rating curves)
# =============================================================================

def calculate_quadratic_fit_points(data, start_year, end_year=None, score_column='score_sp',
                                   liquid_currencies=None, max_coupon=30,
                                   score_range=None, num_points=100,
                                   maturity=10):
    """
    Calculate points along quadratic fit lines for bond coupon vs rating data.
    Fits separate curves for USD, liquid (dominant) currencies, and local currencies.

    Parameters:
        data (DataFrame): Bond data with columns: year, Cpn, score_column, currency_issued
        start_year (int): Starting year filter
        end_year (int, optional): Ending year filter
        score_column (str): Column with credit score ratings
        liquid_currencies (list): Liquid currencies to group. Default: ['USD', 'GBP', 'EUR']
        max_coupon (float): Maximum coupon rate to include
        score_range (tuple, optional): (min, max) score range for fit points
        num_points (int): Number of fit points (used only for float ranges)
        maturity (int or None): Bond maturity (years) to include. Default 10
            (consistent with the regression panels). Pass None to include all
            maturities (e.g., for sub-10y / 10y+ comparisons).

    Returns:
        dict: 'fit_points_df' (DataFrame with score, USD, dominant_currency, local_currency),
              'usd_coefficients', 'liquid_coefficients', 'other_coefficients',
              'liquid_data_count', 'other_data_count', 'total_data_count', 'year_range'
    """
    if liquid_currencies is None:
        liquid_currencies = ['USD', 'GBP', 'EUR']

    if end_year is None:
        mask = (data['year'] == start_year) & (data['Cpn'] < max_coupon)
        year_label = str(start_year)
    else:
        mask = (data['year'] >= start_year) & (data['year'] <= end_year) & (data['Cpn'] < max_coupon)
        year_label = f"{start_year}-{end_year}"
    if maturity is not None and 'maturity' in data.columns:
        mask &= (data['maturity'] == maturity)

    predict_df = data[mask].dropna(subset=[score_column, 'Cpn'])
    if len(predict_df) == 0:
        print(f"No data available for year(s) {year_label}.")
        return None

    # Separate by currency type
    usd_data = predict_df[predict_df['currency_issued'] == 'USD']
    liquid_data = predict_df[predict_df['currency_issued'].isin(liquid_currencies)]
    other_data = predict_df[~predict_df['currency_issued'].isin(liquid_currencies)]

    usd_coefficients = None
    liquid_coefficients = None
    other_coefficients = None

    if len(usd_data) >= 3:
        usd_coefficients = np.polyfit(usd_data[score_column], usd_data['Cpn'], 2)
    if len(liquid_data) >= 3:
        liquid_coefficients = np.polyfit(liquid_data[score_column], liquid_data['Cpn'], 2)
    if len(other_data) >= 3:
        other_coefficients = np.polyfit(other_data[score_column], other_data['Cpn'], 2)

    # Calculate fit points
    fit_points_df = pd.DataFrame()

    if len(predict_df) > 0:
        if score_range is not None:
            min_score, max_score = score_range
            if min_score == int(min_score) and max_score == int(max_score):
                x_range = np.arange(int(min_score), int(max_score) + 1)
            else:
                x_range = np.linspace(min_score, max_score, num_points)
        else:
            x_range = np.linspace(predict_df[score_column].min(),
                                  predict_df[score_column].max(), num_points)

        fit_points_df[score_column] = x_range
        fit_points_df['USD'] = np.polyval(usd_coefficients, x_range) if usd_coefficients is not None else np.nan
        fit_points_df['dominant_currency'] = np.polyval(liquid_coefficients, x_range) if liquid_coefficients is not None else np.nan
        fit_points_df['local_currency'] = np.polyval(other_coefficients, x_range) if other_coefficients is not None else np.nan

    return {
        'fit_points_df': fit_points_df,
        'usd_coefficients': usd_coefficients,
        'liquid_coefficients': liquid_coefficients,
        'other_coefficients': other_coefficients,
        'liquid_data_count': len(liquid_data),
        'other_data_count': len(other_data),
        'total_data_count': len(predict_df),
        'year_range': year_label,
    }


def predict_coupon_from_rating(score, fit_results, currency_type='USD'):
    """
    Predict coupon rate from a credit rating score using fitted quadratic curve.

    Parameters:
        score (float or array): Credit rating score(s)
        fit_results (dict): Output from calculate_quadratic_fit_points()
        currency_type (str): 'USD', 'liquid'/'dominant_currency', or 'other'/'local_currency'

    Returns:
        float or array: Predicted coupon rate(s)
    """
    key_map = {
        'USD': 'usd_coefficients',
        'liquid': 'liquid_coefficients',
        'dominant_currency': 'liquid_coefficients',
        'other': 'other_coefficients',
        'local_currency': 'other_coefficients',
    }
    coeff_key = key_map.get(currency_type)
    if coeff_key is None or fit_results.get(coeff_key) is None:
        raise ValueError(f"No fit available for currency_type='{currency_type}'")
    return np.polyval(fit_results[coeff_key], score)


def generate_predicted_coupon_csv(data, start_year=2011, end_year=2019,
                                  score_column='score_total',
                                  liquid_currencies=None,
                                  exclude_currencies=None,
                                  save=False, maturity=None):
    """
    Generate predicted coupon rates by currency type for a given period.
    Reproduces predicted_coupon_2011_2019.csv with the defaults (bonds of all
    maturities; pass maturity=10 to restrict to 10-year bonds).

    Parameters:
        data (DataFrame): Processed bond/rating data
        start_year (int): Start year
        end_year (int): End year
        score_column (str): Rating score column
        liquid_currencies (list): Dominant currencies. Default: ['USD', 'GBP', 'EUR', 'CHF', 'CAD']
        exclude_currencies (list): Currencies to exclude. Default: ['JPY']
        save (bool): If True, save to PATHS['predicted_cpn_2011_2019']
        maturity (int or None): bond maturity to restrict to; None = all maturities

    Returns:
        DataFrame: Predicted coupon rates with columns: score_column, USD, dominant_currency, local_currency
    """
    if liquid_currencies is None:
        liquid_currencies = ['USD', 'GBP', 'EUR', 'CHF', 'CAD']
    if exclude_currencies is None:
        exclude_currencies = ['JPY']

    data_adj = data[~data['currency_issued'].isin(exclude_currencies)]

    results = calculate_quadratic_fit_points(
        data_adj, start_year, end_year,
        score_column=score_column,
        liquid_currencies=liquid_currencies,
        max_coupon=30,
        score_range=(2, 21),
        maturity=maturity,
    )

    if results is None:
        print("ERROR: Could not generate predicted coupon data.")
        return None

    fit_points_df = results['fit_points_df']

    if save:
        fit_points_df.to_csv(PATHS['predicted_cpn_2011_2019'], index=False)
        print(f"  Saved: {PATHS['predicted_cpn_2011_2019']}")

    return fit_points_df


# =============================================================================
# PLOTTING FUNCTIONS (coupon-rating curves)
# =============================================================================

def plot_bond_coupon_vs_rating(data, start_year, end_year=None, score_column='score_sp',
                              liquid_currencies=['USD', 'GBP', 'EUR'],
                              max_coupon=30, figsize=(12, 8),
                              text_labels_full=None, show_stats=False,
                              show_usd_fit=True, save_path=None, maturity=None,
                              maturity_op='==', clip_fit_to_data=False,
                              title_suffix=None, panel_label=None):
    """
    Plot bond coupon rates vs credit ratings with quadratic fits for different currency groups.

    Parameters:
        data (DataFrame): Bond data with columns: year, Cpn, score_column, currency_issued
        start_year (int): Starting year filter
        end_year (int, optional): Ending year filter
        score_column (str): Column with credit score ratings
        liquid_currencies (list): Liquid currencies to group
        max_coupon (float): Maximum coupon rate to include
        figsize (tuple): Figure size
        text_labels_full (list): Rating labels for x-axis
        show_stats (bool): Whether to print R-squared values
        show_usd_fit (bool): Whether to show USD quadratic fit
        save_path (str, optional): save the figure here (before plt.show()).
        maturity (int or None): bond maturity (years) to restrict to. Default
            None = all maturities, which is what the published Fig. S26 shows.
            Applied to the scatter AND the fit, so the curves are always fitted
            on the points they are drawn over.
        maturity_op (str): how `maturity` is applied - '==' (default), '<=' or '>'.
        title_suffix (str): appended inside the title's parentheses, e.g.
            'all maturities'. Overrides the automatic maturity text.
        panel_label (str): single-letter panel tag drawn top-left, for composing
            this plot as one panel of a multi-panel figure.
        clip_fit_to_data (bool): draw each fitted curve only across the rating
            range where that currency group actually has observations, instead
            of extrapolating across the whole CC-AAA axis.

    Returns:
        dict: fit_points_df, coefficients, counts, R-squared values
    """
    if text_labels_full is None:
        text_labels_full = RATING_LABELS['full']

    if end_year is None:
        mask = (data['year'] == start_year) & (data['Cpn'] < max_coupon)
        year_label = str(start_year)
    else:
        mask = (data['year'] >= start_year) & (data['year'] <= end_year) & (data['Cpn'] < max_coupon)
        year_label = f"{start_year}-{end_year}"
    # Maturity restriction applies to both the scatter and the fit
    if maturity is not None and 'maturity' in data.columns:
        if maturity_op == '<=':
            mask &= (data['maturity'] <= maturity)
        elif maturity_op == '>':
            mask &= (data['maturity'] > maturity)
        else:
            mask &= (data['maturity'] == maturity)
    predict_df = data[mask].dropna(subset=[score_column, 'Cpn'])

    if len(predict_df) == 0:
        print(f"No data available for year(s) {year_label} with the given filters.")
        return None

    fit_results = calculate_quadratic_fit_points(data, start_year, end_year, score_column,
                                                liquid_currencies, max_coupon, score_range=(2, 21),
                                                maturity=maturity if maturity_op == '==' else None)
    if fit_results is None:
        return None

    fit_points_df = fit_results['fit_points_df']
    usd_coefficients = fit_results['usd_coefficients']
    liquid_coefficients = fit_results['liquid_coefficients']
    other_coefficients = fit_results['other_coefficients']

    usd_data = predict_df[predict_df['currency_issued'] == 'USD']
    liquid_data = predict_df[predict_df['currency_issued'].isin(liquid_currencies)]
    other_data = predict_df[~predict_df['currency_issued'].isin(liquid_currencies)]

    results = {}

    if len(usd_data) >= 3:
        usd_coefficients = np.polyfit(usd_data[score_column], usd_data['Cpn'], 2)
        usd_data = usd_data.copy()
        usd_data['predicted_Cpn'] = np.polyval(usd_coefficients, usd_data[score_column])

    if len(liquid_data) >= 3:
        liquid_coefficients = np.polyfit(liquid_data[score_column], liquid_data['Cpn'], 2)
        liquid_data = liquid_data.copy()
        liquid_data['predicted_Cpn'] = np.polyval(liquid_coefficients, liquid_data[score_column])
    else:
        liquid_coefficients = None
        liquid_data = liquid_data.copy() if len(liquid_data) > 0 else pd.DataFrame()

    if len(other_data) >= 3:
        other_coefficients = np.polyfit(other_data[score_column], other_data['Cpn'], 2)
        other_data = other_data.copy()
        other_data['predicted_Cpn'] = np.polyval(other_coefficients, other_data[score_column])
    else:
        other_coefficients = None
        other_data = other_data.copy() if len(other_data) > 0 else pd.DataFrame()

    np.random.seed(42)
    if len(usd_data) > 0:
        usd_data['score_jittered'] = usd_data[score_column] + np.random.normal(0, 0.15, len(usd_data))
    if len(liquid_data) > 0:
        liquid_data['score_jittered'] = liquid_data[score_column] + np.random.normal(0, 0.15, len(liquid_data))
    if len(other_data) > 0:
        other_data['score_jittered'] = other_data[score_column] + np.random.normal(0, 0.15, len(other_data))

    plt.figure(figsize=figsize)

    if len(other_data) > 0:
        sns.scatterplot(x=other_data['score_jittered'], y=other_data['Cpn'],
                        color='pink', alpha=0.4, s=40, label='Other currency bonds',
                        edgecolor='white', linewidth=0.3)

    if len(liquid_data) > 0:
        liquid_label = f"{'/'.join(liquid_currencies)} bonds"
        sns.scatterplot(x=liquid_data['score_jittered'], y=liquid_data['Cpn'],
                        color='blue', alpha=0.5, s=40, label=liquid_label,
                        edgecolor='white', linewidth=0.3)

    if len(usd_data) > 0:
        sns.scatterplot(x=liquid_data['score_jittered'], y=liquid_data['Cpn'],
                        color='green', alpha=0.5, s=40, label='USD bonds',
                        edgecolor='white', linewidth=0.3)

    # Curves are quadratic fits of the scattered points; fit_points_df gives only the x grid.
    if len(fit_points_df) > 0 and not fit_points_df[score_column].isna().all():
        x_grid = fit_points_df[score_column].values

        def _curve(coeffs, group_df):
            """(x, y) for one group's fitted curve, optionally clipped to the
            rating range where that group actually has observations."""
            if coeffs is None:
                return None, None
            x = x_grid
            if clip_fit_to_data and len(group_df) > 0:
                lo, hi = group_df[score_column].min(), group_df[score_column].max()
                x = x[(x >= lo) & (x <= hi)]
                if len(x) == 0:
                    return None, None
            return x, np.polyval(coeffs, x)

        x_other,  other_line  = _curve(other_coefficients,  other_data)
        x_liquid, liquid_line = _curve(liquid_coefficients, liquid_data)
        x_usd,    usd_line    = _curve(usd_coefficients,    usd_data)

        if other_line is not None:
            plt.plot(x_other, other_line, color='pink', linewidth=2.5,
                     label='Quadratic fit (other currencies)', linestyle='-', alpha=0.8)

        if liquid_line is not None:
            liquid_fit_label = f"Quadratic fit ({'/'.join(liquid_currencies)})"
            plt.plot(x_liquid, liquid_line, color='darkblue', linewidth=2.5,
                     label=liquid_fit_label, linestyle='-', alpha=0.9)

        if show_usd_fit and usd_line is not None:
            plt.plot(x_usd, usd_line, color='green', linewidth=2.5,
                     label='Quadratic fit (USD)', linestyle='-', alpha=0.9)
            plt.scatter(x_usd, usd_line, color='darkgreen', s=60,
                        marker='o', alpha=0.8, edgecolor='white', linewidth=1,
                        label='USD fit points', zorder=10)

        if liquid_line is not None:
            plt.scatter(x_liquid, liquid_line, color='navy', s=60,
                        marker='s', alpha=0.8, edgecolor='white', linewidth=1,
                        label='_nolegend_', zorder=10)

        if other_line is not None:
            plt.scatter(x_other, other_line, color='darkred', s=60,
                        marker='^', alpha=0.8, edgecolor='white', linewidth=1,
                        label='_nolegend_', zorder=10)

    try:
        bbb_minus_idx = text_labels_full.index('BBB-') + 2
        plt.axvline(x=bbb_minus_idx, color='red', linestyle='--',
                    alpha=0.7, linewidth=2, label='BBB- threshold')
    except (ValueError, NameError):
        pass

    plt.gca().invert_xaxis()
    tick_positions = list(range(2, 2 + len(text_labels_full)))
    plt.xticks(ticks=tick_positions[:len(text_labels_full)], labels=text_labels_full,
               rotation=45, ha='right')
    plt.xlabel('Credit Rating')
    plt.ylabel('Annual Coupon (%)')
    if title_suffix is not None:
        _mat = f', {title_suffix}'
    else:
        _mat = '' if maturity is None else (
            f', maturity {maturity_op} {maturity}y' if maturity_op in ('<=', '>')
            else f', {maturity}y maturity')
    plt.title(f'Credit Rating vs Bond Coupon Rates by Currency '
              f'({year_label} data, Cpn<{max_coupon}%{_mat})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    if panel_label:
        plt.gcf().text(0.005, 0.985, panel_label, ha='left', va='top',
                       fontsize=22, fontweight='bold')
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    fit_results = calculate_quadratic_fit_points(data, start_year, end_year, score_column,
                                                liquid_currencies, max_coupon, score_range=(2, 21),
                                                maturity=maturity if maturity_op == '==' else None)
    if fit_results is None:
        return None

    results.update({
        'liquid_data_count': len(liquid_data),
        'other_data_count': len(other_data),
        'total_data_count': len(predict_df),
        'year_range': year_label,
        'fit_points_df': fit_points_df,
        'liquid_coefficients': liquid_coefficients,
        'other_coefficients': other_coefficients
    })

    print(f"USD bonds: {len(usd_data)} data points")
    print(f"Liquid currency bonds ({'/'.join(liquid_currencies)}): {len(liquid_data)} data points")
    print(f"Other currency bonds: {len(other_data)} data points")
    print(f"Total data points: {len(predict_df)}")
    print(f"Year range: {year_label}")

    if show_stats:
        def r_squared(y_actual, y_predicted):
            if len(y_actual) == 0:
                return np.nan
            ss_res = np.sum((y_actual - y_predicted) ** 2)
            ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        if usd_coefficients is not None:
            usd_data_temp = usd_data.copy()
            usd_data_temp['predicted_Cpn'] = np.polyval(usd_coefficients, usd_data_temp[score_column])
            usd_r2 = r_squared(usd_data_temp['Cpn'], usd_data_temp['predicted_Cpn'])
            results['usd_r2'] = usd_r2
            print(f"USD bonds R-squared: {usd_r2:.4f}")

        if liquid_coefficients is not None:
            liquid_data_temp = liquid_data.copy()
            liquid_data_temp['predicted_Cpn'] = np.polyval(liquid_coefficients, liquid_data_temp[score_column])
            liquid_r2 = r_squared(liquid_data_temp['Cpn'], liquid_data_temp['predicted_Cpn'])
            results['liquid_r2'] = liquid_r2
            print(f"{'/'.join(liquid_currencies)} R-squared: {liquid_r2:.4f}")

        if other_coefficients is not None:
            other_data_temp = other_data.copy()
            other_data_temp['predicted_Cpn'] = np.polyval(other_coefficients, other_data_temp[score_column])
            other_r2 = r_squared(other_data_temp['Cpn'], other_data_temp['predicted_Cpn'])
            results['other_r2'] = other_r2
            print(f"Other currencies R-squared: {other_r2:.4f}")

    return results


def calculate_quadratic_fit_points_fc(data, start_year, end_year=None, score_column='score_sp',
                                     max_coupon=30, score_range=None, num_points=100,
                                     maturity=10):
    """
    Calculate quadratic fit points for bond coupon vs rating, separated by FC_flag.

    Parameters:
        maturity (int or None): Bond maturity (years) to include. Default 10
            (consistent with the regression panels). Pass None to include all.

    Returns:
        dict: fit_points_df (with FC_domestic, FC_foreign, FC_missing columns),
              coefficients, and counts
    """
    if end_year is None:
        mask = (data['year'] == start_year) & (data['Cpn'] < max_coupon)
        year_label = str(start_year)
    else:
        mask = (data['year'] >= start_year) & (data['year'] <= end_year) & (data['Cpn'] < max_coupon)
        year_label = f"{start_year}-{end_year}"
    if maturity is not None and 'maturity' in data.columns:
        mask &= (data['maturity'] == maturity)

    predict_df = data[mask].dropna(subset=[score_column, 'Cpn'])
    if len(predict_df) == 0:
        print(f"No data available for year(s) {year_label}.")
        return None

    fc_domestic = predict_df[predict_df['FC_flag'] == 0]
    fc_foreign = predict_df[predict_df['FC_flag'] == 1]
    fc_missing = predict_df[predict_df['FC_flag'].isna()]

    domestic_coefficients = None
    foreign_coefficients = None
    missing_coefficients = None

    if len(fc_domestic) >= 3:
        domestic_coefficients = np.polyfit(fc_domestic[score_column], fc_domestic['Cpn'], 2)
    if len(fc_foreign) >= 3:
        foreign_coefficients = np.polyfit(fc_foreign[score_column], fc_foreign['Cpn'], 2)
    if len(fc_missing) >= 3:
        missing_coefficients = np.polyfit(fc_missing[score_column], fc_missing['Cpn'], 2)

    fit_points_df = pd.DataFrame()
    if len(predict_df) > 0:
        if score_range is not None:
            min_score, max_score = score_range
            if min_score == int(min_score) and max_score == int(max_score):
                x_range = np.arange(int(min_score), int(max_score) + 1)
            else:
                x_range = np.linspace(min_score, max_score, num_points)
        else:
            x_range = np.linspace(predict_df[score_column].min(),
                                  predict_df[score_column].max(), num_points)

        fit_points_df[score_column] = x_range
        fit_points_df['FC_domestic'] = np.polyval(domestic_coefficients, x_range) if domestic_coefficients is not None else np.nan
        fit_points_df['FC_foreign'] = np.polyval(foreign_coefficients, x_range) if foreign_coefficients is not None else np.nan
        fit_points_df['FC_missing'] = np.polyval(missing_coefficients, x_range) if missing_coefficients is not None else np.nan

    return {
        'fit_points_df': fit_points_df,
        'domestic_coefficients': domestic_coefficients,
        'foreign_coefficients': foreign_coefficients,
        'missing_coefficients': missing_coefficients,
        'domestic_data_count': len(fc_domestic),
        'foreign_data_count': len(fc_foreign),
        'missing_data_count': len(fc_missing),
        'total_data_count': len(predict_df),
        'year_range': year_label
    }


def plot_bond_coupon_vs_rating_fc(data, start_year, end_year=None, score_column='score_sp',
                                  max_coupon=30, figsize=(12, 8),
                                  text_labels_full=None, show_stats=False):
    """
    Plot bond coupon rates vs credit ratings with quadratic fits by FC_flag groups.

    Returns:
        dict: fit_points_df, coefficients, counts, R-squared values
    """
    if text_labels_full is None:
        text_labels_full = RATING_LABELS['full']

    if end_year is None:
        mask = (data['year'] == start_year) & (data['Cpn'] < max_coupon)
        predict_df = data[mask].dropna(subset=[score_column, 'Cpn'])
        year_label = str(start_year)
    else:
        mask = (data['year'] >= start_year) & (data['year'] <= end_year) & (data['Cpn'] < max_coupon)
        predict_df = data[mask].dropna(subset=[score_column, 'Cpn'])
        year_label = f"{start_year}-{end_year}"

    if len(predict_df) == 0:
        print(f"No data available for year(s) {year_label} with the given filters.")
        return None

    fit_results = calculate_quadratic_fit_points_fc(data, start_year, end_year, score_column,
                                                    max_coupon, score_range=(2, 21))
    if fit_results is None:
        return None

    fit_points_df = fit_results['fit_points_df']
    domestic_coefficients = fit_results['domestic_coefficients']
    foreign_coefficients = fit_results['foreign_coefficients']
    missing_coefficients = fit_results['missing_coefficients']

    fc_domestic = predict_df[predict_df['FC_flag'] == 0]
    fc_foreign = predict_df[predict_df['FC_flag'] == 1]
    fc_missing = predict_df[predict_df['FC_flag'].isna()]

    np.random.seed(42)
    if len(fc_domestic) > 0:
        fc_domestic = fc_domestic.copy()
        fc_domestic['score_jittered'] = fc_domestic[score_column] + np.random.normal(0, 0.15, len(fc_domestic))
    if len(fc_foreign) > 0:
        fc_foreign = fc_foreign.copy()
        fc_foreign['score_jittered'] = fc_foreign[score_column] + np.random.normal(0, 0.15, len(fc_foreign))
    if len(fc_missing) > 0:
        fc_missing = fc_missing.copy()
        fc_missing['score_jittered'] = fc_missing[score_column] + np.random.normal(0, 0.15, len(fc_missing))

    plt.figure(figsize=figsize)

    if len(fc_domestic) > 0:
        sns.scatterplot(x=fc_domestic['score_jittered'], y=fc_domestic['Cpn'],
                        color='blue', alpha=0.6, s=40, label='Local currency bonds (FC_flag=0)',
                        edgecolor='white', linewidth=0.3)
    if len(fc_foreign) > 0:
        sns.scatterplot(x=fc_foreign['score_jittered'], y=fc_foreign['Cpn'],
                        color='red', alpha=0.6, s=40, label='Foreign currency bonds (FC_flag=1)',
                        edgecolor='white', linewidth=0.3)
    if len(fc_missing) > 0:
        sns.scatterplot(x=fc_missing['score_jittered'], y=fc_missing['Cpn'],
                        color='gray', alpha=0.6, s=40, label='Currency unknown (FC_flag=missing)',
                        edgecolor='white', linewidth=0.3)

    if len(fit_points_df) > 0 and not fit_points_df[score_column].isna().all():
        x_range = fit_points_df[score_column].values
        if domestic_coefficients is not None and not fit_points_df['FC_domestic'].isna().all():
            plt.plot(x_range, fit_points_df['FC_domestic'], color='darkblue', linewidth=2.5,
                     label='Quadratic fit (Local currency)', linestyle='-', alpha=0.9)
        if foreign_coefficients is not None and not fit_points_df['FC_foreign'].isna().all():
            plt.plot(x_range, fit_points_df['FC_foreign'], color='darkred', linewidth=2.5,
                     label='Quadratic fit (Foreign currency)', linestyle='-', alpha=0.9)
        if missing_coefficients is not None and not fit_points_df['FC_missing'].isna().all():
            plt.plot(x_range, fit_points_df['FC_missing'], color='black', linewidth=2.5,
                     label='Quadratic fit (Currency unknown)', linestyle='--', alpha=0.9)

    try:
        bbb_minus_idx = text_labels_full.index('BBB-') + 2
        plt.axvline(x=bbb_minus_idx, color='orange', linestyle='--',
                    alpha=0.7, linewidth=2, label='BBB- threshold')
    except (ValueError, NameError):
        pass

    plt.gca().invert_xaxis()
    tick_positions = list(range(2, 2 + len(text_labels_full)))
    plt.xticks(ticks=tick_positions[:len(text_labels_full)], labels=text_labels_full,
               rotation=45, ha='right')
    plt.xlabel('Credit Rating')
    plt.ylabel('Annual Coupon (%)')
    plt.title(f'Credit Rating vs Bond Coupon Rates by Currency Type ({year_label} data, Cpn<{max_coupon}%)')
    plt.legend(loc='upper left')
    plt.ylim(0, None)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    results = {
        'domestic_data_count': len(fc_domestic),
        'foreign_data_count': len(fc_foreign),
        'missing_data_count': len(fc_missing),
        'total_data_count': len(predict_df),
        'year_range': year_label,
        'fit_points_df': fit_points_df,
        'domestic_coefficients': domestic_coefficients,
        'foreign_coefficients': foreign_coefficients,
        'missing_coefficients': missing_coefficients
    }

    print(f"Local currency bonds (FC_flag=0): {len(fc_domestic)} data points")
    print(f"Foreign currency bonds (FC_flag=1): {len(fc_foreign)} data points")
    print(f"Currency unknown (FC_flag=missing): {len(fc_missing)} data points")
    print(f"Total data points: {len(predict_df)}")
    print(f"Year range: {year_label}")

    if show_stats:
        def r_squared(y_actual, y_predicted):
            if len(y_actual) == 0:
                return np.nan
            ss_res = np.sum((y_actual - y_predicted) ** 2)
            ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        if domestic_coefficients is not None:
            fc_domestic_temp = fc_domestic.copy()
            fc_domestic_temp['predicted_Cpn'] = np.polyval(domestic_coefficients, fc_domestic_temp[score_column])
            domestic_r2 = r_squared(fc_domestic_temp['Cpn'], fc_domestic_temp['predicted_Cpn'])
            results['domestic_r2'] = domestic_r2
            print(f"Local currency R-squared: {domestic_r2:.4f}")
        if foreign_coefficients is not None:
            fc_foreign_temp = fc_foreign.copy()
            fc_foreign_temp['predicted_Cpn'] = np.polyval(foreign_coefficients, fc_foreign_temp[score_column])
            foreign_r2 = r_squared(fc_foreign_temp['Cpn'], fc_foreign_temp['predicted_Cpn'])
            results['foreign_r2'] = foreign_r2
            print(f"Foreign currency R-squared: {foreign_r2:.4f}")
        if missing_coefficients is not None:
            fc_missing_temp = fc_missing.copy()
            fc_missing_temp['predicted_Cpn'] = np.polyval(missing_coefficients, fc_missing_temp[score_column])
            missing_r2 = r_squared(fc_missing_temp['Cpn'], fc_missing_temp['predicted_Cpn'])
            results['missing_r2'] = missing_r2
            print(f"Currency unknown R-squared: {missing_r2:.4f}")

    return results
