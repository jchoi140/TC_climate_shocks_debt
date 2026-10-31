"""delta_method.py — CMIP6 ANTH/NOANTH temperature delta and the Tshock_NATH_noANTH shock.

Reads:
  - 0_raw/climate_scenarios/annual_delta_ANTHvsNOANTH.csv  (CMIP6 ANTH/NOANTH, 10 models, 1960–2020)
  - 0_raw/climate_exposures/country_master.csv
  - 1_input/climate_exposures/era5_v4_Tshocks.csv
Writes:
  - 1_input/climate_exposures/era5_v4_Tshocks.csv, in place: only the delta-dependent columns
    (delta, temp_noANTH, Tshock_NATH_noANTH) are recomputed

Methodology:
  1. Per-country slope/intercept of the ANTH−NOANTH temperature gap from
     20-year averages over 1971–1990 (base) and 1995–2014 (end), averaged
     across 10 CMIP6 models.
  2. Linear extrapolation of `delta` over years EXTEND_FROM..EXTEND_TO-1
     (default 1960..2023).
  3. `temp_noANTH = temp - delta` and `meantemp_hist = mean(temp[year<1980])`
     per ISO.
  4. Refit Nath et al. 2024 specification on the panel:
        temp_noANTH ~ L1.temp_noANTH + L2.temp_noANTH
                    + L1.temp_noANTH:meantemp_hist
                    + L2.temp_noANTH:meantemp_hist
                    | ISO + year
     using `linearmodels.PanelOLS` (entity + time effects, cluster-robust
     by ISO; matches R's `fixest::feols` to ~6 decimals).
  5. Tshock_NATH_noANTH = residual; replaces existing column.
  6. All other columns (temp, Tshock_NATH, Tshock_BILAL, Tshock_0, …) are
     passed through unchanged.
"""
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

CMIP_PATH    = "0_raw/climate_scenarios/annual_delta_ANTHvsNOANTH.csv"
COUNTRY_PATH = "0_raw/climate_exposures/country_master.csv"
TEMP_PATH    = "1_input/climate_exposures/era5_v4_Tshocks.csv"
OUT_PATH     = TEMP_PATH

EXTEND_FROM, EXTEND_TO = 1960, 2024  # exclusive end (np.arange convention)

# ===========================================================================
# 1. Country-level slope/intercept from CMIP6 ANTH−NOANTH (in C)
# ===========================================================================
T = pd.read_csv(CMIP_PATH)
T["pw_meantemp1"] -= 273.15  # ANTH historical (K → C)
T["pw_meantemp2"] -= 273.15  # NOANTH (hist-nat / DAMIP)

base = T[(T['Year'] >= 1971) & (T['Year'] <= 1990)]
end_ = T[(T['Year'] >= 1995) & (T['Year'] <= 2014)]
mb = (base.groupby(['Model', 'Country'])[['pw_meantemp1', 'pw_meantemp2']]
          .mean().reset_index()
          .rename(columns={'pw_meantemp1': 'pw1_base', 'pw_meantemp2': 'pw2_base'}))
me = (end_.groupby(['Model', 'Country'])[['pw_meantemp1', 'pw_meantemp2']]
          .mean().reset_index()
          .rename(columns={'pw_meantemp1': 'pw1_end',  'pw_meantemp2': 'pw2_end'}))
mt = mb.merge(me, on=['Model', 'Country'])

data = (mt.groupby('Country')[['pw1_base', 'pw2_base', 'pw1_end', 'pw2_end']]
          .mean().reset_index())
data['diff_base'] = data['pw1_base'] - data['pw2_base']
data['diff_end']  = data['pw1_end']  - data['pw2_end']
data['slope']     = (data['diff_end'] - data['diff_base']) / (2004 - 1980)
data['intercept'] = data['diff_base'] - data['slope'] * 1980

years = np.arange(EXTEND_FROM, EXTEND_TO)
ext = pd.DataFrame({
    'year': np.tile(years, len(data)),
    'Country': np.repeat(data['Country'].values, len(years)),
    'delta': np.concatenate([
        row['slope'] * years + row['intercept'] for _, row in data.iterrows()
    ]),
})
print(f"[delta] {len(years)} years × {ext['Country'].nunique()} countries "
      f"({years.min()}-{years.max()})")

# ===========================================================================
# 2. Map Country → ISO (manual dict + country_master names)
# ===========================================================================
country_iso_dict = {
    'Antigua and Barb.': 'ATG', 'Aruba': 'ABW', 'Bahamas': 'BHS',
    'Bosnia and Herz.': 'BIH', 'Central African Rep.': 'CAF',
    'Congo': 'COG', 'Curaçao': 'CUW', "Côte d'Ivoire": 'CIV',
    'Dem. Rep. Congo': 'COD', 'Dominican Rep.': 'DOM', 'Eq. Guinea': 'GNQ',
    'Greenland': 'GRL', 'Guernsey': 'GGY', 'Hong Kong': 'HKG',
    'Isle of Man': 'IMN', 'Jersey': 'JEY', 'Macao': 'MAC',
    'Marshall Is.': 'MHL', 'Micronesia': 'FSM', 'S. Sudan': 'SSD',
    'Serbia': 'SRB', 'Sint Maarten': 'SXM', 'Solomon Is.': 'SLB',
    'St. Kitts and Nevis': 'KNA', 'St. Vin. and Gren.': 'VCT',
    'Timor-Leste': 'TLS',
    'Trinidad and Tobago': 'TTO', 'United States of America': 'USA',
}
ciso = pd.DataFrame(list(country_iso_dict.items()), columns=['country', 'iso'])
# CMIP country names can match either the `country` or the `SOVEREIGNT` column of country_master.
cm_full = pd.read_csv(COUNTRY_PATH)
name_iso = pd.concat([
    cm_full[['ISO', 'country']].rename(columns={'country': 'name'}).dropna(),
    cm_full[['ISO', 'SOVEREIGNT']].rename(columns={'SOVEREIGNT': 'name'}).dropna(),
]).drop_duplicates(subset=['name'])

t = (ext.merge(name_iso, left_on='Country', right_on='name', how='left')
        .drop(columns=['name'])
        .merge(ciso, left_on='Country', right_on='country', how='left'))
t['ISO'] = t.apply(lambda r: r['iso'] if pd.notna(r['iso']) else r['ISO'], axis=1)
t = (t.drop(columns=['country', 'iso'])
       .query("Country != 'N. Cyprus'")
       .dropna(subset=['ISO']))
delta_iso = t.groupby(['ISO', 'year'], as_index=False)['delta'].mean()

# ===========================================================================
# 3. Swap delta into existing era5 panel; recompute temp_noANTH and meantemp_hist
# ===========================================================================
era5 = pd.read_csv(TEMP_PATH)
print(f"[input]  {TEMP_PATH}: {len(era5):,} rows; "
      f"existing delta non-null = {era5['delta'].notna().sum():,}")

era5 = era5.drop(columns=['delta']).merge(delta_iso, on=['ISO', 'year'], how='left')
era5['temp_noANTH'] = era5['temp'] - era5['delta']
era5 = era5.sort_values(['ISO', 'year']).reset_index(drop=True)
era5['meantemp_hist'] = era5.groupby('ISO')['temp'].transform(
    lambda s: s[era5.loc[s.index, 'year'] < 1980].mean()
)

# ===========================================================================
# 4. Refit Tshock_NATH_noANTH (Nath et al. 2024) with PanelOLS
# ===========================================================================
era5['L1_temp_noANTH'] = era5.groupby('ISO')['temp_noANTH'].shift(1)
era5['L2_temp_noANTH'] = era5.groupby('ISO')['temp_noANTH'].shift(2)
era5['L1_x_mt']        = era5['L1_temp_noANTH'] * era5['meantemp_hist']
era5['L2_x_mt']        = era5['L2_temp_noANTH'] * era5['meantemp_hist']

reg_cols = ['temp_noANTH', 'L1_temp_noANTH', 'L2_temp_noANTH', 'L1_x_mt', 'L2_x_mt']
reg = era5[['ISO', 'year'] + reg_cols].dropna().set_index(['ISO', 'year'])

mod = PanelOLS(
    dependent      = reg['temp_noANTH'],
    exog           = reg[['L1_temp_noANTH', 'L2_temp_noANTH', 'L1_x_mt', 'L2_x_mt']],
    entity_effects = True,
    time_effects   = True,
    drop_absorbed  = True,
)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print("\n[feols-equivalent fit] coefficients:")
print(res.params.to_string())

fitted = (res.predict(reg[['L1_temp_noANTH', 'L2_temp_noANTH', 'L1_x_mt', 'L2_x_mt']],
                      effects=True)
              .reset_index()
              .rename(columns={'predictions': 'fitted'}))
era5 = era5.merge(fitted[['ISO', 'year', 'fitted']], on=['ISO', 'year'], how='left')
era5['Tshock_NATH_noANTH'] = era5['temp_noANTH'] - era5['fitted']

# ===========================================================================
# 5. Drop helper columns, keep the input column order, write CSV
# ===========================================================================
era5 = era5.drop(columns=['L1_temp_noANTH', 'L2_temp_noANTH',
                          'L1_x_mt', 'L2_x_mt', 'fitted'])
orig_cols = pd.read_csv(TEMP_PATH, nrows=0).columns.tolist()
era5 = era5.reindex(columns=orig_cols)
era5.to_csv(OUT_PATH, index=False)

print(f"\n[saved] {OUT_PATH}")
print(f"  total rows:                       {len(era5):,}")
print(f"  delta non-null:                   {era5['delta'].notna().sum():,}")
print(f"  Tshock_NATH_noANTH non-null:      {era5['Tshock_NATH_noANTH'].notna().sum():,}")
print(f"  earliest year with delta:         {era5.loc[era5['delta'].notna(), 'year'].min()}")
print(f"  earliest year with Tshock_noANTH: "
      f"{era5.loc[era5['Tshock_NATH_noANTH'].notna(), 'year'].min()}")
