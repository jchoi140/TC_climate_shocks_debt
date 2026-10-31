"""build_proprietary_aggregates.py — shipped aggregates of the proprietary rating / bond data.

REQUIRES THE PROPRIETARY RAW DATA in 0_raw/credit_ratings/ (see README, "Data"):
  bloomberg/bloomberg_predict_df2.csv, lseg/lseg_predict_df_v2.csv   bond issuance
  sp_ratings_v3.csv, moodys_ratings_v3.csv, fitch_ratings_v3.csv      agency rating histories


Writes to 1_input/credit_ratings/:
  predicted_coupon_2011_2019.csv  quadratic coupon-vs-rating curve, bonds issued 2011-2019 (all
                                  maturities, JPY excluded): coupon by average rating for USD,
                                  dominant-currency and local-currency bonds        -> Fig. 5
  cpn_summary_2011_2019_v2.csv    each country's mean 10-year coupon 2011-2019, by currency type
                                                                                    -> Fig. 5
  bond_currency_summary.csv       each country's bond count and currency shares (all years);
                                  splits dominant- vs local-currency issuers        -> Fig. 5
  rating_grade_2019.csv           investment (IG) / speculative (SG) grade in 2019 by the best of
                                  the three agency ratings (IG = BBB- or above); unrated countries
                                  are absent                                        -> Fig. 5, S31, T11
  rated_countries_by_source.csv   number of rated countries per year, total and per agency
                                                                                    -> Fig. S24

Run:  python3 2_code/data_prep/build_proprietary_aggregates.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import PATHS, require_proprietary  # noqa: E402
from prep import (  # noqa: E402
    load_all_data, process_rating_data, load_rating_panel, create_currency_summary,
    calculate_coupon_summary_2011_2019, generate_predicted_coupon_csv,
)

def main():
    # --- bond-level aggregates ---------------------------------------------
    processed = process_rating_data(load_all_data())   # needs the bond and agency files
    generate_predicted_coupon_csv(processed, save=True)
    calculate_coupon_summary_2011_2019(processed).to_csv(PATHS['cpn_summary_2011_2019'], index=False)
    currency = create_currency_summary(processed, min_year=0)
    currency[['ISO', 'total_bonds', 'FC_percentage', 'USD_percentage',
              'dominant_percentage', 'other_percentage']].to_csv(PATHS['bond_currency_summary'],
                                                                 index=False)

    # --- agency-rating aggregates ------------------------------------------
    best_2019 = (pd.concat([pd.read_csv(require_proprietary(k)).query("year == 2019")[['ISO', 'value']]
                            for k in ('sp_ratings', 'moodys_ratings', 'fitch_ratings')])
                 .groupby('ISO')['value'].max().dropna())
    pd.DataFrame({'ISO': best_2019.index,
                  'grade_2019': ['IG' if v >= 12 else 'SG' for v in best_2019.values]}) \
        .to_csv(PATHS['rating_grade_2019'], index=False)

    panel = load_rating_panel(agencies=True)
    counts = [panel.dropna(subset=['score_total']).drop_duplicates(['ISO', 'year', 'score_total'])
                   .groupby('year').size().rename('count').reset_index().assign(source='Total')]
    for source, col in [("S&P", 'score_sp'), ("Moody's", 'score_moodys'), ("Fitch", 'score_fitch')]:
        counts.append(panel.dropna(subset=[col]).groupby('year').size().rename('count')
                           .reset_index().assign(source=source))
    pd.concat(counts, ignore_index=True)[['year', 'source', 'count']] \
        .to_csv(PATHS['rated_countries_by_source'], index=False)

    for key in ['predicted_cpn_2011_2019', 'cpn_summary_2011_2019', 'bond_currency_summary',
                'rating_grade_2019', 'rated_countries_by_source']:
        print("wrote", os.path.relpath(PATHS[key]))


if __name__ == "__main__":
    require_proprietary('bloomberg')   # fail fast, before the slow loads
    main()
