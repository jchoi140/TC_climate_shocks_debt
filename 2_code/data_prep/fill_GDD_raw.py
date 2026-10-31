"""
fill_GDD_raw.py

Builds 1_input/economic_outcomes/gdd_debt_v2.csv (debt ratios, % of GDP, by country-year) from
the IMF Global Debt Database extract 0_raw/economic_outcomes/GDD_raw.csv:

  1. Fill gaps in GDD_raw (57 values):
     - ps_data (public-sector debt) for the Dominican Republic, 1970-2022, is missing and is
       set to nfps_data (non-financial public sector) - 53 values;
     - four single-year gaps in cg_data (central-government debt) are linearly interpolated
       from the neighbouring years: Dominica 1989, Jamaica 1979, Cote d'Ivoire 1984 and 1998.
  2. Match GDD country names to ISO3 codes via 0_raw/climate_exposures/country_master.csv.
  3. Ecuador: gg_data (general government) is set to nfps_data.
  4. Debt ratios:
       gg_debt_pp      = gg_data, else cg_data
       public_debt_pp  = ps_data, else gg_debt_pp   (outcome of the main LP model)
       private_debt_pp = tot_all_data, else tot_ls_data

Run before prep_data.R, which reads gdd_debt_v2.csv.

Usage:
    python3 2_code/data_prep/fill_GDD_raw.py
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "0_raw", "economic_outcomes", "GDD_raw.csv")
MASTER = os.path.join(ROOT, "0_raw", "climate_exposures", "country_master.csv")
OUT = os.path.join(ROOT, "1_input", "economic_outcomes", "gdd_debt_v2.csv")

DATA_COLS = ["tot_all_data", "hh_all_data", "nfc_all_data", "tot_ls_data", "hh_ls_data",
             "nfc_ls_data", "ps_data", "nfps_data", "gg_data", "cg_data"]

CG_GAPS = [("Dominica", 1989), ("Jamaica", 1979), ("Cote D'Ivoire", 1984), ("Cote D'Ivoire", 1998)]

# GDD country name -> country_master name
COUNTRY_NAMES = {
    "Venezuela, Republica Bolivariana de": "Venezuela",
    "Bahamas, The": "The Bahamas",
    "St. Kitts and Nevis": "St. Kitts & Nevis",
    "St. Vincent and the Grenadines": "St. Vincent & the Grenadines",
    "Iran, I. Rep. Of": "Iran",
    "U.A.E.": "United Arab Emirates",
    "Brunei Darussalam": "Brunei",
    "Hong Kong SAR": "Hong Kong (China)",
    "Korea, Republic of": "South Korea",
    "C.A.R.": "Central African Republic",
    "Congo, Republic of": "Congo",
    "Congo, Dem. Rep. of": "Congo, DRC",
    "Gambia, The": "The Gambia",
    "Cote D'Ivoire": "Cote d'Ivoire",
    "São Tomé and Príncipe": "Sao Tome & Principe",
    "Solomon Islands": "Solomon Is.",
    "Marshall Islands, Rep.": "Marshall Is.",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Russian Federation": "Russia",
    "China, Mainland": "China",
    "Slovak Republic": "Slovakia",
    "Macedonia, FYR": "Macedonia",
    "Taiwan Province of China": "Taiwan",
}


def fill(gdd):
    gdd = gdd.copy()
    dom = (gdd["country"] == "Dominican Republic") & gdd["ps_data"].isna() & gdd["nfps_data"].notna()
    gdd.loc[dom, "ps_data"] = gdd.loc[dom, "nfps_data"]
    for country, year in CG_GAPS:
        rows = gdd["country"] == country
        prev = gdd.loc[rows & (gdd["year"] == year - 1), "cg_data"].values[0]
        nxt = gdd.loc[rows & (gdd["year"] == year + 1), "cg_data"].values[0]
        gdd.loc[rows & (gdd["year"] == year), "cg_data"] = (prev + nxt) / 2
    return gdd


def main():
    gdd = fill(pd.read_csv(RAW, encoding="utf-8-sig"))[["country", "year"] + DATA_COLS]
    gdd["country"] = gdd["country"].replace(COUNTRY_NAMES)

    # Read names as text, so blank codes stay "" (e.g. Kosovo has no ISO code here)
    master = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    isolist = master[["country", "country2", "ISO", "Region.Name", "Sub.region.Name"]].drop_duplicates()
    gdd = (gdd.merge(isolist, on="country", how="left")
              .dropna(subset=["ISO"])
              .drop_duplicates()
              .reset_index(drop=True))

    gdd.loc[gdd["ISO"] == "ECU", "gg_data"] = gdd.loc[gdd["ISO"] == "ECU", "nfps_data"]
    gdd["gg_debt_pp"] = gdd["gg_data"].fillna(gdd["cg_data"])
    gdd["public_debt_pp"] = gdd["ps_data"].fillna(gdd["gg_debt_pp"])
    gdd["private_debt_pp"] = gdd["tot_all_data"].fillna(gdd["tot_ls_data"])

    gdd.to_csv(OUT, index=False, na_rep="NA")
    print(f"wrote {OUT}: {len(gdd)} rows, {gdd['ISO'].nunique()} countries")


if __name__ == "__main__":
    main()
