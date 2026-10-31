"""Compute IG/SG-split country counts per rating agency for the
`tab:supp_rating_agency_coverage` table in `supplementary_dataset_coverage.tex`.

Sovereign countries only. For each agency (S&P, Moody's, Fitch, Any agency), we report two rows
(Investment grade, Speculative grade) using the country's max 2019 rating
(IG = score >= 12). A final "Non-rated" row covers sovereign countries with
no 2019 rating from any of the three agencies.

Columns:
  Total countries / SIDS / TC-exposed (>18 m/s) / TC-exposed (>cat1) /
  >1 deg C warming / First rating > 2000

Run:
    python3 build_rating_coverage_table.py
"""

from __future__ import annotations

import os
import pandas as pd

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

SIDS_ISOS = {
    "ATG", "BHS", "BRB", "BLZ", "CPV", "COM", "COK", "CUB", "DMA", "DOM",
    "FJI", "GRD", "GNB", "GUY", "HTI", "JAM", "KIR", "MDV", "MHL", "FSM",
    "MUS", "NRU", "NIU", "PLW", "PNG", "WSM", "STP", "KNA", "LCA",
    "VCT", "SYC", "SLB", "SUR", "TLS", "TON", "TTO", "TUV", "VUT",
}

IG_THRESHOLD = 12   # rating score >= 12 -> investment grade


def _load_agency(fn):
    path = os.path.join(PROJ, "0_raw/credit_ratings", fn)   # proprietary, not shipped
    if not os.path.exists(path):
        raise FileNotFoundError("Please add the required credit rating or bond data.")
    return pd.read_csv(path).dropna(subset=["value"])


def _exposure_sets():
    tc = pd.read_csv(os.path.join(PROJ, "0_raw/climate_exposures/tc2_maxws.csv"))
    tc80 = (tc[tc["year"] >= 1980]
              .groupby("ISO")
              .agg(maxts=("ts_v2", "max"),
                   maxcat1=("cat1_v2", "max"),
                   maxcat3=("cat3_v2", "max"))
              .reset_index())
    cm = pd.read_csv(os.path.join(PROJ,
        "0_raw/climate_exposures/country_master.csv")).drop_duplicates("ISO")
    europe_subs = {"Northern Europe", "Western Europe",
                   "Eastern Europe", "Southern Europe"}
    extratropical = set(cm.loc[cm["Sub.region.Name"].isin(europe_subs),
                                "ISO"].dropna())

    ts_isos   = set(tc80.loc[tc80["maxts"]   > 0, "ISO"]) - extratropical
    cat1_isos = set(tc80.loc[tc80["maxcat1"] > 0, "ISO"]) - extratropical
    cat3_isos = set(tc80.loc[tc80["maxcat3"] > 0, "ISO"]) - extratropical

    temp = pd.read_csv(os.path.join(PROJ,
        "1_input/climate_exposures/era5_v4_Tshocks.csv"))
    t19 = temp[temp["year"] == 2019].copy()
    t19["warming"] = t19["temp"] - t19["temp_noANTH"]
    warm_isos = set(t19.loc[t19["warming"] > 1.0, "ISO"].dropna())

    sov_isos = set(cm.loc[cm["sovereigncountry"] == 1, "ISO"].dropna())

    return ts_isos, cat1_isos, cat3_isos, warm_isos, sov_isos


def _row_counts(isos, ts_isos, cat1_isos, cat3_isos, warm_isos,
                 first_after_2000_isos):
    return {
        "total":   len(isos),
        "sids":    len(isos & SIDS_ISOS),
        "ts":      len(isos & ts_isos),
        "cat1":    len(isos & cat1_isos),
        "cat3":    len(isos & cat3_isos),
        "warm":    len(isos & warm_isos),
        "first2000": (len(isos & first_after_2000_isos)
                      if first_after_2000_isos is not None else None),
    }


def main():
    ts_isos, cat1_isos, cat3_isos, warm_isos, sov_isos = _exposure_sets()

    agencies = {
        "S&P":    _load_agency("sp_ratings_v3.csv"),
        "Moody's": _load_agency("moodys_ratings_v3.csv"),
        "Fitch":  _load_agency("fitch_ratings_v3.csv"),
    }
    # Sovereign countries only (drops rated territories, e.g. Bermuda, Hong Kong, Isle of Man)
    agencies = {name: df[df["ISO"].isin(sov_isos)] for name, df in agencies.items()}

    # 2019 rating per agency (max if multiple records; usually one)
    rated_2019 = {}
    first_year = {}
    for name, df in agencies.items():
        d19 = df[df["year"] == 2019].groupby("ISO")["value"].max()
        rated_2019[name] = d19
        first_year[name] = df.groupby("ISO")["year"].min()

    # Any-agency 2019: take max across agencies (countries with at least one
    # 2019 rating).
    any_2019 = (pd.concat([s.rename("v") for s in rated_2019.values()],
                           axis=0)
                  .reset_index()
                  .groupby("ISO")["v"].max())
    any_first = (pd.concat([s.rename("y") for s in first_year.values()],
                            axis=0)
                   .reset_index()
                   .groupby("ISO")["y"].min())

    rows = []
    def _add(label, isos, first_after_2000_isos):
        rows.append({"label": label,
                     **_row_counts(isos, ts_isos, cat1_isos, cat3_isos, warm_isos,
                                   first_after_2000_isos)})

    for name, df in agencies.items():
        ever = set(df["ISO"].unique())
        d19 = rated_2019[name]
        ig = set(d19.index[d19 >= IG_THRESHOLD])
        sg = set(d19.index[d19 <  IG_THRESHOLD])
        f2000 = set(first_year[name].index[first_year[name] > 2000])
        _add(f"{name} - Total (ever-rated)", ever, f2000)
        _add(f"{name} - IG", ig, f2000)
        _add(f"{name} - SG", sg, f2000)

    ever_any = set().union(*(set(df["ISO"].unique())
                              for df in agencies.values()))
    ig_any = set(any_2019.index[any_2019 >= IG_THRESHOLD])
    sg_any = set(any_2019.index[any_2019 <  IG_THRESHOLD])
    f2000_any = set(any_first.index[any_first > 2000])
    _add("Any agency - Total (ever-rated)", ever_any, f2000_any)
    _add("Any agency - IG", ig_any, f2000_any)
    _add("Any agency - SG", sg_any, f2000_any)

    # Non-rated: sovereign countries with no 2019 rating from any agency.
    rated_2019_any = set(any_2019.index)
    nr = sov_isos - rated_2019_any
    _add("Non-rated", nr, None)

    df = pd.DataFrame(rows)
    out = os.path.join(PROJ, "3_output", "S_tables",
                       "rating_agency_coverage_counts.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
