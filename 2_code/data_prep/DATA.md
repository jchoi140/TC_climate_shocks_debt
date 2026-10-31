# Data retrieval guide (Bloomberg Terminal / LSEG Workspace)

The licensed datasets below are not included in the repository. Save each one under
`0_raw/credit_ratings/` with the name given in its section; then
`2_code/data_prep/build_proprietary_aggregates.py` rebuilds the shipped aggregates in
`1_input/credit_ratings/`. Code that needs a missing raw file stops with "Please add the required
credit rating or bond data."

## 1. Sovereign credit ratings (Bloomberg Terminal)

- **Function:** `RATC <GO>` (credit rating history) on each sovereign issuer, e.g. `BRAZIL <Govt>`
  `RATC <GO>`; `CSDR <GO>` (sovereign debt ratings monitor) gives the cross-country view.
- **Data fields / API fields:** `RTG_SP`, `RTG_MOODY`, `RTG_FITCH` (long-term foreign-currency
  issuer rating) and the matching outlook fields `RTG_SP_OUTLOOK`, `RTG_MDY_OUTLOOK`,
  `RTG_FITCH_OUTLOOK`; the full history of rating changes and outlook announcements, with dates.
- **Tickers/Universe:** all sovereigns rated by S&P, Moody's or Fitch (S&P 135, Moody's 141,
  Fitch 122 countries in the shipped panel). Gaps were filled using publicly available data from Trading Economics.
- **Date range:** first available rating to 2023 (S&P from 1972, Moody's from 1974, Fitch from 1994).
- **Processing:** map each rating to the 1-21 scale in `0_raw/credit_ratings/ratingscore_key_v2.csv`
  (AAA/Aaa = 21 … C = 1; positive outlook +0.5, negative −0.5, marked `**` / `*` in the key).
  Save as `0_raw/credit_ratings/{sp,moodys,fitch}_ratings_v3.csv` with columns `ISO, year, value`.

## 2. Sovereign 10-year bond yields (Bloomberg Terminal)

- **Function:** `WB <GO>` (World Bond Markets), 10-year tenor, local-currency and USD
  sovereign curves. The output columns follow its labels (`Brazil 10Y`, `Brazil (USD) 10Y`, …).
- **Data fields / API fields:** `PX_LAST` on the generic yield ticker (mid yield to maturity, %).
- **Tickers/Universe:** 69 series in total; Columns are named `<Country> 10Y` or
  `<Country> (USD) 10Y`; 
- **Date range:** 1970-07-20 to 2024-10-18
- **Frequency:** daily (business days).
- **Processing:** Save as `0_raw/credit_ratings/bondyield_10yr.csv`: one `Date` column
  (`m/d/yy`, newest first) and one value per series.

## 3. Sovereign bond issuance (Bloomberg Terminal)

- **Function:** `SRCH <GO>` (fixed-income search), pulled region by region (Caribbean; Central and
  South America; North America; Europe; Southeast Asia).
- **Search criteria:** Issuer Industry = `GOVT NATIONAL`; Country of
  Risk = the target country; include matured bonds.
- **Data fields (export columns):** Issuer Name, Cntry of Risk, Issue Date, Maturity, Mty Type, Cpn,
  Cpn Freq Des, Amt Issued, Orig Amt Sold, Market Type / Market Issue, Currency, Yield at Issue,
  Issue Price (plus Yld to Mty (Mid), Par Amt, Bid/Ask Price).
- **Date range:** 1969-2024.
- **Frequency:** one row per bond issue.
- **Filters applied afterwards** (SI, *Sovereign bond data*): fixed coupon, bullet maturity
  (`Mty Type = AT MATURITY`), non-zero coupon, private placements dropped (`Market Type =
  PRIV PLACEMENT`); rows returning `#N/A Review` dropped. The analysis keeps 10-year maturities.
  The merged result is saved as `0_raw/credit_ratings/bloomberg/bloomberg_predict_df2.csv`.

## 4. Sovereign bond issuance (LSEG Workspace)

- **App:** Deals Screener → Bonds / New Issues (SDC Platinum new-issues database).
- **Expression / filters:** Issuer/Borrower TRBC Economic Sector = *Government Activity* (also
  government-guaranteed issues under *Financials*); Coupon Type = *Fixed Rate*; Issue Date
  2000-01-01 to 2019-12-31; all Market Areas (Domestic, Euro, International, Global).
- **Output columns:** Issue Date; Issuer/Borrower Name Full; Issuer/Borrower Nation; Proceeds
  Amount Incl Overallotment Sold This Market (USD mn); Offer Price (%); Issuer Domicile Nation;
  Issuer/Borrower TRBC Industry; Issuer/Borrower TRBC Economic Sector; Coupon: Coupon Rate;
  Coupon: Coupon Type; Coupon Payment Frequency; Maturity Type; Maturity: Maturity Date; Actual
  Number Of Years To Maturity; Market Area; Multi Currency Flag; Denominations Currency.
- **Currency:** denomination currency mapped to ISO 4217 codes with lookup table `1_input/credit_ratings/currencies.csv`.
- **Filters applied afterwards:** Maturity Type = *Bullet Maturity*; non-zero coupon; 10-year
  maturity for the coupon analysis. The merged result is saved as
  `0_raw/credit_ratings/lseg/lseg_predict_df_v2.csv`.

## Public download: IBTrACS storm tracks

Fig. 4a overlays storm tracks from IBTrACS v04r01 (public NOAA file, ~140 MB, too large for
GitHub). Download
<https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv>
and save it as `0_raw/climate_exposures/ibtracs/ibtracs.since1980.list.v04r01.csv`. Without it the
map renders without tracks.
