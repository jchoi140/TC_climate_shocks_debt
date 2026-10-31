# Local projections (LP)

LP outputs are produced by one script, `2_code/main/analysis/LP/run_lp.R`, with inputs from
`2_code/main/analysis/LP/model_variants_INPUT.csv`. 

```bash
Rscript 2_code/main/analysis/LP/run_lp.R                         # row 1 = the main model
Rscript 2_code/main/analysis/LP/run_lp.R --variant se_DK         # one row, by its `name`
Rscript 2_code/main/analysis/LP/run_lp.R --all                   # every row
```

Columns (allowed values are listed at the top of the CSV):

| Column | Meaning |
|---|---|
| `name` | short id used with `--variant` |
| `estimator` | `lp` (fixest LP) or `wild_bootstrap` (500-draw wild cluster bootstrap of the same LP) |
| `outcome` | outcome series in `1_input/economic_outcomes/lp_outcomes.csv`|
| `sample` | `1980onwards`, `1970onwards`, `1980_2022`, `unbalanced`, `40yrs_data`, `30yrs_data` |
| `subsets` | country subsets, `;`-separated (one output file each) |
| `exposure` | `bins` (tropical storm / Cat 1-3 / Cat >3) or `avg_ws` |
| `tc_lags`, `temp_controls`, `growth_lags` | 10 or 5 TC lags; temperature controls on/off; 2 or 1 lagged growth terms |
| `fixed_effects`, `time_trends` | `ISO_year` with `linear;no_trend;quadratic`, or `year_trend` / `year` / `region_year` |
| `vcov` | `cluster_ISO`, `cluster_region`, `DK` (Driscoll-Kraay), `hetero` (White), `fixest_default` |
| `tshocks` | temperature-shock definitions, `;`-separated: `NATH`, `Tshock0`, `BILAL`, `temp` |
| `out_folder` | subfolder of `3_output/debtratio/TC+Tshock/coef_df/` (`LP_coefficients`) |
| `tag` | leave empty: the file tag is built from the settings, e.g. `1980onwards_gl2_TC10_T02_DK` |
