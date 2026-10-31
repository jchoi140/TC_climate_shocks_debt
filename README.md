# The debt burden of tropical cyclones and climate change


## Project structure

| Folder | Contents |
|---|---|
| `0_raw/` | raw input data as obtained from the source |
| `1_input/` | processed input panels |
| `2_code/` | analysis and plotting code, split into main and supplementary |
| `3_output/` | outputs written by the analysis code and read by the plotting code |
| `4_figures/` | rendered figures, split into main and supplementary  |

```
TC_climate_shocks_debt/
├── 0_raw/
│   ├── climate_exposures/
│   ├── climate_finance/
│   ├── climate_scenarios/
│   ├── credit_ratings/          
│   ├── economic_outcomes/
│   └── shapefiles/
├── 1_input/
│   ├── climate_exposures/
│   ├── counterfactual_data/
│   ├── credit_ratings/
│   └── economic_outcomes/
├── 2_code/
│   ├── common/                  
│   ├── data_prep/
│   ├── main/
│   │   ├── analysis/
│   │   │   ├── LP/
│   │   │   └── rating_model/
│   │   └── figures/
│   └── supplementary/
│       ├── analysis/
│       │   ├── LP/
│       │   ├── rating_model/
│       │   └── S_tables/
│       └── figures/
├── 3_output/
│   ├── counterfactual_data/
│   ├── debtratio/
│   ├── GDP/
│   ├── LP/
│   ├── rating_alternatives/
│   ├── S_pretrends/
│   ├── S_tables/
│   └── XGB/
├── 4_figures/
│   ├── main/
│   └── supplementary/
├── README.md
├── requirements.txt
└── .here                        
```



## Data

All main and supplementary figures are reproducible with the data included in this repository,
except supplementary figures S25-S26 and Tables T2 and T5-T8, which require agency-level or bond-level data.
Agency-level rating data and bond-level data were obtained via academic
institutional licenses for the Bloomberg Terminal and LSEG Workspace, and restrictions apply to the
distribution of this raw data. Exact query configurations necessary to download the data are
documented in `2_code/data_prep/DATA.md`.

## Pipeline

Run everything from the repository root. Install the Python packages
with `pip install -r requirements.txt` and the R packages listed under **Dependencies** below.

1. **Data prep:** `prep_data.R`
2. **Main analysis:**
   - LP: `2_code/main/analysis/LP/run_lp.R` (main model; see `2_code/main/analysis/LP_README.md`)
   - Rating model: `2_code/main/analysis/rating_model/generate_XGB_predictions.py --seeds 500`
     (seeds 0-499; about 1 min per 5 seeds on a laptop, so the full run is best done on a cluster).
3. **Main figures:** `2_code/main/figures/main_figures.ipynb` — Figs. 1-5
4. **Supplementary analysis** 
   - LP robustness variants: `2_code/main/analysis/LP/run_lp.R --all`
   - Bootstrap and interactions (Figs. S13/S14): `LP/alternatives/*.R`
   - Pre-trends / LP-DiD: `LP/S_pretrends/*.R`
   - SI tables (`S_tables/`)
   - Alternative rating models: `rating_model/alternatives_rating.py`
5. **Supplementary figures** 
   - Figs. S1-S23: `SI_figures1.ipynb`
   - Figs. S24-S33: `SI_figures2.ipynb` 
   - Fig. S1: `sankey_new.ipynb`
   - Figs. S13/S14: `LP/alternatives/2_LP_TC+TEMP_interactions.R`


## Dependencies

- **Python** (3.9.6): `pip install -r requirements.txt` 
- **R** (4.3.1):
  `install.packages(c("here", "fixest", "broom", "dplyr", "tidyr", "tidyverse", "data.table", "ggplot2", "ggh4x", "patchwork", "forcats", "countrycode", "openxlsx"))`


## License

MIT & CC BY 4.0


## Use of AI tools

Anthropic's Claude Code (Opus 5.5) assisted in
the development of code to support analysis, visualizations, and in organizing and documenting
this replication package: restructuring the code and directories, fixing file paths, adapting figure-generation scripts, running end-to-end reproduction checks, and comparing
regenerated figures with the published ones. The research design, data, analysis, and results
are the authors' own, and the authors reviewed and verified all code and outputs.
