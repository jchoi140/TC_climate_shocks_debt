################################################################################
#                     DATA PREPARATION SCRIPT                                  #
################################################################################
# INPUTS:  raw data in 0_raw/, plus 1_input/economic_outcomes/gdd_debt_v2.csv (run fill_GDD_raw.py first)
# OUTPUTS: 1_input/credit_ratings/vix.csv, 1_input/economic_outcomes/{defaults_history_SI,
#          lp_outcomes}.csv
################################################################################

library(dplyr)
library(tidyverse)
library(countrycode)
library(openxlsx)  # For reading Excel files
`%notin%` <- function(x, y) !x %in% y

setwd(here::here("1_input"))

cat("\n=== DATA PREPARATION SCRIPT STARTED ===\n")
cat("Loading and processing raw data files...\n\n")

PATHS = list(country_master = "../0_raw/climate_exposures/country_master.csv")

country_master = read.csv(PATHS$country_master)
sovereign_ISOs <- country_master %>%
  filter(sovereigncountry == 1 | 
         country =='Taiwan') %>% # since IMF GDD records Taiwan debt separately
  select(country, ISO, sovereigncountry)
sovereign_ISOs <- sovereign_ISOs$ISO

# ==============================================================================
# 1. MACRO INDICATORS: VIX and Default History
# ==============================================================================
cat("\nProcessing Macro Indicators...\n")

## 1.1 VIX Volatility Index (FRED) ---------------------------------------------
# OUTPUT: vix.csv → 1_input/credit_ratings/
# DESCRIPTION: Annual average VIX index (1990-2025)

{
  vix_raw <- read.csv("../0_raw/economic_outcomes/VIX_FRED_2025.csv")

  vix_annual <- vix_raw %>%
    mutate(observation_date = as.Date(observation_date, format = "%Y-%m-%d"),
           year = as.numeric(format(observation_date, "%Y"))) %>%
    group_by(year) %>%
    summarise(annual_avg_vix = mean(VIXCLS, na.rm = TRUE))

  write.csv(vix_annual, "credit_ratings/vix.csv", row.names = FALSE)

  cat("✓ Created: vix.csv (annual VIX averages)\n")
  cat(sprintf("  Years: %d-%d\n", min(vix_annual$year), max(vix_annual$year)))
}


## 1.2 Sovereign Default History (BoC-BoE Database) ----------------------------
# OUTPUT: defaults_history_SI.csv → 1_input/economic_outcomes/ (Fig. S default history)
# DESCRIPTION: Indicators for sovereign default events (1960-2020)

{
  # encoding = "UTF-8": keeps accented names (Curaçao, São Tomé) intact in any locale
  defaults_raw <- read.csv("../0_raw/economic_outcomes/defaults_CRAG-Database-05-07-21.csv", encoding = "UTF-8")

  defaults_clean <- defaults_raw %>%
    filter(!country %in% c("Czechoslovakia", "Yugoslavia", "")) %>%
    mutate(ISO = countrycode(country, "country.name", "iso3c")) %>%
    mutate(across(starts_with("X"), as.character)) %>%
    pivot_longer(cols = starts_with("X"),
                 names_to = "year",
                 values_to = "default_amt") %>%
    mutate(year = as.numeric(gsub("X", "", year))) %>%
    pivot_wider(names_from = "type",
                values_from = "default_amt") %>%
    rename("FC_bankloans" = ` - FC bank loans`,
           "FC_bonds" = ` - FC bonds`,
           "LC_debt" = ` - LC debt`,
           "otherprivate" = ` - Other private creditors`) %>%
    select(ISO, year, FC_bankloans, FC_bonds, LC_debt, otherprivate, total) %>%
    mutate(
      default_FC_bankloans = ifelse(FC_bankloans == "****" | suppressWarnings(as.numeric(FC_bankloans)) > 0, 1, 0),
      default_FC_bonds = ifelse(FC_bonds == "****" | suppressWarnings(as.numeric(FC_bonds)) > 0, 1, 0),
      default_LC_debt = ifelse(LC_debt == "****" | suppressWarnings(as.numeric(LC_debt)) > 0, 1, 0),
      default_otherprivate = ifelse(otherprivate == "****" | suppressWarnings(as.numeric(otherprivate)) > 0, 1, 0),
      default_total = ifelse(total == "****" | suppressWarnings(as.numeric(total)) > 0, 1, 0)
    ) %>%
    mutate(across(c(FC_bankloans, FC_bonds, LC_debt, otherprivate, total), as.numeric))

  # Add advanced economies with no default history
  missing_advanced <- c(
    "AUS", "AUT", "BEL", "BHR", "BRN", "CAN", "CHE", "CHN", "CZE", "DEU",
    "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "ISL", "ISR", "ITA", "JPN",
    "KOR", "KWT", "LVA", "LTU", "LUX", "MLT", "NLD", "NOR", "NZL", "OMN",
    "QAT", "SAU", "SGP", "SVK", "SWE", "TWN", "USA", "MYS"
  )

  defaults_full <- defaults_clean %>%
    bind_rows(
      expand.grid(
        ISO = missing_advanced,
        year = 1960:2020,
        stringsAsFactors = FALSE
      ) %>%
        mutate(
          FC_bankloans = 0, FC_bonds = 0, LC_debt = 0, otherprivate = 0, total = 0,
          default_FC_bankloans = 0, default_FC_bonds = 0, default_LC_debt = 0,
          default_otherprivate = 0, default_total = 0
        )
    ) %>%
    arrange(ISO, year)

  write.csv(defaults_full, "./economic_outcomes/defaults_history_SI.csv", row.names = FALSE)

  cat("✓ Created: defaults_history_SI.csv (full history with advanced economies)\n")
}


# ==============================================================================
# 2. GDP AND WEO DEBT
# ==============================================================================
cat("\nProcessing GDP Data...\n")

## 2.1 GDP Per Capita Constant 2015 USD (WDI) ----------------------------------
# OUTPUT: In-memory dataframe (gdp_pcap_wdi_constant2015_USD)
# DESCRIPTION: GDP per capita in constant 2015 USD

{
  wdi_gdp <- read.csv("../0_raw/economic_outcomes/wdi_gdp_constantUSDpcap_2015.csv")

  gdp_pcap_wdi_constant2015_USD <- wdi_gdp %>%
    select(-c(Series.Name, Series.Code, Country.Name)) %>%
    pivot_longer(cols = -c(Country.Code),
                 names_to = "Year",
                 values_to = "Value") %>%
    mutate(
      year = as.integer(substr(Year, 10, nchar(Year) - 1)),
      value = ifelse(Value == "..", NA, Value),
      value = as.numeric(gsub("[^0-9.-]", "", value))
    ) %>%
    select(-c(Year, Value)) %>%
    rename(ISO = Country.Code)

  cat("✓ Processed: gdp_pcap_wdi_constant2015_USD (WDI)\n")
}


## 2.2 General government gross debt (IMF WEO 2025) ---------------------------
weo2025 <- read.csv("../0_raw/economic_outcomes/WEO_2025.csv")

# ISO3 from the IMF series code (e.g. "CIV.GGXCNL.A"); Kosovo and West Bank & Gaza recoded,
# 4-character country-group codes (G001, ...) dropped.
weo2025_ISO = weo2025 %>%
  distinct(COUNTRY, ISO = sub("[.].*$", "", SERIES_CODE)) %>%
  mutate(ISO = recode(ISO, KOS = "XKX", WBG = "PSE")) %>%
  filter(grepl("^[A-Z]{3}$", ISO))

gg_debt_WEO <- weo2025 %>%
  filter(INDICATOR == "Gross debt, General government, Percent of GDP" ,
          UNIT =="Percent") %>%
  select(COUNTRY, starts_with("X")) %>%
  pivot_longer(cols = -c(COUNTRY), names_to = "Year", values_to = "Value") %>%
  mutate(
    year = as.integer(substr(Year, 2, nchar(Year))),
    value = as.numeric(gsub("[^0-9.-]", "", Value))
  ) %>%
  left_join(weo2025_ISO, by=c("COUNTRY")) %>%
  select(-c(Year, Value, COUNTRY)) %>%
  filter(ISO %in% sovereign_ISOs) 


# ==============================================================================
# 3. DEBT DATA: Global Debt Database and Global Macro Database
# ==============================================================================
cat("\nProcessing Debt Data...\n")

## 3.1 Global Debt Database (GDD) - Public & Private Debt ---------------------
# SOURCE: 1_input/economic_outcomes/gdd_debt_v2.csv (built from GDD_raw.csv by fill_GDD_raw.py)
# OUTPUT: In-memory dataframes (public_pp_GDD, private_pp_GDD, cg_debt_pp_GDD, gg_debt_pp_GDD), % of GDP

{
gdd_debt = read.csv("./economic_outcomes/gdd_debt_v2.csv")

cg_debt_pp_GDD <- gdd_debt %>% select(ISO, year, cg_data) %>% rename(value = cg_data)
gg_debt_pp_GDD <- gdd_debt %>% select(ISO, year, gg_debt_pp) %>% rename(value = gg_debt_pp)
public_pp_GDD <- gdd_debt %>% select(ISO, year, public_debt_pp) %>% rename(value = public_debt_pp)
private_pp_GDD <- gdd_debt %>% select(ISO, year, private_debt_pp) %>% rename(value = private_debt_pp)

cat("✓ Processed: public_pp_GDD, private_pp_GDD, cg_debt_pp_GDD, gg_debt_pp_GDD\n")
cat(sprintf("  Countries with data: %d\n", length(unique(gdd_debt$ISO))))
}

#### GLOBAL MACROECONOMIC DATABASE ####
gmd = read.csv("../0_raw/economic_outcomes/GMD_May2025.csv")
gg_debt_GMD = gmd %>%
  select(ISO3, year, govdebt_GDP) %>%
  rename(ISO = ISO3,
        value = govdebt_GDP)


# ==============================================================================
# 4. FISCAL DATA: Kose et al. (2022) fiscal-space database
# ==============================================================================
cat("\nProcessing Fiscal Data...\n")

## 4.1 General government gross debt (% GDP) -----------------------------------
# SOURCE: 0_raw/economic_outcomes/WB_Fiscal-space-data_Oct2022.xlsx (the vintage the published SI used)
# OUTPUT: In-memory dataframe (panel_ggdy)

{
  path <- "../0_raw/economic_outcomes/WB_Fiscal-space-data_Oct2022.xlsx"
  sheets <- openxlsx::getSheetNames(path)
  fs <- lapply(sheets, openxlsx::read.xlsx, xlsxFile = path)
  names(fs) <- sheets

  ggdy <- fs$ggdy # general government gross debt
  panel_ggdy <- ggdy %>%
    select(Country.Code, Country, matches("^\\d{4}$")) %>%
    pivot_longer(cols = matches("^\\d{4}$"),  # year columns
                 names_to = "year",
                 values_to = "value") %>%
    mutate(year = as.integer(year)) %>%
    select(c(Country.Code, Country, value, year)) %>%
    rename(ISO = Country.Code)
  cat("✓ Processed: panel_ggdy (general government debt % GDP)\n")
}


## 5. LP outcome series ---------------------------------------------------------
# OUTPUT: lp_outcomes.csv -> 1_input/economic_outcomes/
# DESCRIPTION: every outcome the local-projection runner (2_code/main/analysis/LP/run_lp.R)
#              can estimate, long format (outcome, ISO, year, value); `outcome` matches
#              model_variants_INPUT.csv.
{
  # External debt ratios (World Bank International Debt Statistics, "World" counterpart rows)
  ed <- read.csv("../0_raw/economic_outcomes/WB_IDS_external_debt_world.csv") %>%
    transmute(ISO = debtor, year, value, Series)
  ed_pp_GNI     <- ed %>% filter(Series == "External debt stocks (% of GNI)") %>% select(-Series)
  ed_pp_exports <- ed %>% filter(Series == "External debt stocks (% of exports of goods, services and primary income)") %>%
    select(-Series)

  lp_outcome_names <- c("public_pp_GDD", "private_pp_GDD", "cg_debt_pp_GDD", "gg_debt_WEO",
                        "panel_ggdy", "gg_debt_GMD", "gdp_pcap_wdi_constant2015_USD",
                        "ed_pp_GNI", "ed_pp_exports")
  lp_outcomes <- bind_rows(lapply(lp_outcome_names, function(o)
    get(o) %>% ungroup() %>% transmute(outcome = o, ISO, year, value)))
  write.csv(lp_outcomes, "./economic_outcomes/lp_outcomes.csv", row.names = FALSE)
  cat(sprintf("✓ Created: lp_outcomes.csv (%d outcomes, %d rows)\n",
              length(lp_outcome_names), nrow(lp_outcomes)))
}
