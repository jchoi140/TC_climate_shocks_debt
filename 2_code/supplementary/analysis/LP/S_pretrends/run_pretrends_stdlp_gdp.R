# =============================================================================
# Focused Standard LP runner for GDP per capita (WDI constant 2015 USD).
# Runs ONLY the "linear" specification on the "full" subset, but extends the
# horizon range to h ∈ [-10, 20] so it aligns with the LP-DiD appendix figures.
# Saves to: full_gdp_pcap_wdi_constant2015_USD_1980onwards_linear_h_extended_TC.csv
# =============================================================================

suppressPackageStartupMessages({
  library(fixest); library(dplyr); library(broom); library(tidyverse)
})

project_root <- here::here()
setwd(project_root)
path_out <- "3_output/S_pretrends/"

cat("Loading data...\n")
era5  <- read.csv("./1_input/climate_exposures/era5_v4_Tshocks.csv")
tc    <- read.csv("./1_input/climate_exposures/tc_maxws_complete.csv")
master <- read.csv("./0_raw/climate_exposures/country_master.csv")
sovereign_ISOs <- master$ISO[master$sovereign == 1]

outcome_data <- read.csv("./0_raw/economic_outcomes/gdp_pcap_wdi_constant2015_USD.csv") %>%
  dplyr::select(ISO, year, value)

panel <- era5 %>%
  rename(temp_pw = temp) %>%
  left_join(tc, by = c("ISO", "year")) %>%
  left_join(outcome_data, by = c("ISO", "year")) %>%
  filter(year >= 1980, year < 2020, ISO %in% sovereign_ISOs) %>%
  group_by(ISO) %>% arrange(ISO, year) %>%
  mutate(meantemp = meantemp_hist, temp = Tshock_NATH,
         outcome_LD1 = log(lag(value, 1)) - log(lag(value, 2)),
         outcome_LD2 = log(lag(value, 2)) - log(lag(value, 3))) %>%
  ungroup()

cat(paste0("Panel: ", n_distinct(panel$ISO), " countries, ",
           min(panel$year), "-", max(panel$year), "\n"))

leads <- 10; lags <- 20
fml_str <- paste0(
  "log_longdiff ~ l(ts_v2, 0:10) + l(cat1_v2, 0:10) + l(cat3_v2, 0:10)",
  " + temp + temp:meantemp + l(temp, 1:2) + l(temp, 1:2):meantemp",
  " + outcome_LD1 + outcome_LD2 | ISO + year + ISO[[year]]")

results <- data.frame()
for (h in c(-leads:-2, 0:lags)) {
  panel_h <- panel %>% group_by(ISO) %>%
    mutate(log_longdiff = if (h >= 0)
             log(lead(value, h)) - log(lag(value, 1))
           else
             log(lag(value, abs(h))) - log(lag(value, 1))) %>%
    ungroup()
  panel_h <- fixest::panel(panel_h, ~ ISO + year, duplicate.method = "first")
  tryCatch({
    mod <- feols(as.formula(fml_str), data = panel_h, vcov = ~ISO, notes = FALSE)
    coefs <- tidy(mod, conf.int = TRUE) %>%
      filter(term %in% c("ts_v2", "cat1_v2", "cat3_v2", "temp", "temp:meantemp")) %>%
      mutate(y_h = h)
    results <- bind_rows(results, coefs)
    cat(sprintf("  h=%d done (n=%d)\n", h, nobs(mod)))
  }, error = function(e) cat(sprintf("  h=%d FAILED: %s\n", h, e$message)))
}

results$Tshock_type <- "NATH"
results$outcome <- "gdp_pcap_wdi_constant2015_USD"
results$specification <- "1980onwards_linear_h_extended"
results$model <- "linear"
results$subset <- "full"

fn <- paste0(path_out,
  "full_gdp_pcap_wdi_constant2015_USD_1980onwards_linear_h_extended_TC.csv")
write.csv(results, fn, row.names = FALSE)
cat(paste0("\nSaved: ", fn, " (", nrow(results), " rows)\n"))
