# =============================================================================
# LP-DiD (debt) for the pre-trend appendix figure (pretrend_appendix_lpdid_combined_v3_debt):
#   lpdid_hi_region7_fe_onesided_{v2, targetonly, allTC}   (clean control [t, t+h])
#   lpdid_hi_region7_fe_prebuf3_{v2, targetonly, allTC}    (clean control [t-3, t+h])
# Full sample of sovereigns, 1980-2019. Output: lpdid_subsets_TC.csv
# The panel and run_lpdid() above `all_results <- data.frame()` are also sourced by
# run_lpdid_gdp.R and run_lpdid_prebuf10.R.
# =============================================================================

library(fixest)
library(dplyr)
library(broom)
library(tidyverse)

project_root <- here::here()
setwd(project_root)
path_out <- "3_output/S_pretrends/"

cat("Loading data...\n")
era5 <- read.csv("./1_input/climate_exposures/era5_v4_Tshocks.csv")
tc <- read.csv("./1_input/climate_exposures/tc_maxws_complete.csv")  # 1970-2022 TC data
master <- read.csv("./0_raw/climate_exposures/country_master.csv")
sovereign_ISOs <- master$ISO[master$sovereigncountry == 1]

outcome_data <- read.csv("./1_input/economic_outcomes/gdd_debt_v2.csv") %>%
  dplyr::select(ISO, year, value = public_debt_pp)

high_cap_rating <- c("AND","ARE","AUS","AUT","BEL","BHR","BHS","BWA","CAN","CHE",
  "CHL","CHN","CYP","CZE","DEU","DNK","ESP","EST","FIN","FRA",
  "GBR","HUN","IRL","ISL","ISR","ITA","JPN","KOR","KWT","LIE",
  "LTU","LUX","LVA","MEX","MLT","MUS","MYS","NLD","NOR","NZL",
  "OMN","POL","PRT","QAT","SAU","SGP","SMR","SVK","SVN","SWE",
  "THA","TTO","USA","ZAF")

panel <- era5 %>%
  rename(temp_pw = temp) %>%
  left_join(tc, by = c("ISO", "year")) %>%
  left_join(outcome_data, by = c("ISO", "year")) %>%
  filter(year >= 1980, year < 2020, ISO %in% sovereign_ISOs) %>%
  group_by(ISO) %>% arrange(ISO, year) %>%
  mutate(
    meantemp = meantemp_hist, temp = Tshock_NATH,
    outcome_LD1 = log(lag(value, 1)) - log(lag(value, 2)),
    outcome_LD2 = log(lag(value, 2)) - log(lag(value, 3)),
    year2 = year^2,
    subregion_year = paste0(Sub.region.Name, "_", year),
    region5_year = paste0(Region.Name, "_", year),
    region7 = case_when(
      Sub.region.Name == "Sub-Saharan Africa" ~ "Sub-Saharan Africa",
      Sub.region.Name == "Latin America and the Caribbean" ~ "Latin America & Caribbean",
      Sub.region.Name %in% c("Northern Africa","Western Asia","Central Asia") ~ "N. Africa & Middle East",
      Sub.region.Name %in% c("Eastern Asia","South-eastern Asia") ~ "East & SE Asia",
      Sub.region.Name == "Southern Asia" ~ "South Asia",
      Sub.region.Name %in% c("Northern Europe","Southern Europe","Western Europe",
                              "Eastern Europe","Northern America") ~ "N. America & Europe",
      Sub.region.Name %in% c("Australia and New Zealand","Melanesia","Micronesia",
                              "Polynesia") ~ "Oceania",
      TRUE ~ "Other"
    ),
    region7_year = paste0(region7, "_", year),
    hi_region = ifelse(ISO %in% high_cap_rating, "High income", Region.Name),
    hi_region_year = paste0(hi_region, "_", year),
    hi_region7 = ifelse(ISO %in% high_cap_rating, "High income", region7),
    hi_region7_year = paste0(hi_region7, "_", year),
    hi_subregion = ifelse(ISO %in% high_cap_rating, "High income", Sub.region.Name),
    hi_subregion_year = paste0(hi_subregion, "_", year)
  ) %>% ungroup()

compute_clean <- function(df, other_vars, h, pre_buffer = 0,
                          target_var = NULL, include_target = FALSE) {
  # Horizon-dependent cleanness (window extends from baseline to outcome), plus
  # optional pre_buffer K years BEFORE the window to guard baseline contamination:
  #   h >= 0: window [t - K, t + h]
  #   h < 0:  window [t + h - K, t - 1]
  # pre_buffer = 0 gives the one-sided window.
  # include_target = TRUE: also count target_var as a shock in the window,
  #   so that controls with prior/future target-bin events are filtered out.
  df <- df %>% arrange(ISO, year)
  df$other_shock <- 0
  for (ov in other_vars) df$other_shock <- df$other_shock + ifelse(df[[ov]] > 0, 1, 0)
  if (include_target && !is.null(target_var)) {
    df$other_shock <- df$other_shock + ifelse(df[[target_var]] > 0, 1, 0)
  }
  df <- df %>% group_by(ISO) %>%
    mutate(other_in_window = {
      os <- other_shock; n <- length(os); result <- rep(NA_real_, n)
      if (h >= 0) {
        for (i in 1:n) {
          si <- max(i - pre_buffer, 1); ei <- min(i + h, n)
          result[i] <- sum(os[si:ei], na.rm = TRUE)
        }
      } else {
        for (i in 1:n) {
          si <- max(i + h - pre_buffer, 1); ei <- i - 1
          result[i] <- if (ei >= si) sum(os[si:ei], na.rm = TRUE) else 0
        }
      }
      result
    }) %>% ungroup()
  df
}

run_lpdid <- function(panel, treatment_var, other_vars, formula_str,
                      leads = 10, lags = 25, model_name = "lpdid",
                      clean_style = "onesided",
                      pre_buffer = 0, include_target = FALSE) {
  # clean_style: "onesided" - horizon-dep window (baseline-to-outcome) + optional pre_buffer
  results <- data.frame()

  for (h in c(-leads:-2, 0:lags)) {
    panel_h <- panel %>% group_by(ISO) %>% arrange(ISO, year) %>%
      mutate(y_cumul = if (h >= 0) log(lead(value, h)) - log(lag(value, 1))
                        else log(lag(value, abs(h))) - log(lag(value, 1))) %>%
      ungroup()
    panel_h <- compute_clean(panel_h, other_vars, h, pre_buffer = pre_buffer,
                              target_var = treatment_var,
                              include_target = include_target)
    panel_h$is_treated <- panel_h[[treatment_var]] > 0
    panel_h$is_clean <- (panel_h[[treatment_var]] == 0) &
      !is.na(panel_h$other_in_window) & (panel_h$other_in_window == 0)

    pf <- panel_h %>% filter((is_treated | is_clean) & !is.na(y_cumul))
    nt <- sum(pf$is_treated); nc <- sum(pf$is_clean)
    ntc <- length(unique(pf$ISO[pf$is_treated]))
    ncc <- length(unique(pf$ISO[pf$is_clean]))
    if (nt < 5 || nc < 10) next
    tryCatch({
      pf <- fixest::panel(pf, ~ ISO + year, duplicate.method = "first")
      mod <- eval(parse(text = paste0("feols(", formula_str, ", data=pf, vcov=~ISO, notes=FALSE)")))
      coefs <- tidy(mod, conf.int = TRUE) %>%
        filter(term == treatment_var) %>%
        mutate(y_h = h, n_treated = nt, n_control = nc,
               n_treated_countries = ntc, n_clean_countries = ncc,
               model = model_name)
      results <- bind_rows(results, coefs)
    }, error = function(e) {
      cat(paste0("  ", model_name, " h=", h, ": ", e$message, "\n"))
    })
  }
  results
}

bins_3bin <- list(
  ts_v2   = c("cat1_v2", "cat3_v2"),
  cat1_v2 = c("ts_v2", "cat3_v2"),
  cat3_v2 = c("ts_v2", "cat1_v2")
)

all_results <- data.frame()

model_name <- "lpdid_hi_region7_fe"
fml_template <- "y_cumul ~ %s + temp + temp:meantemp + outcome_LD1 + outcome_LD2 | ISO + hi_region7_year + ISO[[year]]"

run_one <- function(bin_name, fml, mn, K, ov, inc_tgt) {
  cat(paste0("  ", bin_name, " / ", mn, "\n"))
  res <- run_lpdid(panel, bin_name, ov, fml, model_name = mn,
                   clean_style = "onesided", pre_buffer = K,
                   include_target = inc_tgt)
  if (nrow(res) > 0) { res$subset <- "full"; res$bin_spec <- "3bin" }
  res
}

for (bin_name in names(bins_3bin)) {
  others <- bins_3bin[[bin_name]]
  fml    <- sprintf(fml_template, bin_name)
  # A. v2: bin-specific exclusion sets tuned to event frequency
  #    TS: exclude {Cat1-2, Cat3+}; Cat1-2: exclude {Cat3+}; Cat3+: exclude {Cat3+} (target)
  ov_v2 <- switch(bin_name,
                  "ts_v2"   = c("cat1_v2", "cat3_v2"),
                  "cat1_v2" = c("cat3_v2"),
                  "cat3_v2" = character(0))
  inc_tgt_v2 <- bin_name == "cat3_v2"
  for (K in c(0, 3)) {                 # 0 = onesided [t, t+h]; 3 = prebuf3 [t-3, t+h]
    tag <- if (K == 0) "_onesided" else "_prebuf3"
    all_results <- bind_rows(all_results,
      run_one(bin_name, fml, paste0(model_name, tag, "_v2"),         K, ov_v2,        inc_tgt_v2),
      # B. target bin excluded only
      run_one(bin_name, fml, paste0(model_name, tag, "_targetonly"), K, character(0), TRUE),
      # C. all TC bins excluded
      run_one(bin_name, fml, paste0(model_name, tag, "_allTC"),      K, others,       TRUE))
  }
}

all_results$outcome <- "public_pp_GDD"
fn <- paste0(path_out, "lpdid_subsets_TC.csv")
write.csv(all_results, fn, row.names = FALSE)
cat(paste0("\nSaved: ", fn, " (", nrow(all_results), " rows)\n"))
