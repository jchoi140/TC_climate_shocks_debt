# =============================================================================
# Run the LP-DiD variants needed for the GDP-per-capita appendix figures:
#   - lpdid_hi_region7_fe_onesided_{v2, targetonly, allTC}
#   - lpdid_hi_region7_fe_prebuf3_{v2, targetonly, allTC}
#   - lpdid_hi_region7_fe_prebuf10_{v2, targetonly, allTC}
# Outcome: log(GDP per capita, WDI constant 2015 USD)
# Saves to:
#   lpdid_subsets_TC_gdp.csv   (onesided + prebuf3 variants)
#   lpdid_prebuf10_TC_gdp.csv  (prebuf10 variants)
# Re-uses panel construction + run_lpdid() helper from
# run_pretrends_lpdid_subsets.R, swapping the outcome data source.
# =============================================================================

library(fixest); library(dplyr); library(broom); library(tidyverse)

project_root <- here::here()
setwd(project_root)
path_out <- "3_output/S_pretrends/"

# Source helpers + panel construction from the debt runner, swapping debt for GDP per capita.
source_file <- "2_code/supplementary/analysis/LP/S_pretrends/run_pretrends_lpdid_subsets.R"
src <- readLines(source_file)

old_line <- 'outcome_data <- read.csv("./1_input/economic_outcomes/gdd_debt_v2.csv") %>%'
new_line <- 'outcome_data <- read.csv("./0_raw/economic_outcomes/gdp_pcap_wdi_constant2015_USD.csv") %>%'
old_select <- '  dplyr::select(ISO, year, value = public_debt_pp)'
new_select <- '  dplyr::select(ISO, year, value)'
src <- gsub(old_line, new_line, src, fixed = TRUE)
src <- gsub(old_select, new_select, src, fixed = TRUE)

# Stop sourcing before the loop body (line `all_results <- data.frame()`).
stop_at <- grep("^all_results <- data\\.frame\\(\\)", src)[1]
eval(parse(text = paste(src[1:(stop_at - 1)], collapse = "\n")))

# `panel`, `run_lpdid`, `bins_3bin` are in scope, with GDP as `value`.

model_name <- "lpdid_hi_region7_fe"
fml_template <- "y_cumul ~ %s + temp + temp:meantemp + outcome_LD1 + outcome_LD2 | ISO + hi_region7_year + ISO[[year]]"

run_one <- function(bin_name, others, fml, mn, K, ov, inc_tgt) {
  cat(paste0("  ", bin_name, " / ", mn, "\n"))
  res <- run_lpdid(panel, bin_name, ov, fml, model_name = mn,
                   clean_style = "onesided", pre_buffer = K,
                   include_target = inc_tgt)
  if (nrow(res) > 0) { res$subset <- "full"; res$bin_spec <- "3bin" }
  res
}

results_main <- data.frame()  # onesided + prebuf3
results_p10  <- data.frame()  # prebuf10

for (bin_name in names(bins_3bin)) {
  others <- bins_3bin[[bin_name]]
  fml    <- sprintf(fml_template, bin_name)

  ov_v2 <- switch(bin_name,
                  "ts_v2"   = c("cat1_v2", "cat3_v2"),
                  "cat1_v2" = c("cat3_v2"),
                  "cat3_v2" = character(0))
  inc_tgt_v2 <- bin_name == "cat3_v2"

  # onesided (K=0)
  results_main <- bind_rows(results_main,
    run_one(bin_name, others, fml, paste0(model_name, "_onesided_v2"),
            K = 0, ov = ov_v2,        inc_tgt = inc_tgt_v2),
    run_one(bin_name, others, fml, paste0(model_name, "_onesided_targetonly"),
            K = 0, ov = character(0), inc_tgt = TRUE),
    run_one(bin_name, others, fml, paste0(model_name, "_onesided_allTC"),
            K = 0, ov = others,       inc_tgt = TRUE))

  # prebuf3 (K=3)
  results_main <- bind_rows(results_main,
    run_one(bin_name, others, fml, paste0(model_name, "_prebuf3_v2"),
            K = 3, ov = ov_v2,        inc_tgt = inc_tgt_v2),
    run_one(bin_name, others, fml, paste0(model_name, "_prebuf3_targetonly"),
            K = 3, ov = character(0), inc_tgt = TRUE),
    run_one(bin_name, others, fml, paste0(model_name, "_prebuf3_allTC"),
            K = 3, ov = others,       inc_tgt = TRUE))

  # prebuf10 (K=10)
  results_p10 <- bind_rows(results_p10,
    run_one(bin_name, others, fml, paste0(model_name, "_prebuf10_v2"),
            K = 10, ov = ov_v2,        inc_tgt = inc_tgt_v2),
    run_one(bin_name, others, fml, paste0(model_name, "_prebuf10_targetonly"),
            K = 10, ov = character(0), inc_tgt = TRUE),
    run_one(bin_name, others, fml, paste0(model_name, "_prebuf10_allTC"),
            K = 10, ov = others,       inc_tgt = TRUE))
}

results_main$outcome <- "gdp_pcap_wdi_constant2015_USD"
results_p10$outcome  <- "gdp_pcap_wdi_constant2015_USD"

fn1 <- paste0(path_out, "lpdid_subsets_TC_gdp.csv")
write.csv(results_main, fn1, row.names = FALSE)
cat(paste0("\nSaved: ", fn1, " (", nrow(results_main), " rows)\n"))

fn2 <- paste0(path_out, "lpdid_prebuf10_TC_gdp.csv")
write.csv(results_p10, fn2, row.names = FALSE)
cat(paste0("Saved: ", fn2, " (", nrow(results_p10), " rows)\n"))
