# =============================================================================
# Run only the 3 prebuf10 variants of lpdid_hi_region7_fe needed for the
# pretrend_appendix_lpdid_twosided_h10 plot. Saves to lpdid_prebuf10_TC.csv.
# Re-uses panel construction + run_lpdid() helper from
# run_pretrends_lpdid_subsets.R.
# =============================================================================

library(fixest); library(dplyr); library(broom); library(tidyverse)

project_root <- here::here()
setwd(project_root)
path_out <- "3_output/S_pretrends/"

source_file <- "2_code/supplementary/analysis/LP/S_pretrends/run_pretrends_lpdid_subsets.R"
src <- readLines(source_file)
# Stop sourcing before the loop body (line `all_results <- data.frame()`).
stop_at <- grep("^all_results <- data\\.frame\\(\\)", src)[1]
eval(parse(text = paste(src[1:(stop_at - 1)], collapse = "\n")))

# `panel`, `run_lpdid`, `bins_3bin` are in scope.

model_name <- "lpdid_hi_region7_fe"
fml_template <- "y_cumul ~ %s + temp + temp:meantemp + outcome_LD1 + outcome_LD2 | ISO + hi_region7_year + ISO[[year]]"

all_results <- data.frame()

for (bin_name in names(bins_3bin)) {
  others <- bins_3bin[[bin_name]]
  fml <- sprintf(fml_template, bin_name)

  # v2 (bin-specific exclusion sets, K=10)
  ov_v2 <- switch(bin_name,
                  "ts_v2"   = c("cat1_v2", "cat3_v2"),
                  "cat1_v2" = c("cat3_v2"),
                  "cat3_v2" = character(0))
  inc_tgt_v2 <- bin_name == "cat3_v2"
  mn <- paste0(model_name, "_prebuf10_v2")
  cat(paste0("  ", bin_name, " / ", mn, "\n"))
  res <- run_lpdid(panel, bin_name, ov_v2, fml, model_name = mn,
                   clean_style = "onesided", pre_buffer = 10,
                   include_target = inc_tgt_v2)
  if (nrow(res) > 0) { res$subset <- "full"; res$bin_spec <- "3bin"
    all_results <- bind_rows(all_results, res) }

  # targetonly (K=10, only target bin excluded from window)
  mn <- paste0(model_name, "_prebuf10_targetonly")
  cat(paste0("  ", bin_name, " / ", mn, "\n"))
  res <- run_lpdid(panel, bin_name, character(0), fml, model_name = mn,
                   clean_style = "onesided", pre_buffer = 10,
                   include_target = TRUE)
  if (nrow(res) > 0) { res$subset <- "full"; res$bin_spec <- "3bin"
    all_results <- bind_rows(all_results, res) }

  # allTC (K=10, all TC bins excluded from window)
  mn <- paste0(model_name, "_prebuf10_allTC")
  cat(paste0("  ", bin_name, " / ", mn, "\n"))
  res <- run_lpdid(panel, bin_name, others, fml, model_name = mn,
                   clean_style = "onesided", pre_buffer = 10,
                   include_target = TRUE)
  if (nrow(res) > 0) { res$subset <- "full"; res$bin_spec <- "3bin"
    all_results <- bind_rows(all_results, res) }
}

all_results$outcome <- "public_pp_GDD"
fn <- paste0(path_out, "lpdid_prebuf10_TC.csv")
write.csv(all_results, fn, row.names = FALSE)
cat(paste0("\nSaved: ", fn, " (", nrow(all_results), " rows)\n"))
