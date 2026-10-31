# =============================================================================
# Interaction (heterogeneity) runs for the standard LP and LP-DiD onesided v2
# specifications. Three heterogeneity dimensions, two outcomes each, two
# model types each = 3 x 2 x 2 = 12 estimations total.
#
# Heterogeneity dimensions:
#   1) tc_freq       - country-level Cat1+ year count, median split,
#                      "with-no-cat1" pool included as supplemental clean
#                      controls.
#   2) capacity      - CPIA-mean median split + HI/OECD-not-in-CPIA = high,
#                      "uncovered" sovereigns added as supplemental clean
#                      controls.
#   3) compound_w5   - row-level prior_severe = 1{cat3_v2 > 0 in t-1..t-5}.
#
# For each (outcome, dimension), we save ONE CSV holding both model types side
# by side, distinguished by the `spec_display` column.
#
# Outcomes: debt (public_pp_GDD) and gdp (gdp_pcap_wdi_constant2015_USD).
#
# Model variants per estimation:
#   Standard LP (HI + 5 region x year FE)
#   LP-DiD onesided v2 (HI + 7 region x year FE)
#
# Outputs:
#   3_output/LP/interactions/data/interactions_<dim>_<outcome>.csv  (6 CSVs)
#     dim     in {tc_freq, capacity, compound_w5}
#     outcome in {debt, gdp}
#   3_output/LP/interactions/plots/interactions_lpdid_<outcome>_p.png  (Figs. S13 debt, S14 gdp)
# =============================================================================

suppressPackageStartupMessages({
  library(fixest); library(dplyr); library(broom); library(tidyverse)
  library(ggplot2); library(patchwork)
})

project_root <- here::here()
setwd(project_root)
out_dir_data  <- "3_output/LP/interactions/data/"
out_dir_plots <- "3_output/LP/interactions/plots/"
dir.create(out_dir_data,  recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir_plots, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# Static lookups
# ============================================================================
high_cap_rating <- c("AND","ARE","AUS","AUT","BEL","BHR","BHS","BWA","CAN","CHE",
  "CHL","CHN","CYP","CZE","DEU","DNK","ESP","EST","FIN","FRA",
  "GBR","HUN","IRL","ISL","ISR","ITA","JPN","KOR","KWT","LIE",
  "LTU","LUX","LVA","MEX","MLT","MUS","MYS","NLD","NOR","NZL",
  "OMN","POL","PRT","QAT","SAU","SGP","SMR","SVK","SVN","SWE",
  "THA","TTO","USA","ZAF")
oecd <- c("AUS","AUT","BEL","CAN","CHE","CHL","COL","CRI","CZE","DEU",
          "DNK","ESP","EST","FIN","FRA","GBR","GRC","HUN","IRL","ISL",
          "ISR","ITA","JPN","KOR","LTU","LUX","LVA","MEX","NLD","NOR",
          "NZL","POL","PRT","SVK","SVN","SWE","TUR","USA")

# ============================================================================
# Load shared inputs (era5, tc, master, CPIA)
# ============================================================================
cat("Loading data...\n")
era5         <- read.csv("./1_input/climate_exposures/era5_v4_Tshocks.csv")
tc_complete  <- read.csv("./1_input/climate_exposures/tc_maxws_complete.csv")
tc_v2        <- read.csv("./1_input/climate_exposures/tc_maxws_v2.csv")
master       <- read.csv("./0_raw/climate_exposures/country_master.csv")
sovereign_ISOs <- unique(c(master$ISO[master$sovereigncountry == 1], "TWN"))

twn_rows <- tc_v2 %>% filter(ISO == "TWN") %>%
  mutate(X = NA_integer_, total = NA_real_, totalexposed = NA_real_,
         avg_ws = NA_real_, country2 = "Taiwan",
         Region.Name = "Asia", Sub.region.Name = "Eastern Asia",
         sovereigncountry = 1L) %>%
  dplyr::select(X, year, country, total, totalexposed, ts_v2, cat1_v2, cat3_v2,
                avg_ws, country2, ISO, Region.Name, Sub.region.Name, sovereigncountry)
tc <- bind_rows(tc_complete, twn_rows)

cpia_raw <- read.csv("./1_input/economic_outcomes/CPIA_IDA_indicator.csv",
                     stringsAsFactors = FALSE)
cpia <- cpia_raw[1:86, ]
year_cols <- grep("^X[0-9]{4}", colnames(cpia), value = TRUE)
cpia_long <- cpia %>%
  dplyr::select(ISO = Country.Code, all_of(year_cols)) %>%
  mutate(across(all_of(year_cols), ~ suppressWarnings(as.numeric(.))))
cpia_avg <- cpia_long %>%
  rowwise() %>%
  mutate(cpia_mean = mean(c_across(all_of(year_cols)), na.rm = TRUE)) %>%
  ungroup() %>%
  dplyr::select(ISO, cpia_mean)
cpia_threshold <- median(cpia_avg$cpia_mean, na.rm = TRUE)

cpia_high  <- cpia_avg$ISO[cpia_avg$cpia_mean >= cpia_threshold &
                             cpia_avg$ISO %in% sovereign_ISOs]
cpia_low   <- cpia_avg$ISO[cpia_avg$cpia_mean <  cpia_threshold &
                             cpia_avg$ISO %in% sovereign_ISOs]
hi_oecd_not_in_cpia <- intersect(setdiff(union(high_cap_rating, oecd),
                                         cpia_avg$ISO),
                                 sovereign_ISOs)
cap_high  <- unique(c(cpia_high, hi_oecd_not_in_cpia))
cap_low   <- cpia_low
cap_uncov <- setdiff(sovereign_ISOs,
                     union(union(cpia_avg$ISO, high_cap_rating), oecd))
cat(sprintf("CPIA threshold=%.3f: HIGH=%d, LOW=%d, UNCOVERED=%d\n",
            cpia_threshold, length(cap_high), length(cap_low), length(cap_uncov)))

# ============================================================================
# Panel builder (parameterised by outcome)
# ============================================================================
build_panel <- function(outcome_csv, value_expr) {
  outcome_df <- read.csv(outcome_csv) %>%
    dplyr::select(ISO, year, value = all_of(value_expr))

  panel <- era5 %>%
    rename(temp_pw = temp) %>%
    left_join(tc, by = c("ISO", "year")) %>%
    left_join(outcome_df, by = c("ISO", "year")) %>%
    filter(year >= 1980, year < 2020, ISO %in% sovereign_ISOs) %>%
    group_by(ISO) %>% arrange(ISO, year) %>%
    mutate(
      meantemp    = meantemp_hist,
      temp        = Tshock_NATH,
      outcome_LD1 = log(lag(value, 1)) - log(lag(value, 2)),
      outcome_LD2 = log(lag(value, 2)) - log(lag(value, 3)),
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
      hi_region       = ifelse(ISO %in% high_cap_rating, "High income", Region.Name),
      hi_region_year  = paste0(hi_region, "_", year),
      hi_region7      = ifelse(ISO %in% high_cap_rating, "High income", region7),
      hi_region7_year = paste0(hi_region7, "_", year),
      region7_year    = paste0(region7, "_", year)
    ) %>% ungroup()

  for (k in 1:10) {
    panel <- panel %>% group_by(ISO) %>% arrange(ISO, year) %>%
      mutate(!!paste0("ts_v2_L", k)    := lag(ts_v2,    k),
             !!paste0("cat1_v2_L", k)  := lag(cat1_v2,  k),
             !!paste0("cat3_v2_L", k)  := lag(cat3_v2,  k)) %>% ungroup()
  }
  for (k in 1:2) {
    panel <- panel %>% group_by(ISO) %>% arrange(ISO, year) %>%
      mutate(!!paste0("temp_L", k)          := lag(temp, k),
             !!paste0("temp_meantemp_L", k) := lag(temp, k) * meantemp) %>% ungroup()
  }

  # prior_severe_w5: 1 if cat3_v2 > 0 in any of t-1..t-5 for the same ISO.
  # Drives the `compound_w5` dimension's row-level `high` indicator.
  panel <- panel %>%
    group_by(ISO) %>% arrange(ISO, year) %>%
    mutate(
      severe_now = as.integer(replace_na(cat3_v2 > 0, FALSE)),
      prior_severe_w5 = as.integer(
        Reduce(`|`, lapply(1:5,
                            function(k) replace_na(lag(severe_now, k) == 1, FALSE)))
      )
    ) %>% ungroup() %>%
    dplyr::select(-severe_now)

  panel
}

# ============================================================================
# Cat1+ classification (used by tc_freq dimension)
# ============================================================================
classify_cat1 <- function(panel) {
  tc_base <- panel %>%
    mutate(any_cat1plus = ifelse((cat1_v2 > 0) | (cat3_v2 > 0), 1, 0)) %>%
    group_by(ISO) %>%
    summarize(n_cat1plus_years = sum(any_cat1plus, na.rm = TRUE), .groups = "drop")
  med_threshold <- median(tc_base$n_cat1plus_years[tc_base$n_cat1plus_years >= 1])
  tc_class <- tc_base %>%
    mutate(exposure_group = case_when(
      n_cat1plus_years == 0             ~ "no_tc",
      n_cat1plus_years >  med_threshold ~ "high",
      TRUE                              ~ "low"
    ))
  list(
    high_iso = tc_class$ISO[tc_class$exposure_group == "high"],
    low_iso  = tc_class$ISO[tc_class$exposure_group == "low"],
    notc_iso = tc_class$ISO[tc_class$exposure_group == "no_tc"],
    threshold = med_threshold
  )
}

# ============================================================================
# Estimation helpers
# ============================================================================
extract_low_high <- function(mod, bin_name, h, model_label,
                              n_treated_obs = NA, n_clean_obs = NA,
                              n_treated_countries = NA, n_clean_countries = NA) {
  cf <- coef(mod); vc <- vcov(mod)
  ix_main <- bin_name
  ix_int  <- paste0(bin_name, ":high")
  if (!(ix_main %in% names(cf))) return(data.frame())
  b_low  <- cf[ix_main]
  b_diff <- if (ix_int %in% names(cf)) cf[ix_int] else 0
  b_high <- b_low + b_diff
  v_main <- vc[ix_main, ix_main]
  v_int  <- if (ix_int %in% rownames(vc)) vc[ix_int, ix_int] else 0
  cv     <- if (ix_int %in% rownames(vc)) vc[ix_main, ix_int] else 0
  se_low  <- sqrt(v_main); se_diff <- sqrt(v_int)
  se_high <- sqrt(v_main + v_int + 2 * cv)
  data.frame(
    term = bin_name,
    exposure_group = c("low", "high"),
    estimate = c(b_low, b_high),
    std.error = c(se_low, se_high),
    conf.low  = c(b_low - 1.96 * se_low,  b_high - 1.96 * se_high),
    conf.high = c(b_low + 1.96 * se_low,  b_high + 1.96 * se_high),
    y_h = h, model = model_label,
    diff_estimate = b_diff, diff_se = se_diff,
    diff_pvalue   = if (se_diff > 0) 2 * pnorm(-abs(b_diff / se_diff)) else NA_real_,
    n_treated = n_treated_obs, n_clean = n_clean_obs,
    n_treated_countries = n_treated_countries,
    n_clean_countries   = n_clean_countries
  )
}

run_standard_lp_interact <- function(panel_sub, fe_spec, leads = 10, lags = 25,
                                      model_label) {
  tc_lags <- paste(c(paste0("ts_v2_L", 1:10),
                     paste0("cat1_v2_L", 1:10),
                     paste0("cat3_v2_L", 1:10)), collapse = " + ")
  temp_terms <- "temp + temp:meantemp + temp_L1 + temp_L2 + temp_meantemp_L1 + temp_meantemp_L2"
  controls <- paste(tc_lags, temp_terms, "outcome_LD1 + outcome_LD2", sep = " + ")
  bins <- c("ts_v2", "cat1_v2", "cat3_v2")
  bin_main  <- paste(bins, collapse = " + ")
  bin_intx  <- paste(paste0(bins, ":high"), collapse = " + ")
  bin_terms <- paste(bin_main, bin_intx, sep = " + ")
  results <- data.frame()
  for (h in c(-leads:-2, 0:lags)) {
    panel_h <- panel_sub %>% group_by(ISO) %>% arrange(ISO, year) %>%
      mutate(y_cumul = if (h >= 0) log(lead(value, h)) - log(lag(value, 1))
                        else log(lag(value, abs(h))) - log(lag(value, 1))) %>%
      ungroup()
    fml <- as.formula(paste0("y_cumul ~ ", bin_terms, " + ", controls, " | ", fe_spec))
    tryCatch({
      mod <- feols(fml, data = panel_h, vcov = ~ISO, notes = FALSE)
      for (b in bins) {
        results <- bind_rows(results, extract_low_high(mod, b, h, model_label))
      }
    }, error = function(e) cat(sprintf("  [stdLP] %s h=%d err: %s\n",
                                         model_label, h, e$message)))
  }
  results
}

compute_clean <- function(df, other_vars, h) {
  df <- df %>% arrange(ISO, year)
  df$other_shock <- 0
  for (ov in other_vars) df$other_shock <- df$other_shock + ifelse(df[[ov]] > 0, 1, 0)
  df <- df %>% group_by(ISO) %>%
    mutate(other_in_window = {
      os <- other_shock; n <- length(os); r <- rep(NA_real_, n)
      if (h >= 0) {
        for (i in 1:n) { ei <- min(i + h, n); r[i] <- sum(os[i:ei], na.rm = TRUE) }
      } else {
        for (i in 1:n) { si <- max(i + h, 1); ei <- i - 1
          r[i] <- if (ei >= si) sum(os[si:ei], na.rm = TRUE) else 0 }
      }
      r
    }) %>% ungroup()
  df
}

bins_3bin <- list(
  ts_v2   = c("cat3_v2"),
  cat1_v2 = c("cat3_v2"),
  cat3_v2 = c("cat3_v2")
)

run_lpdid_interact <- function(panel_sub, fe_spec, leads = 10, lags = 25, model_label) {
  results <- data.frame()
  for (bin_name in names(bins_3bin)) {
    other_vars <- bins_3bin[[bin_name]]
    for (h in c(-leads:-2, 0:lags)) {
      panel_h <- panel_sub %>% group_by(ISO) %>% arrange(ISO, year) %>%
        mutate(y_cumul = if (h >= 0) log(lead(value, h)) - log(lag(value, 1))
                          else log(lag(value, abs(h))) - log(lag(value, 1))) %>%
        ungroup()
      panel_h <- compute_clean(panel_h, other_vars, h)
      panel_h$is_treated <- panel_h[[bin_name]] > 0
      panel_h$is_clean   <- (panel_h[[bin_name]] == 0) &
        !is.na(panel_h$other_in_window) & (panel_h$other_in_window == 0)
      pf <- panel_h %>% filter((is_treated | is_clean) & !is.na(y_cumul))
      n_t  <- sum(pf$is_treated); n_c <- sum(pf$is_clean)
      n_tc <- length(unique(pf$ISO[pf$is_treated]))
      n_cc <- length(unique(pf$ISO[pf$is_clean]))
      if (n_t < 5 || n_c < 3) next
      fml <- as.formula(paste0("y_cumul ~ ", bin_name, " + ", bin_name, ":high",
                                 " + temp + temp:meantemp + outcome_LD1 + outcome_LD2 | ",
                                 fe_spec))
      tryCatch({
        pf <- fixest::panel(pf, ~ ISO + year, duplicate.method = "first")
        mod <- feols(fml, data = pf, vcov = ~ISO, notes = FALSE)
        results <- bind_rows(results,
          extract_low_high(mod, bin_name, h, model_label, n_t, n_c, n_tc, n_cc))
      }, error = function(e) cat(sprintf("  [LP-DiD] %s %s h=%d err: %s\n",
                                           model_label, bin_name, h, e$message)))
    }
  }
  results
}

# ============================================================================
# Spec list (always run BOTH model types)
# ============================================================================
specs <- list(
  list(label = "stdLP_hi5", type = "stdLP",
       fe_spec = "ISO + hi_region_year + ISO[[year]]",
       display = "Standard LP (HI + 5 region x year FE)"),
  list(label = "lpdid_hi7", type = "lpdid",
       fe_spec = "ISO + hi_region7_year + ISO[[year]]",
       display = "LP-DiD onesided v2 (HI + 7 region x year FE)")
)

run_all_specs <- function(panel_sub, dim_label) {
  all_results <- data.frame()
  for (spec in specs) {
    cat(sprintf("    [%s] %s\n", dim_label, spec$display))
    res <- if (spec$type == "stdLP")
      run_standard_lp_interact(panel_sub, spec$fe_spec, model_label = spec$label)
    else
      run_lpdid_interact      (panel_sub, spec$fe_spec, model_label = spec$label)
    if (nrow(res) > 0) {
      res$spec         <- spec$label
      res$spec_display <- spec$display
      all_results <- bind_rows(all_results, res)
    }
  }
  all_results
}

# ============================================================================
# Per-outcome driver
# ============================================================================
outcomes <- list(
  list(tag = "debt",
       csv = "./1_input/economic_outcomes/gdd_debt_v2.csv",
       value_col = "public_debt_pp"),
  list(tag = "gdp",
       csv = "./0_raw/economic_outcomes/gdp_pcap_wdi_constant2015_USD.csv",
       value_col = "value")
)

for (oc in outcomes) {
  cat(sprintf("\n=========================================================\n"))
  cat(sprintf("=== Outcome: %s ===\n", oc$tag))
  cat(sprintf("=========================================================\n"))

  panel <- build_panel(oc$csv, oc$value_col)

  # --- Dimension 1: Cat1+ TC frequency (with-no-cat1 pool) ----------------
  tc_cls <- classify_cat1(panel)
  cat(sprintf("[%s] tc_freq: HIGH=%d, LOW=%d, NO-Cat1+=%d (threshold=%g)\n",
              oc$tag, length(tc_cls$high_iso), length(tc_cls$low_iso),
              length(tc_cls$notc_iso), tc_cls$threshold))
  panel_tc <- panel %>%
    mutate(high = as.integer(ISO %in% tc_cls$high_iso)) %>%
    filter(ISO %in% c(tc_cls$high_iso, tc_cls$low_iso, tc_cls$notc_iso))
  res_tc <- run_all_specs(panel_tc, sprintf("%s/tc_freq", oc$tag))
  fn <- sprintf("%sinteractions_tc_freq_%s.csv", out_dir_data, oc$tag)
  write.csv(res_tc, fn, row.names = FALSE)
  cat(sprintf("  saved: %s (%d rows)\n", fn, nrow(res_tc)))

  # --- Dimension 2: CPIA capacity (with-uncovered pool) -------------------
  cat(sprintf("[%s] capacity: HIGH=%d, LOW=%d, UNCOVERED=%d\n",
              oc$tag, length(cap_high), length(cap_low), length(cap_uncov)))
  panel_cap <- panel %>%
    mutate(high = as.integer(ISO %in% cap_high)) %>%
    filter(ISO %in% c(cap_high, cap_low, cap_uncov))
  res_cap <- run_all_specs(panel_cap, sprintf("%s/capacity", oc$tag))
  fn <- sprintf("%sinteractions_capacity_%s.csv", out_dir_data, oc$tag)
  write.csv(res_cap, fn, row.names = FALSE)
  cat(sprintf("  saved: %s (%d rows)\n", fn, nrow(res_cap)))

  # --- Dimension 3: Compounding window=5 ---------------------------------
  cat(sprintf("[%s] compound_w5: prior_severe (cat3 in t-1..t-5) = %d / %d obs\n",
              oc$tag, sum(panel$prior_severe_w5 == 1, na.rm = TRUE),
              nrow(panel)))
  panel_c5 <- panel %>%
    mutate(high = as.integer(prior_severe_w5)) %>%
    filter(ISO %in% sovereign_ISOs)
  res_c5 <- run_all_specs(panel_c5, sprintf("%s/compound_w5", oc$tag))
  fn <- sprintf("%sinteractions_compound_w5_%s.csv", out_dir_data, oc$tag)
  write.csv(res_c5, fn, row.names = FALSE)
  cat(sprintf("  saved: %s (%d rows)\n", fn, nrow(res_c5)))
}

cat("\nAll interaction estimations complete.\n")

# ============================================================================
# Plots: 3-row IRF comparison per outcome.
# Rows: TC frequency, Debt-carrying capacity, Compounding (window=5).
# Columns: TS, Cat 1-2, Cat 3+.
# ============================================================================
term_labels <- c(ts_v2 = "Tropical Storm",
                 cat1_v2 = "Category 1-2",
                 cat3_v2 = "Category 3+")
term_levels <- names(term_labels)

base_theme <- theme_bw() + theme(
  panel.grid.minor   = element_blank(),
  strip.text.y       = element_text(face = "bold", size = 13, lineheight = 0.9),
  strip.background.y = element_rect(fill = "grey85", color = NA),
  axis.title.x       = element_blank(),
  axis.title.y       = element_blank(),
  axis.text          = element_text(size = 10),
  legend.background  = element_rect(fill = alpha("white", 0.85), color = NA),
  legend.key.size    = unit(0.4, "cm"),
  legend.text        = element_text(size = 9),
  legend.title       = element_blank(),
  legend.spacing.y   = unit(0, "cm"),
  legend.margin      = margin(1, 1, 1, 1),
  plot.margin        = margin(2, 4, 2, 4)
)


# ============================================================================
# `_p` plots (Figs. S13 debt, S14 gdp). A star marks post-treatment horizons
# where the interaction term (:high coefficient) is significant at p < 0.05
# (`diff_pvalue`, a formal test, not CI overlap).
# ============================================================================
build_row_p <- function(csv_path, spec_filter, row_label_text,
                         label_high, label_low, color_high, color_low,
                         ylim_v = NULL,
                         show_x_strip = FALSE, show_x_axis = FALSE) {
  df_raw <- read.csv(csv_path) %>%
    filter(spec_display == spec_filter, term %in% term_levels,
           y_h >= -10, y_h <= 20)
  df <- df_raw %>%
    mutate(spec      = ifelse(exposure_group == "high", label_high, label_low),
           tc_label  = factor(term_labels[term], levels = unname(term_labels)),
           row_label = factor(row_label_text, levels = row_label_text))
  lvls <- c(label_high, label_low)
  cols <- setNames(c(color_high, color_low), lvls)
  df$spec <- factor(df$spec, levels = lvls)

  # diff_pvalue is identical for low/high rows (same regression); keep 'low' to dedupe.
  sig <- df_raw %>%
    filter(exposure_group == "low", y_h >= 0,
           !is.na(diff_pvalue), diff_pvalue < 0.05) %>%
    mutate(tc_label  = factor(term_labels[term], levels = unname(term_labels)),
           row_label = factor(row_label_text, levels = row_label_text),
           star      = "*")
  # Place stars near the top of each panel (just under the ylim ceiling).
  if (!is.null(ylim_v) && nrow(sig) > 0) {
    sig$y_anno <- ylim_v[2] - 0.05 * (ylim_v[2] - ylim_v[1])
  }

  p <- ggplot(df, aes(x = y_h, y = estimate,
                       color = spec, fill = spec, group = spec)) +
    geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
    geom_vline(xintercept = -0.5, color = "red", linetype = "dashed", alpha = 0.5) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high),
                alpha = 0.15, color = NA, show.legend = FALSE) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 0.8, show.legend = FALSE)
  if (nrow(sig) > 0 && !is.null(ylim_v)) {
    p <- p + geom_text(data = sig,
                        aes(x = y_h, y = y_anno, label = star),
                        inherit.aes = FALSE,
                        size = 3.5, color = "grey15", fontface = "bold")
  }
  p <- p +
    facet_grid(row_label ~ tc_label) +
    scale_color_manual(values = cols) +
    scale_fill_manual(values = cols, guide = "none") +
    base_theme +
    theme(legend.position      = c(0.02, 0.98),
          legend.justification = c(0, 1))

  if (show_x_strip) {
    p <- p + theme(strip.text.x = element_text(face = "bold", size = 14),
                   strip.background.x = element_rect(fill = "grey90", color = NA))
  } else {
    p <- p + theme(strip.text.x = element_blank(),
                   strip.background.x = element_blank())
  }
  if (show_x_axis) {
    p <- p + theme(axis.title.x = element_text(size = 10)) +
      labs(x = "Horizon (years)")
  }
  p
}

plot_3row_p <- function(outcome_tag, fname) {
  spec_filter <- "LP-DiD onesided v2 (HI + 7 region x year FE)"
  ylim_v <- if (outcome_tag == "debt") c(-0.005, 0.015) else c(-0.003, 0.003)
  p1 <- build_row_p(sprintf("%sinteractions_tc_freq_%s.csv",   out_dir_data, outcome_tag),
                     spec_filter, "TC frequency",
                     "high exposure",                "low exposure",
                     "#D62728", "#1F77B4", ylim_v,
                     show_x_strip = TRUE)
  p2 <- build_row_p(sprintf("%sinteractions_capacity_%s.csv",  out_dir_data, outcome_tag),
                     spec_filter, "Debt-carrying\ncapacity",
                     "high debt carrying capacity", "low debt carrying capacity",
                     "#FF7F0E", "#9467BD", ylim_v)
  p3 <- build_row_p(sprintf("%sinteractions_compound_w5_%s.csv", out_dir_data, outcome_tag),
                     spec_filter, "Compounding\n(prior cat3, t-1..t-5)",
                     "prior cat3 exposure",          "no prior cat3",
                     "#17BECF", "#7F7F7F", ylim_v,
                     show_x_axis = TRUE)
  fixed_y <- coord_cartesian(ylim = ylim_v)
  combined <- (p1 + fixed_y) / (p2 + fixed_y) / (p3 + fixed_y)
  outcome_title <- if (outcome_tag == "debt") "Debt-to-GDP ratio" else "GDP per capita"
  combined <- combined +
    plot_annotation(title = outcome_title,
                    theme = theme(
                      plot.title    = element_text(size = 16, face = "bold",
                                                    hjust = 0, margin = margin(b = 2))))
  ggsave(fname, combined, width = 16, height = 10, dpi = 300, bg = "white")
  cat(sprintf("  saved plot: %s\n", fname))
}

cat("\nGenerating `_p` variant plots with interaction p<0.05 highlighting...\n")
for (oc in outcomes) {
  fname <- sprintf("%sinteractions_lpdid_%s_p.png", out_dir_plots, oc$tag)
  plot_3row_p(oc$tag, fname)
}

cat("\nAll plots written to ", out_dir_plots, "\n", sep = "")
