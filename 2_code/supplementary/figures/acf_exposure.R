# =============================================================================
# SI figure: ACF_plots.png  (fig:ACF) — "ACF Distribution Across Countries"
#
# Per-lag boxplots of per-country autocorrelation (ACF), lags 1-5.
#
#   Top row    — TC exposure (% land area), from tc_maxws_v2.csv:
#                  ts_v2 ("Tropical Storm", >=18 m/s), cat1_v2 ("Category 1"),
#                  cat3_v2 ("Category 3")
#   Bottom row — annual temperature, from era5_v4_Tshocks.csv:
#                  temp ("Temperature", country-mean annual temp, not isolated)
#                  and Tshock_NATH ("Temperature Shock (isolated)", Nath et al.)
#
# Blue dashed lines = mean 95% significance threshold (1.96/sqrt(T_i)) per panel.
# Countries shown = all ISOs in those two files (not the LP estimation panel).
# =============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2); library(patchwork); library(here)
})
setwd(here::here())

tc   <- read.csv("1_input/climate_exposures/tc_maxws_v2.csv")
era5 <- read.csv("1_input/climate_exposures/era5_v4_Tshocks.csv")

MAX_LAG <- 5
MIN_OBS <- 15   # require >= 15 yearly obs per country

# --- per-ISO ACF, returned long: one row per (ISO, lag) -----------------------
compute_acf <- function(data, var_name, max_lag, label) {
  data %>%
    arrange(ISO, year) %>%
    group_by(ISO) %>%
    filter(sum(!is.na(.data[[var_name]])) >= MIN_OBS) %>%
    summarise(
      n        = sum(!is.na(.data[[var_name]])),
      acf_vals = list(acf(.data[[var_name]], lag.max = max_lag,
                          plot = FALSE, na.action = na.pass)$acf[-1]),  # drop lag 0
      .groups  = "drop"
    ) %>%
    unnest_wider(acf_vals, names_sep = "_") %>%
    pivot_longer(cols = starts_with("acf_vals"), names_to = "lag", values_to = "acf") %>%
    mutate(
      lag       = as.numeric(gsub("acf_vals_", "", lag)),
      threshold = 1.96 / sqrt(n),
      variable  = label
    )
}

acf_boxplot_panel <- function(acf_all, ncol = 3) {
  avg_threshold <- acf_all %>%
    group_by(variable) %>%
    summarise(threshold = mean(threshold, na.rm = TRUE), .groups = "drop")
  ggplot(acf_all, aes(x = factor(lag), y = acf)) +
    geom_hline(yintercept = 0, color = "gray") +
    geom_hline(data = avg_threshold, aes(yintercept =  threshold),
               color = "#5B9BD5", linetype = "dashed") +
    geom_hline(data = avg_threshold, aes(yintercept = -threshold),
               color = "#5B9BD5", linetype = "dashed") +
    geom_boxplot(outlier.size = 0.3, linewidth = 0.25) +
    facet_wrap(~variable, ncol = ncol) +
    labs(x = "lags", y = "ACF") +
    theme_minimal()
}

# --- top row: TC exposure (3 facets) -----------------------------------------
acf_tc <- bind_rows(
  compute_acf(tc, "ts_v2",   MAX_LAG, "Tropical Storm"),
  compute_acf(tc, "cat1_v2", MAX_LAG, "Category 1"),
  compute_acf(tc, "cat3_v2", MAX_LAG, "Category 3")
) %>% mutate(variable = factor(variable,
            levels = c("Tropical Storm", "Category 1", "Category 3")))

# --- bottom row: temperature (2 facets) --------------------------------------
acf_temp <- bind_rows(
  compute_acf(era5, "temp",        MAX_LAG, "Temperature"),
  compute_acf(era5, "Tshock_NATH", MAX_LAG, "Temperature Shock (isolated)")
) %>% mutate(variable = factor(variable,
            levels = c("Temperature", "Temperature Shock (isolated)")))

p_tc   <- acf_boxplot_panel(acf_tc,   ncol = 3)
p_temp <- acf_boxplot_panel(acf_temp, ncol = 2)

fig <- (p_tc / p_temp) +
  patchwork::plot_annotation(
    title   = "ACF Distribution Across Countries",
    caption = "Blue dashed lines = 95% significance threshold",
    # Title indented to the panel area, caption right-aligned
    theme   = ggplot2::theme(
      plot.title   = ggplot2::element_text(margin = ggplot2::margin(l = 42, b = 6)),
      plot.caption = ggplot2::element_text(hjust = 1))
  )

dir.create("4_figures/supplementary", recursive = TRUE, showWarnings = FALSE)
ggsave("4_figures/supplementary/ACF_plots.png", fig, width = 14, height = 8, dpi = 300, bg = "white")
cat("Saved 4_figures/supplementary/ACF_plots.png\n")
