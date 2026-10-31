# =============================================================================
# Per-country LP-DiD clean-control classification (Fig. S11, Jamaica) across
# 4 clean-control variants x 3 target bins, at horizon h = 3.
#   Top panel:    TC exposure by bin and year
#   Bottom panel: treated / clean control / excluded status per variant and year
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(tidyr); library(patchwork)
})

project_root <- here::here()
setwd(project_root)
path_plots <- "4_figures/supplementary/S_pretrends/"

H <- 3L

# ------------------------------------------------------------------
# Load panel
# ------------------------------------------------------------------
era5   <- read.csv("./1_input/climate_exposures/era5_v4_Tshocks.csv")
tc     <- read.csv("./0_raw/climate_exposures/tc2_maxws.csv")
master <- read.csv("./0_raw/climate_exposures/country_master.csv")
sovereigns <- master$ISO[master$sovereigncountry == 1]

panel <- era5 %>%
  left_join(tc, by = c("ISO","year")) %>%
  filter(year >= 1980, year < 2020, ISO %in% sovereigns) %>%
  arrange(ISO, year) %>%
  mutate(ts = ifelse(ts_v2 > 0, 1, 0),
         cat12 = ifelse(cat1_v2 > 0, 1, 0),
         cat3  = ifelse(cat3_v2 > 0, 1, 0))

# ------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------
variant_list <- list(
  list(label = "Cat3+ excluded, one-sided [t, t+h]",        short = "a", K = 0, mode = "cat3"),
  list(label = "Target bin excluded, one-sided [t, t+h]",   short = "b", K = 0, mode = "target"),
  list(label = "Cat3+ excluded, two-sided [t-3, t+h]",      short = "c", K = 3, mode = "cat3"),
  list(label = "Target bin excluded, two-sided [t-3, t+h]", short = "d", K = 3, mode = "target")
)

# Caption explaining each short label
version_caption <- paste0(
  "a: Exclude observations with a Cat3+ event in the one-sided window [t, t+h].\n",
  "b: Exclude observations with a TARGET-BIN event in the one-sided window [t, t+h].\n",
  "c: Exclude observations with a Cat3+ event in the two-sided window [t-3, t+h].\n",
  "d: Exclude observations with a TARGET-BIN event in the two-sided window [t-3, t+h]."
)

exclusion_sets <- list(
  "TS target bin"     = list(cat3 = c("cat3"),  target = c("ts")),
  "Cat1-2 target bin" = list(cat3 = c("cat3"),  target = c("cat12")),
  "Cat3+ target bin"  = list(cat3 = c("cat3"),  target = c("cat3"))
)

targets <- list(
  "TS target bin"     = list(target_fn = function(d) d$ts),
  "Cat1-2 target bin" = list(target_fn = function(d) d$cat12),
  "Cat3+ target bin"  = list(target_fn = function(d) d$cat3)
)

target_order  <- c("TS target bin", "Cat1-2 target bin", "Cat3+ target bin")
variant_order <- sapply(variant_list, `[[`, "label")

bin_colors <- c("Tropical Storm" = "#FDAE61",
                "Cat 1-2"        = "#E06666",
                "Cat 3+"         = "#7A0019")

class_colors <- c(
  "Treated (TS)"     = "#FDAE61",
  "Treated (Cat1-2)" = "#E06666",
  "Treated (Cat3+)"  = "#7A0019",
  "Clean control"    = "#009E73",
  "Excluded"         = "#CCCCCC"
)
class_order <- names(class_colors)

# ------------------------------------------------------------------
# Single-country plot
# ------------------------------------------------------------------
make_country_plot <- function(panel, iso, country_name) {
  d <- panel %>% filter(ISO == iso) %>% arrange(year)
  n_d <- nrow(d)
  if (n_d == 0) { cat(sprintf("No data for %s, skipping\n", iso)); return(invisible(NULL)) }

  classif_all <- data.frame()
  for (tgt_name in names(targets)) {
    tgt <- targets[[tgt_name]]
    target_vec <- tgt$target_fn(d)
    excl <- exclusion_sets[[tgt_name]]
    for (v in variant_list) {
      cols_to_check <- excl[[v$mode]]
      shock_sum <- rowSums(d[, cols_to_check, drop = FALSE])
      shock <- ifelse(shock_sum > 0, 1, 0)
      win <- sapply(seq_len(n_d), function(i) {
        si <- max(i - v$K, 1); ei <- min(i + H, n_d)
        sum(shock[si:ei], na.rm = TRUE)
      })
      treated_label <- switch(tgt_name,
        "TS target bin"      = "Treated (TS)",
        "Cat1-2 target bin"  = "Treated (Cat1-2)",
        "Cat3+ target bin"   = "Treated (Cat3+)")
      cls <- case_when(
        target_vec == 1              ~ treated_label,
        target_vec == 0 & win == 0   ~ "Clean control",
        TRUE                         ~ "Excluded"
      )
      classif_all <- bind_rows(classif_all, data.frame(
        year = d$year, classification = cls,
        target = tgt_name, variant = v$label,
        stringsAsFactors = FALSE
      ))
    }
  }

  # Y positions of the bottom-panel rows, with a gap between target groups
  row_spacing <- 0.6
  group_gap   <- 1.2
  variant_short <- sapply(variant_list, `[[`, "short")
  grid_info <- data.frame()
  y_cursor <- 0
  for (g in seq_along(target_order)) {
    group_top <- y_cursor
    for (k in seq_along(variant_order)) {
      grid_info <- bind_rows(grid_info, data.frame(
        target  = target_order[g],
        variant = variant_order[k],
        short   = variant_short[k],
        y_pos   = group_top - (k - 1) * row_spacing,
        stringsAsFactors = FALSE
      ))
    }
    y_cursor <- group_top - (length(variant_order) - 1) * row_spacing - group_gap
  }
  classif_all <- classif_all %>% left_join(grid_info, by = c("target","variant"))
  classif_all$classification <- factor(classif_all$classification, levels = class_order)

  variant_counts <- classif_all %>%
    group_by(target, variant, y_pos) %>%
    summarize(n_clean = sum(classification == "Clean control"),
              .groups = "drop")

  group_labels <- grid_info %>%
    group_by(target) %>%
    summarize(y_center = mean(y_pos), .groups = "drop") %>%
    mutate(target = factor(target, levels = target_order))

  # Divider lines between target groups (not above the top group)
  group_tops <- grid_info %>%
    group_by(target) %>%
    summarize(top = max(y_pos) + row_spacing * 0.5, .groups = "drop") %>%
    arrange(desc(top))
  divider_y <- group_tops$top[-1]

  # TC-exposure long-form
  d_long <- d %>%
    pivot_longer(cols = c(ts_v2, cat1_v2, cat3_v2),
                 names_to = "bin", values_to = "exposure") %>%
    mutate(bin = recode(bin,
                         "ts_v2"   = "Tropical Storm",
                         "cat1_v2" = "Cat 1-2",
                         "cat3_v2" = "Cat 3+"),
           bin = factor(bin, levels = c("Tropical Storm","Cat 1-2","Cat 3+")))

  xlim_vals <- c(1978, 2030)  # data viewport; target labels placed outside via clip = off

  # ----- Top panel: bars only, ylim 0..100 -----
  p_top <- ggplot(d_long, aes(x = year, y = exposure, fill = bin)) +
    geom_col(width = 0.7) +
    scale_fill_manual(values = bin_colors, name = "TC bin") +
    scale_y_continuous(limits = c(0, 100),
                        breaks = c(0, 25, 50, 75, 100),
                        expand = c(0, 0)) +
    scale_x_continuous(breaks = seq(1980, 2020, 5), expand = c(0, 0)) +
    coord_cartesian(xlim = xlim_vals, clip = "off") +
    labs(title = paste0(country_name,
                        ": clean-control classification across 4 variants and 3 target bins (h=3)"),
         y = "TC exposure (% country area)",
         x = NULL) +
    theme_bw() + theme(
      legend.position = "right",
      axis.title.y = element_text(size = 10, margin = margin(r = 8)),
      axis.title.x = element_blank(),
      axis.text.x  = element_text(size = 11),
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 12),
      plot.margin = margin(t = 5, r = 5, b = 2, l = 75)
    )

  # ----- Bottom panel: classification dots -----
  y_min <- min(grid_info$y_pos) - 0.8
  y_max <- max(grid_info$y_pos) + 0.8

  p_bot <- ggplot() +
    geom_point(data = classif_all,
               aes(x = year, y = y_pos, color = classification),
               size = 3.2, shape = 15) +
    geom_hline(yintercept = divider_y, color = "grey75",
               linetype = "dotted", linewidth = 0.4) +
    # Target group labels in the left margin
    geom_text(data = group_labels,
              aes(x = 1975.5, y = y_center, label = target),
              angle = 90, fontface = "bold", size = 4, color = "grey10") +
    # Per-row letter labels (a/b/c/d) just before the data starts
    geom_text(data = grid_info,
              aes(x = 1979, y = y_pos, label = short),
              hjust = 0.5, size = 4, color = "grey20") +
    # n_clean at far right
    geom_text(data = variant_counts,
              aes(x = 2021, y = y_pos,
                  label = paste0("n_clean = ", n_clean)),
              hjust = 0, size = 4.5, color = "grey20") +
    scale_color_manual(values = class_colors, name = "Classification",
                        limits = class_order) +
    scale_x_continuous(breaks = seq(1980, 2020, 5), expand = c(0, 0)) +
    scale_y_continuous(limits = c(y_min, y_max), breaks = NULL) +
    coord_cartesian(xlim = xlim_vals, clip = "off") +
    labs(x = NULL, y = NULL, caption = version_caption) +
    theme_bw() + theme(
      legend.position = "right",
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.5, "cm"),
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid = element_blank(),
      panel.border = element_blank(),
      plot.caption = element_text(size = 12, color = "grey20", hjust = 0,
                                   margin = margin(t = 10)),
      plot.margin = margin(t = 2, r = 5, b = 5, l = 75)
    )

  combined <- p_top / p_bot + plot_layout(heights = c(1.0, 2.4))

  fn <- paste0(path_plots, "pretrends_timeline_", iso, "_comparison.png")
  ggsave(fn, combined, width = 14, height = 11, dpi = 300, bg = "white")
  cat(sprintf("Saved: %s\n", fn))

  cat(sprintf("\n=== Classification counts for %s (%s) at h=3 ===\n",
              country_name, iso))
  print(as.data.frame(variant_counts %>% arrange(target, y_pos) %>%
                        dplyr::select(target, variant, n_clean)),
        row.names = FALSE)
  invisible(combined)
}

# ------------------------------------------------------------------
# Generate plots
# ------------------------------------------------------------------
# Jamaica is SI Fig. S11. To plot other countries, pass their ISO3 codes, e.g.
#   Rscript 2_code/supplementary/figures/pretrends_timeline_JAM_comparison.R DOM BRB KNA
isos <- commandArgs(trailingOnly = TRUE)
if (length(isos) == 0) isos <- "JAM"
for (iso in toupper(isos)) {
  country_name <- master$country[master$ISO == iso][1]
  if (is.na(country_name)) stop("ISO3 code '", iso, "' not in country_master.csv")
  make_country_plot(panel, iso = iso, country_name = country_name)
}
