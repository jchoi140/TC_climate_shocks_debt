# =============================================================================
# pretrend_appendix_lpdid_combined_v3_debt.png — pre-trend appendix figure:
# standard LP (top), one-sided LP-DiD (middle) and two-sided LP-DiD (bottom).
# In the bottom group, solid = pre-window [t-3, t+h], dotted = [t-10, t+h].
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(tidyr); library(patchwork); library(ggh4x)
})

project_root <- here::here()
setwd(project_root)
mo <- "3_output/S_pretrends/"

term_labels <- c(ts_v2 = "Tropical Storm", cat1_v2 = "Category 1-2", cat3_v2 = "Category 3+")
term_levels <- names(term_labels)
ylim_v <- c(-0.005, 0.015)

h_filter <- function(d) {
  d %>% filter(y_h >= -10, y_h <= 20)
}

ld    <- read.csv(paste0(mo, "lpdid_subsets_TC.csv"))
ld10  <- read.csv(paste0(mo, "lpdid_prebuf10_TC.csv"))
get_lpdid <- function(model_name) {
  ld %>% filter(model == model_name, subset == "full", bin_spec == "3bin",
                  term %in% term_levels) %>% h_filter()
}
get_lpdid10 <- function(model_name) {
  ld10 %>% filter(model == model_name, subset == "full", bin_spec == "3bin",
                    term %in% term_levels) %>% h_filter()
}

lpdid_row_levels <- c("A. cat3+ excluded",
                       "B. target bin\nexcluded",
                       "C. all bins excluded")
sl_row_label <- "Standard LP"
row_levels <- c(sl_row_label, lpdid_row_levels)
group_levels <- c(" ",
                   "LP-DiD (clean control [t, t+h])",
                   "LP-DiD (clean control [t-n, t+h])")

# --- Standard LP (top group) -----------------------------------------------
d_std <- read.csv(paste0(mo,
  "full_public_pp_GDD_1980onwards_linear_quadratic_pretrends_TC.csv")) %>%
  filter(model == "linear", subset == "full", term %in% term_levels) %>%
  h_filter() %>%
  mutate(row_label = sl_row_label, color_key = "STD",
         group_label = group_levels[1])

# --- Onesided variants (middle group) --------------------------------------
d_onesided <- bind_rows(
  get_lpdid("lpdid_hi_region7_fe_onesided_v2")        %>% mutate(row_label = lpdid_row_levels[1], color_key = "A_one"),
  get_lpdid("lpdid_hi_region7_fe_onesided_targetonly")%>% mutate(row_label = lpdid_row_levels[2], color_key = "B_one"),
  get_lpdid("lpdid_hi_region7_fe_onesided_allTC")     %>% mutate(row_label = lpdid_row_levels[3], color_key = "C_one")
) %>% mutate(group_label = group_levels[2])

# --- Prebuf3 variants (bottom group, solid line + ribbon) ------------------
d_prebuf3 <- bind_rows(
  get_lpdid("lpdid_hi_region7_fe_prebuf3_v2")        %>% mutate(row_label = lpdid_row_levels[1], color_key = "A_two"),
  get_lpdid("lpdid_hi_region7_fe_prebuf3_targetonly")%>% mutate(row_label = lpdid_row_levels[2], color_key = "B_two"),
  get_lpdid("lpdid_hi_region7_fe_prebuf3_allTC")     %>% mutate(row_label = lpdid_row_levels[3], color_key = "C_two")
) %>% mutate(group_label = group_levels[3])

# --- Prebuf10 variants (bottom group, dotted overlay, no ribbon) -----------
d_prebuf10 <- bind_rows(
  get_lpdid10("lpdid_hi_region7_fe_prebuf10_v2")        %>% mutate(row_label = lpdid_row_levels[1], color_key = "A_two"),
  get_lpdid10("lpdid_hi_region7_fe_prebuf10_targetonly")%>% mutate(row_label = lpdid_row_levels[2], color_key = "B_two"),
  get_lpdid10("lpdid_hi_region7_fe_prebuf10_allTC")     %>% mutate(row_label = lpdid_row_levels[3], color_key = "C_two")
) %>% mutate(group_label = group_levels[3])

apply_factors <- function(df) df %>% mutate(
  tc_label    = factor(term_labels[term], levels = unname(term_labels)),
  row_label   = factor(row_label, levels = row_levels),
  group_label = factor(group_label, levels = group_levels))

d_solid  <- bind_rows(d_std, d_onesided, d_prebuf3) %>% apply_factors()
d_dotted <- d_prebuf10 %>% apply_factors()

color_map <- c(
  "STD"   = "#1F1F1F",
  "A_one" = "#1B5E20", "B_one" = "#66BB6A", "C_one" = "#A5D6A7",
  "A_two" = "#0D47A1", "B_two" = "#42A5F5", "C_two" = "#90CAF9"
)

# --- Panel-local legend (row 5 = bottom group, A row, TS column) -----------
legend_panel <- data.frame(
  group_label = factor(group_levels[3], levels = group_levels),
  row_label   = factor(lpdid_row_levels[1], levels = row_levels),
  tc_label    = factor("Tropical Storm", levels = unname(term_labels))
)
legend_seg <- data.frame(
  legend_panel,
  x = c(-9.5, -9.5), xend = c(-6.5, -6.5),
  y = c(0.0135, 0.0115), yend = c(0.0135, 0.0115),
  lt = c("solid", "dotted")
)
legend_txt <- data.frame(
  legend_panel,
  x = c(-6, -6), y = c(0.0135, 0.0115),
  label = c("[t-3, t+h]", "[t-10, t+h]")
)

p <- ggplot() +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed") +
  geom_vline(xintercept = -0.5, color = "red", linetype = "dashed", alpha = 0.5) +
  # solid: ribbon + line + points
  geom_ribbon(data = d_solid,
              aes(x = y_h, ymin = conf.low, ymax = conf.high,
                  fill = color_key, group = color_key),
              alpha = 0.18, color = NA, show.legend = FALSE) +
  geom_line(data = d_solid,
            aes(x = y_h, y = estimate, color = color_key, group = color_key),
            linewidth = 0.7, show.legend = FALSE) +
  geom_point(data = d_solid,
             aes(x = y_h, y = estimate, color = color_key, group = color_key),
             size = 0.8, show.legend = FALSE) +
  # dotted overlay (prebuf10): line only
  geom_line(data = d_dotted,
            aes(x = y_h, y = estimate, color = color_key, group = color_key),
            linetype = "dotted", linewidth = 0.7, show.legend = FALSE) +
  # panel-local legend
  geom_segment(data = legend_seg,
               aes(x = x, xend = xend, y = y, yend = yend, linetype = lt),
               color = "grey15", linewidth = 0.6, inherit.aes = FALSE) +
  geom_text(data = legend_txt,
            aes(x = x, y = y, label = label),
            hjust = 0, size = 3.2, color = "grey15", inherit.aes = FALSE) +
  facet_nested(group_label + row_label ~ tc_label,
                nest_line = element_line(linetype = 1),
                strip = strip_nested(
                  text_y = elem_list_text(face = c("bold", "bold"), size = c(14, 13)),
                  background_y = elem_list_rect(fill = c("grey75", "grey85")))) +
  scale_color_manual(values = color_map) +
  scale_fill_manual(values  = color_map) +
  scale_linetype_manual(values = c("solid" = "solid", "dotted" = "dotted"),
                         guide = "none") +
  coord_cartesian(ylim = ylim_v) +
  labs(x = "Horizon (years)", y = "Coefficient") +
  theme_bw() + theme(
    strip.text.x       = element_text(face = "bold", size = 14),
    strip.background.x = element_rect(fill = "grey90", color = NA),
    axis.text          = element_text(size = 10),
    axis.title         = element_text(size = 11),
    panel.grid.minor   = element_blank(),
    panel.spacing      = unit(0.4, "lines")
  )

# --- Remove (transparent) outer strip for the Standard LP row --------------
suppressPackageStartupMessages(library(grid))
g <- ggplotGrob(p)
idx <- which(g$layout$name == "strip-r-1")
if (length(idx) > 0) g$grobs[[idx]] <- nullGrob()

fn <- "4_figures/supplementary/S_pretrends/pretrend_appendix_lpdid_combined_v3_debt.png"
ggsave(fn, g, width = 16, height = 16, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", fn))
