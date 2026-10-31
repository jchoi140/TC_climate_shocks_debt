library(dplyr)
library(ggplot2)
library(forcats)
setwd(here::here())
source("2_code/data_prep/categories.R")  # country groups (highincome, oecd, lowincome, sids, ...)
defaults_SI = read.csv("./1_input/economic_outcomes/defaults_history_SI.csv")

remove_advanced <- c(
  "AUS", "AUT", "BEL", "BHR", "BRN", "CAN", "CHE", "CHN", "CZE", "DEU", 
  "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "ISL", "ISR", "ITA", "JPN", 
  "KOR", "KWT", "LVA", "LTU", "LUX", "MLT", "NLD", "NOR", "NZL", "OMN", 
  "QAT", "SAU", "SGP", "SVK", "SWE", "TWN", "USA"
) # advanced economies added with zero defaults for completeness; excluded below

non_na_FC_bonds <- defaults_SI %>%
  filter(!ISO %in% remove_advanced, 
        !is.na(default_FC_bonds)) %>%
  distinct(ISO) %>%
  pull(ISO)

non_na_FC_bankloans <- defaults_SI %>%
  filter(!ISO %in% remove_advanced, 
        !is.na(default_FC_bankloans)) %>%
  distinct(ISO) %>%
  pull(ISO)

non_na_LC_debt <- defaults_SI %>%
  filter(!ISO %in% remove_advanced, 
        !is.na(default_LC_debt)) %>%
  distinct(ISO) %>%
  pull(ISO)


plot_defaults <- function(group) {
  title_name <- deparse(substitute(group))

  default_heatmap_long <- defaults_SI %>%
    dplyr::filter(year >= 1960, year <= 2020, 
                  ISO %in% group) %>% 
    dplyr::select(ISO, year, default_total) %>%
    dplyr::mutate(year = as.integer(year))

  ggplot(default_heatmap_long, aes(x = year, y = fct_rev(ISO), fill = factor(default_total))) +
    geom_tile(color = "white") +
    scale_fill_manual(
      values = c("0" = "#FFCB61", "1" = "#EA5B6F"),
      labels = c("0" = "no default", "1" = "in default"),
      na.value = "grey"
    ) +
    labs(
      title = paste0("Sovereign defaults: ", title_name),
      x = "",
      y = "Country (ISO3)",
      fill = "default status"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(hjust = 1))
}

plot_defaults_type <- function(group, type) {
  group_name <- deparse(substitute(group))
  type_name  <- deparse(substitute(type))
  
  default_heatmap_long <- defaults_SI %>%
    dplyr::filter(year >= 1960, year <= 2020, ISO %in% group) %>%
    dplyr::select(ISO, year, {{ type }}) %>%
    dplyr::mutate(year = as.integer(year))
  
  ggplot(default_heatmap_long, aes(x = year, y = fct_rev(ISO), fill = factor({{ type }}))) +
    geom_tile(color = "white") +
    scale_fill_manual(
      values = c("0" = "#FFCB61", "1" = "#EA5B6F"),
      labels = c("0" = "no default", "1" = "in default"),
      na.value = "grey"
    ) +
    labs(
      title = paste0("Sovereign defaults: ", group_name, " — ", type_name),
      x = "",
      y = "Country (ISO3)",
      fill = "default status"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(hjust = 1))
}


plot_defaults(highincome)
ggsave("4_figures/supplementary/default_history/defaults_highincome.png", plot = last_plot(), width = 8, height = 6, dpi = 300, bg = "white")
plot_defaults(oecd)
ggsave("4_figures/supplementary/default_history/defaults_oecd.png", plot = last_plot(), width = 8, height = 6, dpi = 300, bg = "white")
plot_defaults(lowincome)
ggsave("4_figures/supplementary/default_history/defaults_lowincome.png", plot = last_plot(), width = 8, height = 6, dpi = 300, bg = "white")
plot_defaults(sids)
ggsave("4_figures/supplementary/default_history/defaults_sids.png", plot = last_plot(), width = 8, height = 6, dpi = 300, bg = "white")
plot_defaults(sids_carib)
plot_defaults(sids_pacific)
