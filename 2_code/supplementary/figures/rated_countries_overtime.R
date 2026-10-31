# rated_countries_overtime.png  (SI figure)
#
# Number of sovereign countries with a credit rating, by year, for each rating
# source (Total / S&P / Moody's / Fitch).
#
# Input:  1_input/credit_ratings/rated_countries_by_source.csv (counts per year; a shipped
#         aggregate of the proprietary per-agency ratings, written by
#         2_code/data_prep/build_proprietary_aggregates.py)
# Output: 4_figures/supplementary/rated_countries_overtime.png

library(here)
library(dplyr)
library(ggplot2)

t <- read.csv(here("1_input", "credit_ratings", "rated_countries_by_source.csv")) %>%
  mutate(source = as.factor(source))

p <- ggplot(t, aes(x = year, y = count, color = source)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(limits = c(0, 160)) +
  labs(
    title = "number of sovereign countries with ratings",
    x = "",
    y = "Count",
    color = "Source"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(size = 11))

out <- here("4_figures", "supplementary", "rated_countries_overtime.png")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
ggsave(out, plot = p, width = 7, height = 4, bg = "white")
cat("wrote", out, "\n")
