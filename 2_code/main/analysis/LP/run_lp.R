# =============================================================================
# Local-projection (LP) runner: TC + temperature shocks -> debt / GDP
# =============================================================================
# Every coefficient file is described by one row of model_variants_INPUT.csv
# (same folder). This script turns a row into a panel, a fixest formula and an
# output file name, runs the LP for horizons 0..15 and writes
#   <subset>_<outcome>_<tag>_TC.csv     TC (and temperature) coefficients per horizon
#   <subset>_<outcome>_<tag>_TEMP.csv   temperature effects at mean temperature 5/15/25 C
# where <tag> is built from the row's settings (e.g. 1980onwards_gl2_TC10_T02_DK).
#
# Usage (from anywhere inside the repo):
#   Rscript 2_code/main/analysis/LP/run_lp.R                  # first row (main model)
#   Rscript 2_code/main/analysis/LP/run_lp.R --variant DK     # one named row
#   Rscript 2_code/main/analysis/LP/run_lp.R --all            # every row
#   ... --out-dir /tmp/lp_check                               # write elsewhere (for checks)
#
# Inputs: 1_input/climate_exposures/era5_v4_Tshocks.csv, 0_raw/climate_exposures/tc2_maxws.csv,
#         0_raw/climate_exposures/country_master.csv, 1_input/economic_outcomes/lp_outcomes.csv
#         (written by 2_code/data_prep/prep_data.R), and the country groups in
#         2_code/data_prep/categories.R.
# =============================================================================

suppressPackageStartupMessages({
  library(fixest); library(dplyr); library(broom)
})

root <- here::here()
LAGS <- 15                       # IRF horizons 0..LAGS
MEANTEMPS <- c(5, 15, 25)        # mean temperatures for the TEMP marginal effects
N_BOOT <- 500                    # wild cluster bootstrap draws

# --- inputs -------------------------------------------------------------------
variants <- read.csv(file.path(root, "2_code", "main", "analysis", "LP", "model_variants_INPUT.csv"),
                     stringsAsFactors = FALSE, comment.char = "#", strip.white = TRUE)
era5     <- read.csv(file.path(root, "1_input", "climate_exposures", "era5_v4_Tshocks.csv"))
tc       <- read.csv(file.path(root, "0_raw", "climate_exposures", "tc2_maxws.csv"))
master   <- read.csv(file.path(root, "0_raw", "climate_exposures", "country_master.csv"))
outcomes <- read.csv(file.path(root, "1_input", "economic_outcomes", "lp_outcomes.csv"))

sovereign_ISOs <- master$ISO[master$sovereigncountry == 1]
region_df <- master %>% distinct(ISO, region = Region.Name) %>% distinct(ISO, .keep_all = TRUE)
cats <- new.env()                # oecd, sids, rated, non_rated, ...
sys.source(file.path(root, "2_code", "data_prep", "categories.R"), envir = cats)

# Country subsets (all restricted to sovereign countries).
SUBSETS <- list(
  full            = sovereign_ISOs,
  full2           = c(sovereign_ISOs, "PRI", "ABW"),
  remove_outliers = setdiff(sovereign_ISOs, c("NIC", "STP", "LBR", "SDN", "VEN", "ARG")),
  oecd            = intersect(sovereign_ISOs, cats$oecd),
  `non-oecd`      = setdiff(sovereign_ISOs, cats$oecd),
  sids            = intersect(sovereign_ISOs, cats$sids),
  `non-sids`      = setdiff(sovereign_ISOs, cats$sids),
  rated           = intersect(sovereign_ISOs, cats$rated),
  `non-rated`     = intersect(sovereign_ISOs, cats$non_rated),
  rated_Agrade    = intersect(sovereign_ISOs, master$ISO[master$Ainvestmentgrade == 1]),
  rated_Bgrade    = intersect(sovereign_ISOs, master$ISO[master$Binvestmentgrade == 1])
)

# Temperature-shock definitions (column used as `temp`).
TSHOCKS <- c(NATH = "Tshock_NATH", Tshock0 = "Tshock_0", BILAL = "Tshock_BILAL", temp = "temp_pw")

# Fixed effects by (fixed_effects, time_trend).
FE <- list(
  ISO_year    = c(no_trend = "ISO + year", linear = "ISO + year + ISO[[year]]",
                  quadratic = "ISO + year + ISO[[year]] + ISO[[year2]]"),
  year_trend  = c(linear = "year + ISO[[year]]"),      # no country FE, country trends kept
  year        = c(linear = "year"),
  region_year = c(linear = "region + year + region[[year]]")
)

# Standard errors.
VCOV <- list(cluster_ISO = ~ISO, cluster_region = ~region, DK = "DK", hetero = "hetero",
             fixest_default = NULL)   # fixest default: cluster by the first fixed effect

# Tag suffixes used in the output file names.
SUFFIX_VCOV <- c(cluster_ISO = "", fixest_default = "", DK = "_DK", hetero = "_WHITE",
                 cluster_region = "_cluster_region")
SUFFIX_FE   <- c(ISO_year = "", year_trend = "_yearFEonly", year = "_yearFEonly",
                 region_year = "_year_regionFEonly")

# --- panel --------------------------------------------------------------------
# Samples:
#   1980onwards  1980-2019 (main model)      1970onwards  1970-2019
#   1980_2022    1980-2022                   unbalanced   all years
#   40yrs_data / 30yrs_data  all years, countries with >= 40 / 30 years of outcome data before 2020
build_panel <- function(outcome, sample) {
  out <- outcomes %>% filter(outcome == !!outcome) %>% select(ISO, year, value)
  if (nrow(out) == 0) stop("outcome '", outcome, "' not in lp_outcomes.csv (re-run prep_data.R?)")
  p <- era5 %>% rename(temp_pw = temp) %>%
    left_join(region_df, by = "ISO") %>%
    left_join(tc, by = c("ISO", "year")) %>%
    left_join(out, by = c("ISO", "year"))
  p <- switch(sample,
    `1980onwards` = filter(p, year >= 1980, year < 2020),
    `1970onwards` = filter(p, year >= 1970, year < 2020),
    `1980_2022`   = filter(p, year >= 1980),
    unbalanced    = p,
    `40yrs_data`  = ,
    `30yrs_data`  = {
      n_min <- as.integer(sub("yrs_data", "", sample))
      keep <- p %>% filter(year < 2020, !is.na(value)) %>% count(ISO) %>% filter(n >= n_min)
      filter(p, ISO %in% keep$ISO)
    },
    stop("unknown sample '", sample, "'"))
  p %>% group_by(ISO) %>%
    mutate(meantemp    = meantemp_hist,   # mean temperature before 1980 (state dependence)
           outcome_LD1 = log(lag(value, 1)) - log(lag(value, 2)),
           outcome_LD2 = log(lag(value, 2)) - log(lag(value, 3)))
}

# --- formula ------------------------------------------------------------------
build_formula <- function(v, trend) {
  L <- v$tc_lags
  tc_terms <- if (v$exposure == "avg_ws") sprintf("l(avg_ws, 0:%d)", L) else
    sprintf("l(%s, 0:%d)", c("ts_v2", "cat1_v2", "cat3_v2"), L)
  rhs <- c(tc_terms,
           if (v$temp_controls == "yes")
             c("temp", "temp:meantemp", "l(temp_pw, 1:2)", "l(temp_pw, 1:2):meantemp"),
           paste0("outcome_LD", seq_len(v$growth_lags)))
  fe <- FE[[v$fixed_effects]][[trend]]
  if (is.null(fe)) stop("time trend '", trend, "' not defined for fixed_effects '", v$fixed_effects, "'")
  as.formula(paste("log_longdiff ~", paste(rhs, collapse = " + "), "|", fe))
}

variant_tag <- function(v) {
  if (!is.na(v$tag) && nzchar(v$tag)) return(v$tag)
  paste0(v$sample, "_gl", v$growth_lags, "_TC", v$tc_lags,
         if (v$temp_controls == "yes") "_T02" else "",
         SUFFIX_FE[[v$fixed_effects]], SUFFIX_VCOV[[v$vcov]],
         if (v$exposure == "avg_ws") "_avgws" else "",
         if (v$estimator == "wild_bootstrap") "_boot_wild" else "")
}

# --- estimation ---------------------------------------------------------------
KEEP_TERMS <- c("ts_v2", "cat1_v2", "cat3_v2", "avg_ws", "temp", "temp:meantemp")

fit <- function(fml, data, vcov_spec) {
  if (identical(vcov_spec, "DK")) vcov_spec <- DK(lag = 2)
  if (is.null(vcov_spec)) feols(fml, data, notes = FALSE)
  else feols(fml, data, vcov = vcov_spec, notes = FALSE)
}

# One horizon: plain LP, or wild cluster bootstrap around it.
estimate_horizon <- function(panel3, ll, fml, v) {
  panel4 <- panel3 %>% group_by(ISO) %>%
    mutate(log_longdiff = log(lead(value, n = ll)) - log(lag(value, n = 1)))
  panel4 <- fixest::panel(panel4, ~ ISO + year, duplicate.method = "first")
  mod <- fit(fml, panel4, VCOV[[v$vcov]])
  has_temp <- "temp:meantemp" %in% names(coef(mod))
  cov_tt <- if (has_temp) vcov(mod)["temp", "temp:meantemp"] else NA_real_

  if (v$estimator == "lp") {
    coefs <- tidy(mod, conf.int = TRUE) %>% filter(term %in% KEEP_TERMS) %>% mutate(y_h = ll)
    return(list(coef = coefs, cov = data.frame(covariance = cov_tt, y_h = ll), boot = NULL))
  }

  # Wild cluster bootstrap (Rademacher weights by country) on the fitted model.
  # Seeded per horizon so a re-run gives the same intervals.
  set.seed(20240101 + ll)
  d <- panel4[obs(mod), ]
  d$y_hat <- fitted(mod); d$u_hat <- resid(mod)
  isos <- unique(d$ISO)
  boot <- bind_rows(lapply(seq_len(N_BOOT), function(k) {
    w <- setNames(sample(c(-1, 1), length(isos), replace = TRUE), isos)
    d$log_longdiff <- d$y_hat + d$u_hat * w[as.character(d$ISO)]
    db <- fixest::panel(ungroup(d), ~ ISO + year)
    tidy(fit(fml, db, VCOV[[v$vcov]])) %>% filter(term %in% KEEP_TERMS) %>%
      transmute(term, estimate, bootstrap = k)
  }))
  coefs <- tidy(mod) %>% filter(term %in% KEEP_TERMS) %>%
    transmute(term, orig_estimate = estimate, y_h = ll) %>%
    left_join(boot %>% group_by(term) %>%
                summarise(conf.low = quantile(estimate, 0.025), conf.high = quantile(estimate, 0.975),
                          boot_se = sd(estimate), n_boot = n(), .groups = "drop"), by = "term")
  list(coef = coefs, cov = data.frame(covariance = cov_tt, y_h = ll), boot = mutate(boot, y_h = ll))
}

# Temperature effect at mean temperature mt: b1 + mt * b2 (delta-method SE, or bootstrap CI).
temp_effects <- function(res, v) {
  bind_rows(lapply(MEANTEMPS, function(mt) {
    if (v$estimator == "lp") {
      b1 <- res$coef %>% filter(term == "temp") %>% arrange(y_h)
      b2 <- res$coef %>% filter(term == "temp:meantemp") %>% arrange(y_h)
      cv <- res$cov %>% arrange(y_h)
      data.frame(y_h = b1$y_h, meantemp = mt, est = b1$estimate + mt * b2$estimate,
                 se = sqrt(b1$std.error^2 + mt^2 * b2$std.error^2 + 2 * mt * cv$covariance)) %>%
        mutate(conf.low = est - 1.96 * se, conf.high = est + 1.96 * se)
    } else {
      pt <- res$coef %>% select(term, y_h, orig_estimate) %>%
        tidyr::pivot_wider(names_from = term, values_from = orig_estimate)
      bs <- res$boot %>% tidyr::pivot_wider(names_from = term, values_from = estimate) %>%
        mutate(est = temp + mt * `temp:meantemp`) %>% group_by(y_h) %>%
        summarise(conf.low = quantile(est, 0.025), conf.high = quantile(est, 0.975),
                  boot_se = sd(est), n_boot = n(), .groups = "drop")
      transmute(pt, y_h, meantemp = mt, orig_estimate = temp + mt * `temp:meantemp`) %>%
        left_join(bs, by = "y_h")
    }
  }))
}

run_variant <- function(v, out_root) {
  tag <- variant_tag(v)
  out_dir <- file.path(out_root, v$out_folder)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  panel2 <- build_panel(v$outcome, v$sample)
  trends <- strsplit(v$time_trends, ";")[[1]]
  shocks <- strsplit(v$tshocks, ";")[[1]]
  for (subset in strsplit(v$subsets, ";")[[1]]) {
    if (is.null(SUBSETS[[subset]])) stop("unknown subset '", subset, "'")
    panel3 <- panel2 %>% filter(ISO %in% SUBSETS[[subset]])
    tc_out <- list(); temp_out <- list()
    for (trend in trends) {
      fml <- build_formula(v, trend)
      for (sh in shocks) {
        panel3$temp <- panel3[[TSHOCKS[[sh]]]]
        hs <- lapply(0:LAGS, function(ll) estimate_horizon(panel3, ll, fml, v))
        res <- list(coef = bind_rows(lapply(hs, `[[`, "coef")),
                    cov  = bind_rows(lapply(hs, `[[`, "cov")),
                    boot = bind_rows(lapply(hs, `[[`, "boot")))
        meta <- data.frame(outcome = v$outcome, specification = tag, model = trend,
                           subset = subset, Tshock_type = sh)
        tc_out[[length(tc_out) + 1]] <- cbind(mutate(res$coef, Tshock_type = sh), meta[, 1:4])
        if (v$temp_controls == "yes")
          temp_out[[length(temp_out) + 1]] <- cbind(temp_effects(res, v), meta)
      }
    }
    stem <- file.path(out_dir, paste0(subset, "_", v$outcome, "_", tag))
    write.csv(bind_rows(tc_out), paste0(stem, "_TC.csv"), row.names = FALSE)
    if (length(temp_out)) write.csv(bind_rows(temp_out), paste0(stem, "_TEMP.csv"), row.names = FALSE)
    cat(sprintf("  saved %s_{TC,TEMP}.csv\n", stem))
  }
}

# --- CLI ----------------------------------------------------------------------
# Only when run as a script; source()-ing this file just defines build_panel() etc.
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  out_root <- file.path(root, "3_output", "debtratio", "TC+Tshock", "coef_df")
  if ("--out-dir" %in% args) out_root <- args[which(args == "--out-dir") + 1]
  rows <- if ("--all" %in% args) {
    seq_len(nrow(variants))
  } else if ("--variant" %in% args) {
    which(variants$name == args[which(args == "--variant") + 1])
  } else 1
  if (!length(rows)) stop("no row named '", args[which(args == "--variant") + 1], "' in model_variants_INPUT.csv")
  for (i in rows) {
    v <- variants[i, ]
    cat(sprintf("[%d/%d] %s\n", match(i, rows), length(rows), v$name))
    run_variant(v, out_root)
  }
}
