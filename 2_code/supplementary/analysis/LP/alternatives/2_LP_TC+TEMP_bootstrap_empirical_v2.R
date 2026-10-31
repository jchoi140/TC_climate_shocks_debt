# Main LP model with country-cluster bootstrap confidence intervals

library(data.table)
setDTthreads(1)  # multithreaded copy() can segfault (data.table 1.17.8, R 4.3.1)

# Panel of the main model (public debt, 1980-2019), built by the LP runner's build_panel();
# also defines sovereign_ISOs and loads fixest / dplyr / broom.
source(here::here("2_code", "main", "analysis", "LP", "run_lp.R"))
select <- "public_pp_GDD"
panel2 <- build_panel(select, "1980onwards")
coef_list <- list()
set.seed(20240101)  # country resampling, so a re-run gives the same intervals

path0 = paste0(here::here("3_output"), "/")
path1 = "debtratio/TC+Tshock/coef_df/bootstrap/"
path2 = "1980onwards_gl2_TC10_T02_boot_empirical" 

subset_values <- c("full")

Tshock = list(
    NATH = "Tshock_NATH"
)

models = list( 
    linear = "feols(log_longdiff ~ l(ts_v2, 0:10) + l(cat1_v2, 0:10) + l(cat3_v2, 0:10) + temp + temp:meantemp + l(temp_pw, 1:2) + l(temp_pw, 1:2):meantemp + outcome_LD1 + outcome_LD2 | unique_id + year + unique_id[[year]], dtboot, notes=FALSE)"
)

##### RUNNING THE MODEL ######
nboot = 500  # number of bootstrap resamples (500-1000 recommended)
lags <- 15

coef_terms <- c('ts_v2', 'cat1_v2', 'cat3_v2', 'temp', 'temp:meantemp')
lagwindow <- 0:lags

orig_coef_list <- list()
orig_cov_list <- list()
boot_coef_list <- list()  # Store raw bootstrap estimates
boot_cov_list <- list()

for (subset in subset_values) {
    print(paste("Processing subset:", subset))
    
    panel3 <- switch(subset,
        "full" = panel2 %>% filter(ISO %in% sovereign_ISOs)
    )
    
    panel3_dt <- as.data.table(panel3)
    unique_isos <- unique(panel3$ISO)
    n_isos <- length(unique_isos)

    for (i in seq_along(models)) {
        model <- models[[i]]
        model_name <- names(models)[i]
        print(paste("generating results for:", model_name))
        
        Nboot <- nboot

        boot_coef_results <- vector("list", length(Tshock) * (lags + 1) * Nboot)
        boot_cov_results <- vector("list", length(Tshock) * (lags + 1) * Nboot)
        result_idx <- 1
        
        orig_coef_results <- vector("list", length(Tshock) * (lags + 1))
        orig_cov_results <- vector("list", length(Tshock) * (lags + 1))
        orig_idx <- 1

        for (j in seq_along(Tshock)) {
            Tshock_name <- names(Tshock)[j]
            Tshock_var <- Tshock[[j]]
            print(Tshock_name)
            panel3_dt[, temp := .SD[[Tshock_var]]]

            for (ll in 0:lags) {

                panel4_dt <- panel3_dt[, .(
                    ISO, year, temp, meantemp, ts_v2, cat1_v2, cat3_v2, 
                    temp_pw, outcome_LD1, outcome_LD2, value,
                    log_longdiff = log(shift(value, n = ll, type = "lead")) - log(shift(value, n = 1, type = "lag")),
                    longdiff = shift(value, n = ll, type = "lead") - shift(value, n = 1, type = "lag")
                ), by = ISO]

                # ============================================================
                # ORIGINAL ESTIMATES (full data, no bootstrap)
                # ============================================================
                dtboot_orig <- as.data.frame(panel4_dt)
                dtboot_orig <- dtboot_orig[, !duplicated(names(dtboot_orig))]
                dtboot_orig$unique_id <- dtboot_orig$ISO
                dtboot_orig <- panel(dtboot_orig, ~ unique_id + year)
                
                dtboot <- dtboot_orig
                mod_orig <- eval(parse(text = model))
                
                # Save ONLY the estimate (no conf intervals from analytical SEs)
                mod_tidy_orig <- tidy(mod_orig)
                mod_temp_orig <- mod_tidy_orig[mod_tidy_orig$term %in% coef_terms, c("term", "estimate")]
                mod_temp_orig$y_h <- ll
                mod_temp_orig$Tshock_type <- Tshock_name
                orig_coef_results[[orig_idx]] <- mod_temp_orig
                
                vcov_matrix_orig <- vcov(mod_orig)
                orig_cov_results[[orig_idx]] <- data.frame(
                    covariance = vcov_matrix_orig['temp', 'temp:meantemp'],
                    y_h = ll,
                    Tshock_type = Tshock_name
                )
                orig_idx <- orig_idx + 1

                # ============================================================
                # BOOTSTRAP ITERATIONS - only save estimates
                # ============================================================
                for (k in 1:Nboot) {
                    sampled_isos <- sample(unique_isos, size = n_isos, replace = TRUE)
                    cl_dt <- data.table(iso = sampled_isos)
                    
                    dtboot_dt <- panel4_dt[cl_dt, on = .(ISO = iso), allow.cartesian = TRUE]
                    
                    dtboot <- as.data.frame(dtboot_dt)
                    dtboot <- dtboot[, !duplicated(names(dtboot))]
                    
                    dtboot <- dtboot %>%
                        group_by(ISO, year) %>%
                        mutate(unique_id = paste(ISO, row_number(), sep = "_")) %>%
                        ungroup()
                    
                    dtboot <- panel(dtboot, ~ unique_id + year)
                    mod <- eval(parse(text = model))

                    mod_tidy <- tidy(mod)
                    mod_temp <- mod_tidy[mod_tidy$term %in% coef_terms, c("term", "estimate")]
                    mod_temp$y_h <- ll
                    mod_temp$bootstrap <- k
                    mod_temp$Tshock_type <- Tshock_name
                    
                    boot_coef_results[[result_idx]] <- mod_temp

                    vcov_matrix <- vcov(mod)
                    boot_cov_results[[result_idx]] <- data.frame(
                        covariance = vcov_matrix['temp', 'temp:meantemp'],
                        y_h = ll,
                        bootstrap = k,
                        Tshock_type = Tshock_name
                    )
                    
                    result_idx <- result_idx + 1
                }

                print(ll)
            }

            print(paste("all processed for:", Tshock_name))
        }

        boot_coef_df <- rbindlist(boot_coef_results, use.names = TRUE, fill = TRUE) %>% as.data.frame()
        boot_cov_df <- rbindlist(boot_cov_results, use.names = TRUE) %>% as.data.frame()
        
        orig_coef_df <- rbindlist(orig_coef_results, use.names = TRUE, fill = TRUE) %>% as.data.frame()
        orig_cov_df <- rbindlist(orig_cov_results, use.names = TRUE) %>% as.data.frame()
        
        # ============================================================
        # CONSTRUCT BOOTSTRAP CONFIDENCE INTERVALS FOR _TC.csv
        # ============================================================
        boot_ci <- boot_coef_df %>%
            group_by(term, y_h, Tshock_type) %>%
            summarise(
                conf.low = quantile(estimate, 0.025),
                conf.high = quantile(estimate, 0.975),
                boot_se = sd(estimate),  # bootstrap standard error
                n_boot = n(),
                .groups = "drop"
            )
        
        # Merge original estimate with bootstrap CIs
        coef_Tshocks <- orig_coef_df %>%
            rename(orig_estimate = estimate) %>%
            left_join(boot_ci, by = c("term", "y_h", "Tshock_type"))
        
        coef_Tshocks$outcome <- select
        coef_Tshocks$specification <- path2
        coef_Tshocks$model <- model_name
        coef_Tshocks$subset <- subset

        coef_list[[model_name]] <- coef_Tshocks
        
        # Store for TEMP calculation
        orig_coef_list[[model_name]] <- orig_coef_df
        orig_cov_list[[model_name]] <- orig_cov_df
        boot_coef_list[[model_name]] <- boot_coef_df
        boot_cov_list[[model_name]] <- boot_cov_df
    }

    write.csv(do.call(rbind, coef_list), paste0(path0, path1, subset, "_", select, "_", path2, "_TC.csv"), row.names = FALSE)

    # ============================================================
    # CONSTRUCT BOOTSTRAP CIs FOR _TEMP.csv
    # ============================================================
    
    coef_templist <- list()

    for (i in seq_along(models)) {
        for (j in seq_along(Tshock)) {
            
            # Original estimates
            orig_coef_df <- orig_coef_list[[names(models)[i]]] %>%
                filter(Tshock_type == names(Tshock)[j])
            orig_cov_df <- orig_cov_list[[names(models)[i]]] %>%
                filter(Tshock_type == names(Tshock)[j])
            
            orig_b1 <- orig_coef_df %>% filter(term == "temp") %>% select(y_h, est_b1 = estimate)
            orig_b2 <- orig_coef_df %>% filter(term == "temp:meantemp") %>% select(y_h, est_b2 = estimate)
            
            # Bootstrap estimates
            boot_coef_df <- boot_coef_list[[names(models)[i]]] %>%
                filter(Tshock_type == names(Tshock)[j])
            boot_cov_df <- boot_cov_list[[names(models)[i]]] %>%
                filter(Tshock_type == names(Tshock)[j])
            
            boot_b1 <- boot_coef_df %>% filter(term == "temp") %>% select(y_h, bootstrap, est_b1 = estimate)
            boot_b2 <- boot_coef_df %>% filter(term == "temp:meantemp") %>% select(y_h, bootstrap, est_b2 = estimate)
            
            meantemp_values <- c(5, 15, 25)
            
            temp_results <- lapply(meantemp_values, function(mt) {
                
                # Original estimate for this meantemp
                orig_temp <- orig_b1 %>%
                    inner_join(orig_b2, by = "y_h") %>%
                    mutate(
                        meantemp = mt,
                        orig_estimate = est_b1 + mt * est_b2
                    ) %>%
                    select(y_h, meantemp, orig_estimate)
                
                # Bootstrap distribution of estimates
                boot_temp <- boot_b1 %>%
                    inner_join(boot_b2, by = c("y_h", "bootstrap")) %>%
                    mutate(
                        est = est_b1 + mt * est_b2
                    )
                
                # Calculate bootstrap CIs from distribution
                boot_ci <- boot_temp %>%
                    group_by(y_h) %>%
                    summarise(
                        conf.low = quantile(est, 0.025),
                        conf.high = quantile(est, 0.975),
                        boot_se = sd(est),
                        n_boot = n(),
                        .groups = "drop"
                    )
                
                # Merge original with bootstrap CIs
                orig_temp %>%
                    left_join(boot_ci, by = "y_h")
            })
            
            coef_temp <- bind_rows(temp_results)
            
            coef_temp$outcome <- select
            coef_temp$specification <- path2
            coef_temp$subset <- subset
            coef_temp$model <- names(models)[i]
            coef_temp$Tshock_type <- names(Tshock)[j]
            
            coef_templist[[paste0(names(models)[i], "_", names(Tshock)[j])]] <- coef_temp
        }
    }

    write.csv(do.call(rbind, coef_templist), paste0(path0, path1, subset, "_", select, "_", path2, "_TEMP.csv"), row.names = FALSE)
    print(paste("Results saved for:", subset, "_", path2))
}

print(paste("All data processed successfully for outcome: ", select))