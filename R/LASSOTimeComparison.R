
#' Run timing and consistency checks for the Nesterov LASSO implementations
#'
#' @param n Integer. Rows.
#' @param p Integer. Columns.
#' @param seed Integer. RNG seed.
#' @param plot Logical. If TRUE, make simple diagnostic plots.
#' @param s_step Numeric or NULL. Step size for prox-Nesterov; if NULL the R version auto-computes.
#' @return list with fits and microbenchmark object
#' @export
run_lasso_time_comparison <- function(n = 30, p = 50, seed = 38947,
                                      plot = interactive(), s_step = 0.1) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  Y <- rnorm(n)
  
  std <- standardizeXY(X, Y)  # from LASSO_CoordinateDescent.R
  nX <- nrow(std$Xtilde)
  
  lambda_max <- max(abs(crossprod(std$Xtilde, std$Ytilde))) / nX
  lambda1 <- 0.1 * lambda_max
  
  # C++ path wrapped by Rcpp
  fit_cpp <- fitLASSO_prox_Nesterov(std$Xtilde, std$Ytilde, lambda1,
                                    beta_start = numeric(p),
                                    eps = 1e-10,
                                    s = s_step)
  
  # R path
  fit_r <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lambda1,
                                              beta_start = numeric(p),
                                              eps = 1e-10,
                                              s = s_step)
  
  if (plot) {
    op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
    par(mfrow = c(1, 2))
    plot(fit_cpp$beta, fit_r$beta,
         xlab = "beta (C++ prox-Nesterov)", ylab = "beta (R prox-Nesterov)")
    abline(0, 1, lty = 2)
    if (!is.null(fit_r$fobj_vec) && length(fit_r$fobj_vec) > 1L) {
      plot(fit_r$fobj_vec, type = "l", xlab = "iter", ylab = "objective")
    }
  }
  
  # microbenchmark
  mb <- microbenchmark::microbenchmark(
    cpp = fitLASSO_prox_Nesterov(std$Xtilde, std$Ytilde, lambda1,
                                 beta_start = numeric(p), eps = 1e-10, s = s_step),
    r = fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lambda1,
                                             beta_start = numeric(p), eps = 1e-10, s = s_step),
    times = 20L
  )
  
  list(
    lambda = lambda1,
    fit_cpp = fit_cpp,
    fit_r = fit_r,
    fmin_diff = fit_cpp$fmin - fit_r$fmin,
    bench = mb
  )
}
