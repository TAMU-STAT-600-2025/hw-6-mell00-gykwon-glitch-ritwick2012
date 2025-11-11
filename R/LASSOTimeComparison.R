
#' Run timing and consistency checks for the Nesterov LASSO implementations
#' 
#' @importFrom stats rnorm
#' @importFrom graphics par abline
#' @param n Integer. Rows (ignored if X,Y supplied).
#' @param p Integer. Columns (ignored if X,Y supplied).
#' @param seed Integer. RNG seed used only if X,Y are NULL.
#' @param plot Logical. If TRUE, make simple diagnostic plots.
#' @param s_step Numeric or NULL. Common step size for both solvers.
#' @param X Optional numeric matrix. If provided, Y must also be provided.
#' @param Y Optional numeric vector. If provided, X must also be provided.
#' @return list with fits, microbenchmark, lambda, std, and the X,Y used
#' @export
run_lasso_time_comparison <- function(n = 30, p = 50, seed = 38947,
                                      plot = interactive(), s_step = 0.1,
                                      X = NULL, Y = NULL) {
  if (is.null(X) || is.null(Y)) {
    set.seed(seed)
    X <- matrix(rnorm(n * p), n, p)
    Y <- rnorm(n)
  } else {
    X <- as.matrix(X); storage.mode(X) <- "double"
    Y <- as.numeric(Y)
    n <- nrow(X); p <- ncol(X)
    if (length(Y) != n) stop("length(Y) must equal nrow(X)")
  }
  
  std <- standardizeXY(X, Y)
  nX <- nrow(std$Xtilde)
  lambda_max <- max(abs(crossprod(std$Xtilde, std$Ytilde))) / nX
  lambda1 <- 0.1 * lambda_max
  
  # C++ wrapper expects original X,Y
  fit_cpp <- fitLASSO_prox_Nesterov(X, Y, lambda1,
                                    beta_start = numeric(p),
                                    eps = 1e-10, s = s_step)
  
  # R standardized path uses Xtilde,Ytilde
  fit_r <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lambda1,
                                              beta_start = numeric(p),
                                              eps = 1e-10, s = s_step)
  
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
  
  mb <- microbenchmark::microbenchmark(
    cpp = fitLASSO_prox_Nesterov(X, Y, lambda1,
                                 beta_start = numeric(p), eps = 1e-10, s = s_step),
    r   = fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lambda1,
                                             beta_start = numeric(p), eps = 1e-10, s = s_step),
    times = 20L, unit = "ms"
  )
  
  list(
    lambda = lambda1,
    fit_cpp = fit_cpp,
    fit_r = fit_r,
    fmin_diff = fit_cpp$fmin - fit_r$fmin,
    bench = mb,
    std = std,
    X = X,
    Y = Y
  )
}