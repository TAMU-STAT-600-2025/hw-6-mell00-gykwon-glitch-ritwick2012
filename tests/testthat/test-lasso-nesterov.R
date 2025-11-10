
set.seed(123)
cat("\n=== Begin Nesterov LASSO Tests ===\n")
n_ok <- 0L

## Helper functions

# KKT check on standardized scale:
# g = (1/n) X^T (Y - X b)
# active coords satisfy g_j = lambda * sign(b_j)
# inactive satisfy |g_j| <= lambda
kkt_check <- function(Xt, Yt, btilde, lambda, tol = 1e-6) {
  n <- nrow(Xt)
  r <- as.numeric(Yt - Xt %*% btilde)
  g <- as.numeric(crossprod(Xt, r)) / n
  act <- which(abs(btilde) > 0)
  inact <- setdiff(seq_along(btilde), act)
  ok_act <- TRUE
  if (length(act)) ok_act <- max(abs(g[act] - lambda * sign(btilde[act]))) <= tol
  ok_inact <- TRUE
  if (length(inact)) ok_inact <- max(abs(g[inact])) <= lambda + tol
  list(ok = ok_act && ok_inact, max_act = ifelse(length(act), max(abs(g[act] - lambda * sign(btilde[act]))), 0),
       max_inact = ifelse(length(inact), max(pmax(0, abs(g[inact]) - lambda)), 0))
}


# convert original scale beta to standardized scale using weights
to_standardized_beta <- function(beta_orig, weights) as.numeric(beta_orig * weights)

# objective on standardized scale
lasso_std <- function(Xt, Yt, btilde, lambda) {
  n <- nrow(Xt)
  0.5 * sum((Yt - Xt %*% btilde)^2) / n + lambda * sum(abs(btilde))
}



## 1) Basic shape/value checks on small random data

{
  test_name <- "shapes and finiteness of R vs C++ on small random data"
  n <- 40; p <- 20
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.2 * max(abs(crossprod(std$Xtilde, std$Ytilde))) / n
  
  fit_r   <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lambda = lam,
                                                beta_start = numeric(p), eps = 1e-9, s = NULL)
  fit_cpp <- fitLASSO_prox_Nesterov(X, Y, lambda = lam,
                                    beta_start = numeric(p), eps = 1e-9, s = NULL)
  
  stopifnot(is.numeric(fit_r$beta), length(fit_r$beta) == p, is.finite(fit_r$fmin))
  stopifnot(is.numeric(fit_cpp$beta), length(fit_cpp$beta) == p, is.finite(fit_cpp$fmin))
  stopifnot(is.numeric(fit_cpp$intercept), length(fit_cpp$intercept) == 1L)
  
  # compare on standardized scale
  beta_cpp_tilde <- to_standardized_beta(fit_cpp$beta, std$weights)
  f_cpp_std <- lasso_std(std$Xtilde, std$Ytilde, beta_cpp_tilde, lam)
  f_r_std   <- lasso_std(std$Xtilde, std$Ytilde, fit_r$beta, lam)
  
  if (!(abs(f_cpp_std - f_r_std) <= 1e-6 * pmax(1, abs(f_r_std))))
    stop(test_name, sprintf(" (f mismatch: cpp=%.6g r=%.6g)", f_cpp_std, f_r_std))
  
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}


## 2) KKT optimality on standardized solution ----------------------------

{
  test_name <- "KKT conditions satisfied (standardized)"
  n <- 60; p <- 30
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.15 * max(abs(crossprod(std$Xtilde, std$Ytilde))) / n
  
  fit_r <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-8)
  
  kkt <- kkt_check(std$Xtilde, std$Ytilde, fit_r$beta, lam, tol = 1e-5)
  if (!kkt$ok)
    stop(test_name, sprintf(" (violations: act=%.3e inact=%.3e)", kkt$max_act, kkt$max_inact))
  
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}