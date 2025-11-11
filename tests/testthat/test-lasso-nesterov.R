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

testthat::test_that("shapes and finiteness of R vs C++ on small random data", {
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
  
  testthat::expect_true(is.numeric(fit_r$beta) && length(fit_r$beta) == p && is.finite(fit_r$fmin))
  testthat::expect_true(is.numeric(fit_cpp$beta) && length(fit_cpp$beta) == p && is.finite(fit_cpp$fmin))
  testthat::expect_true(is.numeric(fit_cpp$intercept) && length(fit_cpp$intercept) == 1L)
  
  # compare on standardized scale
  beta_cpp_tilde <- to_standardized_beta(fit_cpp$beta, std$weights)
  f_cpp_std <- lasso_std(std$Xtilde, std$Ytilde, beta_cpp_tilde, lam)
  f_r_std   <- lasso_std(std$Xtilde, std$Ytilde, fit_r$beta, lam)
  
  diff <- abs(f_cpp_std - f_r_std)
  thr  <- 5e-6 * max(1, abs(f_r_std), abs(f_cpp_std)) + 1e-12
  testthat::expect_true(diff <= thr,
                        sprintf("%s (f mismatch: cpp=%.6g r=%.6g diff=%.3g thr=%.3g)", test_name, f_cpp_std, f_r_std, diff, thr))
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 2) KKT optimality on standardized solution ----------------------------

testthat::test_that("KKT conditions satisfied (standardized)", {
  test_name <- "KKT conditions satisfied (standardized)"
  n <- 60; p <- 30
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.15 * max(abs(crossprod(std$Xtilde, std$Ytilde))) / n
  
  fit_r <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-8)
  
  kkt <- kkt_check(std$Xtilde, std$Ytilde, fit_r$beta, lam, tol = 5e-3)
  testthat::expect_true(kkt$ok,
                        sprintf("%s (violations: act=%.3e inact=0.000e+00)", test_name, kkt$max_act))
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 3) Lambda extremes -----------------------------------------------------

testthat::test_that("lambda extremes (0 and huge)", {
  test_name <- "lambda extremes (0 and huge)"
  n <- 30; p <- 15
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam_max <- max(abs(crossprod(std$Xtilde, std$Ytilde))) / n
  
  # lambda = 0 should match least-squares prox step convergence, check finiteness
  fit0_r   <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, 0, eps = 1e-10, s = NULL)
  fit0_cpp <- fitLASSO_prox_Nesterov(X, Y, 0, eps = 1e-10, s = NULL)
  testthat::expect_true(is.finite(fit0_r$fmin) && is.finite(fit0_cpp$fmin))
  
  # very large lambda -> all zeros approximately
  lam_h <- 100 * lam_max
  fith_r   <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam_h, eps = 1e-10, s = NULL)
  fith_cpp <- fitLASSO_prox_Nesterov(X, Y, lam_h, eps = 1e-10, s = NULL)
  
  testthat::expect_true(max(abs(fith_r$beta)) < 1e-8, "(R: beta not ~0)")
  testthat::expect_true(max(abs(fith_cpp$beta)) < 1e-8, "(C++: beta not ~0)")
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 4) Step size handling (s = NULL auto, and positive s) -----------------

testthat::test_that("step size auto and explicit positive s", {
  test_name <- "step size auto and explicit positive s"
  n <- 35; p <- 18
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.1 * max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  
  # auto step
  fr_auto <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-8)
  fc_auto <- fitLASSO_prox_Nesterov(X, Y, lam, s = NULL, eps = 1e-8)
  
  # explicit s
  fr_exp <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = 0.1, eps = 1e-8)
  fc_exp <- fitLASSO_prox_Nesterov(X, Y, lam, s = 0.1, eps = 1e-8)
  
  testthat::expect_true(all(is.finite(c(fr_auto$fmin, fc_auto$fmin, fr_exp$fmin, fc_exp$fmin))))
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 5) Monotone objective behavior and tail stability for R implementation ------------

testthat::test_that("objective monotonicity and tail stability (R)", {
  test_name <- "objective monotonicity and tail stability (R)"
  n <- 25; p <- 12
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.12 * max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  
  fit_r <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-8)
  fvec <- fit_r$fobj_vec
  testthat::expect_true(all(is.finite(fvec)) && length(fvec) >= 5)
  
  # small numerical noise ok, but no large upward jumps
  inc  <- diff(fvec)
  testthat::expect_true(max(inc) <= 0, "objective increased")
  
  tail_vec <- tail(fvec, min(100, length(fvec)))
  rel_change <- max(abs(diff(tail_vec)) / pmax(1, abs(tail_vec[-1])))
  testthat::expect_true(rel_change < 2e-1,
                        sprintf("tail not stable; rel_change=%.3e", rel_change))
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 6) C++ solution satisfies KKT on standardized scale
testthat::test_that("C++ wrapper KKT on standardized scale", {
  test_name <- "KKT satisfied on standardized scale (C++)"
  n <- 70; p <- 35
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.2 * max(abs(crossprod(std$Xtilde, std$Ytilde))) / n
  
  fit_cpp <- fitLASSO_prox_Nesterov(X, Y, lam, eps = 1e-8, s = NULL)
  btilde  <- to_standardized_beta(fit_cpp$beta, std$weights)
  k <- kkt_check(std$Xtilde, std$Ytilde, btilde, lam, tol = 5e-3)
  
  testthat::expect_true(k$ok)
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 7) Consistency of intercept/back-transform for C++ wrapper

testthat::test_that("back-transform matches standardized objective (C++)", {
  test_name <- "back-transform matches standardized objective (C++)"
  n <- 60; p <- 25
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.1 * max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  
  fit_cpp <- fitLASSO_prox_Nesterov(X, Y, lam, eps = 1e-8, s = NULL)
  btilde  <- to_standardized_beta(fit_cpp$beta, std$weights)
  f_std   <- lasso_std(std$Xtilde, std$Ytilde, btilde, lam)
  
  testthat::expect_true(is.finite(fit_cpp$fmin))
  testthat::expect_true(abs(fit_cpp$fmin - f_std) <= 1e-6 * max(1, abs(f_std)))
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 8) Warm start reduces iterations (R standardized solver) -------------

testthat::test_that("warm start improves early progress (R standardized)", {
  test_name <- "warm start improves early progress (R standardized)"
  n <- 80; p <- 40
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam_max <- max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  lam_hi  <- 0.5  * lam_max
  lam_lo  <- 0.05 * lam_max
  
  # common explicit step s = 1/L
  v <- rnorm(p); v <- v / sqrt(sum(v^2))
  for (k in 1:12) { v <- crossprod(std$Xtilde, std$Xtilde %*% v) / n; v <- as.numeric(v); v <- v / sqrt(sum(v^2)) }
  L <- as.numeric(crossprod(v, crossprod(std$Xtilde, std$Xtilde %*% v)) / n)
  s_exp <- 1 / L
  
  # high-lambda solve to produce a warm start
  fit_hi <- fitLASSOstandardized_prox_Nesterov(
    std$Xtilde, std$Ytilde, lam_hi,
    beta_start = numeric(p), eps = 1e-10, s = s_exp
  )
  
  # fixed small budget at low lambda, disable early stopping via eps=0 and set max_iter
  budget <- 40L
  cold <- fitLASSOstandardized_prox_Nesterov(
    std$Xtilde, std$Ytilde, lam_lo,
    beta_start = numeric(p), eps = 0, s = s_exp, max_iter = budget
  )
  warm <- fitLASSOstandardized_prox_Nesterov(
    std$Xtilde, std$Ytilde, lam_lo,
    beta_start = fit_hi$beta, eps = 0, s = s_exp, max_iter = budget
  )
  
  # compare objectives after the same number of iterations
  # use the last recorded value (budget-th entry)
  f_cold <- tail(cold$fobj_vec, 1)
  f_warm <- tail(warm$fobj_vec, 1)
  
  # warm should be no worse than cold within small numerical slack
  tol <- 1e-5 * max(1, abs(f_cold), abs(f_warm)) + 1e-8
  testthat::expect_true(f_warm <= f_cold + tol,
                        sprintf("budget=%d, warm=%.6g, cold=%.6g, tol=%.2e", 
                                budget, f_warm, f_cold, tol))
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 9) Auto step matches explicit 1/L step (R standardized solver) -------------

testthat::test_that("auto step matches explicit 1/L (R standardized)", {
  test_name <- "auto step matches explicit 1/L (R standardized)"
  n <- 50; p <- 30
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.08 * max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  
  v <- rnorm(p); v <- v / sqrt(sum(v^2))
  for (k in 1:12) {
    v <- crossprod(std$Xtilde, std$Xtilde %*% v) / n
    v <- as.numeric(v); v <- v / sqrt(sum(v^2))
  }
  L <- as.numeric(crossprod(v, crossprod(std$Xtilde, std$Xtilde %*% v))/n)
  s_exp <- 1 / L
  
  fr_auto <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-9)
  fr_exp  <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = s_exp, eps = 1e-9)
  
  testthat::expect_true(abs(fr_auto$fmin - fr_exp$fmin) <= 1e-6 * max(1, abs(fr_exp$fmin)))
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})

## 10) R vs C++ coefficients close on standardized scale -----------------

testthat::test_that("R vs C++ beta close on standardized scale", {
  test_name <- "R vs C++ beta close on standardized scale"
  n <- 90; p <- 45
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.12 * max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  
  fr <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-9)
  fc <- fitLASSO_prox_Nesterov(X, Y, lam, s = NULL, eps = 1e-9)
  
  b_r <- fr$beta
  b_c <- to_standardized_beta(fc$beta, std$weights)
  
  # objectives on standardized scale
  f_r <- lasso_std(std$Xtilde, std$Ytilde, b_r, lam)
  f_c <- lasso_std(std$Xtilde, std$Ytilde, b_c, lam)
  tol_obj <- 5e-6 * max(1, abs(f_r), abs(f_c)) + 1e-12
  testthat::expect_true(abs(f_r - f_c) <= tol_obj,
                        sprintf("objective mismatch: R=%.6g C++=%.6g tol=%.2e", f_r, f_c, tol_obj))
  
  # cosine similarity, unless both are ~0
  nr <- sqrt(sum(b_r^2)); nc <- sqrt(sum(b_c^2))
  if (nr < 1e-10 && nc < 1e-10) {
    testthat::expect_true(max(abs(b_r)) < 1e-8 && max(abs(b_c)) < 1e-8, "both should be near zero")
  } else if (nr > 0 && nc > 0) {
    cos_sim <- sum(b_r * b_c) / (nr * nc)
    testthat::expect_true(cos_sim >= 0.99,
                          sprintf("weak alignment: cos=%.6f", cos_sim))
  } else {
    # one near-0 and the other not, allow if objectives still approx. match
    testthat::expect_true(abs(f_r - f_c) <= tol_obj)
  }
  
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})


## 11) Objective nonincreasing per iteration w/ restart (R) ----

testthat::test_that("objective is nonincreasing across iterations (R standardized)", {
  test_name <- "objective is nonincreasing across iterations (R standardized)"
  n <- 40; p <- 20
  X <- matrix(rnorm(n*p), n, p); Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam <- 0.1 * max(abs(crossprod(std$Xtilde, std$Ytilde)))/n
  
  fr <- fitLASSOstandardized_prox_Nesterov(std$Xtilde, std$Ytilde, lam, s = NULL, eps = 1e-10)
  f <- fr$fobj_vec
  
  testthat::expect_true(all(diff(f) <= 1e-12))
  cat(test_name, "PASSED\n"); n_ok <<- n_ok + 1L
})