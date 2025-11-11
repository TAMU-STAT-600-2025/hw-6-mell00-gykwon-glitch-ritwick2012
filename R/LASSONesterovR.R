# Fit LASSO on standardized data using proximal gradient + Nesterov acceleration
# Xtilde: centered, column-scaled X (n x p)
# Ytilde: centered Y (length n)
# lambda: >= 0
# beta_start: optional length-p numeric
# eps: stopping tolerance on relative L1 change
# s: step size; if NULL, s = 1/L with L ≈ ||X||_op^2 / n
fitLASSOstandardized_prox_Nesterov <- function(
    Xtilde, Ytilde, lambda, beta_start = NULL, eps = 1e-4, s = NULL,
    max_iter = 5000L, bt_inc = 2.0, tol_dec = 1e-12
){
  if (is.null(dim(Xtilde))) stop("Xtilde must be a numeric matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde)) stop("Xtilde and Ytilde must be numeric")
  Xtilde <- as.matrix(Xtilde); storage.mode(Xtilde) <- "double"
  Ytilde <- as.numeric(Ytilde)
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("Dimensions of Xtilde and Ytilde do not match")
  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0) stop("lambda must be a nonnegative scalar")
  
  if (is.null(beta_start)) beta_start <- numeric(p)
  if (length(beta_start) != p) stop("beta_start must have length p")
  beta_start <- as.numeric(beta_start)
  
  f_eval <- function(b){
    r <- Ytilde - as.vector(Xtilde %*% b)
    0.5 * sum(r * r) / n + lambda * sum(abs(b))
  }
  g_eval <- function(b){
    r <- Ytilde - as.vector(Xtilde %*% b)
    list(g = 0.5 * sum(r * r) / n, r = r)
  }
  
  # initial L (if s given, L = 1/s)
  if (is.null(s)) {
    v <- rnorm(p); v <- v / sqrt(sum(v * v))
    for (k in 1:10) {
      v <- crossprod(Xtilde, Xtilde %*% v) / n
      v <- as.numeric(v); nv <- sqrt(sum(v * v))
      if (!is.finite(nv) || nv == 0) { v[] <- 0; v[1] <- 1; break }
      v <- v / nv
    }
    L <- as.numeric(crossprod(v, crossprod(Xtilde, Xtilde %*% v)) / n)
    if (!is.finite(L) || L <= 0) L <- 1
  } else {
    if (!is.numeric(s) || length(s) != 1L || s <= 0) stop("step size s must be positive")
    L <- 1 / s
  }
  
  x_curr <- beta_start
  y_curr <- x_curr
  t_prev <- 1
  
  f_curr <- f_eval(x_curr)
  if (!is.finite(f_curr)) stop("objective is not finite at initialization")
  
  f_hist <- numeric(max_iter + 1L)
  f_hist[1L] <- f_curr
  converged <- FALSE
  iters_done <- max_iter
  
  for (iter in 1:max_iter) {
    # grad at y
    gy <- g_eval(y_curr)
    grad <- -(crossprod(Xtilde, gy$r) / n)[,1]
    
    # backtracking for sufficient decrease model
    Lk <- L
    repeat {
      sk <- 1 / Lk
      z  <- y_curr - sk * grad
      x_next <- soft(z, lambda * sk)
      
      f_next <- f_eval(x_next)
      dx <- x_next - y_curr
      Q  <- gy$g + sum(grad * dx) + 0.5 * Lk * sum(dx * dx) + lambda * sum(abs(x_next))
      
      if (f_next <= Q + tol_dec * (1 + abs(Q))) break
      Lk <- Lk * bt_inc
    }
    L <- Lk
    
    # momentum proposal
    t_next <- 0.5 * (1 + sqrt(1 + 4 * t_prev * t_prev))
    y_next <- x_next + ((t_prev - 1) / t_next) * (x_next - x_curr)
    
    # HARD MONOTONE GUARD: never allow f to increase
    if (f_next > f_curr) {
      x_next <- x_curr
      y_next <- x_curr
      t_next <- 1
      f_next <- f_curr
    }
    
    # log post-update objective
    f_hist[iter + 1L] <- f_next
    
    # convergence: relative L1 change
    if (sum(abs(x_next - x_curr)) / (1 + sum(abs(x_curr))) < eps) {
      x_curr <- x_next
      f_curr <- f_next
      converged <- TRUE
      iters_done <- iter
      f_hist <- f_hist[seq_len(iter + 1L)]
      break
    }
    
    # state update
    x_curr <- x_next
    y_curr <- y_next
    t_prev <- t_next
    f_curr <- f_next
    
    if (iter == max_iter) f_hist <- f_hist[seq_len(iter + 1L)]
  }
  
  list(beta = x_curr, fmin = f_curr, iters = iters_done,
       converged = converged, fobj_vec = f_hist)
}
