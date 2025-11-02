# Fit LASSO on standardized data using proximal gradient + Nesterov acceleration
# Xtilde - centered & column-scaled X (n x p)
# Ytilde - centered Y (n)
# lambda - >= 0
# beta_start - optional p-vector
# eps - stopping tolerance on relative L1 change
# s - step size (try ~ 1/L); if NULL, auto-choose s = 1/L with L = ||X||_op^2 / n

fitLASSOstandardized_prox_Nesterov <- function(
    Xtilde, Ytilde, lambda, beta_start = NULL, eps = 1e-4, s = 1e-2
){
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("Dimensions of X and Y don't match")
  if (!is.numeric(lambda) || lambda < 0) stop("Only non-negative values of lambda are allowed!")
  if (is.null(beta_start)) beta_start <- rep(0, p)
  if (length(beta_start) != p) stop("Supplied initial starting point has length different from p!")
  
  # auto step size if requested
  if (is.null(s)) {
    d1 <- svd(Xtilde, nu = 0, nv = 0)$d[1]
    L  <- (d1 * d1) / n
    s  <- 1 / (L + 1e-12)
  }
  if (!is.numeric(s) || s <= 0) stop("step size s must be positive")
  
  # Helper objective using existing lasso()
  f_eval <- function(b) as.numeric(lasso(Xtilde, Ytilde, b, lambda))
  
  # Initialize Nesterov state
  x_curr <- as.numeric(beta_start)
  y_curr <- x_curr
  t_prev <- 1
  max_iter <- 5000L
  
  f_curr <- f_eval(x_curr)
  if (!is.finite(f_curr)) stop("objective is not finite at initialization")
  f_hist <- numeric(0)
  
  converged <- FALSE
  for (iter in 1:max_iter) {
    # Track pre-update objective for plotting
    f_hist <- c(f_hist, f_curr)
    
    # Gradient of g(b) = (1/(2n))||Y - Xb||^2 at y_curr: ∇g = -(1/n) X^T (Y - X y_curr)
    r    <- Ytilde - as.vector(Xtilde %*% y_curr) # n-vector
    grad <- -drop(crossprod(Xtilde, r)) / n # p-vector
    
    # proximal step (soft threshold)
    z      <- y_curr - s * grad
    x_next <- soft(z, lambda * s)
    
    # Nesterov acceleration
    t_next <- (1 + sqrt(1 + 4 * t_prev^2)) / 2
    y_next <- x_next + ((t_prev - 1) / t_next) * (x_next - x_curr)
    
    # monotone safeguard (restart momentum if objective increased)
    f_next <- f_eval(x_next)
    if (!is.finite(f_next)) {
      # fall back to no momentum
      t_next <- 1
      y_next <- x_next
      f_next <- f_eval(x_next)  # recompute (same)
    } else if (f_next > f_curr) {
      # adaptive restart: drop momentum this round
      t_next <- 1
      y_next <- x_next
      f_next <- f_eval(x_next)
    }
    
    # convergence check (relative L1 change)
    if (sum(abs(x_next - x_curr)) / (1 + sum(abs(x_curr))) < eps) {
      x_curr <- x_next
      f_curr <- f_next
      converged <- TRUE
      break
    }
    
    # update state
    x_curr <- x_next
    y_curr <- y_next
    t_prev <- t_next
    f_curr <- f_next
  }
  
  list(
    beta = x_curr,
    fmin = f_curr,
    iters = if (converged) iter else max_iter,
    converged = converged,
    fobj_vec = f_hist
  )
}
