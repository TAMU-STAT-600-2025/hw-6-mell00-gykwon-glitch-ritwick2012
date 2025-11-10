
#' @title LASSO via proximal gradient with Nesterov acceleration
#' @param X numeric matrix
#' @param Y numeric vector, length nrow(X)
#' @param lambda nonnegative scalar
#' @param beta_start numeric vector length ncol(X)
#' @param eps convergence tolerance
#' @param s step size; if NULL, estimated
#' @return A list with:
#' \describe{
#'   \item{beta}{Numeric length p, coefficients on original scale.}
#'   \item{intercept}{Numeric scalar.}
#'   \item{fmin}{Objective value on standardized scale.}
#'   \item{obj}{Optional objective trace if available; may be NULL.}
#'   \item{lambda, eps, step}{Echoed inputs.}
#' }
#' @export
fitLASSO_prox_Nesterov <- function(X, Y, lambda,
                                   beta_start = NULL, eps = 1e-4, s = NULL) {
  
  # Basic checks and type coercion
  if (is.null(dim(X))) stop("X must be a numeric matrix")
  if (!is.numeric(X) || !is.numeric(Y)) stop("X and Y must be numeric")
  X <- as.matrix(X); storage.mode(X) <- "double"
  Y <- as.numeric(Y)
  n <- nrow(X); p <- ncol(X)
  if (length(Y) != n) stop("length(Y) must equal nrow(X)")
  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0)
    stop("lambda must be a nonnegative scalar")
  
  # Standardize
  std <- standardizeXY(X, Y)
  Xtilde <- std$Xtilde; Ytilde <- std$Ytilde
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  
  # Start vector
  if (is.null(beta_start)) beta_start <- numeric(p)
  if (length(beta_start) != p) stop("beta_start must have length ncol(X)")
  beta_start <- as.numeric(beta_start) 
  
  # Step size: estimate if not supplied
  if (is.null(s)) {
    v <- rnorm(p); v <- v / sqrt(sum(v^2))
    for (k in 1:10) {
      v <- crossprod(Xtilde, Xtilde %*% v) / n
      v <- as.numeric(v); v <- v / sqrt(sum(v^2))
    }
    L <- as.numeric(crossprod(v, crossprod(Xtilde, Xtilde %*% v)) / n)
    s <- 1 / L
  }
  
  # Call C++
  res <- fitLASSOstandardized_prox_Nesterov_c(Xtilde, Ytilde, lambda,
                                              beta_start, eps, s)
  
  # Handle return type
  if (is.list(res) && !is.null(res$beta)) {
    beta_tilde <- as.numeric(res$beta)
    obj <- res$obj
  } else {
    beta_tilde <- as.numeric(res)
    obj <- NULL
  }
  if (length(beta_tilde) != p) stop("C++ returned wrong-length beta")
  
  # Objective on standardized scale
  fmin <- lasso(Xtilde, Ytilde, beta_tilde, lambda)
  
  # Back-transform
  beta <- beta_tilde / std$weights
  intercept <- std$Ymean - sum(std$Xmeans * beta)
  
  list(beta = beta, intercept = intercept, fmin = fmin, obj = obj,
       lambda = lambda, eps = eps, step = s)
}
