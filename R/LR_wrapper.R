#' Title Multiclass Logistic Regression
#'
#' @param X Numeric matrix of training data (n x p).
#' @param y Numeric response vector of length n.
#' @param numIter Number of Newton iterations (default 50).
#' @param eta Damping / learning rate (default 0.1).
#' @param lambda Ridge regularization parameter (default 1).
#' @param beta_init Optional initial beta values (p x K matrix). If NULL, initialized to zeros.
#'
#' @return
#' A list with:
#' \item{beta}{p x K matrix of estimated coefficients}
#' \item{objective}{Vector of objective function values at each iteration}
#' @export
#'
#' @examples
#' set.seed(1)
#' n <- 50
#' p <- 3
#' K <- 3
#' X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
#' y <- sample(0:(K-1), n, replace=TRUE)
#' out <- LRMultiClass(X, y, numIter = 20)
#' out$beta
#' out$objective
LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  # --- Input checks ---
  if(!is.matrix(X)) stop("X must be a numeric matrix")
  n <- nrow(X); p <- ncol(X)
  if(length(y) != n) stop("Length of y must match rows of X")
  if(any(X[,1] != 1)) stop("First column of X must be all 1s (intercept)")
  if(eta <= 0) stop("eta must be positive")
  if(lambda < 0) stop("lambda must be non-negative")
  
  K <- length(unique(y))
  if(is.null(beta_init)) {
    beta_init <- matrix(0, nrow = p, ncol = K)
  } else {
    if(!all(dim(beta_init) == c(p, K))) stop("beta_init dimensions must be p x K")
  }
  
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}



