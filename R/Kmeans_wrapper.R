#' K-means Algorithm
#'
#' This function performs K-means clustering using an RcppArmadillo-based implementation.
#'
#' @param X matrix[n, p] A numeric matrix with n observations (rows) and p features (columns).
#' @param K integer The number of clusters.
#' @param M matrix[K, p] Optional initial cluster centers. If \code{NULL} (default), 
#'   K centers are randomly chosen from the rows of \code{X}.
#' @param numIter integer The maximum number of iterations (default = 100).
#'
#' @return An integer vector \code{Y} of length \code{n}, where \code{Y[i]} is the assigned
#'   cluster number for the i-th observation.
#' @export
#'
#' @examples
#' n1 <- 40
#' p1 <- 20
#' K1 <- 5
#' X1 <- matrix(rnorm(n1 * p1), n1, p1)
#' MyKmeans(X1, K1, NULL, 100)
#'
#' n2 <- 50
#' p2 <- 30
#' K2 <- 10
#' X2 <- matrix(rnorm(n2 * p2), n2, p2)
#' MyKmeans(X2, K2, NULL, 50)

MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  X = as.matrix(X)
  n = nrow(X) # number of rows in X
  p = ncol(X) # number of columns in X
  
  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  if (is.null(M)) {
    M <- X[sample.int(n, K), , drop = FALSE]
  } else if ((nrow(M) != K || ncol(M) != p)) { # Check compatibility of M
    stop("M must be K x p matrix.") # Return the error message
  }
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # change cluster label to integer
  Y <- as.integer(Y)
  
  # Return the class assignments
  return(Y)
}


