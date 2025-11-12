#' Title
#'
#' @param X 
#' @param K 
#' @param M 
#' @param numIter 
#'
#' @return Explain return
#' @export
#'
#' @examples
#' # Give example
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