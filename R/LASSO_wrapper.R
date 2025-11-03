#' Title
#'
#' @param X 
#' @param Y
#' @param lambda 
#' @param beta_start 
#' @param eps 
#' @param s 
#'
#' @returns
#' @export
#'
#' @examples
fitLASSO_prox_Nesterov <- function(X, Y, lambda, 
                                   beta_start = NULL, eps = 0.0001, s = 0.01){
  
  # Compatibility checks from ProximalExamples
  if (is.null(dim(X))) stop("X must be a numeric matrix")
  if (!is.numeric(X) || !is.numeric(Y)) stop("X and Y must be numeric")
  n <- nrow(X); p <- ncol(X)
  if (length(Y) != n) stop("length(Y) must equal nrow(X)")
  if (!is.numeric(lambda) || lambda < 0) stop("lambda must be nonnegative")
  
  # Center and standardize X,Y as in HW4
  std <- standardizeXY(X, Y)  # from LASSO_CoordinateDescent.R
  Xtilde <- std$Xtilde
  Ytilde <- std$Ytilde
  
  # Initialize beta_init
  if (is.null(beta_start)) {
    beta_start <- rep(0, p)
  } else if (length(beta_start) != p) {
    stop("beta_start must have length equal to ncol(X)")
  }
  
  # Call C++ fitLASSOstandardized_prox_Nesterov_c function to implement the algorithm
  beta_tilde = fitLASSOstandardized_prox_Nesterov_c(Xtilde, Ytilde, lambda, beta_start, eps, s)
  
  # Perform back scaling and centering to get original intercept and coefficient vector
  
  # Return 
  # beta - the solution (without center or scale)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}