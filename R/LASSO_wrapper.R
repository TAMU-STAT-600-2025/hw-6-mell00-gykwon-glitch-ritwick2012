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
  
  # Compatibility checks from ProximalExamples and initialization of beta_init
  
  # Center and standardize X,Y as in HW4
  
  # Call C++ fitLASSOstandardized_prox_Nesterov_c function to implement the algorithm
  beta_tilde = fitLASSOstandardized_prox_Nesterov_c(Xtilde, Ytilde, lambda, beta_start, eps, s)
  
  # Perform back scaling and centering to get original intercept and coefficient vector
  
  # Return 
  # beta - the solution (without center or scale)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}