// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// Xtilde - centered and scaled X, n x p
// Ytilde - centered Y, n x 1
// lambda - tuning parameter
// beta0 - p vector of starting point for coordinate-descent algorithm, optional
// eps - precision level for convergence assessment, default 0.0001
// s - step size for proximal gradient
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_prox_Nesterov_c(const arma::mat& Xtilde, const arma::colvec& Ytilde,
                                                  double lambda, const arma::colvec& beta_start, 
                                                double eps = 0.0001, double s = 0.01){
  // All input is assumed to be correct
  
  // Initialize some parameters
  int n = Xtilde.n_rows, p = Xtilde.n_cols;
  arma::colvec beta(p);

  return beta;
}
