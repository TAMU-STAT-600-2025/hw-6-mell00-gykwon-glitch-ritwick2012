// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                      const arma::mat& M, int numIter = 100){
  // All input is assumed to be correct
  
  // Initialize some parameters
  int n = X.n_rows;
  int p = X.n_cols;
  arma::uvec Y(n); // to store cluster assignments
  
  // Initialize any additional parameters if needed
  arma::mat current_M = M; // K x p
  
  // For loop with kmeans algorithm
  for (int it = 0; it < numIter; ++it){
    // Compute argmin over k of ||x_i - mu_k||^2 = ||mu_k||^2 - 2 x_i^T mu_k (+ const)
    arma::rowvec m2 = arma::sum(arma::square(current_M), 1).t();     // 1 x K, ||mu_k||^2
    arma::mat mat_dist = arma::repmat(m2, n, 1) - 2.0 * X * current_M.t(); // n x K
    Y = arma::index_min(mat_dist, 1); // 0-based, first min on ties
    
    // ----- Update step (including empty check) -----
    arma::mat new_mean(K, p, arma::fill::zeros);
    for (int k = 0; k < K; ++k) {
      arma::uvec idx = arma::find(Y == (arma::uword)k);
      if (idx.n_elem == 0) {
        Rcpp::stop("A cluster became empty; please change the initial centers M.");
      } else if (idx.n_elem == 1) {
        new_mean.row(k) = X.row(idx[0]);
      } else {
        new_mean.row(k) = arma::mean(X.rows(idx), 0); // 1 x p
      }
    }
    
    // Convergence check
    if (arma::accu(arma::square(current_M - new_mean)) == 0.0) {
      break;
    } 
    // update and iteration
    current_M = new_mean;
  }
  // add 1 to match index with R
  Y += 1;
  // Returns the vector of cluster assignments
  return(Y);
}



