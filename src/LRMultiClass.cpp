// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1){
  // All input is assumed to be correct
  
  // Initialize some parameters
  int K = arma::max(y) + 1; // number of classes
  int p = X.n_cols;
  int n = X.n_rows;

  
  arma::mat beta = beta_init; // to store betas and be able to change them if needed
  arma::vec objective(numIter + 1); // to store objective values
  
  // Identity matrix for ridge term
  arma::mat I_p = arma::eye(p, p);
  
  // Softmax function
  auto softmax = [](const arma::mat& X, const arma::mat& beta){
    arma::mat scores = X * beta;
    arma::mat max_row = arma::repmat(arma::max(scores, 1), 1, scores.n_cols);
    arma::mat exp_scores = arma::exp(scores - max_row);
    return exp_scores.each_col() / arma::sum(exp_scores, 1);
  };
  
  // Objective function: -log likelihood + ridge
  auto obj_fn = [&](const arma::mat& beta){
    arma::mat P = softmax(X, beta);
    double loglik = 0.0;
    for(int i=0; i<n; i++){
      loglik += std::log(P(i, y[i]));
    }
    double reg = 0.5 * lambda * arma::accu(beta % beta);
    return -loglik + reg;
  };
  
  // Objective at start
  objective[0] = obj_fn(beta);
  
  
  // Initialize anything else that you may need
  
  // Newton's method cycle - implement the update EXACTLY numIter iterations
  
  for(int t=0; t<numIter; t++){
    
    arma::mat P = softmax(X, beta);
    
    for(int k=0; k<K; k++){
      arma::vec pk = P.col(k);
      arma::vec yk = arma::conv_to<arma::vec>::from(y == k);
      arma::vec gk = X.t() * (pk - yk) + lambda * beta.col(k);
      arma::vec wk = pk % (1 - pk);
      arma::mat Hk = X.t() * (X.each_col() % wk) + lambda * I_p;
      arma::vec delta = arma::solve(Hk, gk);
      beta.col(k) -= eta * delta;
    }
    
    objective[t+1] = obj_fn(beta);
  }
  
  // Create named list with betas and objective values
  return Rcpp::List::create(Rcpp::Named("beta") = beta,
                            Rcpp::Named("objective") = objective);
}