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


// Soft-threshold helper function

// [[Rcpp::export]]
double soft_c(double a, double lambda){
  if (a >  lambda) return a - lambda;
  if (a < -lambda) return a + lambda;
  return 0.0;
}

// Main Nesterov LASSO function

// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_prox_Nesterov_c(const arma::mat& Xtilde, const arma::colvec& Ytilde,
                                                  double lambda, const arma::colvec& beta_start, 
                                                double eps = 0.0001, double s = 0.01){
  // All input is assumed to be correct
  int n = Xtilde.n_rows;
  int p = Xtilde.n_cols;
  arma::colvec beta(p);
  arma::colvec x_curr = beta_start;
  arma::colvec y_curr = x_curr;
  arma::colvec grad(p);
  arma::colvec z(p);
  arma::colvec r(n);

  double t_prev = 1.0;
  int max_iter = 5000;

  // Objective function
  auto f_eval = [&](const arma::colvec& b){
    arma::colvec r2 = Ytilde - Xtilde * b;
    double rss = 0.0;
    for (int i = 0; i < n; ++i) {
      double val = r2.at(i);
      rss += val * val;
    }
    rss = rss / (2.0 * n);
    
    double l1 = 0.0;
    for (int j = 0; j < p; ++j) {
      double v = b.at(j);
      l1 += std::sqrt(v * v);
    }
    return rss + lambda * l1;
  };
  
  double f_curr = f_eval(x_curr);
  double f_next = f_curr;
  
  
  // Main loop
  for (int iter = 0; iter < max_iter; ++iter) {
    // r = Y - X * y_curr
    r = Ytilde - Xtilde * y_curr;
    
    // grad = -(1/n) X^T r
    for (int j = 0; j < p; ++j) grad.at(j) = 0.0;
    for (int j = 0; j < p; ++j){
      double acc = 0.0;
      for (int i = 0; i < n; ++i) acc += Xtilde.at(i,j) * r.at(i);
      grad.at(j) = -acc / n;
    }
    
    // z = y_curr - s * grad
    for (int j = 0; j < p; ++j) z.at(j) = y_curr.at(j) - s * grad.at(j);
    
    // x_next = soft(z, lambda*s)
    arma::colvec x_next(p);
    for (int j = 0; j < p; ++j) x_next.at(j) = soft_c(z.at(j), lambda * s);
    
    // Nesterov
    double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t_prev * t_prev));
    double mom = (t_prev - 1.0) / t_next;
    arma::colvec y_next(p);
    for (int j = 0; j < p; ++j)
      y_next.at(j) = x_next.at(j) + mom * (x_next.at(j) - x_curr.at(j));
    
    // monotone restart
    f_next = f_eval(x_next);
    if (!std::isfinite(f_next) || f_next > f_curr){
      for (int j = 0; j < p; ++j) y_next.at(j) = x_next.at(j);
      t_next = 1.0;
      f_next = f_eval(x_next);
    }
    
    // convergence: relative L1
    double num = 0.0, den = 1.0;
    for (int j = 0; j < p; ++j){
      double diff = x_next.at(j) - x_curr.at(j);
      num += std::sqrt(diff * diff);
      den += std::sqrt(x_curr.at(j) * x_curr.at(j));
    }
    if (num / den < eps){
      for (int j = 0; j < p; ++j) beta.at(j) = x_next.at(j);
      return beta;
    }
    
    // update (element-wise)
    for (int j = 0; j < p; ++j) x_curr.at(j) = x_next.at(j);
    for (int j = 0; j < p; ++j) y_curr.at(j) = y_next.at(j);
    f_curr = f_next;
    t_prev = t_next;
    
    if (iter == max_iter - 1)
      for (int j = 0; j < p; ++j) beta.at(j) = x_curr.at(j);
  }
  
  

  return beta;
}
