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


// Soft-threshold
inline double soft_c(const double a, const double lam) {
  if (a >  lam) return a - lam;
  if (a < -lam) return a + lam;
  return 0.0;
}

// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_prox_Nesterov_c(const arma::mat& Xtilde,
                                                  const arma::colvec& Ytilde,
                                                  const double lambda,
                                                  const arma::colvec& beta_start,
                                                  const double eps = 1e-4,
                                                  const double s = 1e-2) {
  const int n = Xtilde.n_rows;
  const int p = Xtilde.n_cols;
  if (Ytilde.n_elem != static_cast<unsigned>(n))
    Rcpp::stop("Dimensions of Xtilde and Ytilde do not match");
  if (lambda < 0.0) Rcpp::stop("lambda must be nonnegative");
  if (beta_start.n_elem != static_cast<unsigned>(p))
    Rcpp::stop("beta_start must have length p");
  if (!(s > 0.0)) Rcpp::stop("step size s must be positive");
  if (!(eps > 0.0)) Rcpp::stop("eps must be positive");
  
  const int max_iter = 5000;
  
  arma::colvec x_curr = beta_start;
  arma::colvec y_curr = x_curr;
  
  // Objective: (1/2n)||Y - Xb||^2 + lambda||b||_1
  auto f_eval = [&](const arma::colvec& b) -> double {
    arma::colvec r = Ytilde - Xtilde * b;
    const double rss = arma::dot(r, r) / (2.0 * n);
    const double l1  = arma::sum(arma::abs(b));
    return rss + lambda * l1;
  };
  
  double f_curr = f_eval(x_curr);
  if (!std::isfinite(f_curr)) Rcpp::stop("objective not finite at initialization");
  
  double t_prev = 1.0;
  arma::colvec grad(p), z(p), x_next(p), y_next(p);
  
  for (int iter = 0; iter < max_iter; ++iter) {
    // r = Y - X y; grad = -(1/n) X^T r
    arma::colvec r = Ytilde - Xtilde * y_curr;
    grad = -(Xtilde.t() * r) / static_cast<double>(n);
    
    // Proximal step
    z = y_curr - s * grad;
    for (int j = 0; j < p; ++j) x_next[j] = soft_c(z[j], lambda * s);
    
    // Nesterov update
    const double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t_prev * t_prev));
    const double mom = (t_prev - 1.0) / t_next;
    y_next = x_next + mom * (x_next - x_curr);
    
    // Monotone restart
    double f_next = f_eval(x_next);
    if (!std::isfinite(f_next) || f_next > f_curr) {
      y_next = x_next;
      f_next = f_eval(x_next);
      // also reset momentum
      t_prev = 1.0;
    } else {
      t_prev = t_next;
    }
    
    // Convergence: relative L1 change
    double num = arma::accu(arma::abs(x_next - x_curr));
    double den = 1.0 + arma::accu(arma::abs(x_curr));
    if (num / den < eps) return x_next;
    
    // Update state
    x_curr = x_next;
    y_curr = y_next;
    f_curr = f_next;
  }
  
  return x_curr; // reached max_iter
}