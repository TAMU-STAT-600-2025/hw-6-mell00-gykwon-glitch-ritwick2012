library(testthat)
library(GroupHW)


#Test 1


test_that("LRMultiClass returns beta matrix and objective vector of correct size", {
  set.seed(123)
  n <- 30
  p <- 3
  K <- 2
  X <- cbind(1, matrix(rnorm(n * (p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 10)
  
  # All output checks
  testthat::expect_true(is.list(out))
  testthat::expect_true("beta" %in% names(out))
  testthat::expect_true("objective" %in% names(out))
  testthat::expect_equal(dim(out$beta), c(p, K))
  testthat::expect_length(out$objective, 11) # numIter + 1
})


#Test 2


test_that("Objective decreases over iterations", {
  set.seed(456)
  n <- 25
  p <- 4
  K <- 3
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 5)
  
  # Objective should not increase
  testthat::expect_true(all(diff(out$objective) <= 1e-6 + 1e-8))
})


#Test 3


test_that("Handles beta_init correctly", {
  set.seed(789)
  n <- 20
  p <- 3
  K <- 3
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  beta_init <- matrix(0.5, nrow = p, ncol = K)
  out <- LRMultiClass(X, y, beta_init = beta_init, numIter = 5)
  
  testthat::expect_equal(dim(out$beta), c(p, K))
})


#Test 4


test_that("Error when first column of X is not all 1s", {
  set.seed(101)
  n <- 20
  p <- 3
  K <- 2
  X <- matrix(rnorm(n*p), n, p)
  y <- sample(0:(K-1), n, replace = TRUE)
  
  testthat::expect_error(
    LRMultiClass(X, y),
    "First column of X must be all 1s"
  )
})


#Test 5


test_that("Error when eta <= 0 or lambda < 0", {
  set.seed(102)
  n <- 15
  p <- 3
  K <- 2
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  testthat::expect_error(LRMultiClass(X, y, eta = 0))
  testthat::expect_error(LRMultiClass(X, y, lambda = -1))
})


#Test 6


test_that("K=1 returns single-column beta matrix", {
  set.seed(111)
  n <- 20
  p <- 4
  K <- 1
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- rep(0, n)  # all in one class
  
  out <- LRMultiClass(X, y, numIter = 5)
  
  testthat::expect_equal(dim(out$beta), c(p, 1))
  testthat::expect_length(out$objective, 6) # numIter + 1
})


#Test 7


test_that("LRMultiClass works when n < p", {
  set.seed(333)
  n <- 3
  p <- 5
  K <- 2
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 5)
  
  testthat::expect_equal(dim(out$beta), c(p, K))
  testthat::expect_length(out$objective, 6)
})


#Test 8


test_that("Error when beta_init has wrong dimensions", {
  set.seed(555)
  n <- 20
  p <- 3
  K <- 2
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  beta_wrong <- matrix(0, nrow = p, ncol = K + 1) # wrong number of columns
  
  testthat::expect_error(
    LRMultiClass(X, y, beta_init = beta_wrong),
    "beta_init dimensions must be p x K"
  )
})


#Test 9


test_that("Single sample per class still returns correct dimensions", {
  set.seed(204)
  n <- 3
  p <- 2
  K <- 3
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- 0:2
  
  out <- LRMultiClass(X, y, numIter = 3)
  
  testthat::expect_equal(dim(out$beta), c(p, K))
  testthat::expect_length(out$objective, 4)
})


#Test 10

test_that("LRMultiClass works with small eta and large lambda", {
  set.seed(201)
  n <- 30
  p <- 3
  K <- 3
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 10, eta = 0.01, lambda = 10)
  
  testthat::expect_equal(dim(out$beta), c(p, K))
  testthat::expect_length(out$objective, 11)
  # Check objective decreases or stays almost the same (small step size)
  testthat::expect_true(all(diff(out$objective) <= 1e-5 + 1e-8))
})


#Test 11


test_that("Predicted probabilities from softmax sum to 1 for each observation", {
  set.seed(321)
  n <- 50
  p <- 4
  K <- 3
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 10)
  beta <- out$beta
  
  # Compute probabilities
  scores <- X %*% beta
  scores <- sweep(scores, 1, apply(scores, 1, max)) # numerical stability
  exp_scores <- exp(scores)
  P <- exp_scores / rowSums(exp_scores)
  
  # Each row should sum to 1
  testthat::expect_true(all(abs(rowSums(P) - 1) < 1e-8))
})


#Test 12 


test_that("C++ wrapper is faster than pure R implementation", {
  
  old <- getOption("warn")
  options(warn = -1)   
  
  suppressPackageStartupMessages(library(microbenchmark))
  
  
  set.seed(123)
  n <- 100
  p <- 10
  K <- 3
  X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  # Pure R Newton implementation
  LRMultiClass_R <- function(X, y, beta_init = NULL, numIter = 20, eta = 0.1, lambda = 1){
    n <- nrow(X); p <- ncol(X)
    K <- length(unique(y))
    if(is.null(beta_init)) beta_init <- matrix(0, nrow = p, ncol = K)
    beta <- beta_init
    objective <- numeric(numIter + 1)
    
    softmax <- function(X, beta){
      scores <- X %*% beta
      scores <- sweep(scores, 1, apply(scores, 1, max)) # numeric stability
      exp_scores <- exp(scores)
      exp_scores / rowSums(exp_scores)
    }
    
    obj_fn <- function(beta){
      P <- softmax(X, beta)
      loglik <- sum(log(P[cbind(1:n, y + 1)]))
      reg <- 0.5 * lambda * sum(beta^2)
      -loglik + reg
    }
    
    objective[1] <- obj_fn(beta)
    
    for(t in 1:numIter){
      P <- softmax(X, beta)
      for(k in 1:K){
        pk <- P[,k]
        yk <- as.numeric(y == (k-1))
        gk <- t(X) %*% (pk - yk) + lambda * beta[,k]
        wk <- pk * (1 - pk)
        Hk <- t(X) %*% (X * wk) + lambda * diag(p)
        delta <- solve(Hk, gk)
        beta[,k] <- beta[,k] - eta * delta
      }
      objective[t+1] <- obj_fn(beta)
    }
    
    list(beta = beta, objective = objective)
  }
  
  # Microbenchmark both
  mb <- microbenchmark(
    R_only = LRMultiClass_R(X, y, numIter = 20),
    Cpp_via_wrapper = LRMultiClass(X, y, numIter = 20),
    times = 5
  )
  
  print(mb)
  
  # Test: C++ should be at least 3x faster than pure R
  mean_R <- mean(mb$time[mb$expr == "R_only"])
  mean_Cpp <- mean(mb$time[mb$expr == "Cpp_via_wrapper"])
  
  expect_lt(mean_Cpp, mean_R / 3)
})

# Test 13
test_that("LRMultiClass handles extremely large X values without NaN", {
  set.seed(701)
  n <- 20
  p <- 4
  K <- 3
  
  X <- cbind(1, matrix(1e6 * rnorm(n * (p-1)), n, p-1))  # large-scale predictors
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 5)
  
  testthat::expect_false(any(is.nan(out$objective)))
  testthat::expect_false(any(is.nan(out$beta)))
})

# Test 14

test_that("LRMultiClass handles very large lambda", {
  set.seed(705)
  n <- 40
  p <- 5
  K <- 3
  
  X <- cbind(1, matrix(rnorm(n * (p-1)), n, p-1))
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 10, lambda = 1e6)
  
  # coefficients should shrink almost to zero
  testthat::expect_true(max(abs(out$beta)) < 1e-3)
})

# Test 15

test_that("LRMultiClass handles nearly separable classes", {
  set.seed(703)
  n <- 60
  p <- 3
  K <- 2
  
  x_raw <- rnorm(n)
  # y almost perfectly determined by sign(x)
  y <- ifelse(x_raw > 0, 1, 0)
  
  # large signal to exaggerate separability
  X <- cbind(1, x_raw, 10 * x_raw)
  
  out <- LRMultiClass(X, y, numIter = 25, lambda = 1)
  
  testthat::expect_equal(dim(out$beta), c(p, K))
  testthat::expect_false(any(is.nan(out$objective)))
})


# Test 16

test_that("LRMultiClass handles collinear predictors", {
  set.seed(702)
  n <- 30
  p <- 4
  K <- 3
  
  z <- rnorm(n)
  X <- cbind(1, z, z, z)  # identical columns
  y <- sample(0:(K-1), n, replace = TRUE)
  
  out <- LRMultiClass(X, y, numIter = 10)
  
  # ensure the algorithm runs and objective finite
  testthat::expect_equal(dim(out$beta), c(p, K))
  testthat::expect_false(any(is.infinite(out$objective)))
})

