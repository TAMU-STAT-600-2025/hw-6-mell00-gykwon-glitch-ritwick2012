library(testthat)

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


