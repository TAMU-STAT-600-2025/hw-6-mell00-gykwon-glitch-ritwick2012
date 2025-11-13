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

