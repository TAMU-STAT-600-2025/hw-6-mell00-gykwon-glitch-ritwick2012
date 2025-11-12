testthat::test_that("basic run returns 1..K labels of length n", {
  set.seed(6001)
  n <- 60 
  p <- 2
  K <- 3
  X <- rbind(
    matrix(rnorm(n/3 * p, mean = 0, sd = 0.25), n/3, p),
    matrix(rnorm(n/3 * p, mean = 3, sd = 0.25), n/3, p),
    matrix(rnorm(n/3 * p, mean = 6, sd = 0.25), n/3, p)
  )
  
  Y <- MyKmeans(X, K, M = NULL, numIter = 100)
  
  # Check that the result is an integer vector of length n
  testthat::expect_type(Y, "integer")
  testthat::expect_length(Y, n)
  # All labels must be between 1 and K
  testthat::expect_true(all(Y >= 1L & Y <= K))
  # Ensure every point was assigned to some cluster
  tab <- table(Y)
  testthat::expect_equal(sum(tab), n)
})


testthat::test_that("determinism with same seed & M=NULL", {
  set.seed(6002)
  n <- 45
  p <- 3
  K <- 4
  X <- matrix(rnorm(n * p), n, p)
  
  # If the seed and initialization are identical, output must match
  set.seed(42);  Y1 <- MyKmeans(X, K, M = NULL, numIter = 50)
  set.seed(42);  Y2 <- MyKmeans(X, K, M = NULL, numIter = 50)
  testthat::expect_identical(Y1, Y2)
})

testthat::test_that("determinism when M is provided", {
  set.seed(6003)
  n <- 40
  p <- 4
  K <- 3
  X <- matrix(rnorm(n * p), n, p)
  M0 <- X[sample.int(n, K, replace = FALSE), , drop = FALSE]
  
  # Using the same initial centers must yield identical assignments
  Y1 <- MyKmeans(X, K, M = M0, numIter = 80)
  Y2 <- MyKmeans(X, K, M = M0, numIter = 80)
  testthat::expect_identical(Y1, Y2)
})

testthat::test_that("error when M has incompatible dimension", {
  set.seed(6004)
  n <- 20
  p <- 3
  K <- 3
  X <- matrix(rnorm(n * p), n, p)
  
  # M has wrong number of rows (K-1 instead of K)
  Mbad <- matrix(rnorm((K - 1) * p), K - 1, p)
  testthat::expect_error(
    MyKmeans(X, K, M = Mbad, numIter = 10),
    "M must be K x p"
  )
})

