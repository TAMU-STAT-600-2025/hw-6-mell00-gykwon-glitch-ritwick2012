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


testthat::test_that("empty-cluster triggers a clear error", {
  set.seed(6005)
  # Two points and three initial centers -> empty cluster must exist as K > X
  X <- rbind(c(0, 0), c(10, 10))
  K <- 3
  M_bad <- rbind(c(0, 0), c(5, 5), c(10, 10))
  # Expect the algorithm to stop when a cluster becomes empty
  testthat::expect_error(
    MyKmeans(X, K, M = M_bad, numIter = 10),
    "A cluster became empty"
  )
})

testthat::test_that("empty-cluster triggers a clear error", {
  set.seed(7005)
  n <- 120; p <- 2; K <- 3
  X <- rbind(
    matrix(rnorm(n/2 * p, mean = 0, sd = 0.3), n/2, p),
    matrix(rnorm(n/2 * p, mean = 6, sd = 0.3), n/2, p)
  )
  
  # Two reasonable centers near data, one absurdly far away (bad input)
  M_bad <- rbind(
    c(0, 0),
    c(6, 6),
    c(100, 100)  # Empty cluster occurs with a dominated far-away center (nobody chooses this)
  )
  
  testthat::expect_error(
    MyKmeans(X, K, M = M_bad, numIter = 50),
    "A cluster became empty"
  )
})

testthat::test_that("K=1 places all points in cluster 1", {
  set.seed(6006)
  n <- 25
  p <- 5
  K <- 1
  X <- matrix(rnorm(n * p), n, p)
  M <- matrix(colMeans(X), K, p)
  
  # All points should belong to cluster 1
  Y <- MyKmeans(X, K, M = M, numIter = 30)
  testthat::expect_true(all(Y == 1L))
  testthat::expect_length(Y, n)
})

testthat::test_that("Close to perfect ARI on well-separated two-Gaussian mixture", {
  if (!requireNamespace("mclust", quietly = TRUE)) {
    testthat::skip("mclust not installed")
  }
  set.seed(7001)
  X_two <- rbind(
    matrix(rnorm(100 * 3, mean = -10, sd = 1), 100, 3),
    matrix(rnorm(100 * 3, mean =  10, sd = 1), 100, 3)
  )
  true_label <- c(rep(1L, 100), rep(2L, 100))
  Y <- as.integer(MyKmeans(X_two, 2, M = NULL, numIter = 50))
  ari <- mclust::adjustedRandIndex(Y, true_label)
  testthat::expect_equal(ari, 1, tolerance = 0.05)
})



