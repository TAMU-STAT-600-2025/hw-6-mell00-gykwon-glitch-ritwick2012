#########################################
# Source the functions and necessary libraries
#########################################
# Lasso functions for coordinate descent
source("LASSO_CoordinateDescent.R")

# Nesterov LASSO function R wrapper fitLASSO_prox_Nesterov
source("LASSO_wrapper.R")

# Functions written in the starter code
source("LASSOProximalExample.R")

# For later time comparisons
library(bench)
library(microbenchmark)

#########################################
# Generate toy example dataset, standardize
#########################################
set.seed(38947)
p = 50
n = 30
Y <- rnorm(n)
X <- matrix(rnorm(n*p), n, p)
out <- standardizeXY(X,Y)
lambda_max <- max(abs(crossprod(out$Xtilde, out$Ytilde))/nrow(X))
lambda1 <- 0.1 * lambda_max

#########################################
# Test Nesterov acceleration
##########################################

out_coord <- fitLASSO_prox_Nesterov(out$Xtilde, out$Ytilde, lambda1,
                                  beta_start = rep(0, p), eps = 1e-10)

out_prox2 <- fitLASSOstandardized_prox_Nesterov(out$Xtilde, out$Ytilde, beta_start = rep(0, p),lambda = lambda1, eps = 1e-10, s = 0.1)
out_coord$fmin - out_prox2$fmin
plot(out_coord$beta, out_prox2$beta)
plot(out_prox2$fobj_vec[-c(1:40)])

# Compare objectives
out_coord$fmin - out_prox2$fmin
plot(out_coord$beta, out_prox2$beta)
abline(0, 1, col = "red", lty = 2)

# Check the implementation time
micro_out <- microbenchmark(
  fitLASSOstandardized_prox_Nesterov(out$Xtilde, out$Ytilde, beta_start = rep(0, p),lambda = lambda1, eps = 1e-10),
  fitLASSO_prox_Nesterov(out$Xtilde, out$Ytilde, beta_start = rep(0, p),lambda = lambda1, eps = 1e-10, s = 0.1)
)

print(micro_out)

