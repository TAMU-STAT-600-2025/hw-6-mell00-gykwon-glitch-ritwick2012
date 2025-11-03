#########################################
# Source the functions and necessary libraries
#########################################
# Lasso functions for coordinate descent
source("LASSO_CoordinateDescent.R")
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


#########################################
# Test Nesterov acceleration
##########################################

out_prox2 <- fitLASSOstandardized_prox_Nesterov(out$Xtilde, out$Ytilde, beta_start = rep(0, p),lambda = lambda1, eps = 1e-10, s = 0.1)
out_coord$fmin - out_prox2$fmin
plot(out_coord$beta, out_prox2$beta)
plot(out_prox2$fobj_vec[-c(1:40)])

# Check the implementation time
micro_out <- microbenchmark(
  fitLASSOstandardized_prox_Nesterov(out$Xtilde, out$Ytilde, beta_start = rep(0, p),lambda = lambda1, eps = 1e-10, s = 0.1),
  fitLASSO_prox_Nesterov(out$Xtilde, out$Ytilde, beta_start = rep(0, p),lambda = lambda1, eps = 1e-10, s = 0.1)
)

print(micro_out)

