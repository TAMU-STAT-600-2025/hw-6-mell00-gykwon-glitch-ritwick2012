
set.seed(123)
cat("\n=== Begin Nesterov LASSO Tests ===\n")
n_ok <- 0L

## Helper functions

# KKT check on standardized scale:
# g = (1/n) X^T (Y - X b)
# active coords satisfy g_j = lambda * sign(b_j)
# inactive satisfy |g_j| <= lambda
kkt_check <- function(Xt, Yt, btilde, lambda, tol = 1e-6) {
  n <- nrow(Xt)
  r <- as.numeric(Yt - Xt %*% btilde)
  g <- as.numeric(crossprod(Xt, r)) / n
  act <- which(abs(btilde) > 0)
  inact <- setdiff(seq_along(btilde), act)
  ok_act <- TRUE
  if (length(act)) ok_act <- max(abs(g[act] - lambda * sign(btilde[act]))) <= tol
  ok_inact <- TRUE
  if (length(inact)) ok_inact <- max(abs(g[inact])) <= lambda + tol
  list(ok = ok_act && ok_inact, max_act = ifelse(length(act), max(abs(g[act] - lambda * sign(btilde[act]))), 0),
       max_inact = ifelse(length(inact), max(pmax(0, abs(g[inact]) - lambda)), 0))
}