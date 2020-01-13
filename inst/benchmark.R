library(coop)
library(rbenchmark)
library(MPCC)

cols <- c("test", "replications", "elapsed", "relative")
reps <- 25

m <- 4000
n <- 1000
x <- matrix(rnorm(m*n), m, n)

benchmark(cor(x), pcor(x), PCC(x), replications=reps, columns=cols)