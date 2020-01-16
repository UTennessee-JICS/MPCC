library(coop)
library(rbenchmark)
library(MPCC)

cols <- c("test", "replications", "elapsed", "relative")
reps <- 25

m <- 4000
n <- 1000
x <- matrix(rnorm(m*n), m, n)

benchmark(cor(x), pcor(x), PCC(x), replications=reps, columns=cols)

library(MPCC)
library(rbenchmark)

cols <- c("test", "replications", "elapsed", "relative")
reps <- 25

set.seed(1)

mAB <- genAB(p = 1000, n = 50000, m = 1000, missing = 0.4)
benchmark(PCC(mAB[["A"]], mAB[["B"]]), replications=reps, columns=cols)
