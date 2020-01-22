# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

library(MPCC)
MPCCinfo()

set.seed(1)
x <- 10

mA <- matrix(runif(x * x), x, x)

ref <- cor(mA)
mpcc <- PCC(mA)

if (sum(round(mpcc - ref, 12)) != 0) {
  stop("Inaccurate results for 10x10 matrix")
}

