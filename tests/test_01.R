# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# Compare MPCC versus cor() function, mA = 5 x 5, mB = 5 x 5, no missing data
library(MPCC)

set.seed(1)
x <- 5

mA <- matrix(runif(x * x), x, x)
mB <- matrix(runif(x * x), x, x)

ref <- cor(mA, mB)
mpcc <- PCC(mA, mB)

if (sum(round(mpcc$res - ref, 12)) != 0) {
  stop("Inaccurate results for 5x5 matrix")
}

