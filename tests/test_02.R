# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# Compare MPCC versus cor() function, mA = 15 x 5, mB = 15 x 10, no missing data
library(MPCC)

set.seed(1)
mAB <- genAB(p = 5, n = 15, m = 10)

ref <- cor(mAB[["A"]], mAB[["B"]])
mpcc <- PCC(mAB[["A"]], mAB[["B"]])

if (sum(round(mpcc$res - ref, 12)) != 0) {
  stop("Inaccurate results for 5x10 matrix")
}

