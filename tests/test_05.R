# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# Compare MPCC versus cor() function, mA = 15 x 5, mB = 15 x 10, no missing data
library(MPCC)
MPCCinfo()

set.seed(1)
mAB <- genAB(p = 5, n = 15, m = 10, missing = 1)

ref <- cor(mAB[["A"]], mAB[["B"]], use="pair")
mpcc <- PCC(mAB[["A"]], mAB[["B"]], debugOn=TRUE)

if (sum(round(mpcc$res - ref, 12),na.rm = TRUE) != 0) {
  stop("Inaccurate results for 5x10 matrix")
}

