# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

library(MPCC)
MPCCinfo()

set.seed(1)
mAB <- genAB(p = 5, n = 15, m = 10, missing = 0.4)

ref <- cor(mAB[["A"]], use="pair")
mpcc <- PCC(mAB[["A"]], debugOn=TRUE)

if (sum(round(mpcc$res - ref, 12),na.rm = TRUE) != 0) {
  stop("Inaccurate results for 10x15 matrix")
}

