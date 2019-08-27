# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# Compare MPCC versus cor() function, mA = 15 x 5, mB = 15 x 10, no missing data
library(MPCC)

set.seed(1)
mAB <- genAB(p = 2000, n = 2000, m = 2000)

s.ref <- proc.time()[3]
ref <- cor(mAB[["A"]], mAB[["B"]])
(proc.time()[3] - s.ref)

s.mpcc <- proc.time()[3]
mpcc <- PCC(mAB[["A"]], mAB[["B"]])
(proc.time()[3] - s.mpcc)

s.mpcc <- proc.time()[3]
mpcc <- PCC.naive(mAB[["A"]], mAB[["B"]])
(proc.time()[3] - s.mpcc)


if (sum(round(mpcc - ref, 12)) != 0) {
  stop("Inaccurate results for 5x10 matrix")
}

