# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# Compare MPCC versus cor() function, mA = 15 x 5, mB = 15 x 10, no missing data
library(MPCC)
MPCCinfo()

set.seed(1)
mAB <- genAB(p = 200, n = 300, m = 200)

s.ref <- proc.time()[3]
ref <- cor(mAB[["A"]], mAB[["B"]])
(proc.time()[3] - s.ref)

s.mpcc <- proc.time()[3]
mpcc <- PCC(mAB[["A"]], mAB[["B"]])
(proc.time()[3] - s.mpcc)

if (sum(round(mpcc - ref, 12)) != 0) {
  stop("Inaccurate results for 200x200 matrix")
}

s.mpcc <- proc.time()[3]
mpcc <- PCC.naive(mAB[["A"]], mAB[["B"]])
(proc.time()[3] - s.mpcc)


if (sum(round(mpcc - ref, 12)) != 0) {
  stop("Inaccurate results for 200x200 matrix")
}

