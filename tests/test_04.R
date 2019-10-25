# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

library(MPCC)

set.seed(1)

vA <- rep(NA, 35)
vB <- rep(NA, 35)

vA[2:26] <- rnorm(25,10,15)
vB[2:26] <- rnorm(25,10,15)

M <- cbind(vA, vB)

ref <- cor(M, M, use="pair")
naive <- PCC.naive(M, M)
mpcc <- PCC(M, M)

if (sum(round(mpcc - ref, 12),na.rm = TRUE) != 0) {
  stop("Inaccurate results for missing data test")
}
