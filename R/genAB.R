# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# Generate A and B matrices, using p, n, m as input size for A (p*n) and B (m*n)
genAB <- function(p = 100, n = 150, m = 20, missing = 0) {
  mA <- matrix(rnorm(p * n), n, p)
  mB <- matrix(rnorm(m * n), n, m)
  if (missing > 0) {
    mA[sample(p*n, (p * n * missing))] <- NA
    mB[sample(m*n, (m * n * missing))] <- NA
  }
  return(list(A = mA, B = mB))
}

