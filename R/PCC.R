#
# MPCC call function to R
#

# PCC matrix wrapper
PCC <- function(aM, bM, use = NULL) {
  res <- .C("R_pcc_matrix", aM = as.double(aM),
                            bM = as.double(t(bM)),
                            n = as.integer(ncol(aM)), # nInd
                            m = as.integer(nrow(aM)), # nPhe A
                            p = as.integer(nrow(bM)), # nPhe B
                            res = as.double(rep(0, nrow(aM) * nrow(bM))), NAOK = TRUE, package = "MPCC")
  return(res)
}

# PCC naive wrapper
PCC.naive <- function(aM, bM, use = NULL) {
  res <- .C("R_pcc_naive", aM = as.double(aM),
                           bM = as.double(t(bM)),
                           n = as.integer(ncol(aM)), # nInd
                           m = as.integer(nrow(aM)), # nPhe A
                           p = as.integer(nrow(bM)), # nPhe B
                           res = as.double(rep(0, nrow(aM) * nrow(bM))), NAOK = TRUE, package = "MPCC")
  return(res)
}

# Generate A and B matrices, using p,n,m as input size for A (p*n) and B (m*n)
genAB <- function(p = 100, n = 150, m = 20, missing = 0) {
  mA <- matrix(rnorm(p * n), n, p)
  mB <- matrix(rnorm(m * n), n, m)
  if (missing > 0) {
    mA[sample(p*n, (p * n * missing))] <- NA
    mB[sample(m*n, (m * n * missing))] <- NA
  }
  return(list(A = mA, B = mB))
}

# run the PNM timing
runPNM <- function(fun = cor, p = 100, n = 150, m = 20, missing = 0, use = "everything") {
  mAB <- genAB(p,n,m,missing)

  start <- proc.time()[3] # Start time
  res <- fun(mAB[["A"]], mAB[["B"]], use = use)
  elapsed <- proc.time()[3] - start # Add time information

  return(list(res = res, time = elapsed))
}

testRunPNM <- function(){
 require(MPCC)
 res0 <- genAB()
 full.cor <- runPNM(fun = cor)
 full.pcc <- runPNM(fun = PCC)
 m05.cor <- runPNM(fun = cor, missing = 0.05, use = "pair")
 m05.pcc <- runPNM(fun = PCC, missing = 0.05, use = "pair")
 m10.cor <- runPNM(fun = cor, missing = 0.10, use = "pair")
 m10.pcc <- runPNM(fun = PCC, missing = 0.10, use = "pair")
}

# Test function to compare to the standard R implementation
test <- function() {
    require(MPCC)
    set.seed(1)

    # Percentage of missing data
    missing = 0.05

    times.ref <- c() # time for cor reference (no missing data)
    times.ref.missing <- c() # time for cor reference (missing data)
    times.pcc <- c() # time for the pcc matrix version
    sizes <- c(50, 100, 150, 200, 250, 300, 400)
    for(x in sizes) {
      time.ref <- c()
      time.ref.missing <- c()
      time.pcc <- c()
      for(y in 1:25) {
        mA <- matrix(runif(x * x), x, x)
        mB <- matrix(runif(x * x), x, x)
        nmis <- (x * x * missing)
        s.ref <- proc.time()[3] # Start time
        cor(mA, mB)
        time.ref <- c(time.ref, (proc.time()[3] - s.ref)) # Add time information
        s.pcc <- proc.time()[3] # Start time
        PCC(mA, mB)
        time.pcc <- c(time.pcc, (proc.time()[3] - s.pcc)) # Add time information

        mA[sample(x*x, nmis)] <- NA

        s.ref.missing <- proc.time()[3] # Start time
        cor(mA, mB, use = "pair")
        time.ref.missing <- c(time.ref.missing, (proc.time()[3] - s.ref.missing)) # Add time information
      }
      times.ref <- cbind(times.ref, time.ref)
      times.pcc <- cbind(times.pcc, time.pcc)
      times.ref.missing <- cbind(times.ref.missing, time.ref.missing)
      cat("Done", x, "\n")
    }
    colnames(times.ref) <- sizes
    colnames(times.pcc) <- sizes
    colnames(times.ref.missing) <- sizes

    # Calculate the numbers median / stdev
    ref.med <- apply(times.ref, 2, median)
    ref.sd <- apply(times.ref, 2, sd)

    pcc.med <- apply(times.pcc, 2, median)
    pcc.sd <- apply(times.pcc, 2, sd)

    ref.med.missing <- apply(times.ref.missing, 2, median)
    ref.sd.missing <- apply(times.ref.missing, 2, sd)

    write.table(times.ref, "times.ref.txt", sep = "\t", quote = FALSE)
    write.table(times.ref.missing, "times.ref.missing.txt", sep = "\t", quote = FALSE)
    write.table(times.pcc, "times.pcc.txt", sep = "\t", quote = FALSE)

    cat(paste(sizes), "\n", file = "speedup.txt")
    cat("nonmissing", paste(ref.med / pcc.med), "\n", file = "speedup.txt", append = TRUE)
    cat("missing", paste(ref.med.missing / pcc.med), "\n", file = "speedup.txt", append = TRUE)

    # Make the plot
    mY <- max(c(times.ref.missing, times.ref, times.pcc))
    png("my.png", width=1024, height = 800)
      plot(c(min(sizes), max(sizes)), c(0, mY), t = 'n', xlab = "size", ylab = "time (s)")

      polygon(x = c(sizes, rev(sizes)), y = c(ref.med + ref.sd, rev(ref.med - ref.sd)), col="blue")
      points(x = sizes, y = ref.med, t = 'l', col="blue", lwd=2)

      polygon(x = c(sizes, rev(sizes)), y = c(pcc.med + pcc.sd, rev(pcc.med - pcc.sd)), col="orange")
      points(x = sizes, y = pcc.med, t = 'l', col="orange", lwd=2)

      polygon(x = c(sizes, rev(sizes)), 
              y = c(ref.med.missing + ref.sd.missing, rev(ref.med.missing - ref.sd.missing)), col="lightblue")
      points(x = sizes, y = ref.med.missing, t = 'l', col="lightblue", lwd=2)

      legend("topleft", c("R cor()", "R cor(5% missing)", "Matrix PCC"), fill = c("blue", "lightblue", "orange"))
    dev.off()
}

