# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# MPCC call function to R
#

# Test function to compare to the standard R implementation
testImprovement <- function() {
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

