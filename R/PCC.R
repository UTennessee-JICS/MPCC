# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# MPCC call function to R
#
testBug <-function(){
    require(MPCC)
    set.seed(1)
    vA <- rep(NA,35)
    vB <- rep(NA,35)
    vA[2:26] <- rnorm(25,10,15)
    vB[2:26] <- rnorm(25,10,15)
    M <- cbind(vA,vB)
    #ref <- cor(M,M, use="pair")
    naive <- PCC.naive(M,M)
    mpcc <- PCC(M,M)
    #ref
    #naive
    mpcc
}

# Test function to compare to the standard R implementation
test_MPCC_Scaling <- function() {
    require(MPCC)
    library(OpenMPController)
    #library(parallel)
    #numCores <- omp_get_max_threads()
    #print(numCores)

    set.seed(1)

    # Percentage of missing data
    missing = 0.05
    times.pccm <- c() # time for the pcc matrix version
    matsize <- 2000
    #matsize <- 4
    threads <- c(1,2)
    print(matsize) 
    mA <- matrix(runif(matsize * matsize), matsize, matsize)
    mB <- matrix(runif(matsize * matsize), matsize, matsize)
    nmis <- (matsize * matsize * missing)
    # create missing data
    mA[sample(matsize*matsize, nmis)] <- NA

    for(nthreads in threads) {
      omp_set_num_threads(nthreads)
      time.pccm <- c()

      for(y in 1:1) {
        #run matrix version
        s.pccm <- proc.time()[3] # Start time
        #change number of threads
        PCC(mA, mB, nthreads)
        #PCC.naive(mA, mB, nthreads)
        time.pccm <- c(time.pccm, (proc.time()[3] - s.pccm)) # Add time information
        cat("time=",time.pccm,"\n")
      }

      times.pccm <- cbind(times.pccm, time.pccm)
      cat("Finished ", nthreads, " threads \n")
    }

    colnames(times.pccm) <- threads

    pccm.med <- apply(times.pccm, 2, median)
    pccm.sd <- apply(times.pccm, 2, sd)

    #write.table(times.pccm, "times.pccm.txt", sep = "\t", quote = FALSE)

    # Make the plot
    mY <- max(c(times.pccm))
    png("plots/timing.png", width=1024, height = 800)
      plot(c(min(threads), max(threads)), c(0, mY), t = 'n', xlab = "threads", ylab = "time (s)")
      polygon(x = c(threads, rev(threads)), y = c(pccm.med + pccm.sd, rev(pccm.med - pccm.sd)), col="orange")
      points(x = threads, y = pccm.med, t = 'l', col="orange", lwd=2)
      legend("topleft", c("Matrix PCC"), fill = c("orange"))
    dev.off()
}


# Test function to compare to the standard R implementation
testImprovement <- function() {

    library(parallel)
    numCores <- detectCores()
    print(numCores)

    require(MPCC)
    set.seed(1)

    # Percentage of missing data
    missing = 0.05

    ##times.ref <- c() # time for cor reference (no missing data)
    times.ref.missing <- c() # time for cor reference (missing data)
    times.pccn <- c() # time for the pcc naive version
    times.pccm <- c() # time for the pcc matrix version
    times.pccv <- c() # time for the pcc vector version
    sizes <- c(50, 250, 1000, 2000, 4000)
    for(x in sizes) {
      ##time.ref <- c()
      time.ref.missing <- c()
      time.pccn <- c()
      time.pccm <- c()
      time.pccv <- c()

      for(y in 1:1) {
        mA <- matrix(runif(x * x), x, x)
        mB <- matrix(runif(x * x), x, x)
        nmis <- (x * x * missing)

        #s.ref <- proc.time()[3] # Start time
        #cor(mA, mB)
        #time.ref <- c(time.ref, (proc.time()[3] - s.ref)) # Add time information

        # create missing data (create new matrix and missing data for each iteration) 
        mA[sample(x*x, nmis)] <- NA

        #run naive R cor version
        s.ref.missing <- proc.time()[3] # Start time
        cor(mA, mB, use = "pair")
        time.ref.missing <- c(time.ref.missing, (proc.time()[3] - s.ref.missing)) # Add time information

        #run naive optimized pcc version
        s.pccn <- proc.time()[3] # Start time
        PCC.naive(mA, mB)
        time.pccn <- c(time.pccn, (proc.time()[3] - s.pccn)) # Add time information

        #run matrix version
        s.pccm <- proc.time()[3] # Start time
        PCC(mA, mB)
        time.pccm <- c(time.pccm, (proc.time()[3] - s.pccm)) # Add time information

        #run vector version
        s.pccv <- proc.time()[3] # Start time
        PCC.vector(mA, mB)
        time.pccv <- c(time.pccv, (proc.time()[3] - s.pccv)) # Add time information

      }


      ##times.ref <- cbind(times.ref, time.ref)
      times.ref.missing <- cbind(times.ref.missing, time.ref.missing)
      times.pccn <- cbind(times.pccn, time.pccn)
      times.pccm <- cbind(times.pccm, time.pccm)
      times.pccv <- cbind(times.pccv, time.pccv)
      cat("Done", x, "\n")
    }

    ##colnames(times.ref) <- sizes
    colnames(times.ref.missing) <- sizes
    colnames(times.pccn) <- sizes
    colnames(times.pccm) <- sizes
    colnames(times.pccv) <- sizes

    ## Calculate the numbers median / stdev
    ##ref.med <- apply(times.ref, 2, median)
    ##ref.sd <- apply(times.ref, 2, sd)

    ref.med.missing <- apply(times.ref.missing, 2, median)
    ref.sd.missing <- apply(times.ref.missing, 2, sd)

    pccn.med <- apply(times.pccn, 2, median)
    pccn.sd <- apply(times.pccn, 2, sd)

    pccm.med <- apply(times.pccm, 2, median)
    pccm.sd <- apply(times.pccm, 2, sd)

    pccv.med <- apply(times.pccv, 2, median)
    pccv.sd <- apply(times.pccv, 2, sd)


    ##write.table(times.ref, "times.ref.txt", sep = "\t", quote = FALSE)
    write.table(times.ref.missing, "output/times.ref.missing.txt", sep = "\t", quote = FALSE)
    write.table(times.pccn, "output/times.pccn.txt", sep = "\t", quote = FALSE)
    write.table(times.pccm, "output/times.pccm.txt", sep = "\t", quote = FALSE)
    write.table(times.pccv, "output/times.pccv.txt", sep = "\t", quote = FALSE)

    cat(paste(sizes), "\n", file = "output/speedup.txt")
    ##cat("nonmissing", paste(ref.med / pcc.med), "\n", file = "speedup.txt", append = TRUE)
    cat("missing naive", paste(ref.med.missing / pccn.med), "\n", file = "output/speedup.txt", append = TRUE)
    cat("missing mat", paste(ref.med.missing / pccm.med), "\n", file = "output/speedup.txt", append = TRUE)
    cat("missing vec", paste(ref.med.missing / pccv.med), "\n", file = "output/speedup.txt", append = TRUE)

    # Make the plot
    ##mY <- max(c(times.ref.missing, times.ref, times.pccm, times.pccv))
    mY <- max(c(times.ref.missing, times.pccn, times.pccm, times.pccv))
    png("plots/plots.png", width=1024, height = 800)
      plot(c(min(sizes), max(sizes)), c(0, mY), t = 'n', xlab = "size", ylab = "time (s)")

      ##polygon(x = c(sizes, rev(sizes)), y = c(ref.med + ref.sd, rev(ref.med - ref.sd)), col="blue")
      ##points(x = sizes, y = ref.med, t = 'l', col="blue", lwd=2)

      polygon(x = c(sizes, rev(sizes)), y = c(ref.med.missing + ref.sd.missing, rev(ref.med.missing - ref.sd.missing)), col="lightblue")
      points(x = sizes, y = ref.med.missing, t = 'l', col="lightblue", lwd=2)

      polygon(x = c(sizes, rev(sizes)), y = c(pccn.med + pccn.sd, rev(pccn.med - pccn.sd)), col="blue")
      points(x = sizes, y = pccn.med, t = 'l', col="blue", lwd=2)

      polygon(x = c(sizes, rev(sizes)), y = c(pccm.med + pccm.sd, rev(pccm.med - pccm.sd)), col="orange")
      points(x = sizes, y = pccm.med, t = 'l', col="orange", lwd=2)

      polygon(x = c(sizes, rev(sizes)), y = c(pccv.med + pccv.sd, rev(pccv.med - pccv.sd)), col="green")
      points(x = sizes, y = pccv.med, t = 'l', col="green", lwd=2)

      legend("topleft", c("R cor(5% missing)", "Naive PCC", "Matrix PCC", "Vector PCC"), fill = c("lightblue", "blue", "orange", "green"))
    dev.off()
}

