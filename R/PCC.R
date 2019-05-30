#
# MPCC call function to R
#

PCC <- function(aM, bM) {
  res <- .C("R_pcc_matrix", aM = as.double(aM),
                            bM = as.double(bM),
                            n = as.integer(ncol(aM)), # nInd
                            m = as.integer(nrow(aM)), # nPhe A
                            p = as.integer(nrow(bM)), # nPhe B
                            res = as.double(rep(0, nrow(aM) * nrow(bM))), package = "MPCC")
  return(res)
}

test <- function(){
    library(MPCC)
    set.seed(1)
    times.ref <- c()
    times.pcc <- c()
    sizes <- c(100, 150, 200, 250, 300, 400, 500, 1000)
    for(x in sizes) {
      time.ref <- 0
      time.pcc <- 0
      for(y in 1:10) {
        mA <- matrix(runif(x*x),x,x) 
        mB <- matrix(runif(x*x),x,x) 
        s.ref <- proc.time()[3]                             # Start time
        cor(mA, mB)
        time.ref <- time.ref + (proc.time()[3] - s.ref)     # Add time information
        s.pcc <- proc.time()[3]                             # Start time
        PCC(mA, mB)
        time.pcc <- time.pcc + (proc.time()[3] - s.pcc)     # Add time information
      }
      times.ref <- c(times.ref, time.ref)
      times.pcc <- c(times.pcc, time.pcc)
      cat("Done", x, "\n")
    }
    mY <- max(c(times.ref, times.pcc))
    plot(c(min(sizes), max(sizes)), c(0, mY), t = 'n')
    points(sizes, times.ref, t = 'l', col="orange")
    points(sizes, times.pcc, t = 'l', col="green")
}

