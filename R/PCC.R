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
    mA <- matrix(runif(100),10,10) 
    mB <- matrix(runif(100),10,10) 
    PCC(mA, mB)
}

