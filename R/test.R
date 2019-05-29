#
# MPCC call function to R
#

corMxM <- function(aM, bM) {
  res <- .C("R_pcc_matrix", aM = as.double(x),
                            bM = as.double(x),
                            n = as.integer(ncol(aM)), # nInd
                            m = as.integer(nrow(aM)), # nPhe A
                            p = as.integer(nrow(bM)), # nPhe B
                            res = as.double(rep(0, nrow(aM) * nrow(bM))), PACKAGE="MPCC")
  return(res)
}

