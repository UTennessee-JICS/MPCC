# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# PCC matrix c wrapper
PCC <- function(aM, bM, nthreads=1, use = NULL) {
  res <- .C("R_pcc_matrix", aM = as.double(aM),
                            bM = as.double(bM),
                            n = as.integer(nrow(aM)), # nInd
                            m = as.integer(ncol(aM)), # nPhe A
                            p = as.integer(ncol(bM)), # nPhe B
                            res = as.double(rep(0, ncol(aM) * ncol(bM))), 
                            nthreads = as.integer(nthreads),
                            NAOK = TRUE, package = "MPCC")
  res$res <- matrix(res$res, ncol(aM), ncol(bM), byrow=TRUE)
  return(res)
}

# PCC naive c wrapper
PCC.naive <- function(aM, bM, nthreads=1, use = NULL) {
  res <- .C("R_pcc_naive", aM = as.double(aM),
                           bM = as.double(bM),
                           n = as.integer(nrow(aM)), # nInd
                           m = as.integer(ncol(aM)), # nPhe A
                           p = as.integer(ncol(bM)), # nPhe B
                           res = as.double(rep(0, ncol(aM) * ncol(bM))), 
                           nthreads = as.integer(nthreads),
                           NAOK = TRUE, package = "MPCC")
  res$res <- matrix(res$res, ncol(aM), ncol(bM), byrow=TRUE)
  return(res)
}

# PCC vector c wrapper
PCC.vector <- function(aM, bM, nthreads=1, use = NULL) {
  res <- .C("R_pcc_vector", aM = as.double(aM),
                            bM = as.double(bM),
                            n = as.integer(nrow(aM)), # nInd
                            m = as.integer(ncol(aM)), # nPhe A
                            p = as.integer(ncol(bM)), # nPhe B
                            res = as.double(rep(0, ncol(aM) * ncol(bM))), 
                            nthreads = as.integer(nthreads),
                            NAOK = TRUE, package = "MPCC")
  res$res <- matrix(res$res, ncol(aM), ncol(bM), byrow=TRUE)
  return(res)
}

