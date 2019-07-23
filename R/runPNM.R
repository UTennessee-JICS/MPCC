# copyright (c) - HU-Berlin / UTHSC / JICS by Danny Arends

# run the PNM timing
runPNM <- function(fun = cor, p = 100, n = 150, m = 20, missing = 0, use = "everything") {
  mAB <- genAB(p, n, m, missing)

  start <- proc.time()[3] # Start time
  res <- fun(mAB[["A"]], mAB[["B"]], use = use)
  elapsed <- proc.time()[3] - start # Add time information

  return(list(res = res, time = elapsed))
}

# run the PNM timing
runAB <- function(fun = cor, mAB, use = "everything") {
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
