#
# Analysis and creation of figures for the applicationnote
#

library(MPCC)

setwd("/home/danny/MPCCtime")

MPCCinfo()

if(file.exists("timings.txt")){
  results <- read.table("timings.txt", sep = "\t", header=TRUE, colClasses = "numeric")
}else{
  results <- c()
}

for(p in seq(250, 4000, 250)){
  for(m in seq(250, 4000, 250)){
    for(n in seq(250, 10000, 250)){
      for(nM in seq(0.0, 0.5, 0.1)){
        ii <- which(results[,"p"] == p & results[,"m"] == m & results[,"n"] == n & results[,"missing"] == as.numeric(round(nM,1)))
        if(length(ii) > 0){
          cat("Skip", m, " ", n, " ", p, nM, "\n")
          next;
        }
        time.ref <- c()
        time.naive <- c()
        time.matrix <- c()

        nreps <- 5
        if(m > 1500) nreps <- 2  # Reduce the number of reps for large matrices
        if(p > 1500) nreps <- 2  # Reduce the number of reps for large matrices
        
        for (x in 1:nreps) {
          matrices <- genAB(p, n, m)

          s.ref <- proc.time()[3]
          res.ref <- cor(matrices$A, matrices$B, use = "pair")
          time.ref <- c(time.ref, (proc.time()[3] - s.ref))
      
          s.naive <- proc.time()[3]
          res.naive <- PCC.naive(matrices$A, matrices$B, use = "pair")
          time.naive <- c(time.naive, (proc.time()[3] - s.naive))
      
          s.matrix <- proc.time()[3]
          res.matrix <- PCC(matrices$A, matrices$B, use = "pair")
          time.matrix <- c(time.matrix, (proc.time()[3] - s.matrix))
        }
        results <- rbind(results, c(p, n, m, nM, round(mean(time.ref),3), round(mean(time.naive),3), round(mean(time.matrix), 3)))
        colnames(results) <- c("p", "n", "m", "missing", "cor", "naive", "matrix")
        write.table(results, "timings.txt", sep = "\t", quote=FALSE, row.names=FALSE)
      }
      cat("Done", m, " ", n, " ", p, "\n")
    }
  }
}
colnames(results) <- c("p", "n", "m", "missing", "cor", "naive", "matrix")