setwd("D:/Ddrive/Github/MPCC/inst/applicationnote/data")
mdata <- read.csv("MPCC_SpeedUp.txt", header = TRUE,sep="\t", skip = 2)

plot(c(0, max(mdata[, "n"])), c(0, 80), t = 'n', xlab = 'N', ylab = "Speedup", xaxs = "i", yaxs = "i", main = "Speedup: MPCC() versus cor(), A = 250 x N")

maxm <- max(unique(mdata[,"m"]))

angle = 25
density = 100
i <- 1
for(m in seq(250, 2500, 500)){
  min.openblas <- c()
  sd.openblas <- c()
  min.mkl <- c()
  max.openblas <- c()
  max.mkl <- c()

  inM <- mdata[which(mdata[, "m"] == m),]
  for(n in unique(inM[, "n"])){
     inMN <- inM[which(inM[, "n"] == n),]
     min.openblas <- c(min.openblas, min(inMN[, "matrix.1"]))
     sd.openblas <- c(sd.openblas, sd(inMN[, "matrix.1"]))
     max.openblas <- c(max.openblas, max(inMN[, "matrix.1"]))
     min.mkl <- c(min.mkl, min(inMN[, "matrix.3"]))
     max.mkl <- c(max.mkl, max(inMN[, "matrix.3"]))
  }
  
  mean.openblas <- (min.openblas + max.openblas) / 2.0
  mean.mkl <- (min.mkl + max.mkl) / 2.0
  
  x <- c(unique(inM[, "n"]), rev(unique(inM[, "n"])))
  #polygon(x, c(max.openblas, rev(min.openblas)),border = "blue", col=rgb(0,0,1,0.5), density = density, angle = angle)
  points(unique(inM[, "n"]), smooth(mean.openblas), col=rgb(0,0,1,1), t = 'l', lwd=i)
  for(x in 1:length(min.openblas)){
    points(c(unique(inM[, "n"])[x], unique(inM[, "n"])[x]), c(smooth(mean.openblas)[x] - sd.openblas[x], smooth(mean.openblas)[x] + sd.openblas[x]), col=rgb(0,0,1,1), t = 'l', lwd=1, lty=3)
  }
  #points(unique(inM[, "n"]), mean.mkl, col=rgb(1,0,0,0.5), t = 'l', lwd=i)
  #polygon(x, c(max.mkl, rev(min.mkl)),border = "red", col=rgb(1,0,0,0.5), density = density, angle = angle)
  density <- density - 10
  angle <- angle + 45
  i <- i + 0.5
}
legend("topleft", paste0("B = ", seq(250, 2500, 500), " x N"), lwd=seq(1,4,0.5))

#legend("topleft", c("openBLAS", "MKL"), fill = c(rgb(0,0,1,0.5), rgb(1,0,0,0.5)))





for(m in unique(mdata[,"m"])){
  for(na in seq(0.0,0.5,0.1)){
    med.openblas <- c(med.openblas, median(mdata[ii, "matrix.3"]))
    med.mkl <- c(med.mkl, median(mdata[ii, "matrix.5"]))
  }
    ii <- which(mdata[, "m"] == m & mdata[, "p"] == p & mdata[, "missing"] == na)
    points(mdata[ii, "n"], mdata[ii, "matrix.3"], t = 'l', col = rgb(0, 0, 1, 0.2), lty=1)
    points(mdata[ii, "n"], mdata[ii, "matrix.5"], t = 'l', col = rgb(1, 0, 0, 0.2), lty=1)
    
    points(mdata[ii, "n"], med.openblas, t = 'l', col = rgb(0, 0, 1, 0.9), lty=1, lwd=2)
    points(mdata[ii, "n"], med.mkl, t = 'l', col = rgb(1, 0, 0, 0.9), lty=1, lwd=2)
  }
}


plot(c(0, max(mdata[, "n"])), c(0, 40), t = 'n', xlab = 'n', ylab = "Speedup", xaxs = "i", yaxs = "i")


plot(c(0, max(mdata[, "m"]) * max(mdata[, "n"]) * max(mdata[, "p"])), c(0, 40), t = 'n', xlab = 'n', ylab = "Speedup")

for(na in c(0)){
  ii <- which(mdata[, "missing"] == na)
    
  points(mdata[ii, "m"] * mdata[ii, "n"] * mdata[ii, "p"], mdata[ii, "matrix.3"], t = 'l', col = rgb(0,1 - na,0), lty=(m / 250))
  points(mdata[ii, "m"] * mdata[ii, "n"] * mdata[ii, "p"], mdata[ii, "naive.3"], t = 'l', col = rgb(0,0,1 - na), lty=(m / 250))
  points(mdata[ii, "m"] * mdata[ii, "n"] * mdata[ii, "p"], mdata[ii, "matrix.5"], t = 'l', col = rgb(1 - na,0,0), lty=(m / 250))
}