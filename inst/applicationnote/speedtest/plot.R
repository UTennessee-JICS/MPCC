setwd("D:/Ddrive/Github/MPCC/inst/applicationnote/speedtest")
mdata <- read.csv("MPCC_SpeedUp.txt", header = TRUE,sep="\t", skip = 2)

postscript("whatever.eps", width = 8, height = 6, paper = "special")

op <- par(mar = c(5, 4.5, 0.5, 2) + 0.1)

plot(c(0, 10050), c(0, 220), t = 'n', xlab = 'shared dimension (n)', ylab = expression('Speedup '[(latency)]), xaxs = "i", yaxs = "i", las=2, xaxt='n')
axis(1, at = c(250,seq(1000,10000, 1000)), c(250,seq(1000,10000, 1000)))

maxm <- max(unique(mdata[,"m"]))

angle = 25
density = 100
i <- 1
for(m in c(250, 500, 1000, 1500, 2500)){
  min.openblas <- c()
  sd.openblas <- c()
  max.openblas <- c()

  inM <- mdata[which(mdata[, "m"] == m),]
  for(n in unique(inM[, "n"])){
     inMN <- inM[which(inM[, "n"] == n),]
     min.openblas <- c(min.openblas, min(inMN[, "matrix.1"]))
     sd.openblas <- c(sd.openblas, sd(inMN[, "matrix.1"]))
     max.openblas <- c(max.openblas, max(inMN[, "matrix.1"]))
  }
  
  mean.openblas <- (min.openblas + max.openblas) / 2.0
  
  x <- c(unique(inM[, "n"]), rev(unique(inM[, "n"])))
  #polygon(x, c(max.openblas, rev(min.openblas)),border = "blue", col=rgb(0,0,1,0.5), density = density, angle = angle)
  for(x in 1:length(min.openblas)){
    points(c(unique(inM[, "n"])[x], unique(inM[, "n"])[x]), c(smooth(mean.openblas)[x] - sd.openblas[x], smooth(mean.openblas)[x] + sd.openblas[x]), col=rgb(i/5,i/5,i/5,1), t = 'l', lwd=1, lty=1)
  }
  if(i == 1){
    points(unique(inM[, "n"]), smooth(mean.openblas), col=rgb(0,1,0,1), t = 'l', lwd=i)
  }else{
    points(unique(inM[, "n"]), smooth(mean.openblas), col=rgb(0,0,1,1), t = 'l', lwd=i)
  }

  density <- density - 10
  angle <- angle + 45
  i <- i + 0.5
}

i <- 1.5
for(m in c(2000, 3500)){
  min.openblas <- c()
  sd.openblas <- c()
  max.openblas <- c()

  inM <- mdata[which(mdata[, "p"] == m),]
  for(n in unique(inM[, "n"])){
     inMN <- inM[which(inM[, "n"] == n),]
     min.openblas <- c(min.openblas, min(inMN[, "matrix.1"]))
     sd.openblas <- c(sd.openblas, sd(inMN[, "matrix.1"]))
     max.openblas <- c(max.openblas, max(inMN[, "matrix.1"]))
  }
  
  mean.openblas <- (min.openblas + max.openblas) / 2.0
 
  x <- c(unique(inM[, "n"]), rev(unique(inM[, "n"])))
  #polygon(x, c(max.openblas, rev(min.openblas)),border = "blue", col=rgb(0,0,1,0.5), density = density, angle = angle)
  for(x in 1:length(min.openblas)){
    points(c(unique(inM[, "n"])[x], unique(inM[, "n"])[x]), c(smooth(mean.openblas)[x] - sd.openblas[x], smooth(mean.openblas)[x] + sd.openblas[x]), col=rgb(i/3,i/3,i/3,1), t = 'l', lwd=1, lty=1)
  }
  points(unique(inM[, "n"]), smooth(mean.openblas), col=rgb(0,1,0,1), t = 'l', lwd=i)

  density <- density - 10
  angle <- angle + 45
  i <- i + 0.5
}
op <- par(cex = 0.8)
legend("topleft", c(paste0("A=", c(500, 1000, 1500, 2500), "*n, B=250*n"), "A=250*n, B=250*n", "A=2000*n, B=2000*n", "A=3500*n, B=3500*n"), lwd=c(seq(1,2.5,0.5), 1, 1.5, 2), col=c("blue", "blue", "blue", "blue", "green", "green", "green"))

dev.off()

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