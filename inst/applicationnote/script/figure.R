#
# Analysis and creation of figures for the applicationnote
#

library(MPCC)
library(RColorBrewer)

setwd("/home/danny/appnote")

#
# Scenario 1) Genotype correlation
#

bxd.geno <- read.table("http://www.genenetwork.org/genotypes/BXD.geno", sep = "\t",
                       skip = 21, header = TRUE, row.names = 2, colClasses = "character", stringsAsFactors = FALSE)
bxd.map <- bxd.geno[, 1:3]
bxd.geno <- bxd.geno[, -c(1:3)]

bxd.num <- apply(bxd.geno, 1, function(x){
  as.numeric(factor(x, levels = c("B", "D"), exclude = c("H", "U")))
})
rownames(bxd.num) <- colnames(bxd.geno)

time.ref <- c()
time.naive <- c()
time.matrix <- c()
for(x in 1:1){
  s.ref <- proc.time()[3]
  res.ref <- cor(bxd.num, use = "pair")
  time.ref <- c(time.ref, (proc.time()[3] - s.ref))

  cat("ref\n")
  
  s.naive <- proc.time()[3]
  res.naive <- PCC.naive(bxd.num, bxd.num, use = "pair")
  time.naive <- c(time.naive, (proc.time()[3] - s.naive))

  cat("naive\n")
  
  s.matrix <- proc.time()[3]
  res.matrix <- PCC(bxd.num, bxd.num, use = "pair")
  time.matrix <- c(time.matrix, (proc.time()[3] - s.matrix))
  
  cat("matrix\n")
}

mean(time.ref)
mean(time.naive)
mean(time.matrix)

mean(time.ref) / mean(time.matrix)
mean(time.ref) / mean(time.naive)

onChr1 <- rownames(bxd.map[which(bxd.map[,"Chr"] == 1),])
op <- par(cex=0.85)

mypalette <- brewer.pal(9, "RdBu")
cols <- colorRampPalette(mypalette)(200)
image(x = as.numeric(bxd.map[onChr1, "Mb"]), y = as.numeric(bxd.map[onChr1, "Mb"]), z = res.matrix[onChr1,onChr1], 
      breaks = seq(-1,1,0.01), col=cols, xaxt='n', yaxt='n', xlab="Position (Mb)", ylab="Position (Mb)", main="BXD genotype correlation (Chromosome 1)")
box()
axis(1, at = seq(0, 195, 25), round(seq(0, 195, 25), 0))
axis(2, at = seq(0, 195, 25), round(seq(0, 195, 25), 0), las=2)
legend("bottomright", legend = format(round(seq(1, -1, -0.5),2), nsmall = 2), fill = rev(colorRampPalette(mypalette)(length(seq(-1, 1, 0.5)))), bg = "white", border = "white")

setwd("D:/Ddrive/Github/MPCC/inst/applicationnote/script")
speedupData <- read.table("comparison_matrix.csv", sep="\t", header=TRUE)

unique(speedupData[, "mat_dim"])

#Speedup <- c(0, 25, 65, 75, 85, 95)
Dimension <- unique(speedupData[, "mat_dim"])
plot(c(0,9000), c(0, 120), t = 'n', xlab= "Matrix Dimensions (n^2)", ylab= "Speedup", main="Runtime speedup of MPCC versus cor()", xaxt='n', yaxt='n', xaxs='i', yaxs='i')
axis(1, at = seq(0, 9000, 1000), seq(0, 9000, 1000), 0)
axis(2, at = seq(0, 120, 20), seq(0, 120, 20), las=2)
for(h in seq(20, 160, 20)){ abline(h = h, col="gray"); }
for(v in seq(1000, 9000, 1000)){ abline(v = v, col="gray"); }
cnt <- 1
for(x in c(0.05, 0.1, 0.25, 0.5)){
  subspeed <- speedupData[speedupData[, "pct_miss_data"] == x,]
  Speedup <- subspeed[, "timeRCOR"] / subspeed[, "MPCC_nthreads.40_only_A_missing"]
  Speedup[Speedup == Inf] <- 1
  points(x = Dimension, y = Speedup, t = 'b', pch=19, col=cnt)
  Speedup <- subspeed[, "timeRCOR"] / subspeed[, "timeMPCC"]
  Speedup[Speedup == Inf] <- 1
  points(x = Dimension, y = Speedup, t = 'b', lty=2, pch=15, col=cnt)
  cnt = cnt + 1
}
box()
legend("topleft", legend = c("5%","10%","25%","50%", "threads: 1", "threads: 40"), fill=c(1:4, NA, NA), lty=c(NA,NA,NA,NA,2,1), pch=c(NA,NA,NA,NA, 15, 19), bg = "white", border = "white")


#
# Gene expression co-expression data for benchmarking speedup
# We increase the size of the data set by including data chromosome per chromosome
#
library(MPCC)
library(BXDtools)
library(RColorBrewer)

setwd("/home/danny/appnote")

data('bxd.phenotypes', package='BXDtools', envir=environment())

bxd.gn112 <- read.csv("bxd.gn112.txt", sep="\t")

data.s <- which(colnames(bxd.gn112) == "B6D2F1")
data.e <- which(colnames(bxd.gn112) == "D2B6F1")

gn112.map <- bxd.gn112[, 1:(data.s-1)]
gn112.data <- bxd.gn112[, data.s:data.e]
gn112.data <- t(apply(gn112.data, 2, function(x){ as.numeric(x) }))

gn112.matched <- which(rownames(gn112.data) %in% colnames(bxd.phenotypes))
gn112.data <- gn112.data[gn112.matched,]

bxd.matched <- which(colnames(bxd.phenotypes) %in% rownames(gn112.data))
bxd.phenotypes <- t(bxd.phenotypes[,bxd.matched])
bxd.phenotypes <- bxd.phenotypes[rownames(gn112.data), ]

bxd.phenotypes[1,1:2] <- c(0,0)

mm <- PCC(bxd.phenotypes[1:33, 1:2], gn112.data[1:33, 1:2])
cc <- cor(bxd.phenotypes[1:33, 1:2], gn112.data[1:33, 1:2], use = "pair")

chrs <- c(1:19, "X", "Y", "M")

times.ref <- c()
times.naive <- c()
times.matrix <- c()
for(chr in chrs[1:4]){
  time.ref <- c()
  time.naive <- c()
  time.matrix <- c()
  for(x in 1:5){
    todo <- which(chrs == chr)
    onchrs <- which(gn112.map[,"Chr"] %in% chrs[1:todo])
    gn112.subset <- gn112.data[,onchrs]
    cat("Chromosome: ", chr, ", nprobes: ", ncol(gn112.subset), "\n")
    
    #s.ref <- proc.time()[3]
    #res.ref <- cor(gn112.subset, use = "pair")
    #time.ref <- c(time.ref, (proc.time()[3] - s.ref))

    s.naive <- proc.time()[3]
    res.naive <- PCC.naive(gn112.subset, gn112.subset, use = "pair")
    time.naive <- c(time.naive, (proc.time()[3] - s.naive))

    s.matrix <- proc.time()[3]
    res.matrix <- PCC(gn112.subset, gn112.subset, use = "pair")
    time.matrix <- c(time.matrix, (proc.time()[3] - s.matrix))
    cat("Done\n")
  }
  times.ref <- cbind(times.ref, time.ref)
  times.naive <- cbind(times.naive, time.naive)
  times.matrix <- cbind(times.matrix, time.matrix)
}
colnames(times.ref) <- chrs
colnames(times.naive) <- chrs
colnames(times.matrix) <- chrs

speedup.naive <- apply(times.ref, 2, mean) / apply(times.naive, 2, mean)
speedup.matrix <- apply(times.ref, 2, mean) / apply(times.matrix, 2, mean)

write.table(times.ref, "ref.txt",sep="\t")
write.table(times.naive, "naive.txt",sep="\t")
write.table(times.matrix, "matrix.txt",sep="\t")


s.matrix <- proc.time()[3]
res.matrix <- PCC(gn112.data[,1:2000], gn112.data, use = "pair")
proc.time()[3] - s.matrix


s.cor <- proc.time()[3]
res.cor <- cor(gn112.data[,1:2000], gn112.data, use = "pair")
proc.time()[3] - s.cor





