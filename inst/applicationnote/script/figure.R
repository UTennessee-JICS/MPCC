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

mypalette <- brewer.pal(9, "RdBu")
cols <- colorRampPalette(mypalette)(200)
image(x = as.numeric(bxd.map[onChr1, "Mb"]), y = as.numeric(bxd.map[onChr1, "Mb"]), z = res.matrix[onChr1,onChr1], 
      breaks = seq(-1,1,0.01), col=cols, xaxt='n', yaxt='n', xlab="Position (Mb)", ylab="Position (Mb)", main="BXD genotype correlation (Chr 1)")
box()
axis(1, at = seq(0, 195, 25), round(seq(0, 195, 25), 0))
axis(2, at = seq(0, 195, 25), round(seq(0, 195, 25), 0), las=2)
legend("bottomright", legend = format(round(seq(1, -1, -0.25),2), nsmall = 2), title = "Pearson's ρ", fill = rev(colorRampPalette(mypalette)(length(seq(-1, 1, 0.25)))), bg = "white", border = "white")

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





