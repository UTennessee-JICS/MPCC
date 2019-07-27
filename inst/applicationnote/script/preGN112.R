setwd("/home/danny/appnote")

bxd.gn112 <- read.csv("http://datafiles.genenetwork.org/download/GN112/GN112_MeanDataAnnotated_rev081815.txt", sep = "\t",
                       skip = 33, header = TRUE, colClasses = "character", stringsAsFactors = FALSE)

badProbes <- which(is.na(as.numeric(bxd.gn112[, "Mb"])) | bxd.gn112[, "Mb"] == "0.0" | bxd.gn112[, "Mb"] == "1.0")
bxd.gn112 <- bxd.gn112[-badProbes,]

norder <- sort(as.numeric(bxd.gn112[, "Mb"]), index.return=TRUE)$ix
bxd.gn112 <- bxd.gn112[norder,]

chrs <- c(1:19, "X", "Y", "M")
bxd.gn112.ordered <- c()
for(chr in chrs){
  onchr <- which(bxd.gn112[, "Chr"] == chr)
  bxd.gn112.ordered <- rbind(bxd.gn112.ordered, bxd.gn112[onchr,])
}
bxd.gn112 <- bxd.gn112.ordered
write.table(bxd.gn112, "bxd.gn112.txt", sep="\t", quote=FALSE)
