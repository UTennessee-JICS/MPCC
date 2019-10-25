#https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-27/files/raw/
# Get all 24 files from:
# wget https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-27/E-MTAB-27.raw.X.zip

source("https://bioconductor.org/biocLite.R")
biocLite("affy")
biocLite("mgu74av2.db")
biocLite("mgu74av2probe")
biocLite("affyPLM")

library("affy")
library("mgu74av2.db")
library("mgu74av2probe")
library("affyPLM")

setwd("~/appnote/cel")
mdata <- ReadAffy()
eset <- mas5(mdata)

esetQnorm <- normalize.ExpressionSet.quantiles(eset, transfn = "log")

boxplot(log2(assayData(esetQnorm)$exprs[,1:10]))

write.table(assayData(eset)$exprs, "expr_E-MTAB-27_mas5.txt", sep = "\t", quote = FALSE)
write.table(log2(assayData(esetQnorm)$exprs), "expr_E-MTAB-27_mas5_norm.txt", sep = "\t", quote = FALSE)

annot <- AnnotationDbi::select(
  x       = mgu74av2.db,
  keys    = rownames(eset),
  columns = c("PROBEID", "ENSEMBL", "ENTREZID", "SYMBOL", "GENENAME", "MGI"),
  keytype = "PROBEID"
)

write.table(annot, "mgu74av2_annotation.txt", sep = "\t", quote = FALSE)


expressions <- log2(assayData(esetQnorm)$exprs)

library(MPCC)

s.matrix <- proc.time()[3]
aSamples <- PCC(expressions, expressions)
proc.time()[3] - s.matrix

s.matrix <- proc.time()[3]
aProbes <- PCC(t(expressions), t(expressions))
proc.time()[3] - s.matrix

s.ref <- proc.time()[3]
aProbesRef <- cor(t(expressions), t(expressions))
proc.time()[3] - s.ref

