\name{PCC}
\alias{PCC}
\alias{PCC.naive}
\title{PCC - Matrix pearson correlation }
\description{
  Fast missing data agnostic pearson correlation computation on large matrices.
}
\usage{
PCC(aM, bM, use = NULL)
PCC.naive(aM, bM, use = NULL, debugOn = FALSE)
}
\arguments{
  \item{aM}{ Matrix aM, of size (n x m) }
  \item{bM}{ Matrix bM, of size (n x p), if bM is set to null column-wise auto correlation of the aM matrix is computed. }
  \item{use}{ The use parameter is ignored by the mpcc algorithm, it is provided for backwards compatibility with the cor() function }
  \item{debugOn}{ Used for debugging the C-code }
}
\value{
  Returns a matrix of correlations between matrix aM and bM, or when bM is null a matrix of column-wise autocorrelation of aM.
}
\details{
  When the package is compiled with MKL support, the computation will use the optimized
  version of the MPCC correlation algorithm which is missing data agnostic. When 
  compiled without MPCC a warning is issued to inform the user. When MKL is not available 
  the algorithm adds multi-core support to the standard cor function. Missing data is 
  handled comparable to the "pairwise.complete.obs" methodology of the cor() function.
}
\examples{
  require(MPCC)
  rmatrices <- genAB()
  result <- PCC(rmatrices$A, rmatrices$B)
}
\seealso{
  TODO
}
\references{
  TODO
}
\author{ 
  Danny Arends \email{Danny.Arends@gmail.com}\cr
  Maintainer: Danny Arends \email{Danny.Arends@gmail.com} 
}
\keyword{methods}
