#ifndef __MPCC_LAPACK_H__
  #define __MPCC_LAPACK_H__
  
  enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
  enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

  void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
              const int *k, const double *restrict alpha, const double *restrict a,
              const int *lda, const double *restrict b, const int *ldb,
              const double *beta, double *restrict c, const int *ldc);
              
  void daxpy_(const int N, const double alpha, const double *X,
                   const int incX, double *Y, const int incY);

#endif
