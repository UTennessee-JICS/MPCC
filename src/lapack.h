#ifndef MPCC_LAPACK_H
  #define MPCC_LAPACK_H
  #include <stddef.h>

  // This file binds the external LAPACK/BLAS functionality we need by calling the Rstubs
  enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
  enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

  void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
              const int *k, const double *__restrict alpha, const double *__restrict a,
              const int *lda, const double *__restrict b, const int *ldb,
              const double *beta, double *__restrict c, const int *ldc);
  void daxpy_(const int *n, const double *da, const double *dx, const int *incx,
              double *dy, const int *incy);
#endif
