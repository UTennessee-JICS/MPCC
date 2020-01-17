#ifndef MPCC_LAPACK_H
#define MPCC_LAPACK_H
#include <stddef.h>

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

void dgemm_(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
void daxpy_(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY);
#endif
