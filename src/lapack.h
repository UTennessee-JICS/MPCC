#ifndef MPCC_LAPACK_H
  #define MPCC_LAPACK_H
  #include <stddef.h>

  // This file binds the external LAPACK/BLAS functionality we need by calling the Rstubs
  enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
  enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

#endif
