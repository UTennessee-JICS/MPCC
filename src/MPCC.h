/******************************************************************//**
 * \file MPCC.h
 * \brief Global definitions and includes
 *
 **********************************************************************/
#ifndef __MPCC_H__
  #define __MPCC_H__

  #define BILLION  1000000000L 

  #ifdef MKL // MKL build is requested, import MKL
      #include <mkl.h>
      #include <omp.h>
  #else
    #define NOMKL 1
    #include <math.h>
    #ifndef CUBLAS
       #include <cblas.h>
    #endif
  #endif

  //#ifdef CUBLAS
  //    #include <math.h>
  //    //#include <cublas.h>
  //#endif

  #ifdef STANDALONE // Completely standalone (TODO: Implement LIB)
    // #error "Completely standalone (TODO: export as R-bound DYNLIB)"
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <cfloat>
    #include <cmath>
    #include <iostream>
    #include <fstream>
    #include <time.h>
    #include <assert.h>

    #define info(format, ...) { \
        printf(format, __VA_ARGS__); \
        fflush(stdout); }
    #define err(format, ...) { \
        printf(format, __VA_ARGS__); \
        exit(-1); }
  #else //Define for R
    #define DOUBLE 1
    #include <R.h>
    #ifdef NOMKL // Compiling for R not using the Intel MKL so use our lapack.h
      #include "lapack.h"
      #include <R_ext/Lapack.h>
    #endif
    #include <Rmath.h>
    #include <string>

    #define info(format, ...) { \
        Rprintf(format, __VA_ARGS__);}
    #define err(format, ...) { \
        REprintf(format, __VA_ARGS__);}
  #endif

  #ifndef CUBLAS
       #define TRANS CblasTrans
       #define NOTRANS CblasNoTrans
       #define LEAD_PARAM CblasRowMajor
  #else
      #define TRANS CUBLAS_OP_T
      #define NOTRANS CUBLAS_OP_N //also transpose since cublas gemm is column major
  #endif


  #if DOUBLE
      #define DataType double
      #ifdef NOMKL // Use our MKL substitution functions and wrapper
        #ifndef CUBLAS
          #define VSQR vSqr
          #define VMUL vMul
          #define VSQRT vSqrt
          #define VDIV vDiv
          #define GEMM dgemm_wrap
          #define AXPY daxpy_wrap
        #else
          #define VSQR vSqr
          #define VMUL vMul
          #define VSQRT vSqrt
          #define VDIV vDiv
          #define GEMM cublasDgemm
          #define AXPY cublasDaxpy
        #endif
      #else // Use MKL functions
        #define VSQR vdSqr
        #define VMUL vdMul
        #define VSQRT vdSqrt
        #define VDIV vdDiv
        #define GEMM cblas_dgemm
        #define AXPY cblas_daxpy
      #endif
  #else
      #define DataType float
      #ifdef NOMKL // Use our MKL substitution functions
        #ifndef CUBLAS
          #define VSQR vSqr
          #define VMUL vMul
          #define VSQRT vSqrt
          #define VDIV vDiv
          #define GEMM cblas_sgemm
          #define AXPY cblas_saxpy
        #else
          #define VSQR vSqr
          #define VMUL vMul
          #define VSQRT vSqrt
          #define VDIV vDiv
          #define GEMM cublasSgemm
          #define AXPY cublasSaxpy
        #endif
        // #error "R does not provide float versions of GEMM and AXPY"
      #else // Use MKL functions
        #define VSQR vsSqr
        #define VMUL vsMul
        #define VSQRT vsSqrt
        #define VDIV  vsDiv
        #define GEMM cblas_sgemm
        #define AXPY cblas_saxpy
      #endif
  #endif

  // Defines for the NOMKL / MKL allocators
  #ifdef NOMKL
      #define FREE free
      #define ALLOCATOR(n, type, align) calloc(n, type)
  #else
      #define FREE mkl_free
      #define ALLOCATOR(n, type, align) mkl_calloc(n, type, align)
  #endif
  
  #ifndef CUBLAS  
    #define CHECKNA std::isnan
    #define MISSING_MARKER std::nan("1")
    #define NANF NAN //ceb
  #else //cuda versions
    #define CHECKNA isnan
    #define MISSING_MARKER nanf("1")
    #define NANF NAN //ceb
  #endif
  // Forward declaration of the functions
  void vSqr (int l, DataType* in, DataType* out);
  void vMul (int l, DataType* in1, DataType* in2, DataType* out);
  void vSqrt (int l, DataType* in, DataType* out);
  void vDiv (int l, DataType* in1, DataType* in2, DataType* out);
  void vMulSameAB(int m, int p, DataType* in1, DataType* in2, DataType* out);

  int pcc_matrix(int m, int n, int p, DataType* A, DataType* B, DataType* P);
  int pcc_vector(int m, int n, int p, DataType* A, DataType* B, DataType* P);
  int pcc_naive(int m, int n, int p, DataType* A, DataType* B, DataType* P);

#endif //__MPCC_H__
