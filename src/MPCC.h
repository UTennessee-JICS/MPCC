/******************************************************************//**
 * \file MPCC.h
 * \brief Global definitions and includes
 *
 **********************************************************************/
#ifndef __MPCC_H__
  #define __MPCC_H__

    #define BILLION  1000000000L

    #ifdef MKL // Disable the mkl as needed
      #include <mkl.h>
    #else
      #define NOMKL 1
      #include <math.h>
    #endif

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
        
      #define CHECKNA std::isnan
        
    #else
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
        error(format, __VA_ARGS__);}
      
      #define CHECKNA std::isnan
    #endif

  #if DOUBLE
    #define DataType double
    #ifdef NOMKL // Use our MKL substitution functions
      #define VSQR vSqr
      #define VMUL vMul
      #define VSQRT vSqrt
      #define VDIV vDiv
      #define GEMM dgemm_wrap
      #define AXPY daxpy_wrap
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
      #define VSQR vSqr
      #define VMUL vMul
      #define VSQRT vSqrt
      #define VDIV vDiv    
      // R does not provide float versions of GEMM and AXPY
      //#define GEMM cblas_sgemm
      //#define AXPY cblas_saxpy
    #else // Use MKL functions
      #define VSQR vsSqr
      #define VMUL vsMul
      #define VSQRT vsSqrt
      #define VDIV  vsDiv
      #define GEMM cblas_sgemm
      #define AXPY cblas_saxpy
    #endif
  #endif

// Defines for the MKL/ non-MKL allocators
#ifdef NOMKL
  #define FREE free
  #define ALLOCATOR(x,y,z) calloc(x,y)
#else
  #define FREE mkl_free
  #define ALLOCATOR(x,y,z) mkl_calloc(x,y,z)
#endif
  
#ifdef __MINGW32__
    #define NANF nan("1")
#else
    #if __STDC_VERSION__ == 201112L
      #define NANF std::nan("1")
    #else
      #define NANF nan("1")
    #endif
#endif
    
#define MISSING_MARKER NANF

    // Forward declaration of the functions
    int pcc_matrix(int m, int n, int p, DataType* A, DataType* B, DataType* P);
    int pcc_vector(int m, int n, int p, DataType* A, DataType* B, DataType* P);
    int pcc_naive(int m, int n, int p, DataType* A, DataType* B, DataType* P);

#endif //__MPCC_H__

