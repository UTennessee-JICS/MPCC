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
      #include <Rmath.h>
      #include <string>

      #define info(format, ...) { \
        Rprintf(format, __VA_ARGS__);}
      #define err(format, ...) { \
        error(format, __VA_ARGS__);}
      
      #define CHECKNA ISNA
    #endif

  #if DOUBLE
    #define DataType double
    #define VSQR vdSqr
    #define VMUL vdMul
    #define VSQRT vdSqrt
    #define VDIV vdDiv
    #define GEMM cblas_dgemm
    #define AXPY cblas_daxpy
  #else
    #define DataType float
    #define VSQR vsSqr
    #define VMUL vsMul
    #define VSQRT vsSqrt
    #define VDIV  vsDiv
    #define GEMM cblas_sgemm
    #define AXPY cblas_saxpy
  #endif

#ifdef __MINGW32__
    #define NANF nan("1")
#else
    #define NANF std::nan("1")
#endif
    
#define MISSING_MARKER NANF

    // Forward declaration of the functions
    int pcc_matrix(int m, int n, int p, DataType* A, DataType* B, DataType* P);
    int pcc_vector(int m, int n, int p, DataType* A, DataType* B, DataType* P);
    int pcc_naive(int m, int n, int p, DataType* A, DataType* B, DataType* P);

#endif //__MPCC_H__

