// Register Dynamic Symbols

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "R_init.h"

void R_init_mpcc(DllInfo* info) {
    R_registerRoutines(info, NULL, NULL, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
    #ifndef NOMKL
      info("[INFO] Library compiled with Intel MKL support: %d\n", 1);
    #else
      info("[INFO] Library compiled with R/BLAS support: %d\n", 1);
    #endif
}

