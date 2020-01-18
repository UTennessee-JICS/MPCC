#include "interface.h"

extern "C" {

  // C compute backend information
  void R_backendinfo() {
      #ifndef NOMKL
        info("[INFO] Library compiled with Intel MKL support: %d\n", 1);
      #else
        #ifdef NOBLAS
          info("[WARN] Library compiled with threading only: %d\n", 1);
        #else
          info("[INFO] Library compiled with R/BLAS support: %d\n", 1);
        #endif
      #endif
  }

  // Wrap the matrix version into a C call
  void R_pcc_matrix(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res) {
    #ifdef NOBLAS
      pcc_naive((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
    #else
      pcc_matrix((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
    #endif
  }

  // Wrap the naive version into a C call
  void R_pcc_naive(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res) {
    pcc_naive((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
  }
}

