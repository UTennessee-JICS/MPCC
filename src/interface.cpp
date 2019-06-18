#include "interface.h"

extern "C" {

  // Wrap the matrix version into a C call
  void R_pcc_matrix(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res) {
    #ifndef NOMKL
    pcc_matrix((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
    #else
    info("[WARNING] Library compiled with NO Intel MKL support: %d\n", 0);
    pcc_naive((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
    #endif
  }

  // Wrap the naive version into a C call
  void R_pcc_naive(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res) {
    pcc_naive((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
  }
}

