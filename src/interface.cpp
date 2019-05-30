#include "interface.h"

extern "C" {

  void R_pcc_matrix(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res){
    //info("In C!!!! n = %d\n", (int)(*nptr));
    pcc_matrix((int)(*mptr), (int)(*nptr), (int)(*pptr), aM, bM, res);
  }
}

