/******************************************************************//**
 * \file interface.h
 * \brief Definition of the interfaces to R, and the updateR helper function
 *
 **********************************************************************/
#ifndef __INTERFACE_H__
  #define __INTERFACE_H__

  #include "MPCC.h"

  /** Function to 'update' R, checks user input and can flushes console. */
  void    updateR(bool flush);
  /** R interface to perform a CTL scan and permutations on phenotype 'phenotype' */
  extern "C" {
    void R_pcc_matrix(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res, int* thptr);
    void R_pcc_naive( double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res, int* thptr); 
    void R_pcc_vector(double* aM, double* bM, int* nptr, int* mptr, int* pptr, double* res, int* thptr); 
  }

#endif //__INTERFACE_H__

