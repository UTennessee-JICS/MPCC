#include "MPCC.h"

//This function is an implementation of a pairwise vector * vector correlation.
//A is matrix of X vectors and B is transposed matrix of Y vectors:
// C = [N sum(AB) - (sumA)(sumB)] /
//     sqrt[ (N sumA^2 - (sum A)^2)[ (N sumB^2 - (sum B)^2) ]
//int pcc_naive(int m, int n, int p, int count,
int pcc_naive(int m, int n, int p,
	      DataType* A, DataType* B, DataType* C)
{
  DataType sab,sa,sb,saa,sbb;
  int nn;
  int i,j,k;
  int count=1;

  //sum_i( x[i]-x_mean[i])*(y[i]-y_mean[i]) ) /
  //     [ sqrt( sum_i(x[i]-x_mean[i])^2 ) sqrt(sum_i(y[i]-y_mean[i])^2 ) ]
  for (int ii=0; ii<count; ii++) {
    //Disabled pragma, because of weird sorting order errors #pragma omp parallel for private (i,j,k)
    for (i=0; i<m; i++) {
      for (j=0; j<p; j++) {

        sa=0.0;
        sb=0.0;
        saa=0.0;
        sbb=0.0;
        sab=0.0;
        nn=n;

        for (k=0; k<n; k++) {
          //if missing data exists decrement divisor for mean calculation
          if (std::isnan(A[i*n+k]) || std::isnan(B[j*n+k])){
             nn--;
          }
          else{
             //compute components of PCC function
             sa  += A[i*n+k];
             sb  += B[j*n+k];
             sab += A[i*n+k] * B[j*n+k];
             saa += A[i*n+k] * A[i*n+k];
             sbb += B[j*n+k] * B[j*n+k];
          }
        }
          
        if(nn>1){//Note edge case: if nn==1 then denominator is Zero! (saa==sa*sa, sbb==sb*sb)
          //C[i*p+j] = (nn*sab - sa*sb) / sqrt( (nn*saa - sa*sa)*(nn*sbb - sb*sb) );
          C[i*p+j] = (sab - sa*sb/nn) / sqrt( (saa - sa*sa/nn)*(sbb - sb*sb/nn) );
          if( sqrt( (saa - sa*sa/nn)*(sbb - sb*sb/nn) ) ==0.0){printf("Error: R[%d,%d] denominator is zero! sa[%d]=%e sb[%d]=%e \n",i,j,i,sa,j,sb);}
        }
        else{/*printf("Error, no correlation possible for rows A[%d], B[%d]\n",i,j);*/ C[i*p+j]=0.0;}
        //else{/*printf("Error, no correlation possible for rows A[%d], B[%d]\n",i,j);*/ C[i*p+j]=NANF;}
      }
    }
  }
  return 0;
}

