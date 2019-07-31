#include "MPCC.h"
#include <omp.h>
#include <unistd.h>
//This function is an implementation of a pairwise vector * vector correlation.
//A is matrix of X vectors and B is transposed matrix of Y vectors:
// C = [N sum(AB) - (sumA)(sumB)] /
//     sqrt[ (N sumA^2 - (sum A)^2)[ (N sumB^2 - (sum B)^2) ]
int pcc_naive(int m, int n, int p, DataType* A, DataType* B, DataType* C, int nthreads=1)
{
  DataType sab,sa,sb,saa,sbb;
  int nn;
  int i,j,k;
  //int count=1;
  //int tid;

  omp_set_num_threads(nthreads);
  mkl_set_num_threads(nthreads);

  //sum_i( x[i]-x_mean[i])*(y[i]-y_mean[i]) ) /
  //     [ sqrt( sum_i(x[i]-x_mean[i])^2 ) sqrt(sum_i(y[i]-y_mean[i])^2 ) ]
  double wtime = omp_get_wtime();
  #pragma omp parallel default(shared) private(i,j,k,sa,sb,saa,sbb,sab,nn)
  {
    #pragma omp for schedule(static) nowait
    for (i=0; i<m; ++i) {
      //printf("tid=%d, m=%d i=%d\n",omp_get_thread_num(),m,i);
      for (j=0; j<p; ++j) {
        sa=0.0;
        sb=0.0;
        saa=0.0;
        sbb=0.0;
        sab=0.0;
        nn=n;
        //int mm=0;
        for (k=0; k<n; ++k) {
          //if missing data exists decrement divisor for mean calculation
          if (std::isnan(A[i*n+k]) || std::isnan(B[j*n+k])){
             nn--;
          }
          else
          {
             //compute components of PCC function
             sa  += A[i*n+k];
             sb  += B[j*n+k];
             sab += A[i*n+k] * B[j*n+k];
             saa += A[i*n+k] * A[i*n+k];
             sbb += B[j*n+k] * B[j*n+k];
          }
        }

        if(nn>1){//Note edge case: if nn==1 then denominator is Zero! (saa==sa*sa, sbb==sb*sb)
          C[i*p+j] = (nn*sab - sa*sb) / sqrt( (nn*saa - sa*sa)*(nn*sbb - sb*sb) );
        }
        else{/*printf("Error, no correlation possible for rows A[%d], B[%d]\n",i,j);*/ C[i*p+j]=0.0;}
      }
    }
  }
  wtime = omp_get_wtime()-wtime;
  printf( "Naive: Time taken by thread %d is %f \n", omp_get_thread_num(), wtime);

  DataType gflops = (8/1.0e9)*m*n*p/wtime;
  FILE *f = fopen("timing/MPCC_naive_timing_matsize.txt","a");
  fprintf(f, "%d  %f %e %d %lu \n", m, wtime, gflops, nthreads, sizeof(DataType));
  fclose(f);
  return 0;
}

