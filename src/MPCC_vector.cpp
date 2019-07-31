//Pearsons correlation coeficient for two vectors x and y
// r = sum_i( x[i]-x_mean[i])*(y[i]-y_mean[i]) ) /
//     [ sqrt( sum_i(x[i]-x_mean[i])^2 ) sqrt(sum_i(y[i]-y_mean[i])^2 ) ]

// Matrix reformulation and algebraic simplification for improved algorithmic efficiency
// where A is matrix of X vectors and B is transposed matrix of Y vectors:
// R = [N sum(AB) - (sumA)(sumB)] /
//     sqrt[ (N sumA^2 - (sum A)^2)[ (N sumB^2 - (sum B)^2) ]

//This code computes correlation coefficient between all row/column pairs of two matrices 
// ./MPCC MatA_filename MatB_filename 

#include "MPCC.h"
using namespace std;

#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#define __assume_aligned(var,size){ __builtin_assume_aligned(var,size); }
#define DEV_CHECKPT printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); fflush(stdout); 

#ifndef NAIVE //default use matrix version
  #define NAIVE 0
#endif

#ifndef DOUBLE //default to float type
  #define DOUBLE 0
#endif

#ifndef NOMKL

//This function uses bit arithmetic to mask vectors prior to performing a number of FMA's
//The intention is to improve upon the matrix x matrix missing data PCC algorithm by reducing uneccessary computations
// and maximizing the use of vector register arithmetic.
//A is matrix of X vectors and B is transposed matrix of Y vectors:
//P = [ N*sum(AB) - (sumA)(sumB)] /
//    sqrt[ (N sumA^2 -(sum A)^2)[ ( N sumB^2 - (sum B)^2) ]
//P = (N*SAB - SA*SB)/Sqrt( (N*SAA - (SA)^2) * (N*SBB - (SB)^2)  )

int pcc_vector(int m, int n, int p,
	       DataType* A, DataType* B, DataType* P, int nthreads=1)	       
{
  int tid;
  omp_set_num_threads(nthreads);
  //printf("get_omp_max_threads()=%d\n",omp_get_max_threads());
  mkl_set_num_threads(nthreads);
  //printf("mkl_get_max_threads()=%d\n",mkl_get_max_threads());
  double wtime = omp_get_wtime();

  int i,j,k;
  int stride = ((n-1)/64 +1);
  DataType alpha=1.0;
  DataType beta=0.0;
  int count =1;

  bool transposeB = true; //assume this is always true. 
  CBLAS_TRANSPOSE transB=CblasNoTrans;
  int ldb=p;
  if(transposeB){
    transB=CblasTrans;
    ldb=n;
  }

  //we need to compute terms using FMA and masking and then assemble terms to 
  // construct the PCC value for each row column pair

  unsigned long zeros = 0UL;
  unsigned long ones = ~(0UL); //bitwise complement of 0 is all 1's (maxint)

  //We wamt to perform operations on vectors of float or double precision values. 
  //By constructing a mask in the form of a vector of floats or doubles, in which 
  // the values we want to keep are masked by a number which translates to a 111..1 (i.e. maxint for 32 or 64 bit values),
  // and the values we want to ignore are masked by a 000..0 bit string, we can simply apply a bitwise AND
  // to get a resultant vector on which we can perform an FMA operation

  //We also want to use this bitmask to compute a sum of non-missing data for each row-column pair

  //We can create a bitmask for both A and B matrices by doing the following:
  //  Initialize an Amask and Bmask matrix to all 1 bit strings. While reading A or B, 
  //  where there is a Nan bit string in A or B, place a 0 bit string in the corresponding mask matrix

  //To compute the various terms, eg, sum of A, for i=1..m, j=1..p, SA[i,j] = Sum_k(A[i] & Bmask[j]) 
  // reduce (sum) operation will have to be hand coded via openMP parallel reduce with pragma simd inside loop
  // horizontal_add returns sum of components of vector but is not efficient. Breaks SIMD ideal

  //How to create Amask and Bmask
  //In the process of creating masks for A and B we can either replace Nans with 0 as we go in A and B, or we can then apply the masks to 
  // to the matrices after we compute them.

  // We can create a matrix mask starting with a mask matrix containing all maxint bit strings except where A has missing data in which case the mask will contain a zero.
  // We can set these mask values to zero in the appropriate locations by traversing A or B and applying the isnan function in an if statement. 
  // If false, then set Amask or Bmask respectively to ones at that location (masks initialized to zeros)
  // (there may be a more efficent way to do this)

  //After computing masks and N, we want to replace Nan's in A and B with 0.
  // We can do this by masking A with Amask and B with Bmask 
  
  unsigned long* Amask = (unsigned long*)mkl_calloc( m*n, sizeof(unsigned long), 64 );
  __assume_aligned(Amask, 64);
  #pragma omp for private(i)
  for(i=0; i<m*n; ++i){
     if(std::isnan(A[i])){ A[i]=0;}
     else{Amask[i]=ones;}
  }

  unsigned long* Bmask = ( unsigned long*)mkl_calloc( n*p, sizeof(unsigned long), 64 );
  __assume_aligned(Bmask, 64);    
  #pragma omp for private(i)
  for(i=0; i<p*n; ++i){
     if(std::isnan(B[i])){ B[i]=0;}
     else{Bmask[i]=ones;}
  }

  //The masks can then be used by employing a bitwise (AND) &, between the elements of the mask and the elements of the matrix or vector we want to mask. 
  // The elements masked with a zero bit string will return a zero, the elements masked with the 1 bit string will return the original element.

  //How to compute N?
  // N[i,j] contains the number of valid pairs (no Nan's) of elements when comparing rows A[i] and B[j].
  // If we apply a bitwise (AND) & to compare rows A[i] and B[j] the result will be a bit vector of 1's for all element 
  // comparisons that don't include Nan's and zero's elsewhere. To sum up the valid element pairs, we could simply loop over the 
  // resulting vector and count up the maxints.
  //(There may be a faster way to sum values for N using bit ops)

  
  //SAB may best be done by a GEMM operation, though the symmetric portion of the computation can be reduced by 
  //  eliminating the lower triangular redundancy.
  
  DataType* SAB =   ( DataType*) mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SAB, 64);
  //SAB = A*B
  GEMM(CblasRowMajor, CblasNoTrans, transB, m, p, n, alpha, A, n, B, ldb, beta, SAB, p);

  unsigned long one = 1UL;
  //DataType tmp[n];//  = (DataType*)mkl_calloc( n, sizeof(DataType), 64 );
  //DataType tmp2[n];// = (DataType*)mkl_calloc( n, sizeof(DataType), 64 );
  //unsigned long BmA_bitstring[n];// = (unsigned long*)mkl_calloc( n, sizeof(unsigned long), 64 );
  //unsigned long AmB_bitstring[n];// = (unsigned long*)mkl_calloc( n, sizeof(unsigned long), 64 );

  //N contains the number of elements used in each row column PCC calculation (after missing values are removed)
  DataType* N =   (DataType*) mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(N, 64);
  DataType* SA =  (DataType*) mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SA, 64);
  DataType* SB =  (DataType*) mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SB, 64);
  DataType* SAA = (DataType*) mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SAA, 64);
  DataType* SBB = (DataType*) mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SBB, 64);

  unsigned long* Aul=NULL;

  unsigned long AmB_bitstring;
  unsigned long BmA_bitstring;
  DataType tmp;
  DataType tmp2;
  
#if 1
  #pragma omp parallel default(shared) private(i,j,k,BmA_bitstring,AmB_bitstring,tmp,tmp2)
  {
  #pragma omp for schedule(static) nowait
  for(i=0; i<m; ++i){
    for(j=0; j<p; ++j){

       DataType sa=0;
       DataType saa=0;
       DataType sb=0;
       DataType sbb=0;
       int nn=0;
      
       for(k=0;k<n;++k){
          if(Amask[i*n+k] & Bmask[j*n+k]) nn++;
          BmA_bitstring = Bmask[j*n+k] & *(reinterpret_cast<unsigned long*>(&(A[i*n+k])));
          tmp= *(reinterpret_cast<DataType*>(&(BmA_bitstring)));
          sa  += tmp;
          saa += tmp*tmp;
          AmB_bitstring = Amask[i*n+k] & *(reinterpret_cast<unsigned long*>(&(B[j*n+k])));
          tmp2=*(reinterpret_cast<DataType*>(&(AmB_bitstring)));
          sb  += tmp2;
          sbb += tmp2*tmp2;
       }
       N[i*p+j]=nn;
       SA[i*p+j]=sa;
       SAA[i*p+j]=saa;
       SB[i*p+j]=sb;
       SBB[i*p+j]=sbb;
    }
  }
  }
#endif
  
  //allocate and initialize and align memory needed to compute PCC

  //variables used for performance timing
  struct timespec startGEMM, stopGEMM;
  double accumGEMM;

  //Compute PCC terms and assemble
  DataType* SASB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
  DataType* NSAB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
   
  DataType* SASA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 
  DataType* NSAA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    
  DataType* SBSB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );    
  DataType* NSBB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );   
    
  DataType* DENOM = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
  DataType* DENOMSqrt =( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 

  //Compute and assemble composite terms

  //SASB=SA*SB
  VMUL(m*p,SA,SB,SASB);
  //N*SAB
  VMUL(m*p,N,SAB,NSAB); 
  //NSAB=(-1)SASB+NSAB  (numerator)
  AXPY(m*p,(DataType)(-1), SASB,1, NSAB,1); 

  //(SA)^2
  VSQR(m*p,SA,SASA);
  //N(SAA)
  VMUL(m*p,N,SAA,NSAA); //ceb
  //NSAA=NSAA-SASA (denominator term 1)
  AXPY(m*p,(DataType)(-1), SASA,1, NSAA,1);

  //(SB)^2
  VSQR(m*p,SB,SBSB);
  //N(SBB)
  VMUL(m*p,N,SBB,NSBB);
  //SBB=NSBB-SBSB
  AXPY(m*p,(DataType)(-1), SBSB,1, NSBB,1);

  //DENOM=NSAA*NSBB (element wise multiplication)
  VMUL(m*p,NSAA,NSBB,DENOM);

  #pragma omp for private(i)
  for(int i=0;i<m*p;++i){
     if(DENOM[i]==0.){DENOM[i]=1;}//numerator will be 0 so to prevent inf, set denom to 1
  }
  //sqrt(DENOM)
  VSQRT(m*p,DENOM,DENOMSqrt);

  //P=NSAB/DENOMSqrt (element wise division)
  VDIV(m*p,NSAB,DENOMSqrt,P);   

  mkl_free(SASA);
  mkl_free(SASB);
  mkl_free(SBSB);
  mkl_free(NSAB);
  mkl_free(NSAA);
  mkl_free(NSBB);
  mkl_free(DENOM);
  mkl_free(DENOMSqrt); 

  mkl_free(N);
  mkl_free(SA);
  mkl_free(SAA);
  mkl_free(SB);
  mkl_free(SBB);
  mkl_free(SAB);

  wtime = omp_get_wtime()-wtime;
  printf( "Vector: Time taken by thread %d is %f \n", omp_get_thread_num(), wtime);
  FILE *f;
  f = fopen("./timing/MPCC_vector_timing_matsize.txt","a");
  fprintf(f, "%d  %f %d\n", m, wtime, nthreads);
  fclose(f);
  return 0;
};

#endif
