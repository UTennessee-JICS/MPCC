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

#include <bitset>
#include <iostream>
//AVX intrinsics
#include <smmintrin.h>
#include <immintrin.h>

using namespace std;

#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#define __assume_aligned(var,size){ __builtin_assume_aligned(var,size); }
#define DEV_CHECKPT printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); fflush(stdout); 

// This function is an implementation of a bitsum of an unsigned long n;
// same as popcount64 algorithm
inline int bitsum(uint64_t x)
{
#if 0
   int v=0;
   while(x != 0){
      x &= x-1;
      v++;
   }
  return v;
#endif
#if 1 //intrinsic bit counter for 64 bit string
    return _mm_popcnt_u64(x);
#endif
}

#if 0
inline void CSA(uint64_t* h, uint64_t* l, uint64_t a, uint64_t b, uint64_t c)
{
   uint64_t u = (a ^ b);
   *h = (a & b) | (u & c);
   *l = (u ^ c);
}

inline __m256i count(__m256i v)
{
   __m256i lookup = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
   __m256i low_mask = _mm256_set1_epi8(0x0f);
   __m256i lo = _mm256_and_si256(v, low_mask);
   __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v,4), low_mask);
   __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
   __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
   __m256i total = _mm256_add_epi8(popcnt1, popcnt2);
   return _mm256_sad_epu8(total, _mm256_setzero_si256());
}

inline uint64_t avx_hs(__m256i* d, uint64_t size)
{
   __m256i total = _mm256_setzero_si256();
   __m256i ones = _mm256_setzero_si256();
   __m256i twos = _mm256_setzero_si256();
   __m256i fours = _mm256_setzero_si256();
   __m256i eights = _mm256_setzero_si256();
   __m256i sixteens = _mm256_setzero_si256();
   __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;
   for(uint64_t i= 0; i < size; i+=16){
      CSA(&twosA,    &ones,   ones,   d[i],    d[i+1]);
      CSA(&twosB,    &ones,   ones,   d[i+1],  d[i+3]);
      CSA(&foursA,   &twos,   twos,   twosA,   twosB);
      CSA(&twosA,    &ones,   ones,   d[i+4],  d[i+5]);
      CSA(&twosB,    &ones,   ones,   d[i+6],  d[i+7]);
      CSA(&foursB,   &twos,   twos,   twosA,   twosB);
      CSA(&eightsA,  &fours,  fours,  foursA,  foursB);
      CSA(&twosA,    &ones,   ones,   d[i+8],  d[i+9]);
      CSA(&twosB,    &ones,   ones,   d[i+10], d[i+11]);
      CSA(&foursA,   &twos,   twos,   twosA,   twosB);
      CSA(&twosA,    &ones,   ones,   d[i+12], d[i+13]);
      CSA(&twosB,    &ones,   ones,   d[i+14], d[i+15]);
      CSA(&foursB,   &twos,   twos,   twosA,   twosB);
      CSA(&eightsB,  &fours,  fours,  foursA,  foursB);
      CSA(&sixteens, &eights, eights, eightsA, eightsB);
      total = _mm256_add_epi64(total, count(sixteens));
   }
   total = _mm256_slli_epi64(total, 4);
   total = _mm256_add_epi64(total, _mm256_slli_epi64(count(eights), 3));
   total = _mm256_add_epi64(total, _mm256_slli_epi64(count(fours), 2));
   total = _mm256_add_epi64(total, _mm256_slli_epi64(count(twos), 1));
   total = _mm256_slli_epi64(total, count(ones));
   return _mm256_extract_epi64(total, 0)
        + _mm256_extract_epi64(total, 1)
        + _mm256_extract_epi64(total, 2)
        + _mm256_extract_epi64(total, 3);
}
#endif


#ifndef NOMKL
//This function is the implementation of a matrix x matrix algorithm which computes a matrix of PCC values
//but increases the arithmetic intensity of the naive pairwise vector x vector correlation
//A is matrix of X vectors and B is transposed matrix of Y vectors:
//P = [ sum(AB) - (sumA)(sumB)/N] /
//    sqrt[ ( sumA^2 -(1/N) (sum A/)^2)[ ( sumB^2 - (1/N)(sum B)^2) ]
int pcc_matrix(int m, int n, int p,
	       DataType* A, DataType* B, DataType* P, int nthreads=1)	       
{
  double wtime = omp_get_wtime();
  omp_set_num_threads(nthreads);
  printf("get_omp_max_threads()=%d\n",omp_get_max_threads());
  mkl_set_num_threads(nthreads);
  printf("mkl_get_max_threads()=%d\n",mkl_get_max_threads());

  int i,j,k;
  int stride = ((n-1)/64 +1);
  DataType alpha=1.0;
  DataType beta=0.0;
  int count =1;
  bool transposeB = true; //assume this is always true. 
  //allocate and initialize and align memory needed to compute PCC
  DataType *N = (DataType *) mkl_calloc( m*p,sizeof( DataType ), 64 );
  __assume_aligned(N, 64);
  unsigned long *M = (unsigned long *) mkl_calloc( m*p, sizeof( unsigned long ), 64 );
  __assume_aligned(M, 64);
  DataType* SA =    ( DataType*)mkl_calloc( m*p, sizeof(DataType), 64 ); 
  __assume_aligned(SA, 64);
  DataType* AA =    ( DataType*)mkl_calloc( m*n, sizeof(DataType), 64 ); 
  __assume_aligned(AA, 64);
  DataType* SAA =   ( DataType*)mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SAA, 64);
  DataType* SB =    ( DataType*)mkl_calloc( m*p, sizeof(DataType), 64 ); 
  __assume_aligned(SB, 64);
  DataType* BB =    ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 ); 
  __assume_aligned(BB, 64);
  DataType* SBB =   ( DataType*)mkl_calloc( m*p, sizeof(DataType), 64 ); 
  __assume_aligned(SBB, 64);
  DataType* SAB =   ( DataType*)mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(SAB, 64);
  DataType* UnitA = ( DataType*)mkl_calloc( m*n, sizeof(DataType), 64 );
  __assume_aligned(UnitA, 64);
  DataType* UnitB = ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 );
  __assume_aligned(UnitB, 64);  
  unsigned long *amask=(unsigned long*)mkl_calloc( m*stride, sizeof(unsigned long), 64);
  __assume_aligned(amask, 64);
  unsigned long *bmask=(unsigned long*)mkl_calloc( p*stride, sizeof(unsigned long), 64);
  __assume_aligned(bmask, 64);

  //if any of the above allocations failed, then we have run out of RAM on the node and we need to abort
  if ( (N == NULL) | (M == NULL) | (SA == NULL) | (AA == NULL) | (SAA == NULL) | (SB == NULL) | (BB == NULL) | 
      (SBB == NULL) | (SAB == NULL) | (UnitA == NULL) | (UnitB == NULL) | (amask == NULL) | (bmask == NULL)) {
    printf( "\n ERROR: Can't allocate memory for intermediate matrices. Aborting... \n\n");
    mkl_free(N);
    mkl_free(M);
    mkl_free(SA);
    mkl_free(AA);
    mkl_free(SAA);
    mkl_free(SB);
    mkl_free(BB);
    mkl_free(SBB);
    mkl_free(SAB);
    mkl_free(UnitA);
    mkl_free(UnitB);
    mkl_free(amask);
    mkl_free(bmask);
    exit (0);
  } 

  //deal with missing data
  for (int ii=0; ii<count; ii++) {

    //If element in A or B has missing data,
    // add a 1 to the bit column k location for row i
    
    //initialize data mask for matrix A to 0's
    #pragma omp parallel for private (i)
    for (i=0; i< m*stride; i++) { amask[ i ]=0UL; }

    //initialize data mask for matrix B to 0's
    #pragma omp parallel for private (j)   
    for (j=0; j< p*stride; j++) { bmask[ j ]=0UL; }

    //if element in A is missing, flip bit of corresponding col to 1
    #pragma omp parallel for private (i,k)
    for (i=0; i<m; i++){
      int mm=0;
      for (k=0; k<n; k++) {	
	if (std::isnan(A[i*n+k])) {
          mm++;
	  amask[i*stride +k/64] |= (1UL << (n-k-1)%64);
	}
      }
    }

    //if element in B is missing, flip bit of corresponding col to 1
    #pragma omp parallel for private (j,k)
    for (j=0; j<p; j++) {
      for (k=0; k<n; k++) {	
	if (std::isnan(B[j*n+k])) {
	  bmask[j*stride +k/64] |= (1UL << (n-k-1)%64);
	}
      }
    }

    //For all A,B pairs if either A or B has a missing data bit set,
    // a logical OR between row A[i] and column B[j] row bit masks will
    // return a 1 in the bit mask M[i,j]
    // For each row*col pair in A*B comparison, sum up the number of missing values using the bitstring
    #pragma omp parallel for private (i,j,k)
    for (i=0; i<m; i++){
      for (j=0; j<p; j++){
#if 1        
	for(k=0; k<stride; ++k){
	  M[i*p+j] += bitsum((amask[ i*stride+k ] | bmask[ j*stride+k ]));
	}
#else//use vectorized bitsum algorithm

#endif
      }
    }

    //Compute the number of non missing data for every row/column pair.
    //This is done by subtracting the number of elements in a row by the number of
    // missing data bits set for the row/column pair.
    #pragma omp parallel for private(i)
    for(i=0; i<m*p; i++){
      N[i] = n-M[i];
    }
    mkl_free(M);

    //Zero out values that are marked as missing.
    // For subsequent calculations of PCC terms, we need to replace
    // missing value markers with 0 in matrices so that they dont contribute to 
    // the sums
    #pragma omp parallel for private(i)
    for (i=0; i<m*n; i++) {
      if (std::isnan(A[i])) { A[i]=0.0; }
      else{ UnitA[i]=1; }
    }
    VSQR(m*n,A,AA);

    //Zero out values that are marked as missing.
    // For subsequent calculations of PCC terms, we need to replace
    // missing value markers with 0 in matrices so that they dont contribute to
    // the sums
    #pragma omp parallel for private(j)
    for (j=0; j<n*p; j++) {
      if (std::isnan(B[j])) { B[j]=0.0; }
      else{ UnitB[j]=1; }
    }
    VSQR(n*p,B,BB);

    //Compute PCC terms and assemble
    CBLAS_TRANSPOSE transB=CblasNoTrans;
    int ldb=p;
    if(transposeB){
      transB=CblasTrans;
      ldb=n;
    }

    //SA = A*UnitB
    //Compute sum of A for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, A, n, UnitB, ldb, beta, SA, p); 

    //SB = UnitA*B
    //Compute sum of B for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, UnitA, n, B, ldb, beta, SB, p); 

    //SAA = AA*UnitB
    //Compute sum of AA for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, AA, n, UnitB, ldb, beta, SAA, p); 

    //SBB = UnitA*BB
    //Compute sum of BB for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, UnitA, n, BB, ldb, beta, SBB, p); 

    mkl_free(UnitA);
    mkl_free(UnitB);
    mkl_free(AA);
    mkl_free(BB);

    //SAB = A*B
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, A, n, B, ldb, beta, SAB, p); 

    DataType* SASB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* NSAB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); //ceb
   
    DataType* SASA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 
    DataType* NSAA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); //ceb
    
    DataType* SBSB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );    
    DataType* NSBB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); //ceb   
    
    DataType* DENOM = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* DENOMSqrt =( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 

    //Compute and assemble composite terms

    //SASB=SA*SB
    VMUL(m*p,SA,SB,SASB);
    //N*SASB
    VMUL(m*p,N,SAB,NSAB);
    //NSAB=(-1)SASB+NSAB  (numerator)
    AXPY(m*p,(DataType)(-1), SASB,1, NSAB,1); //ceb

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
    //NSBB=NSBB-SBSB (denominator term 2)
    AXPY(m*p,(DataType)(-1), SBSB,1, NSBB,1);

    //DENOM=NSAA*NSBB (element wise multiplication)
    VMUL(m*p,NSAA,NSBB,DENOM);
#if 0
    for(int i=0;i<m*p;++i){
       if(DENOM[i]==0.){DENOM[i]=1;}//numerator will be 0 so to prevent inf, set denom to 1
    }
#endif
    //sqrt(DENOM)
    VSQRT(m*p,DENOM,DENOMSqrt);
    //P=NSAB/DENOMSqrt (element wise division)
    VDIV(m*p,NSAB,DENOMSqrt,P);   

    mkl_free(SASB);
    mkl_free(NSAB);
    mkl_free(SASA);
    mkl_free(NSAA);
    mkl_free(SBSB);
    mkl_free(NSBB);
    mkl_free(DENOM);
    mkl_free(DENOMSqrt); 
  }

  mkl_free(N);
  mkl_free(SA);
  mkl_free(SAA);
  mkl_free(SB);
  mkl_free(SBB);
  mkl_free(SAB);

  wtime = omp_get_wtime()-wtime;
  printf( "Matrix: Time taken by thread %d is %f  \n", omp_get_thread_num(), wtime);

  DataType gflops = (10/1.0e9)*m*n*p/wtime;
  FILE *f;
  f = fopen("./timing/MPCC_matrix_timing_matsize.txt","a");
  fprintf(f, "%d  %f %e %d %lu \n", m, wtime, gflops, nthreads, sizeof(DataType));
  fclose(f);
  return 0;
};
#endif

