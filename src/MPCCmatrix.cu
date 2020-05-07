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
#include <cuda.h>
#include <cublas_v2.h>//ceb

using namespace std;
int numBlocks;
int threadsPerBlock;

__global__ void ElemVecMultKernel(int N, DataType* a, DataType*b, DataType*c)
{
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ c[index] = a[index] * b[index]; }
}
__global__ void ElemVecDivKernel(int N, DataType*a, DataType*b, DataType*c)
{
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ c[index] = a[index] * 1.0/b[index]; }
}
__global__ void ElemVecSqrtKernel(int N, DataType*a, DataType*c)
{
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ c[index] = sqrt(a[index]); }
}

  // MKL substitute functions vSqr
  void vSqr (int l, DataType* in, DataType* out) {
    ElemVecMultKernel<<< numBlocks, threadsPerBlock >>>(l, in, in, out);
  }
  // MKL substitute functions vMul
  void vMul (int l, DataType* in1, DataType* in2, DataType* out) {
    ElemVecMultKernel<<< numBlocks, threadsPerBlock >>>(l, in1, in2, out);
  }
  // MKL substitute functions vSqrt
  void vSqrt (int l, DataType* in, DataType* out) {
    ElemVecSqrtKernel<<< numBlocks, threadsPerBlock >>>(l, in, out);
  }
  // MKL substitute functions vDiv
  void vDiv (int l, DataType* in1, DataType* in2, DataType* out) {
    ElemVecDivKernel<<< numBlocks, threadsPerBlock >>>(l, in1, in2, out);
  }

// vMul function when input matrix A == B



//This function is the implementation of a matrix x matrix algorithm which computes a matrix of PCC values
//but increases the arithmetic intensity of the naive pairwise vector x vector correlation
//A is matrix of X vectors and B is transposed matrix of Y vectors:
//P = [ sum(AB) - (sumA)(sumB)/N ] /
//    sqrt([ (sumA^2 -(1/N)(sum A)^2) ][ (sumB^2 - (1/N)(sum B)^2) ])
//
//P = [ N*sum(AB) - (sumA)(sumB)] /
//    sqrt([ (N*sumA^2 - (sum A)^2) ][ (N*sumB^2 - (sum B)^2) ])
//#ifdef CUBLAS
int pcc_matrix(int m, int n, int p,
               DataType* A, DataType* B, DataType* P)
{
  // Unused variable warning: int stride = ((n-1)/64 +1);
  int i,j,k;
  DataType alpha = 1.0;
  DataType negalpha = -1.0;
  DataType beta = 0.0;
  int count = 1;
  bool transposeB = true; //assume this is always true. 
  bool sameAB = (p == 0) ? true : false;
  if(p == 0)  p = m;

  //
  
  threadsPerBlock=256;
  numBlocks=(m*p+threadsPerBlock-1)/threadsPerBlock;
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasStatus_t stat;
  stat = cublasCreate(&handle);
  //checkError(cublasCreate(&handle), "cublasCreate() error!\n");

  //info("before calloc\n",1);
  //allocate and initialize and align memory needed to compute PCC
  //DataType *N = (DataType *) ALLOCATOR( m*p,sizeof( DataType ), 64 ); //__assume_aligned(N, 64);
  //DataType* SA =    ( DataType*)ALLOCATOR( m*p, sizeof(DataType), 64 ); //__assume_aligned(SA, 64);
  //DataType* AA =    ( DataType*)ALLOCATOR( m*n, sizeof(DataType), 64 ); //__assume_aligned(AA, 64);
  //DataType* SAA =   ( DataType*)ALLOCATOR( m*p, sizeof(DataType), 64 ); //__assume_aligned(SAA, 64);
  //DataType* SB =    sameAB ? SA : ( DataType*)ALLOCATOR( m*p, sizeof(DataType), 64 ); //__assume_aligned(SB, 64);
  //DataType* BB =    sameAB ? AA : ( DataType*)ALLOCATOR( n*p, sizeof(DataType), 64 ); //__assume_aligned(BB, 64);
  //DataType* SBB =   sameAB ? SAA : ( DataType*)ALLOCATOR( m*p, sizeof(DataType), 64 ); //__assume_aligned(SBB, 64);
  //DataType* SAB =   ( DataType*)ALLOCATOR( m*p, sizeof(DataType), 64 );//__assume_aligned(SAB, 64);
  DataType* UnitA = ( DataType*)ALLOCATOR( m*n, sizeof(DataType), 64 );//__assume_aligned(UnitA, 64);
  DataType* UnitB = sameAB ? UnitA : ( DataType*)ALLOCATOR( n*p, sizeof(DataType), 64 );//__assume_aligned(UnitB, 64);  
  DataType *amask=(DataType*)ALLOCATOR( m*n, sizeof(DataType), 64);//__assume_aligned(amask, 64);
  DataType *bmask= sameAB ? amask : (DataType*)ALLOCATOR( n*p, sizeof(DataType), 64);//__assume_aligned(bmask, 64);

  //info("after calloc\n",1);

  //if any of the above allocations failed, then we have run out of RAM on the node and we need to abort
  //if ( (N == NULL) | (SA == NULL) | (AA == NULL) | (SAA == NULL) | (SB == NULL) | (BB == NULL) | 
  //    (SBB == NULL) | (SAB == NULL) | (UnitA == NULL) | (UnitB == NULL) | (amask == NULL) | (bmask == NULL)) {
  //  err("ERROR: Can't allocate memory for intermediate matrices. Aborting...\n", 1);
  //  FREE(N);
  //  FREE(SA);
  //  FREE(AA);
  //  FREE(SAA);
  //  FREE(SAB);
  //  FREE(UnitA);
  //  FREE(amask);
  //  if(!sameAB) { //only do it when A and B are not same
  //    FREE(SB);
  //    FREE(BB);
  //    FREE(SBB);
  //    FREE(UnitB);
  //    FREE(bmask);
  //  }
  //  exit(0);
  //} 

  //create GPU device memory arrays
  DataType *d_A, *d_B, *d_P;
  cudaMalloc(&d_A,m*n*sizeof(DataType));
  cudaMalloc(&d_B,n*p*sizeof(DataType));
  cudaMalloc(&d_P,m*n*sizeof(DataType));

  DataType *d_amask, *d_bmask, *d_UnitA, *d_UnitB, *d_N;
  cudaMalloc(&d_amask,m*n*sizeof(DataType));
  cudaMalloc(&d_bmask,n*p*sizeof(DataType));
  cudaMalloc(&d_UnitA,m*n*sizeof(DataType));
  cudaMalloc(&d_UnitB,n*p*sizeof(DataType));
  cudaMalloc(&d_N,m*p*sizeof(DataType));

  //info("before deal missing data\n",1);
  //deal with missing data
  for (int ii=0; ii<count; ii++) {
    //if element in A is missing, set amask and A to 0
    //#pragma omp parallel for private (i,k)
    for (i=0; i<m; i++) {
      for (k=0; k<n; k++) {
        amask[ i*n + k ] = 1.0;
        if (CHECKNA(A[i*n+k])) { 
          amask[i*n + k] = 0.0;
          A[i*n + k] = 0.0; // set A to 0.0 for subsequent calculations of PCC terms
        } else {
          UnitA[i*n + k] = 1.0;
        }
      }
    }

    if (!sameAB) { //only do it when A and B are not same
      //if element in B is missing, set bmask and B to 0
      //#pragma omp parallel for private (j,k)
      for (j=0; j<p; j++) {
        for (k=0; k<n; k++) {
          bmask[ j*n + k ] = 1.0;
          if (CHECKNA(B[j*n+k])) { 
            bmask[j*n + k] = 0.0;
            B[j*n + k] = 0.0; // set B to 0.0 for subsequent calculations of PCC terms
          } else {
            UnitB[j*n + k] = 1.0;
          }
        }
      }
    }



    //Do a bunch of cuda memcpy here
    cudaMemcpy(d_amask,amask,m*n*sizeof(DataType),cudaMemcpyHostToDevice);
    cudaMemcpy(d_bmask,bmask,n*p*sizeof(DataType),cudaMemcpyHostToDevice);
    //GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_amask, n, d_bmask, n, &beta, d_N, p);
    GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_amask, n, d_bmask, p, &beta, d_N, p);
    //cudaMemcpy(N,d_N,n*p*sizeof(DataType),cudaMemcpyDeviceToHost);
    FREE(amask);
    FREE(bmask);

    cudaMemcpy(d_UnitA,UnitA,m*n*sizeof(DataType),cudaMemcpyHostToDevice);
    cudaMemcpy(d_UnitB,UnitB,n*p*sizeof(DataType),cudaMemcpyHostToDevice);
    FREE(UnitA);
    FREE(UnitB);

    cudaMemcpy(d_A,A,m*n*sizeof(DataType),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,n*p*sizeof(DataType),cudaMemcpyHostToDevice);


    DataType *d_AA, *d_BB;
    cudaMalloc(&d_AA,m*n*sizeof(DataType));
    VSQR(m*n,d_A,d_AA);//need cuda versions for elementwise functions

    //vsSqr(n*p,B,BB);
    //if (!sameAB) { 
    cudaMalloc(&d_BB,n*p*sizeof(DataType));
    VSQR(n*p,d_B,d_BB); 
    //} // Only perform VSQR when A and B are not same

    //CBLAS_TRANSPOSE transB=NOTRANS;
    //int ldb=p;
    //if(transposeB){
    //  ldb=n;
    //}

    
    //SA = A*UnitB
    //Compute sum of A for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    DataType *d_SA,*d_SB;
    cudaMalloc(&d_SA,m*n*sizeof(DataType));
    //GEMM(handle, NOTRANS, TRANS,  m, p, n, &alpha, d_A, n, d_UnitB, ldb, &beta, d_SA, p); 
    GEMM(handle, NOTRANS, TRANS,  m, p, n, &alpha, d_A, n, d_UnitB, p, &beta, d_SA, p); 

    //SB = UnitA*B
    //Compute sum of B for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //if (!sameAB) { //only do it when A and B are not same
    cudaMalloc(&d_SB,n*p*sizeof(DataType));
    //GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_UnitA, n, d_B, ldb, &beta, d_SB, p);
    GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_UnitA, n, d_B, p, &beta, d_SB, p);
    //}


    //SAA = AA*UnitB
    //Compute sum of AA for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    DataType *d_SAA,*d_SBB,*d_SAB;
    cudaMalloc(&d_SAA,m*n*sizeof(DataType));
    //GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_AA, n, d_UnitB, ldb, &beta, d_SAA, p); 
    GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_AA, n, d_UnitB, p, &beta, d_SAA, p); 

    //SBB = UnitA*BB
    //Compute sum of BB for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //if (!sameAB) { //only do it when A and B are not same
    cudaMalloc(&d_SBB,n*p*sizeof(DataType));
    //GEMM(handle, NOTRANS, TRANS,m, p, n, &alpha, d_UnitA, n, d_BB, ldb, &beta, d_SBB, p); 
    GEMM(handle, NOTRANS, TRANS,m, p, n, &alpha, d_UnitA, n, d_BB, p, &beta, d_SBB, p); 
    //}


    //SAB = A*B
    cudaMalloc(&d_SAB,m*p*sizeof(DataType));
    //GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_A, n, d_B, ldb, &beta, d_SAB, p); 
    GEMM(handle, NOTRANS, TRANS, m, p, n, &alpha, d_A, n, d_B, p, &beta, d_SAB, p); 

    //clock_gettime(CLOCK_MONOTONIC, &stopGEMM);
    //accumGEMM =  (TimeSpecToSeconds(&stopGEMM)- TimeSpecToSeconds(&startGEMM));
    //printf("All(5) GEMMs (%e)s GFLOPs=%e \n", accumGEMM, 5*(2/1.0e9)*m*n*p/accumGEMM);

    //DataType* SASB = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 );
    //DataType* NSAB = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 ); //ceb
   
    //DataType* SASA = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 ); 
    //DataType* NSAA = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 ); //ceb
    
    //DataType* SBSB = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 );    
    //DataType* NSBB = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 ); //ceb   
    
    //DataType* DENOM = ( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 );
    //DataType* DENOMSqrt =( DataType*)ALLOCATOR( m*p,sizeof(DataType), 64 ); 


    //Compute and assemble composite terms
    //SASB=SA*SB
    //if (!sameAB) {
    DataType *d_SASB;
    cudaMalloc(&d_SASB,m*p*sizeof(DataType));
    VMUL(m*p,d_SA,d_SB,d_SASB);
    //} else {
    //  vMulSameAB(m,p,d_SA,d_SB,d_SASB);
    //}

    //NSAB=N*SAB
    DataType *d_NSAB;
    cudaMalloc(&d_NSAB,m*p*sizeof(DataType));
    VMUL(m*p,d_N,d_SAB,d_NSAB); //ceb
    //NSAB=(-1)NSAB+SASB  (numerator)
    AXPY(handle, m*p, &negalpha, d_SASB, 1, d_NSAB, 1); //ceb

    //(SA)^2
    DataType *d_SASA;
    cudaMalloc(&d_SASA,m*p*sizeof(DataType));
    VSQR(m*p,d_SA,d_SASA);

    //N(SAA)
    DataType *d_NSAA;
    cudaMalloc(&d_NSAA,m*p*sizeof(DataType));
    VMUL(m*p,d_N,d_SAA,d_NSAA); //ceb
    //NSAA=NSAA-SASA (denominator term 1)
    AXPY(handle, m*p, &negalpha, d_SASA, 1, d_NSAA, 1);

    //(SB)^2
    //if (!sameAB) {
    DataType *d_SBSB;
    cudaMalloc(&d_SBSB,m*p*sizeof(DataType));
    VSQR(m*p,d_SB,d_SBSB);
    //} 
    //else {
    //  //#pragma omp parallel for private(i, j)
    //  for(int i = 0; i < m; i++) {
    //    for(int j = 0; j < p; j++) {
    //      SBSB[i*m + j] = SB[j*m + i] * SB[j*m + i];
    //    }
    //  }
    //}

    //N(SBB)
    //if (!sameAB) {
    DataType *d_NSBB;
    cudaMalloc(&d_NSBB,m*p*sizeof(DataType));
    VMUL(m*p,d_N,d_SBB,d_NSBB);
    //} else {
    //  vMulSameAB(m, p, d_N, d_SBB, d_NSBB);
    //}
    //NSBB=NSBB-SBSB (denominatr term 2)
    AXPY(handle, m*p, &negalpha, d_SBSB, 1, d_NSBB, 1);

    //DENOM=NSAA*NSBB (element wise multiplication)
    DataType *d_DENOM;
    cudaMalloc(&d_DENOM,m*p*sizeof(DataType));
    VMUL(m*p,d_NSAA,d_NSBB,d_DENOM);
    //#pragma omp parallel for private (i)
    //for (int i = 0;i < m*p;++i) {
    //   if(DENOM[i]==0.){DENOM[i]=1;}//numerator will be 0 so to prevent inf, set denom to 1
    //}

    //sqrt(DENOM)
    DataType *d_DENOMSqrt;
    cudaMalloc(&d_DENOMSqrt,m*p*sizeof(DataType));
    VSQRT(m*p,d_DENOM,d_DENOMSqrt);

    //P=NSAB/DENOMSqrt (element wise division)
    VDIV(m*p,d_NSAB,d_DENOMSqrt,d_P);   
    cudaMemcpy(P,d_P,m*n*sizeof(DataType),cudaMemcpyDeviceToHost);

    //FREE(SASB);
    //FREE(NSAB);
    //FREE(SASA);
    //FREE(NSAA);
    //FREE(SBSB);
    //FREE(NSBB);
    //FREE(DENOM);
    //FREE(DENOMSqrt); 
  }

  //FREE(N);
  //FREE(SA);
  //FREE(SAA);
  //if (!sameAB) { //only do it when A and B are not same
    //FREE(SB);
    //FREE(SBB);
  //}
  //FREE(SAB);

  cublasDestroy(handle);

  return 0;
};
//#endif //ndef CUBLAS
//#endif //noblas

