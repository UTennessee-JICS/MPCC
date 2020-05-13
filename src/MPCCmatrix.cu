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

#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void inline checkError(cublasStatus_t status, const char *msg){
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("%s", msg);
        exit(EXIT_FAILURE);
    }
}

//Define global vars for launching GPU kernels
int numBlocks;
int threadsPerBlock;

//Define GPU Kernels
__global__ void ElemVecMultKernel(int N, DataType* a, DataType*b, DataType*c){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ c[index] = a[index] * b[index]; }
}
__global__ void ElemVecDivKernel(int N, DataType*a, DataType*b, DataType*c){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ c[index] = a[index]/b[index]; }
}
__global__ void ElemVecSqrtKernel(int N, DataType*a, DataType*c){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ c[index] = sqrt(a[index]); }
}

__global__ void FixDenomKernel(int N, DataType*a){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if (index < N){ if(a[index]==0.0){a[index] = 1.0;} }
}

__global__ void CheckMissingDataKernel(int N, DataType* A, DataType* AMask, DataType* UnitA){
   int index = threadIdx.x + blockDim.x * blockIdx.x;
   if(index < N){
      AMask[index]=1.0;
      if (CHECKNA(A[index])) {
        AMask[index] = 0.0;
        A[index] = 0.0; // set A to 0.0 for subsequent calculations of PCC terms
      } else {
        UnitA[index] = 1.0;
      }
   }
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
// MKL substitute functions vDiv
void fixDenom (int l, DataType* in) {
  FixDenomKernel<<< numBlocks, threadsPerBlock >>>(l, in);
}
// MKL substitute functions vDiv
void CheckMissingData (int l, DataType* in1, DataType* in2, DataType* in3) {
  CheckMissingDataKernel<<< numBlocks, threadsPerBlock >>>(l, in1, in2, in3);
}

//This function is the implementation of a matrix x matrix algorithm which computes a matrix of PCC values
//but increases the arithmetic intensity of the naive pairwise vector x vector correlation
//A is matrix of X vectors and B is transposed matrix of Y vectors:
//P = [ N*sum(AB) - (sumA)(sumB)] /
//    sqrt([ (N*sumA^2 - (sum A)^2) ][ (N*sumB^2 - (sum B)^2) ])
int pcc_matrix(int m, int n, int p,
               DataType* A, DataType* B, DataType* P)
{
  // Unused variable warning: int stride = ((n-1)/64 +1);
  DataType alpha = 1.0;
  DataType negalpha = -1.0;
  DataType beta = 0.0;
  //bool transposeB = true; //assume this is always true. 
  bool sameAB = (p == 0) ? true : false;
  if(p == 0)  p = m;

  //need to detect optimal threads per block  
  threadsPerBlock=256;
  numBlocks=(m*p+threadsPerBlock-1)/threadsPerBlock;
  // setup execution parameters
  //int devID = findCudaDevice(argc, (const char **)argv);
  //cudaDeviceProp props;
  //checkCudaErrors(cudaGetDevice(&devID));
  //checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  //int block_size = (props.major < 2) ? 16 : 32;
  //dim3 threads(block_size, block_size);
  //dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
  //printf("numBlocks=%d, threadsPerBlock=%d\n",numBlocks,threadsPerBlock);

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  checkError(cublasCreate(&handle), "cublasCreate() error!\n");

  //allocate and initialize and align memory needed to compute PCC
  DataType* UnitA = ( DataType*)ALLOCATOR( m*n, sizeof(DataType), 64 );//__assume_aligned(UnitA, 64);
  DataType* UnitB = sameAB ? UnitA : ( DataType*)ALLOCATOR( n*p, sizeof(DataType), 64 );//__assume_aligned(UnitB, 64);  
  DataType *amask=(DataType*)ALLOCATOR( m*n, sizeof(DataType), 64);//__assume_aligned(amask, 64);
  DataType *bmask= sameAB ? amask : (DataType*)ALLOCATOR( n*p, sizeof(DataType), 64);//__assume_aligned(bmask, 64);

  DataType *d_A, *d_B, *d_P;
  DataType *d_amask, *d_bmask, *d_UnitA, *d_UnitB, *d_N;

  //info("before deal missing data\n",1);
  //deal with missing data
  cudaMalloc(&d_A,m*n*sizeof(DataType));
  cudaMalloc(&d_amask,m*n*sizeof(DataType));
  cudaMalloc(&d_UnitA,m*n*sizeof(DataType));
  gpuErrchk(cudaMemcpy(d_A,A,m*n*sizeof(DataType),cudaMemcpyHostToDevice));
  CheckMissingData(m*n, d_A, d_amask, d_UnitA);

  //if (!sameAB) { //only do it when A and B are not same
  //if element in B is missing, set bmask and B to 0
  cudaMalloc(&d_B,p*n*sizeof(DataType));
  cudaMalloc(&d_bmask,p*n*sizeof(DataType));
  cudaMalloc(&d_UnitB,p*n*sizeof(DataType));
  gpuErrchk(cudaMemcpy(d_B,B,p*n*sizeof(DataType),cudaMemcpyHostToDevice));
  CheckMissingData(p*n, d_B, d_bmask, d_UnitB);
  //}

  //Do a bunch of cuda memcpy here
  gpuErrchk(cudaMalloc(&d_N,m*p*sizeof(DataType)));
  checkError(GEMM(handle, TRANS, NOTRANS, p, m, n, &alpha, d_bmask, n, d_amask, n, &beta, d_N, p),"GEMM error\n");

  gpuErrchk(cudaFree(d_amask));
  gpuErrchk(cudaFree(d_bmask));

  FREE(amask);
  FREE(bmask);

  FREE(UnitA);
  FREE(UnitB);

  DataType *d_AA, *d_BB;
  gpuErrchk(cudaMalloc(&d_AA,m*n*sizeof(DataType)));
  VSQR(m*n,d_A,d_AA);//need cuda versions for elementwise functions

  //vsSqr(n*p,B,BB);
  //if (!sameAB) { 
  gpuErrchk(cudaMalloc(&d_BB,n*p*sizeof(DataType)));
  VSQR(n*p,d_B,d_BB); 
  //} // Only perform VSQR when A and B are not same

  //SA = A*UnitB
  //Compute sum of A for each AB row col pair.
  // This requires multiplication with a UnitB matrix which acts as a mask 
  // to prevent missing data in AB pairs from contributing to the sum
  DataType *d_SA;
  gpuErrchk(cudaMalloc(&d_SA,m*p*sizeof(DataType)));
  checkError(GEMM(handle, TRANS, NOTRANS, p, m, n, &alpha, d_UnitB, n, d_A, n, &beta, d_SA, p),"GEMM error\n"); 

  //SB = UnitA*B
  //Compute sum of B for each AB row col pair.
  // This requires multiplication with a UnitA matrix which acts as a mask 
  // to prevent missing data in AB pairs from contributing to the sum
  //if (!sameAB) { //only do it when A and B are not same
  DataType *d_SB;
  gpuErrchk(cudaMalloc(&d_SB,m*p*sizeof(DataType)));
  checkError(GEMM(handle, TRANS, NOTRANS, p, m, n, &alpha, d_B, n, d_UnitA, n, &beta, d_SB, p),"GEMM error\n");
  //}

  //SAB = A*B
  DataType *d_SAB;
  gpuErrchk(cudaMalloc(&d_SAB,m*p*sizeof(DataType)));
  checkError(GEMM(handle, TRANS, NOTRANS, p, m, n, &alpha, d_B, n, d_A, n, &beta, d_SAB, p),"GEMM error\n"); 

  //free up some memory
  gpuErrchk(cudaFree(d_A));
  gpuErrchk(cudaFree(d_B));

  //SAA = AA*UnitB
  //Compute sum of AA for each AB row col pair.
  // This requires multiplication with a UnitB matrix which acts as a mask 
  // to prevent missing data in AB pairs from contributing to the sum
  DataType *d_SAA,*d_SBB;
  gpuErrchk(cudaMalloc(&d_SAA,m*p*sizeof(DataType)));
  checkError(GEMM(handle, TRANS, NOTRANS, p, m, n, &alpha, d_UnitB, n, d_AA, n, &beta, d_SAA, p),"GEMM error\n"); 

  gpuErrchk(cudaFree(d_UnitB));

  //SBB = UnitA*BB
  //Compute sum of BB for each AB row col pair.
  // This requires multiplication with a UnitA matrix which acts as a mask 
  // to prevent missing data in AB pairs from contributing to the sum
  //if (!sameAB) { //only do it when A and B are not same
  gpuErrchk(cudaMalloc(&d_SBB,m*p*sizeof(DataType)));
  checkError(GEMM(handle, TRANS, NOTRANS, p, m, n, &alpha, d_BB, n, d_UnitA, n, &beta, d_SBB, p),"GEMM error\n"); 
  //}

  gpuErrchk(cudaFree(d_UnitA));

  //Compute and assemble composite terms
  //SASB=SA*SB
  //if (!sameAB) {
  DataType *d_SASB;
  gpuErrchk(cudaMalloc(&d_SASB,m*p*sizeof(DataType)));
  VMUL(m*p,d_SA,d_SB,d_SASB);
  //} else {
  //  vMulSameAB(m,p,d_SA,d_SB,d_SASB);
  //}

  //NSAB=N*SAB
  DataType *d_NSAB;
  gpuErrchk(cudaMalloc(&d_NSAB,m*p*sizeof(DataType)));
  VMUL(m*p,d_N,d_SAB,d_NSAB); //ceb

  gpuErrchk(cudaFree(d_SAB));

  //NSAB=NSAB-SASB  (numerator)
  AXPY(handle, m*p, &negalpha, d_SASB, 1, d_NSAB, 1); //ceb

  gpuErrchk(cudaFree(d_SASB));

  //(SA)^2
  DataType *d_SASA;
  gpuErrchk(cudaMalloc(&d_SASA,m*p*sizeof(DataType)));
  VSQR(m*p,d_SA,d_SASA);

  gpuErrchk(cudaFree(d_SA));

  //N(SAA)
  DataType *d_NSAA;
  gpuErrchk(cudaMalloc(&d_NSAA,m*p*sizeof(DataType)));
  VMUL(m*p,d_N,d_SAA,d_NSAA); //ceb

  gpuErrchk(cudaFree(d_SAA));

  //NSAA=NSAA-SASA (denominator term 1)
  AXPY(handle, m*p, &negalpha, d_SASA, 1, d_NSAA, 1);

  gpuErrchk(cudaFree(d_SASA));

  //(SB)^2
  //if (!sameAB) {
  DataType *d_SBSB;
  gpuErrchk(cudaMalloc(&d_SBSB,m*p*sizeof(DataType)));
  VSQR(m*p,d_SB,d_SBSB);

  gpuErrchk(cudaFree(d_SB));

  //N(SBB)
  //if (!sameAB) {
  DataType *d_NSBB;
  gpuErrchk(cudaMalloc(&d_NSBB,m*p*sizeof(DataType)));
  VMUL(m*p,d_N,d_SBB,d_NSBB);

  gpuErrchk(cudaFree(d_SBB));
  gpuErrchk(cudaFree(d_N));
  //} else {
  //  vMulSameAB(m, p, d_N, d_SBB, d_NSBB);
  //}
  //NSBB=NSBB-SBSB (denominatr term 2)
  AXPY(handle, m*p, &negalpha, d_SBSB, 1, d_NSBB, 1);

  gpuErrchk(cudaFree(d_SBSB));

  //DENOM=NSAA*NSBB (element wise multiplication)
  DataType *d_DENOM;
  gpuErrchk(cudaMalloc(&d_DENOM,m*p*sizeof(DataType)));
  VMUL(m*p,d_NSAA,d_NSBB,d_DENOM);

  gpuErrchk(cudaFree(d_NSAA));
  gpuErrchk(cudaFree(d_NSBB));
  fixDenom(m*p, d_DENOM);

  //sqrt(DENOM)
  DataType *d_DENOMSqrt;
  gpuErrchk(cudaMalloc(&d_DENOMSqrt,m*p*sizeof(DataType)));
  VSQRT(m*p,d_DENOM,d_DENOMSqrt);

  gpuErrchk(cudaFree(d_DENOM));

  //P=NSAB/DENOMSqrt (element wise division)
  //This CUDA version loses precision compared to MKL version here due to divsion
  cudaMalloc(&d_P,m*p*sizeof(DataType));
  VDIV(m*p,d_NSAB,d_DENOMSqrt,d_P);   
  gpuErrchk(cudaMemcpy(P,d_P,m*p*sizeof(DataType),cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_P));

  cublasDestroy(handle);

  return 0;
};

