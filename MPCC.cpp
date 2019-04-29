//Pearsons correlation coeficient for two vectors x and y
// r = sum_i( x[i]-x_mean[i])*(y[i]-y_mean[i]) ) /
//     [ sqrt( sum_i(x[i]-x_mean[i])^2 ) sqrt(sum_i(y[i]-y_mean[i])^2 ) ]
//
//This code computes correlation coefficient between all row/column pairs of two matrices 
// ./MPCC 100 10000 10000 7    3     3
//        m     n    p    seed naive matrix
//                             trip  trip
//                             count count


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <mkl.h>

#include <iostream>
#include <fstream>
using namespace std;

#define BILLION  1000000000L

#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#define __assume_aligned(var,size){ __builtin_assume_aligned(var,size); }

#define DEV_CHECKPT  printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); fflush(stdout); 

#define USE_OMP 1
#define MISSING_DATA 1
#define MISSING_MARKER 2

#define DataType float

static DataType TimeSpecToSeconds(struct timespec* ts)
{
  return (DataType)ts->tv_sec + (DataType)ts->tv_nsec / 1000000000.0;
};
 
int bitsum(unsigned long m){
  int c=0;
  int mm=m;
  for (c=0; mm; ++c) { mm&=mm-1;}
  return c;
}

void initialize(int &n, int &m, int &p, int seed,
		DataType **A, DataType **B,
		DataType **C, DataType **P)
{
  // A is n x m (tall and skinny) row major order
  // B is m x p (short and fat) row major order
  // C,P is n x p (big and square) row major order
  *A = (DataType *)mkl_calloc( n*m,sizeof( DataType ), 64 ); 
  if (*A == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix A. Aborting... \n\n");
    mkl_free(*A);
    exit (0);
  }
  *B = (DataType *)mkl_calloc( m*p,sizeof( DataType ), 64 );
  if (*B == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix B. Aborting... \n\n");
    mkl_free(*B);
    exit (0);
  }
  *C = (DataType *)mkl_calloc( n*p,sizeof( DataType ), 64 ); 
  if (*C == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix C. Aborting... \n\n");
    mkl_free(*C);
    exit (0);
  }
  *P = (DataType *)mkl_calloc( n*p,sizeof( DataType ), 64 ); 
  if (*P == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix P. Aborting... \n\n");
    mkl_free(*P);
    exit (0);
  }
  //*M = (unsigned long *)mkl_calloc( n*p,sizeof( unsigned long ), 64 );
  
  __assume_aligned(A, 64);
  __assume_aligned(B, 64);
  __assume_aligned(C, 64);
  __assume_aligned(P, 64);
  __assume(m%16==0);
 
  //create some synthetic matrices for correlation
  srand(seed);
  DataType randmax_recip=1/(DataType)RAND_MAX;

  //ceb Should write  out matrix and read in for future use.
  // check for input file(s)
  fstream mat_A_file;
  mat_A_file.open("matrixA.dat",ios::in | ios::binary);
  if(mat_A_file){
     // if found then read
     mat_A_file.read(reinterpret_cast<char*>(&n), sizeof(int));
     mat_A_file.read(reinterpret_cast<char*>(&m), sizeof(int));
     mat_A_file.read(reinterpret_cast<char*>(*A), sizeof(DataType)*n*m);
     mat_A_file.close();
  }
  else{ //else compute and then write matrix A
    //random assignemnt of threads gives inconsistent values, so keep serial
    int i;
    #pragma omp parallel for private (i)
    for (i=0; i<n*m; i++) {
      (*A)[i]=(DataType)rand()*randmax_recip;
    }
    //add some missing value markers
    //ceb if missing data causes number of pairs compared to be <2, the result is divide by zero
    (*A)[0]          = MISSING_MARKER;
    (*A)[n*m-1]      = MISSING_MARKER;
    (*A)[((n-1)*m-1)]= MISSING_MARKER;
   
    //write matrix to file
    mat_A_file.open("matrixA.dat",ios::out | ios::binary);
    mat_A_file.write(reinterpret_cast<char*>(&n), sizeof(int));
    mat_A_file.write(reinterpret_cast<char*>(&m), sizeof(int));
    mat_A_file.write(reinterpret_cast<char*>(*A), sizeof(DataType)*n*m);   
    mat_A_file.close();
  }
 
  //ceb Should write  out matrix and read in for future use.
  // check for input file(s)
  fstream mat_B_file;
  mat_B_file.open("matrixB.dat",ios::in | ios::binary);
  if(mat_B_file){
     // if found then read
     mat_B_file.read(reinterpret_cast<char*>(&m), sizeof(int));
     mat_B_file.read(reinterpret_cast<char*>(&p), sizeof(int));
     mat_B_file.read(reinterpret_cast<char*>(*B), sizeof(DataType)*m*p);
     mat_B_file.close();
  }
  else{ //else compute and then write matrix B
    int i;
    //random assignemnt of threads gives inconsistent values, so keep serial
    #pragma omp parallel for private (i)
    for (i=0; i<m*p; i++) {
      (*B)[i]=(DataType)rand()*randmax_recip;
    }
    //add some missing value markers
    //ceb if missing data causes number of pairs compared to be <2, the result is divide by zero
    (*B)[0]          = MISSING_MARKER;
    (*B)[m*p-1]      = MISSING_MARKER;
    (*B)[((m-1)*p-1)]= MISSING_MARKER;
   
    //write matrix to file
    mat_B_file.open("matrixB.dat",ios::out | ios::binary);
    mat_B_file.write(reinterpret_cast<char*>(&m), sizeof(int));
    mat_B_file.write(reinterpret_cast<char*>(&p), sizeof(int));
    mat_B_file.write(reinterpret_cast<char*>(*B), sizeof(DataType)*m*p);   
    mat_B_file.close();
  }

#if 0
  for (int i=0; i<n; i++) { for(int j=0;j<m;++j){printf("A[%d,%d]=%e\n",i,j,(*A)[i*m+j]);}}
  for (int i=0; i<m; i++) { for(int j=0;j<p;++j){printf("B[%d,%d]=%e\n",i,j,(*B)[i*p+j]);}}
#endif 
  return;
}


// r = sum_i( x[i]-x_mean[i])*(y[i]-y_mean[i]) ) /
//     [ sqrt( sum_i(x[i]-x_mean[i])^2 ) sqrt(sum_i(y[i]-y_mean[i])^2 ) ]

// reformulation:
// r = [N sum(XY) - (sumX)(sumY)] /
//     sqrt[ (N sumX^2 - (sum X)^2)[ (N sumY^2 - (sum Y)^2) ]

int pcc_naive(int n, int m, int p, int count_naive,
	      DataType* A, DataType* B, DataType* C)
{
  DataType sab,sa,sb,saa,sbb;
  int mm;
  int i,j,k;

  for (int ii=0; ii<count_naive; ii++) {
    #pragma omp parallel for private (i,j,k)
    for (i=0; i<n; i++) {  
      for (j=0; j<p; j++) {

	sa=0.0;
	sb=0.0;
	saa=0.0;
	sbb=0.0;	
      	sab=0.0;
	mm=m;
	
	for (k=0; k<m; k++) {
	  //compute divisor for mean calculation
	  if ((A[i*m+k] != MISSING_MARKER) && (B[k*p+j] != MISSING_MARKER)){   
	    sa  += A[i*m+k];
	    sb  += B[k*p+j];  
	    sab += A[i*m+k] * B[k*p+j];
	    saa += A[i*m+k] * A[i*m+k];
	    sbb += B[k*p+j] * B[k*p+j];
	  }  
	  else { mm--; }	  
	}
	
	//if(mm>1)//if mm==1 then denominator is Zero!
	{
	  C[i*p+j] = (mm*sab - sa*sb) / sqrt( (mm*saa - sa*sa)*(mm*sbb - sb*sb) );
	}
      }
    }
  }
  return 0;
}


int pcc_matrix(int n, int m, int p, int count_matrix,
	       DataType* A, DataType* B, DataType* P)	       
{
  int i,j,k;
  int stride = ((m-1)/64 +1);
  DataType alpha=1.0;
  DataType beta=0.0;

  DataType *N = (DataType *) mkl_calloc( n*p,sizeof( DataType ), 64 );
  __assume_aligned(N, 64);
  unsigned long *M = (unsigned long *) mkl_calloc( n*p, sizeof( unsigned long ), 64 );
  __assume_aligned(M, 64);
  DataType* SA =    ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 ); 
  __assume_aligned(SA, 64);
  DataType* AA =    ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 ); 
  __assume_aligned(AA, 64);
  DataType* SAA =   ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 );
  __assume_aligned(SAA, 64);
  DataType* SB =    ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 ); 
  __assume_aligned(SB, 64);
  DataType* BB =    ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 ); 
  __assume_aligned(BB, 64);
  DataType* SBB =   ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 ); 
  __assume_aligned(SBB, 64);
  DataType* SAB =   ( DataType*)mkl_calloc( n*p, sizeof(DataType), 64 );
  __assume_aligned(SAB, 64);
  DataType* UnitA = ( DataType*)mkl_calloc( n*m, sizeof(DataType), 64 );
  __assume_aligned(UnitA, 64);
  DataType* UnitB = ( DataType*)mkl_calloc( m*p, sizeof(DataType), 64 );
  __assume_aligned(UnitB, 64);  
  unsigned long *amask=(unsigned long*)mkl_calloc( n*stride, sizeof(unsigned long), 64);
  __assume_aligned(amask, 64);
  unsigned long *bmask=(unsigned long*)mkl_calloc( p*stride, sizeof(unsigned long), 64);
  __assume_aligned(bmask, 64);
  if (N == NULL | M == NULL | SA == NULL | AA == NULL | SAA == NULL | SB == NULL | BB == NULL | 
      SBB == NULL | SAB == NULL | UnitA == NULL | UnitB == NULL | amask == NULL | bmask == NULL) {
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

  for (int ii=0; ii<count_matrix; ii++) {
    // Deal with missing data: zero the value in 2 places (here and in either sa or sb), adjust sample size


    //If element in A or B has missing data,
    // add a 1 to the bit column k location for row i
    #pragma omp parallel for private (i)
    for (i=0; i< n*stride; i++) { amask[ i ]=0UL; }
    
    #pragma omp parallel for private (j)   
    for (j=0; j< p*stride; j++) { bmask[ j ]=0UL; }
    
    #pragma omp parallel for private (i,k)
    for (i=0; i<n; i++) {
      for (k=0; k<m; k++) {	
	if (A[i*m+k]==MISSING_MARKER) {
	  amask[i*stride +k/64] |= (1UL << (m-k-1)%64);
	}
      }
    }

    #pragma omp parallel for private (j,k)
    for (j=0; j<p; j++) {
      for (k=0; k<m; k++) {	
	if (B[k*p+j]==MISSING_MARKER) {
	  bmask[j*stride +k/64] |= (1UL << (m-k-1)%64);
	}
      }
    }
 
#if 0
    for (i=0; i<n; i++) {
      for (k=0; k<m; k++) {
	printf("amask[%d,%d]=%llu\n",i,k,amask[i*((m-1)/64+1)+k/64]);
      }
    }
    for (j=0; j<p; j++) {
      for (k=0; k<m; k++) {
	printf("bmask[%d,%d]=%llu\n",j,k,bmask[j*((m-1)/64+1)+k/64]);
      }
    }
#endif

    //For all A,B pairs if either A or B has a missing data bit set,
    // a logical OR between row A[i] and column B[j] row bit masks will
    // return a 1 in the bit mask M[i,j]
    #pragma omp parallel for private (i,j,k)
    for (i=0; i<n; i++){
      for (j=0; j<p; j++){
	for(k=0; k<stride; ++k){
	  //M[i*m+j] += popcount64((amask[ i*stride+k ] |
	//			  bmask[ j*stride+k ]));
	  M[i*m+j] += bitsum((amask[ i*stride+k ] | bmask[ j*stride+k ]));
	}
      }
    }
  
    //Compute the number of non missing data for every row/column pair.
    //This is done by subtracting the number of elements in a row by the number of
    // missing data bits set for the row/column pair.
    unsigned long ul_m = m;
    #pragma omp parallel for private(i)
    for(i=0; i<n*p; i++){
      N[i] = ul_m-M[i];
    }
    //Zero out values that are marked as missing
    #pragma omp parallel for private(i)
    for (i=0; i<n*m; i++) {
      if (A[i] == MISSING_MARKER) { A[i]=0.0; }
      else{ UnitA[i]=1; }
    }
    vsSqr(n*m,A,AA);

    //Zero out values that are marked as missing
    #pragma omp parallel for private(j)
    for (j=0; j<m*p; j++) {
      if (B[j] == MISSING_MARKER) { B[j]=0.0; }
      else{ UnitB[j]=1; }
    }
    vsSqr(m*p,B,BB);


    struct timespec startSGEMM, stopSGEMM;
    double accumSGEMM;

    //SA = A*UnitB
    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, p, m, alpha, A, m, UnitB, p, beta, SA, p); 
    clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    printf("SGEMM A*UB m=%d GFLOPs=%e \n",m, (2/1.0e9)*m*m*m/accumSGEMM );

    //SB = UnitA*B
    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, p, m, alpha, UnitA, m, B, p, beta, SB, p); 
    clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    printf("SGEMM UA*B m=%d GFLOPs=%e \n",m, (2/1.0e9)*m*m*m/accumSGEMM );

    //SAA = AA*UnitB
    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, p, m, alpha, AA, m, UnitB, p, beta, SAA, p); 
    clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    printf("SGEMM AA*UB m=%d GFLOPs=%e \n",m, (2/1.0e9)*m*m*m/accumSGEMM );

    //SBB = UnitA*BB
    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, p, m, alpha, UnitA, m, BB, p, beta, SBB, p); 
    clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    printf("SGEMM UA*BB m=%d GFLOPs=%e \n",m, (2/1.0e9)*m*m*m/accumSGEMM );

    mkl_free(UnitA);
    mkl_free(UnitB);
    mkl_free(AA);
    mkl_free(BB);

    //SAB = A*B
    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, p, m, alpha, A, m, B, p, beta, SAB, p); 
    clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    printf("SGEMM A*B m=%d GFLOPs=%e \n",m, (2/1.0e9)*m*m*m/accumSGEMM );

    DataType* NSAB = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 ); 
    DataType* SASB = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 );
    
    DataType* NSAA = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 );
    DataType* SASA = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 ); 
    
    DataType* NSBB = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 );
    DataType* SBSB = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 );    
    
    DataType* DENOM = ( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 );
    DataType* DENOMSqrt =( DataType*)mkl_calloc( n*p,sizeof(DataType), 64 ); 

    //NSAB=N*SAB
    vsMul(n*p,N,SAB,NSAB);
    //SASB=SA*SB
    vsMul(n*p,SA,SB,SASB);
    //NSAB=(-1)SASB+NSAB
    cblas_saxpy(n*p,(DataType)(-1), SASB,1, NSAB,1);
    //element by element multiplication of vector X by Y, return in Z
    vsMul(n*p,N,SAA,NSAA);
    //element by element multiplication of vector X by Y, return in Z
    //vsMul(n*p,SA,SA,SASA);
    vsSqr(n*p,SA,SASA);

    //NSAA=(-1)SASA+NSAA
    cblas_saxpy(n*p,(DataType)(-1), SASA,1, NSAA,1);
    //element by element multiplication of vector X by Y, return in Z
    vsMul(n*p,N,SBB,NSBB);    
    //element by element multiplication of vector X by Y, return in Z
    //vsMul(n*p,SB,SB,SBSB);
    vsSqr(n*p,SB,SBSB);

    //NSBB=(-1)SBSB+NSBB
    cblas_saxpy(n*p,(DataType)(-1), SBSB,1, NSBB,1);
    //element by element multiplication of vector X by Y, return in Z
    vsMul(n*p,NSAA,NSBB,DENOM);
    //element by element sqrt of vector A
    vsSqrt(n*p,DENOM,DENOMSqrt);
    //element by element division of vector X by Y, return in Z
    vsDiv(n*p,NSAB,DENOMSqrt,P);   

    mkl_free(SASA);
    mkl_free(SASB);
    mkl_free(SBSB);
    mkl_free(NSAB);
    mkl_free(NSAA);
    mkl_free(NSBB);
    mkl_free(DENOM);
    mkl_free(DENOMSqrt); 
  }

  mkl_free(N);
  mkl_free(M);
  mkl_free(amask);
  mkl_free(bmask);
  mkl_free(SA);
  //mkl_free(UnitA);
  mkl_free(SAA);
  //mkl_free(AA);
  mkl_free(SB);
  //mkl_free(UnitB);
  mkl_free(SBB);
  //mkl_free(BB);
  mkl_free(SAB);
  
  return 0;
};


int main (int argc, char **argv) {
  //16384 = 1024*16
  //32768 = 2048*16
  //40960 = 2560*16 too large (for skylake)
  //49152 = 3072*16 too large
  //65536 = 4096*16 too large
  //int m=16*100;
  //int m=16*2048;
  int m=16*1500;
  //int m=16*1024;
  int n=m;
  int p=m;
  int count_naive=1;
  int count_matrix=1;
  int seed =1;
   
  if(argc>1){ m = atoi(argv[1]); }
  if(argc>2){ n = atoi(argv[2]); }
  if(argc>3){ p = atoi(argv[3]); }
  if(argc>3){ seed = atoi(argv[4]); }
  if(argc>5){ count_naive = atoi(argv[5]); }
  if(argc>6){ count_matrix = atoi(argv[6]); }
  
  struct timespec startN,stopN;
  struct timespec startM,stopM;

  // A is n x m (tall and skinny) row major order

  // B is m x p (short and fat) row major order

  // C is n x p (big and square) row major order

  DataType* A;
  DataType* B;
  DataType* C; 
  DataType* P;
  DataType accumC,accumP;
  
DEV_CHECKPT
  initialize(n, m, p, seed, &A, &B, &C, &P);
DEV_CHECKPT
#if 0
  clock_gettime(CLOCK_MONOTONIC, &startN);
  pcc_naive(n, m, p, count_naive, A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &stopN);
  accumC =  (TimeSpecToSeconds(&stopN)- TimeSpecToSeconds(&startN))/DataType(count_naive);
#endif
DEV_CHECKPT
#if 1  
  clock_gettime(CLOCK_MONOTONIC, &startM);
  pcc_matrix(n, m, p, count_matrix, A, B, P);
  clock_gettime(CLOCK_MONOTONIC, &stopM);
  //accumP =  (TimeSpecToSeconds(&stopM)- TimeSpecToSeconds(&startM))/DataType(count_matrix);
  accumP =  (TimeSpecToSeconds(&stopM)- TimeSpecToSeconds(&startM));
#endif
DEV_CHECKPT
  DataType diff_2norm=0.0;
  DataType P_2norm=0.0;
  DataType C_2norm=0.0;
#if 0  
  for (int i=0; i<n*p; i++) {
    diff_2norm+=(P[i]-C[i])*(P[i]-C[i]);
    P_2norm+=P[i]*P[i];
    C_2norm+=C[i]*C[i];
  }
  diff_2norm=sqrt(diff_2norm);
DEV_CHECKPT
  
  P_2norm=sqrt(P_2norm);
  C_2norm=sqrt(C_2norm);
#endif
  //printf("NaiveV 2Norm = %lf in %e s\n",C_2norm, accumC);
  printf("MatrixV 2Norm = %lf in %e s m=%d GFLOPs=%e \n",P_2norm, accumP,m, (5*2/1.0e9)*m*m*m/accumP );
  //printf("Diff 2Norm = %lf\n",diff_2norm);

  return 0;

}





















