//Pearsons correlation coeficient for two vectors x and y
// r = sum_i( x[i]-x_mean[i])*(y[i]-y_mean[i]) ) /
//     [ sqrt( sum_i(x[i]-x_mean[i])^2 ) sqrt(sum_i(y[i]-y_mean[i])^2 ) ]

// Matrix reformulation and algebraic simplification for improved algorithmic efficiency
// where A is matrix of X vectors and B is transposed matrix of Y vectors:
// R = [N sum(AB) - (sumA)(sumB)] /
//     sqrt[ (N sumA^2 - (sum A)^2)[ (N sumB^2 - (sum B)^2) ]

//This code computes correlation coefficient between all row/column pairs of two matrices 
// ./MPCC MatA_filename MatB_filename 

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <mkl.h>
#include <assert.h>
#include <iostream>
#include <fstream>

using namespace std;
#define BINARY_INPUT 0
#define BILLION  1000000000L

#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#define __assume_aligned(var,size){ __builtin_assume_aligned(var,size); }
#define DEV_CHECKPT printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); fflush(stdout); 

#define MISSING_MARKER FP_NAN

#define DataType float

static DataType TimeSpecToSeconds(struct timespec* ts){
  return (DataType)ts->tv_sec + (DataType)ts->tv_nsec / 1000000000.0;
};
static DataType TimeSpecToNanoSeconds(struct timespec* ts){
  return (DataType)ts->tv_sec*1000000000.0 + (DataType)ts->tv_nsec;
};
 
int bitsum(unsigned long n){
  int c=0;
  int nn=n;
  for (c=0; nn; ++c) { nn&=nn-1;}
  return c;
}

void initialize(int &m, int &n, int &p, int seed,
		DataType **A, DataType **B,
		DataType **C,
		char* matA_filename,
 		char* matB_filename,
		bool &transposeB){
  // A is m x n (tall and skinny) row major order
  // B is n x p (short and fat) row major order
  // C, P is m x p (big and square) row major order

  //if matA_filename exists, read in dimensions
  // check for input file(s)
  fstream mat_A_file;
  float tmp;
#if BINARY_INPUT //binary file
  mat_A_file.open(matA_filename,ios::in | ios::binary);
  if(mat_A_file){
     // if found then read
     mat_A_file.read(reinterpret_cast<char*>(&m), sizeof(int));
     mat_A_file.read(reinterpret_cast<char*>(&n), sizeof(int));
     mat_A_file.close();
  }
#else //text file
  mat_A_file.open(matA_filename,ios::in);
  if(mat_A_file){
     // if found then read
     mat_A_file >> tmp;
     m=tmp;
     mat_A_file >> tmp;
     n=tmp;
printf("m=%d n=%d\n",m,n);
     mat_A_file.close();
  }
#endif
  //else use default value for m,n

  *A = (DataType *)mkl_calloc( m*n,sizeof( DataType ), 64 ); 
  if (*A == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix A. Aborting... \n\n");
    mkl_free(*A);
    exit (0);
  }

  //if matB_filename exists, read in dimensions
  // check for input file(s)
  fstream mat_B_file;
  int _n=n;
#if BINARY_INPUT //binary file
  mat_B_file.open(matB_filename,ios::in | ios::binary);
  if(mat_B_file){
     // if found then read
     mat_B_file.read(reinterpret_cast<char*>(&_n), sizeof(int));
     mat_B_file.read(reinterpret_cast<char*>(&p), sizeof(int));
     mat_B_file.close();
  }
#else //text file
  mat_B_file.open(matB_filename,ios::in);
  if(mat_B_file){
     // if found then read
     mat_B_file >> tmp;
     _n=tmp;
     mat_B_file >> tmp;
     p=tmp;
printf("_n=%d p=%d\n",_n,p);
     mat_B_file.close();
  }
#endif

  //check to see if we need to transpose B
  transposeB=false;
  if(_n !=n && n==p){//then transpose matrix B
     p=_n;
     _n=n; 
     transposeB=true; 
     printf("Transposing B for computational efficiency in SGEMMs\n");
     printf("transposed _n=%d p=%d\n",_n,p);
   }

  //check that inner dimension matches
  assert(n==_n);

  //else use default value for n,p
  *B = (DataType *)mkl_calloc( n*p,sizeof( DataType ), 64 );
  if (*B == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix B. Aborting... \n\n");
    mkl_free(*B);
    exit (0);
  }

  printf("m=%d n=%d p=%d\n",m,n,p);

  *C = (DataType *)mkl_calloc( m*p,sizeof( DataType ), 64 ); 
  if (*C == NULL ) {
    printf( "\n ERROR: Can't allocate memory for matrix C. Aborting... \n\n");
    mkl_free(*C);
    exit (0);
  }
  
  __assume_aligned(A, 64);
  __assume_aligned(B, 64);
  __assume_aligned(C, 64);
  __assume(m%16==0);
 

  //setup random numbers to create some synthetic matrices for correlation
  // if input files do not exist
  srand(seed);
  DataType randmax_recip=1/(DataType)RAND_MAX;

  //Input currently hard coded as binary files matrixA.dat and matrixB.dat
  //Or text files input as command line parameters.
  //with the form:
  //int rows, cols 
  //float elems[rows:cols]

  //If input files do not exist, generate synthetic matrices of given dimensions A[m,p] B[p,n] 
  //  with randomly assigned elements from [0,1] and then add missing values 
  //  in selected locations (currently denoted by the value 2) 
  //These matrices are of type single precision floating point values
  
  // check for input file(s)
#if BINARY_INPUT //binary
  mat_A_file.open(matA_filename,ios::in | ios::binary);
  if(mat_A_file){
     // if found then read
     mat_A_file.read(reinterpret_cast<char*>(&m), sizeof(int));
     mat_A_file.read(reinterpret_cast<char*>(&n), sizeof(int));
     mat_A_file.read(reinterpret_cast<char*>(*A), sizeof(DataType)*m*n);
     mat_A_file.close();
  }
#else
  mat_A_file.open(matA_filename,ios::in);
  if(mat_A_file){
     // if found then read
     mat_A_file >> tmp;
     mat_A_file >> tmp;
     for(int i=0;i<m*n;++i) mat_A_file >> (*A)[i];
     mat_A_file.close();
  }
#endif
  else{ //else compute and then write matrix A
    //random assignemnt of threads gives inconsistent values, so keep serial
    int i;
    #pragma omp parallel for private (i)
    for (i=0; i<m*n; i++) {
      (*A)[i]=(DataType)rand()*randmax_recip;
    }
    //add some missing value markers
    //Note edge case: if missing data causes number of pairs compared to be <2, the result is divide by zero
    (*A)[0]          = MISSING_MARKER;
    (*A)[m*n-1]      = MISSING_MARKER;
    (*A)[((m-1)*n-1)]= MISSING_MARKER;

    //write matrix to file
#if BINARY_INPUT //binary   
    mat_A_file.open(matA_filename,ios::out | ios::binary);
    mat_A_file.write(reinterpret_cast<char*>(&m), sizeof(int));
    mat_A_file.write(reinterpret_cast<char*>(&n), sizeof(int));
    mat_A_file.write(reinterpret_cast<char*>(*A), sizeof(DataType)*m*n);   
#else
    mat_A_file.open(matA_filename,ios::out);
    mat_A_file << m;
    mat_A_file << n;
    for(int i=0;i<m*n;++i) mat_A_file << (*A)[i] << '\n';
#endif
    mat_A_file.close();
  }
 
  //ceb Should write  out matrix and read in for future use.
  // check for input file(s)
#if BINARY_INPUT
  mat_B_file.open(matB_filename,ios::in | ios::binary);
  if(mat_B_file){
     // if found then read
     mat_B_file.read(reinterpret_cast<char*>(&n), sizeof(int));
     mat_B_file.read(reinterpret_cast<char*>(&p), sizeof(int));
     mat_B_file.read(reinterpret_cast<char*>(*B), sizeof(DataType)*n*p);
     mat_B_file.close();
  }
#else
  mat_B_file.open(matB_filename,ios::in);
  if(mat_B_file){
     // if found then read
     mat_B_file >> tmp;
     mat_B_file >> tmp;
     for(int i=0; i<n*p; ++i) mat_B_file >> (*B)[i];
     mat_B_file.close();


  }
#endif
  else{ //else compute and then write matrix B
    int i;
    //random assignemnt of threads gives inconsistent values, so keep serial
    #pragma omp parallel for private (i)
    for (i=0; i<n*p; i++) {
      (*B)[i]=(DataType)rand()*randmax_recip;
    }
    //add some missing value markers
    //ceb if missing data causes number of pairs compared to be <2, the result is divide by zero
    (*B)[0]          = MISSING_MARKER;
    (*B)[n*p-1]      = MISSING_MARKER;
    (*B)[((n-1)*p-1)]= MISSING_MARKER;
   
    //write matrix to file
#if BINARY_INPUT
    mat_B_file.open(matB_filename,ios::out | ios::binary);
    mat_B_file.write(reinterpret_cast<char*>(&n), sizeof(int));
    mat_B_file.write(reinterpret_cast<char*>(&p), sizeof(int));
    mat_B_file.write(reinterpret_cast<char*>(*B), sizeof(DataType)*n*p);   
#else
    mat_B_file.open(matB_filename,ios::out);
    mat_B_file << n;
    mat_B_file << p;
    for(int i=0; i<n*p; ++i) mat_B_file << (*B)[i];
#endif
    mat_B_file.close();
  }
#if 0
  for (int i=0; i<m; i++) { for(int j=0;j<n;++j){printf("A[%d,%d]=%e\n",i,j,(*A)[i*n+j]);}}
  for (int i=0; i<n; i++) { for(int j=0;j<p;++j){printf("B[%d,%d]=%e\n",i,j,(*B)[i*p+j]);}}
#endif
  return;
}


//This function is an implementation of a pairwise vector * vector correlation.
//A is matrix of X vectors and B is transposed matrix of Y vectors:
// C = [N sum(AB) - (sumA)(sumB)] /
//     sqrt[ (N sumA^2 - (sum A)^2)[ (N sumB^2 - (sum B)^2) ]
int pcc_naive(int m, int n, int p, int count,
	      DataType* A, DataType* B, DataType* C)
{
  DataType sab,sa,sb,saa,sbb;
  int nn;
  int i,j,k;

  for (int ii=0; ii<count; ii++) {
    #pragma omp parallel for private (i,j,k)
    for (i=0; i<m; i++) {  
      for (j=0; j<p; j++) {

	sa=0.0;
	sb=0.0;
	saa=0.0;
	sbb=0.0;	
      	sab=0.0;
	nn=n;
	
	for (k=0; k<n; k++) {
	  //compute components of PCC function
	  if ((A[i*n+k] != MISSING_MARKER) && (B[k*p+j] != MISSING_MARKER)){   
	    sa  += A[i*n+k];
	    sb  += B[k*p+j];  
	    sab += A[i*n+k] * B[k*p+j];
	    saa += A[i*n+k] * A[i*n+k];
	    sbb += B[k*p+j] * B[k*p+j];
	  }  
	  else //decrement divisor for mean calculation
          { nn--; }	  
	}
	
	if(nn>1){//Note edge case: if nn==1 then denominator is Zero! (saa==sa*sa, sbb==sb*sb)
	  C[i*p+j] = (nn*sab - sa*sb) / sqrt( (nn*saa - sa*sa)*(nn*sbb - sb*sb) );
	}
	else{printf("Error, no correlation possible for rows A[%d], B[%d]\n",i,j); C[i*p+j]=1./0.;}
      }
    }
  }
  return 0;
}

//This function is the implementation of a matrix x matrix algorithm which computes a matrix of PCC values
//but increases the arithmetic intensity of the naive pairwise vector x vector correlation
//A is matrix of X vectors and B is transposed matrix of Y vectors:
//P = [N sum(AB) - (sumA)(sumB)] /
//    sqrt[ (N sumA^2 - (sum A)^2)[ (N sumB^2 - (sum B)^2) ]
int pcc_matrix(int m, int n, int p, int count,
	       DataType* A, DataType* B, bool transposeB, DataType* P)	       
{
  int i,j,k;
  int stride = ((n-1)/64 +1);
  DataType alpha=1.0;
  DataType beta=0.0;

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
    for (i=0; i<m; i++) {
      for (k=0; k<n; k++) {	
	if (A[i*n+k]==MISSING_MARKER) {
	  amask[i*stride +k/64] |= (1UL << (n-k-1)%64);
	}
      }
    }

    //printf("p=%d n=%d\n",p,n);

    //if element in B is missing, flip bit of corresponding col to 1
    #pragma omp parallel for private (j,k)
    for (j=0; j<p; j++) {
      for (k=0; k<n; k++) {	
	if (B[j*n+k]==MISSING_MARKER) {
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
	for(k=0; k<stride; ++k){
	  M[i*p+j] += bitsum((amask[ i*stride+k ] | bmask[ j*stride+k ]));
	}
      }
    }
  
    //Compute the number of non missing data for every row/column pair.
    //This is done by subtracting the number of elements in a row by the number of
    // missing data bits set for the row/column pair.
    unsigned long ul_n = n;
    #pragma omp parallel for private(i)
    for(i=0; i<m*p; i++){
      N[i] = ul_n-M[i];
    }

    //Zero out values that are marked as missing.
    // For subsequent calculations of PCC terms, we need to replace
    // missing value markers with 0 in matrices so that they dont contribute to 
    // the sums
    #pragma omp parallel for private(i)
    for (i=0; i<m*n; i++) {
      if (A[i] == MISSING_MARKER) { A[i]=0.0; }
      else{ UnitA[i]=1; }
    }
    vsSqr(m*n,A,AA);

    //Zero out values that are marked as missing.
    // For subsequent calculations of PCC terms, we need to replace
    // missing value markers with 0 in matrices so that they dont contribute to
    // the sums
    #pragma omp parallel for private(j)
    for (j=0; j<n*p; j++) {
      if (B[j] == MISSING_MARKER) { B[j]=0.0; }
      else{ UnitB[j]=1; }
    }
    vsSqr(n*p,B,BB);

    //variables used for performance timing
    struct timespec startSGEMM, stopSGEMM;
    double accumSGEMM;

    //Compute PCC terms and assemble
     
    //SA = A*UnitB
    //Compute sum of A for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);

    CBLAS_TRANSPOSE transB=CblasNoTrans;
    int ldb=p;
    if(transposeB){
      transB=CblasTrans;
      ldb=n;
    }

    clock_gettime(CLOCK_MONOTONIC, &startSGEMM);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, A, n, UnitB, ldb, beta, SA, p); 

    //clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    //accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    //printf("SGEMM A*UB GFLOPs=%e \n", (2/1.0e9)*m*n*p/accumSGEMM );

    //SB = UnitA*B
    //Compute sum of B for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum

    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, UnitA, n, B, ldb, beta, SB, p); 

    //clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    //accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    //printf("SGEMM UA*B GFLOPs=%e \n", (2/1.0e9)*m*n*p/accumSGEMM );

    //SAA = AA*UnitB
    //Compute sum of AA for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //clock_gettime(CLOCK_MONOTONIC, &startSGEMM);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, AA, n, UnitB, ldb, beta, SAA, p); 

    //clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    //accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    //printf("SGEMM AA*UB GFLOPs=%e \n", (2/1.0e9)*m*n*p/accumSGEMM );

    //SBB = UnitA*BB
    //Compute sum of BB for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //clock_gettime(CLOCK_MONOTONIC, &startSGEMM);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, UnitA, n, BB, ldb, beta, SBB, p); 

    //clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    //accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    //printf("SGEMM UA*BB GFLOPs=%e \n", (2/1.0e9)*m*n*p/accumSGEMM );

    mkl_free(UnitA);
    mkl_free(UnitB);
    mkl_free(AA);
    mkl_free(BB);

    //SAB = A*B
    //clock_gettime(CLOCK_MONOTONIC, &startSGEMM);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, A, n, B, ldb, beta, SAB, p); 

    //clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    //accumSGEMM =  (TimeSpecToSeconds(&stopSGEMM)- TimeSpecToSeconds(&startSGEMM));
    //printf("SGEMM A*B GFLOPs=%e \n", (2/1.0e9)*m*n*p/accumSGEMM );

    clock_gettime(CLOCK_MONOTONIC, &stopSGEMM);
    accumSGEMM =  (TimeSpecToNanoSeconds(&stopSGEMM)- TimeSpecToNanoSeconds(&startSGEMM));
    printf("All(5) SGEMMs (%e)s GFLOPs=%e \n", accumSGEMM, 5*(2/1.0e9)*m*n*p/accumSGEMM);

    DataType* NSAB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 
    DataType* SASB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    
    DataType* NSAA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* SASA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 
    
    DataType* NSBB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* SBSB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );    
    
    DataType* DENOM = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* DENOMSqrt =( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 

    //Compute and assemble composite terms

    //NSAB=N*SAB
    vsMul(m*p,N,SAB,NSAB);

    //SASB=SA*SB
    vsMul(m*p,SA,SB,SASB);

    //NSAB=(-1)SASB+NSAB
    cblas_saxpy(m*p,(DataType)(-1), SASB,1, NSAB,1);
    //element by element multiplication of vector X by Y, return in Z
    vsMul(m*p,N,SAA,NSAA);
    //element by element multiplication of vector X by Y, return in Z
    vsSqr(m*p,SA,SASA);

    //NSAA=(-1)SASA+NSAA
    cblas_saxpy(m*n,(DataType)(-1), SASA,1, NSAA,1);
    //element by element multiplication of vector X by Y, return in Z
    vsMul(m*p,N,SBB,NSBB);    
    //element by element multiplication of vector X by Y, return in Z
    //vsMul(m*p,SB,SB,SBSB);
    vsSqr(m*p,SB,SBSB);

    //NSBB=(-1)SBSB+NSBB
    cblas_saxpy(m*p,(DataType)(-1), SBSB,1, NSBB,1);

    //element by element multiplication of vector X by Y, return in Z
    vsMul(m*p,NSAA,NSBB,DENOM);

    //element by element sqrt of vector DENOM
    vsSqrt(m*p,DENOM,DENOMSqrt);
    //element by element division of vector X by Y, return in Z
    vsDiv(m*p,NSAB,DENOMSqrt,P);   

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
  mkl_free(SAA);
  mkl_free(SB);
  mkl_free(SBB);
  mkl_free(SAB);
  
  return 0;
};


int main (int argc, char **argv) {
  //ceb testing with various square matrix sizes
  //16384 = 1024*16
  //32768 = 2048*16
  //40960 = 2560*16 too large (for skylake)

  //set default values 
  int m=100;//16*1500;//24000^3 for peak performance on skylake
  int n=50;
  int p=75;
  int count=1;
  int seed =1; 
  char* matA_filename="matA.dat";
  char* matB_filename="matB.dat";
 
  if(argc>1){ matA_filename = argv[1]; }
  if(argc>2){ matB_filename = argv[2]; }
  
  struct timespec startPCC,stopPCC;
  // A is n x p (tall and skinny) row major order
  // B is p x m (short and fat) row major order
  // R is n x m (big and square) row major order
  DataType* A;
  DataType* B; 
  DataType* R;
  DataType accumR;
  
  bool transposeB=false;
  initialize(m, n, p, seed, &A, &B, &R, matA_filename, matB_filename, transposeB);
#if 0
  clock_gettime(CLOCK_MONOTONIC, &startPCC);
  pcc_naive(m, n, p, count, A, B, R);
  clock_gettime(CLOCK_MONOTONIC, &stopPCC);
  accumR =  (TimeSpecToSeconds(&stopPCC)- TimeSpecToSeconds(&startPCC));
#else  
  clock_gettime(CLOCK_MONOTONIC, &startPCC);
  pcc_matrix(m, n, p, count, A, B, transposeB, R);
  clock_gettime(CLOCK_MONOTONIC, &stopPCC);
  accumR =  (TimeSpecToSeconds(&stopPCC)- TimeSpecToSeconds(&startPCC));
#endif

  DataType R_2norm=0.0;
#if 0
  for (int i=0; i<n*p; i++) { R_2norm+=R[i]*R[i]; }
  R_2norm=sqrt(R_2norm);
#endif
  printf("Matrix 2Norm = %lf in %e s m=%d n=%d p=%d GFLOPs=%e \n",R_2norm, accumR, m,n,p, (5*2/1.0e9)*m*m*m/accumR);

  return 0;
}

