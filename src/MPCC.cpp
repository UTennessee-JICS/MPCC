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
#define BILLION  1000000000L

#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#define __assume_aligned(var,size){ __builtin_assume_aligned(var,size); }
#define DEV_CHECKPT printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); fflush(stdout); 

//#define NANF std::nanf("1")
#define NANF std::nan("1")
#define MISSING_MARKER NANF

#ifndef NAIVE //default use matrix version
  #define NAIVE 0
#endif

#ifndef DOUBLE //default to float type
  #define DOUBLE 0
#endif

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
};

DataType convert_to_val(string text)
{
    DataType val;
    if(text=="nan" || text=="NaN" || text=="NAN"){ val = NANF;}
    else{ val = atof(text.c_str());}
    return val;
};

#ifndef USING_R

void initialize(int &m, int &n, int &p, int seed,
		DataType **A, 
                DataType **B,
		DataType **C,
		char* matA_filename,
 		char* matB_filename,
		bool &transposeB)
{
  // A is m x n (tall and skinny) row major order
  // B is n x p (short and fat) row major order
  // C, P is m x p (big and square) row major order

  //if matA_filename exists, read in dimensions
  // check for input file(s)
  std::string text;
  DataType val;
  fstream mat_A_file;

  mat_A_file.open(matA_filename,ios::in);
  if(mat_A_file.is_open()){
     // if found then read
     std::getline(mat_A_file, text);
     m = convert_to_val(text);
     std::getline(mat_A_file, text);
     n = convert_to_val(text);
printf("m=%d n=%d\n",m,n);
     mat_A_file.close();
  }
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
  mat_B_file.open(matB_filename,ios::in);
  if(mat_B_file.is_open()){
     // if found then read
     std::getline(mat_B_file, text);
     _n = convert_to_val(text);
     std::getline(mat_B_file, text);
     p = convert_to_val(text);
printf("_n=%d p=%d\n",_n,p);
     mat_B_file.close();
  }

  //check to see if we need to transpose B
  transposeB=false;
  if(_n !=n && n==p){//then transpose matrix B
     p=_n;
     _n=n; 
     transposeB=true; 
     printf("Transposing B for computational efficiency in GEMMs\n");
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
  //__assume(m%16==0);
 

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
  mat_A_file.open(matA_filename,ios::in);
  if(mat_A_file.is_open()){
     // if found then read
     std::getline(mat_A_file, text);
     //m = convert_to_val(text);
     std::getline(mat_A_file, text);
     //n = convert_to_val(text);
     for(int i=0;i<m*n;++i){ 
        std::getline(mat_A_file, text);
        (*A)[i] = convert_to_val(text);
	//if(isnan((*A)[i])){printf("A[%d]==NAN\n",i);} 
     }
     mat_A_file.close();
  }
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
    mat_A_file.open(matA_filename,ios::out);
    mat_A_file << m;
    mat_A_file << n;
    for(int i=0;i<m*n;++i) mat_A_file << (*A)[i] << '\n';
    mat_A_file.close();
  }
 
  //ceb Should write  out matrix and read in for future use.
  // check for input file(s)
  mat_B_file.open(matB_filename,ios::in);
  if(mat_B_file.is_open()){
     std::getline(mat_B_file, text);
     //m = convert_to_val(text);
     std::getline(mat_B_file, text);
     //n = convert_to_val(text);
     for(int i=0;i<n*p;++i){
        std::getline(mat_B_file, text);
        (*B)[i] = convert_to_val(text);
        //if(isnan((*B)[i]) ){printf("B[%d]==NAN\n",i);}
     }
     mat_B_file.close();
  }
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
    mat_B_file.open(matB_filename,ios::out);
    mat_B_file << n;
    mat_B_file << p;
    for(int i=0; i<n*p; ++i) mat_B_file << (*B)[i];
    mat_B_file.close();
  }
#if 0
  for (int i=0; i<m; i++) { for(int j=0;j<n;++j){printf("A[%d,%d]=%e\n",i,j,(*A)[i*n+j]);}}
  for (int i=0; i<n; i++) { for(int j=0;j<p;++j){printf("B[%d,%d]=%e\n",i,j,(*B)[i*p+j]);}}
#endif
  return;
};

#endif

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

//This function is the implementation of a matrix x matrix algorithm which computes a matrix of PCC values
//but increases the arithmetic intensity of the naive pairwise vector x vector correlation
//A is matrix of X vectors and B is transposed matrix of Y vectors:
//P = [ sum(AB) - (sumA)(sumB)/N] /
//    sqrt[ ( sumA^2 -(1/N) (sum A/)^2)[ ( sumB^2 - (1/N)(sum B)^2) ]
int pcc_matrix(int m, int n, int p,
	       DataType* A, DataType* B, DataType* P)	       
{
  int i,j,k;
  int stride = ((n-1)/64 +1);
  DataType alpha=1.0;
  DataType beta=0.0;
  int count =1;
  bool transposeB = true; //assume this is always true. 
  info("before calloc\n",1);
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

  info("after calloc\n",1);

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

  info("before deal missing data\n",1);

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
	if (std::isnan(A[i*n+k])) {
	  amask[i*stride +k/64] |= (1UL << (n-k-1)%64);
	}
      }
    }

    //printf("p=%d n=%d\n",p,n);

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
      N[i] = 1./(ul_n-M[i]);
      if(std::isnan(N[i])) {printf("N[%d]=%e\n",i,N[i]); N[i]=1;}
    }
    info("Call mkl_free\n",1);
    mkl_free(M);
    info("After Call mkl_free\n",1);

    //Zero out values that are marked as missing.
    // For subsequent calculations of PCC terms, we need to replace
    // missing value markers with 0 in matrices so that they dont contribute to 
    // the sums
    #pragma omp parallel for private(i)
    for (i=0; i<m*n; i++) {
      if (std::isnan(A[i])) { A[i]=0.0; }
      else{ UnitA[i]=1; }
    }
    info("VSQR\n",1);
    //vsSqr(m*n,A,AA);
    VSQR(m*n,A,AA);

    info("before zero out\n",1);

    //Zero out values that are marked as missing.
    // For subsequent calculations of PCC terms, we need to replace
    // missing value markers with 0 in matrices so that they dont contribute to
    // the sums
    #pragma omp parallel for private(j)
    for (j=0; j<n*p; j++) {
      if (std::isnan(B[j])) { B[j]=0.0; }
      else{ UnitB[j]=1; }
    }
    //vsSqr(n*p,B,BB);
    VSQR(n*p,B,BB);

    //variables used for performance timing
    struct timespec startGEMM, stopGEMM;
    double accumGEMM;

    info("before PCC terms\n",1);

    //Compute PCC terms and assemble
     

    CBLAS_TRANSPOSE transB=CblasNoTrans;
    int ldb=p;
    if(transposeB){
      transB=CblasTrans;
      ldb=n;
    }

    clock_gettime(CLOCK_MONOTONIC, &startGEMM);
    
    //SA = A*UnitB
    //Compute sum of A for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, A, n, UnitB, ldb, beta, SA, p); 

    //SB = UnitA*B
    //Compute sum of B for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, UnitA, n, B, ldb, beta, SB, p); 


    //SAA = AA*UnitB
    //Compute sum of AA for each AB row col pair.
    // This requires multiplication with a UnitB matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, AA, n, UnitB, ldb, beta, SAA, p); 

    //SBB = UnitA*BB
    //Compute sum of BB for each AB row col pair.
    // This requires multiplication with a UnitA matrix which acts as a mask 
    // to prevent missing data in AB pairs from contributing to the sum
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, UnitA, n, BB, ldb, beta, SBB, p); 

    mkl_free(UnitA);
    mkl_free(UnitB);
    mkl_free(AA);
    mkl_free(BB);

    //SAB = A*B
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, transB,
    GEMM(CblasRowMajor, CblasNoTrans, transB,
		m, p, n, alpha, A, n, B, ldb, beta, SAB, p); 

    clock_gettime(CLOCK_MONOTONIC, &stopGEMM);
    accumGEMM =  (TimeSpecToSeconds(&stopGEMM)- TimeSpecToSeconds(&startGEMM));
    //printf("All(5) GEMMs (%e)s GFLOPs=%e \n", accumGEMM, 5*(2/1.0e9)*m*n*p/accumGEMM);

    DataType* SASB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* NSASB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); //ceb
   
    DataType* SASA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 
    DataType* NSASA = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); //ceb
    
    DataType* SBSB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );    
    DataType* NSBSB = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); //ceb   
    
    DataType* DENOM = ( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 );
    DataType* DENOMSqrt =( DataType*)mkl_calloc( m*p,sizeof(DataType), 64 ); 

    //Compute and assemble composite terms

    //SASB=SA*SB
    //vsMul(m*p,SA,SB,SASB);
    VMUL(m*p,SA,SB,SASB);
    //N*SASB
    //vsMul(m*p,N,SASB,NSASB); //ceb
    VMUL(m*p,N,SASB,NSASB); //ceb

    //SAB=(-1)NSASB+SAB  (numerator)
    //cblas_saxpy(m*p,(DataType)(-1), NSASB,1, SAB,1); //ceb
    AXPY(m*p,(DataType)(-1), NSASB,1, SAB,1); //ceb

    //(SA)^2
    //vsSqr(m*p,SA,SASA);
    VSQR(m*p,SA,SASA);
    //N(SA)^2
    //vsMul(m*p,N,SASA,NSASA); //ceb
    VMUL(m*p,N,SASA,NSASA); //ceb
    //SAA=SAA-NSASA (denominator term 1)
    //cblas_saxpy(m*p,(DataType)(-1), NSASA,1, SAA,1);
    AXPY(m*p,(DataType)(-1), NSASA,1, SAA,1);

    //(SB)^2
    //vsSqr(m*p,SB,SBSB);
    VSQR(m*p,SB,SBSB);
    //N(SB)^2
    //vsMul(m*p,N,SBSB,NSBSB);
    VMUL(m*p,N,SBSB,NSBSB);
    //SBB=SBB-NSBSB
    //cblas_saxpy(m*p,(DataType)(-1), NSBSB,1, SBB,1);
    AXPY(m*p,(DataType)(-1), NSBSB,1, SBB,1);

    //DENOM=SAA*SBB (element wise multiplication)
    //vsMul(m*p,SAA,SBB,DENOM);
    VMUL(m*p,SAA,SBB,DENOM);
    for(int i=0;i<m*p;++i){
       if(DENOM[i]==0.){DENOM[i]=1;}//numerator will be 0 so to prevent inf, set denom to 1
    }
    //sqrt(DENOM)
    //vsSqrt(m*p,DENOM,DENOMSqrt);
    VSQRT(m*p,DENOM,DENOMSqrt);
    //P=SAB/DENOMSqrt (element wise division)
    //vsDiv(m*p,SAB,DENOMSqrt,P);   
    VDIV(m*p,SAB,DENOMSqrt,P);   

    mkl_free(SASA);
    mkl_free(SASB);
    mkl_free(SBSB);
    mkl_free(NSASB);
    mkl_free(NSASA);
    mkl_free(NSBSB);
    mkl_free(DENOM);
    mkl_free(DENOMSqrt); 
  }

  mkl_free(N);
  mkl_free(SA);
  mkl_free(SAA);
  mkl_free(SB);
  mkl_free(SBB);
  mkl_free(SAB);

  return 0;
};

#ifndef USING_R

int main (int argc, char **argv) {
  //ceb testing with various square matrix sizes
  //16384 = 1024*16
  //32768 = 2048*16
  //40960 = 2560*16 too large (for skylake)

  //set default values 
  int m=64;//16*1500;//24000^3 for peak performance on skylake
  int n=16;
  int p=32;
  int count=1;
  int seed =1; 
  char* matA_filename;//="matA.dat";
  char* matB_filename;//="matB.dat";
 
  if(argc>1){ matA_filename = argv[1]; }
  if(argc>2){ matB_filename = argv[2]; }
  
  struct timespec startPCC,stopPCC;
  // A is n x p (tall and skinny) row major order
  // B is p x m (short and fat) row major order
  // R is n x m (big and square) row major order
  DataType* A;
  DataType* B; 
  DataType* R;
  DataType* diff;
  DataType* C;
  DataType accumR;
   
  bool transposeB=false;
  initialize(m, n, p, seed, &A, &B, &R, matA_filename, matB_filename, transposeB);
  //C = (DataType *)mkl_calloc( m*p,sizeof( DataType ), 64 );
  clock_gettime(CLOCK_MONOTONIC, &startPCC);
#if NAIVE
  printf("naive PCC implmentation\n");
  //pcc_naive(m, n, p, count, A, B, C);
  pcc_naive(m, n, p, A, B, R);
#else  
  printf("matrix PCC implmentation\n");
  //pcc_matrix(m, n, p, count, A, B, transposeB, R);
  pcc_matrix(m, n, p, A, B, R);
#endif
  clock_gettime(CLOCK_MONOTONIC, &stopPCC);
  accumR =  (TimeSpecToSeconds(&stopPCC)- TimeSpecToSeconds(&startPCC));



#if 0
  //read in results file for comparison
  fstream test_file;
  //test_file.open("results_6k_x_29k_values.txt",ios::in);
  test_file.open("6kvs28k.txt",ios::in);
  //test_file.open("flat.txt",ios::in);
  if(test_file.is_open()){
     float tmp;
     // if found then read
     int dim1,dim2;
     test_file >> tmp; 
     dim1 = tmp;
     test_file >> tmp;
     dim2 = tmp;
     printf("dim1=%d dim2=%d dim1*dim2=%d\n",dim1,dim2,dim1*dim2);
     C = (DataType *)mkl_calloc( dim1*dim2,sizeof( DataType ), 64 );
     for(int i=0;i<dim1*dim2;++i) test_file >> C[i];
     test_file.close();
  }
#endif 

#if 0
    //write R matrix to file
    fstream mat_R_file;
    mat_R_file.open("MPCC_computed.txt",ios::out);
    mat_R_file << m << '\n';
    mat_R_file << p << '\n';
    for(int i=0;i<m*p;++i) mat_R_file << R[i] << '\n';
    mat_R_file.close();
#endif
 
  DataType R_2norm = 0.0;
  DataType C_2norm = 0.0;
  DataType diff_2norm = 0.0;
  DataType relativeNorm = 0.0;

#if 0
  for (int i=0; i<m*p; i++) { C_2norm += C[i]*C[i]; }
  C_2norm=sqrt(C_2norm);
  for (int i=0; i<m*p; i++) { R_2norm += R[i]*R[i]; }
  R_2norm=sqrt(R_2norm);
  diff = (DataType *)mkl_calloc( m*p,sizeof( DataType ), 64 );
  for (int i=0; i<m*p; i++) { 
     diff[i]=pow(C[i]-R[i],2);
     diff_2norm += diff[i]; 
  }

  diff_2norm = sqrt(diff_2norm);
  relativeNorm = diff_2norm/R_2norm;
  printf("R_2Norm=%e, C_2Norm=%e, diff_2norm=%e relativeNorm=%e\n", R_2norm, C_2norm, diff_2norm, relativeNorm);
  printf("relative diff_2Norm = %e in %e s m=%d n=%d p=%d GFLOPs=%e \n", relativeNorm, accumR, m,n,p, (5*2/1.0e9)*m*n*p/accumR);
#endif


#if 0
    //write R matrix to file
    fstream diff_file;
    diff_file.open("diff.txt",ios::out);
    diff_file << m << '\n';
    diff_file << p << '\n';
    for(int i=0;i<m*p;++i) diff_file << R[i] << " " << C[i] << " " <<diff[i] << '\n';
    diff_file.close();
#endif

  printf("completed in %e seconds, size: m=%d n=%d p=%d GFLOPs=%e \n",accumR, m,n,p, (5*2/1.0e9)*m*n*p/accumR);

  return 0;
}

#endif

