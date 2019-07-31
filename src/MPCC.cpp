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
//##ifdef USING_R
//using namespace Rcpp;
//#endif

#define __assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#define __assume_aligned(var,size){ __builtin_assume_aligned(var,size); }
#define DEV_CHECKPT printf("Checkpoint: %s, line %d\n", __FILE__, __LINE__); fflush(stdout); 

//#ifndef NAIVE //default use matrix version
//  #define NAIVE 0
//#endif

//#ifndef DOUBLE //default to float type
//  #define DOUBLE 0
//#endif

static DataType TimeSpecToSeconds(struct timespec* ts){
  return (DataType)ts->tv_sec + (DataType)ts->tv_nsec / 1000000000.0;
}

static DataType TimeSpecToNanoSeconds(struct timespec* ts){
  return (DataType)ts->tv_sec*1000000000.0 + (DataType)ts->tv_nsec;
}

// This function convert a string to datatype (double or float);
DataType convert_to_val(string text)
{
    DataType val;
    if(text=="nan" || text=="NaN" || text=="NAN"){ val = NANF;}
    else{ val = atof(text.c_str());}
    return val;
}

#ifndef USING_R
// This function initialized the matrices for m, n, p sized A and the B and result (C) matrices
// Not part of the R interface since R initializes the memory
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
  //DataType val;
  fstream mat_A_file;

  mat_A_file.open(matA_filename,ios::in);
  if(mat_A_file.is_open()){
     // if found then read
     std::getline(mat_A_file, text);
     m = convert_to_val(text);
     std::getline(mat_A_file, text);
     n = convert_to_val(text);
     //printf("m=%d n=%d\n",m,n);
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
     //printf("_n=%d p=%d\n",_n,p);
     mat_B_file.close();
  }

  //check to see if we need to transpose B
  transposeB=false;
  if(_n !=n && n==p){//then transpose matrix B
     p=_n;
     _n=n; 
     transposeB=true; 
     //printf("Transposing B for computational efficiency in GEMMs\n");
     //printf("transposed _n=%d p=%d\n",_n,p);
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
    mat_A_file << m << '\n';
    mat_A_file << n << '\n';
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
    mat_B_file << n << '\n';
    mat_B_file << p << '\n';
    for(int i=0; i<n*p; ++i) mat_B_file << (*B)[i] << '\n';
    mat_B_file.close();
  }
#if 0
  for (int i=0; i<m; i++) { for(int j=0;j<n;++j){printf("A[%d,%d]=%e\n",i,j,(*A)[i*n+j]);}}
  for (int i=0; i<n; i++) { for(int j=0;j<p;++j){printf("B[%d,%d]=%e\n",i,j,(*B)[i*p+j]);}}
#endif
  return;
};
#endif


#ifndef USING_R
int main (int argc, char **argv) {
  //ceb testing with various square matrix sizes
  //32768 = 2048*16
  //40960 = 2560*16 too large (for skylake)
  printf("sizeof(DataType)=%lu\n",sizeof(DataType));
  //set default values 
  int m=40000;//16*1500;//24000^3 for peak performance on skylake
  int n=40000;
  int p=40000;
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
  DataType* diff;
  DataType* C;
  DataType accumR;

  int matsize=10000; 
  //for(int matsize=40; matsize<=40000; matsize*=10){   
  //for(matsize=40000; matsize<=40000; matsize++)
  {   
     printf("testing square matrix size %d\n",matsize);
     m=matsize; n=matsize; p=matsize;
     bool transposeB=false;
     char strbufA[80];
     sprintf(strbufA , "data/matA_%d_%d_%d.dat", m,n,p);
     matA_filename = strbufA;
     char strbufB[80];
     sprintf(strbufB , "data/matB_%d_%d_%d.dat", m,n,p);
     matB_filename = strbufB;
     initialize(m, n, p, seed, &A, &B, &R, matA_filename, matB_filename, transposeB);
     //C = (DataType *)mkl_calloc( m*p,sizeof( DataType ), 64 );
     
     int maxthreads=omp_get_max_threads();
     for(int nthreads=maxthreads; nthreads<=maxthreads; nthreads++){
        printf("nthreads=%d\n",nthreads);
#if 0
        printf("naive PCC implmentation\n");
        pcc_naive(m, n, p, A, B, R, nthreads);
#endif
#if 0
        printf("vector PCC implmentation\n");
        pcc_vector(m, n, p, A, B, R, nthreads);
#endif
#if 1
        printf("matrix PCC implmentation\n");
        pcc_matrix(m, n, p, A, B, R, nthreads);
#endif
     }
  }
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
 
  //DataType R_2norm = 0.0;
  //DataType C_2norm = 0.0;
  //DataType diff_2norm = 0.0;
  //DataType relativeNorm = 0.0;

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

  //printf("completed in %e seconds, size: m=%d n=%d p=%d GFLOPs=%e \n",accumR, m,n,p, (5*2/1.0e9)*m*n*p/accumR);

  return 0;
}
#endif
