#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "cs160validate.h"

#define ARGS 3 
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define uSECtoSEC 1.0E-6
#define THRESH 1e-14
#define SCALE 100.0
#define TRUE 1
#define FALSE 0

// Macro to define index into a linear array for 2D indexing. Stored 
// row by row.
#define IDX(i,j,n) ((i*n)+j)

/* return a clock value with usec precision */
double get_clock() {
    struct timeval tv;
    int status;
    status = gettimeofday(&tv, (void *) 0);
    if(status<0)
        printf("gettimeofday error");
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

void printMatrix(double *A, int N)
{
	int i,j;
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++)
			printf("%lf ", A[IDX(i,j,N)]);
		printf("\n");
	}
}

/* Multiply A*A^T.  
 * double *result - result matrix
 * double * A - source matrix
 * int N - size of matrix (N x N);
 * int lowerT - A is lower triangular
 */
int multT(double *result, double *A, int N, int lowerT, int threadCount)
{
	int i,j,k;
	bzero(result, N*N*sizeof(double));

	int blocksize = (int) N *  0.1;
	if (blocksize == 0) 
		blocksize = 1;
	# pragma omp parallel for schedule(dynamic, blocksize) num_threads(threadCount)
	for(i = 0; i < N; i++)
	{
		/* Result is symmetric, just compute upper triangle */
		for(j = i; j < N; j++) 
		{
			double sum = 0.0;
			/* if A is lower Triangular don't multiply zeroes */
			for(k = 0; k < (!lowerT ? N : j+1); k++)
			{
				sum += A[IDX(i,k,N)] * A[IDX(j,k,N)];
			}
			result[IDX(i,j,N)] = sum;
			result[IDX(j,i,N)] = sum; /*enforce symmetry */
		}
	}
	//printf("PRINT OUT RESULT IN MULT\n\n");
	//printMatrix(result,N);
	return 0;
}
	
/* Validate that A ~= L*L^T 
 * double * A -  NxN symmetric positive definite matrix
 * double * L -  NxN lower triangular matrix such that L*L^T ~= A
 * int N      -  size of matrices
 * thresh     -  threshold considered to be zero (1e-14 good number)
 *
 * Returns # of elements where the residual abs(A(i,j) - LL^T(i,j)) > thresh
*/
int validate(double *A, double * L, int N, double thresh, int threadCount)
{
	double *R = malloc(N*N * sizeof(double));
	multT(R,L,N,TRUE, threadCount);
	int badcount = 0;
	int i,j;
	double rdiff; /* relative difference */
	
	int blocksize = (int) N * N * 0.1;
	if (blocksize == 0)
		blocksize = 1;
	# pragma omp parallel for collapse(2) schedule(dynamic, blocksize) num_threads(threadCount)
	for (i = 0 ; i < N; i++)
	{
		for(j = 0; j < N; j ++)
		{
			rdiff = fabs((R[IDX(i,j,N)] - A[IDX(i,j,N)])/A[IDX(i,j,N)]);
			if (rdiff > thresh)
			{
				printf("(%d,%d):R(i,j):%f,A(i,j):%f (delta: %f)\n",
					i,j,R[IDX(i,j,N)],A[IDX(i,j,N)],
					rdiff);

				badcount++; 
			}
		}
	}

	free(R);
	return badcount;
}
/* Initialize the N X N  array with Random numbers
 * In such a way that the resulting matrix in Symmetric
 * Positive definite (hence Cholesky factorization is valid)
 * args:
 * 	int N - size of the array
 * 	int trueRandom  - nonzero, seed with time of day, 0 don't seed
 *	double **A - N x N double array, allocated by caller
*/
void init_array(int N, int trueRandom, double *A, int threadCount) {
	//int i,j,k;
	int i, j;
	struct drand48_data rbuf;
	if (trueRandom)
		srand48_r((long int) time(NULL),&rbuf);
	else
		srand48_r(1L,&rbuf);
	
	double *B = calloc(N * N, sizeof(double));

	int blocksize = (int) N  * 0.1;
	if (blocksize == 0) 
		blocksize = 1;
	# pragma omp parallel for collapse(2) schedule(dynamic, blocksize) num_threads(threadCount)
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++) 
		{
			drand48_r(&rbuf,&B[IDX(i,j,N)]);
			B[IDX(i,j,N)] *= SCALE;
			//B[IDX(i, j, N)] = i * j + 3 * j -1;
		}
	}

	/* Compute B*B^T to get symmetric, positive definite*/
	multT(A,B,N,0, threadCount);
	free (B);
}

/* Compute the Cholesky Decomposition of A 
 * L - NxN result matrix, Lower Triangular L*L^T = A
 * A - NxN symmetric, positive definite matrix A
 * N - size of matrices;
 */
void cholesky(double *L, double *A, int N, int threadCount)
{
	int i,j,k;
	bzero(L,N*N*sizeof(double));
	double temp;
	int blocksize = (int) N * 0.1;
	if (blocksize == 0) 
		blocksize = 1;
	# pragma omp parallel for private(temp, i, j, k) schedule(dynamic, blocksize) num_threads(threadCount)

	for (i = 0; i < N; i++){
		for (j = 0; j < (i+1); j++) {
			temp = 0;
			/* Inner product of ith row of L, jth row of L */
			for (k = 0; k < j; k++)
				temp += L[IDX(i,k,N)] * L[IDX(j,k,N)];
			if (i == j)
				L[IDX(i,j,N)] = sqrt(A[IDX(i,i,N)] - temp);
			else {
				L[IDX(i,j,N)] = (A[IDX(i,j,N)] - temp)/ L[IDX(j,j,N)];
			}
	  	}
	}
}


int main(int argc, char* argv[]) {

	int n;
	int threadCount;
	//int i,j,k;
	//double ts, te; /* Starting time and ending time */
	double tis, tie;
	double tvs, tve;
	double tcs, tce;
	double *A, *L;
	//double temp;
	if(argc < ARGS) 
	{
		fprintf(stderr,"Wrong # of arguments.\nUsage: %s array_size thread_count\n",argv[0]);
		return -1;
	}
	n = atoi(argv[1]);
	threadCount = atoi(argv[2]);
	A = (double *)malloc(n*n*sizeof(double));
	L = (double *)calloc(n*n,sizeof(double));
	printf("Initializing \n");
	tis = get_clock();
	init_array(n,0,A, threadCount);
	tie = get_clock();
	//printf("Initial matrix:\n");
	//printMatrix(A,n);
	printf("Computing the Cholesky Factorization of random %dX%d Matrix\n",n,n);

	tcs = get_clock();
	/*Serial decomposition*/
	cholesky(L,A,n, threadCount);
	tce = get_clock();
	//printf("Decomposed matrix:\n");
	//printMatrix(L,n);
	printf("Validating\n");
	tvs = get_clock();
	int badcount = validate(A,L,n,THRESH, threadCount);
	tve = get_clock();
	printf("Elapsed time:\n\tinitialization=%e\n", tie - tis);
	printf("\tcholesky=%e\n", tce - tcs);
	printf("\tvalidate=%e\n", tve - tvs);
	if (badcount == 0)
		printf("solution validates\n");
	else
		printf("solution is invalid, %d elements above threshold\n",badcount);
	cs160validate(A,L,n,THRESH);
	free(A);
	free(L);
	return badcount;
}
