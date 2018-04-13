#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "blockCholesky.h"
#include "cs160validate.h"
#include <omp.h>
#define ARGS 3 
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define uSECtoSEC 1.0E-6
#define THRESH 1e-13
#define SCALE 100.0
#define TRUE 1
#define FALSE 0

// Macro to define index into a linear array for 2D indexing. Stored 
// row by row.
#define IDX(i,j,n) ((i*n)+j)

//global variables to keep track of timing
double tis, tie;
double tvs, tve;
double tcs, tce;
int badcount;
int nthreads;
/*
 * Returns the current clock reading
 */
double get_clock() {
	struct timeval tv;
	int status;
	status = gettimeofday(&tv, (void*) 0);
	if(status<0)
		printf("gettimeofday error");
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


/*
 * Print the contents of a matrix
 *
 * Parameters: A - the matrix to be printed
 *             N - the number of row/col in matrix A
 *
 */
void printMatrix(double *A, int N) {
	int i,j;
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			printf("%lf ", A[IDX(i, j, N)]);
		printf("\n");
	}
}


/*
 * Initialize a random (blocksize * kblocks) x (blocksize * kblocks) matrix
 * that is symmetric and positive definite
 *
 * Parameters: A         - the array to return
 *             blocksize - the number of row/col in each block of blockstruct
 * 	       kblocks   - the number of row/col blocks in the blockstruct
 *
 */
int init_array(double *A, int blocksize, int kblocks) {
	int i, j;
	int N = blocksize * kblocks;
	struct drand48_data rbuf;

	double *B = calloc(N * N,sizeof(double));
	
	srand48_r(1L,&rbuf);
	# pragma omp parallel for collapse(2) schedule(dynamic,blocksize) num_threads(nthreads)
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			drand48_r(&rbuf, &B[IDX(i, j, N)]);
			B[IDX(i, j, N)] *= SCALE;
		}
	}

	multT(A, B, N, 0);

	free(B);
	return 0;

}


/*
 * Multiply an N x N matrix with its transpose, A * AT
 *
 * Parameters: result - result of the matrix multiplication
 *             A      - the matrix
 *             N      - width and height of matrix A
 *
 */
int multT(double *result, double *A, int N, int lowerT){
	int i, j, k;
	bzero(result, N * N * sizeof(double));
	int blocksize = (int) N * 0.1;
	if (blocksize == 0)
		blocksize = 1;
	# pragma omp parallel for schedule(dynamic, blocksize) num_threads(nthreads)
	for(i = 0; i < N; i++) {
		for(j = i; j < N; j++) {
			double sum = 0.0;
			for(k = 0; k < (!lowerT ? N : (j + 1)); k++) {
				sum += A[IDX(i, k, N)] * A[IDX(j, k, N)];
			}
			result[IDX(i, j, N)] = sum;
			result[IDX(j, i, N)] = sum;
		}
	}	
	return 0;
}


/*
 * Multiply a non square matrix with a non square matrix
 *
 * Parameters: result - result of the matrix multiplication
 *             A      - the left side matrix
 *             B      - the right side matrix
 *             width  - width of matrix A and the height of matrix B
 *             height - height of matrix A and the width of matrix B
 *
 */
int multTNonSquare(double *result, double *A, double *B, int width, int height, int lowerT){
	int i, j, k;
	bzero(result, width * height * sizeof(double));
	
	# pragma omp parallel for private(i,j) schedule(dynamic) num_threads(nthreads)
	for(i = 0; i < height; i++) {
		for(j = i; j < height; j++) {
			double sum = 0.0;
			for (k = 0; k < (!lowerT ? width : (j + 1)); k++) {
				sum += A[IDX(i, k, width)] * B[IDX(j, k, width)];
			}
			result[IDX(i, j, height)] = sum;
			result[IDX(j, i, height)] = sum;
		}
	}
	return 0;
}


/*
 * Computes the block cholesky factorization by initialization a random kblocks * blocksize
 * matrix, performs the factorization, and then validates the result
 *
 * Parameters: kblocks   - the number of row/col blocks in the blockstruct
 *             blocksize - the number of row/col in each block of blockstruct
 *
 */
void blockCholesky(int kblocks, int blocksize){
	int i, j, a, b;	

	blockstruct *** A = (blockstruct ***) malloc(kblocks * sizeof(blockstruct **));
	blockstruct *** L = (blockstruct ***) malloc(kblocks * sizeof(blockstruct **));
	blockstruct *** Amaster = (blockstruct ***) malloc(kblocks * sizeof(blockstruct **));

	for(i = 0; i < kblocks; i++) {
		A[i] =(blockstruct **) malloc(kblocks * sizeof(blockstruct *));
		L[i] =(blockstruct **) malloc(kblocks * sizeof(blockstruct *));
		Amaster[i] = (blockstruct **) malloc(kblocks * sizeof(blockstruct *));
	}
	
	/* Initilize all the structs so we can use 2D reference*/ 
	for( i = 0; i < kblocks; i++) {
		for(j = 0; j < kblocks; j++) {
			blockstruct * temp =(blockstruct *) malloc(sizeof(blockstruct));
			temp->block = (double *) malloc(blocksize * blocksize * sizeof(double));
			temp->row = i;
			temp->col = j;
			A[i][j] = temp;

			blockstruct * temp1 =(blockstruct *) malloc(sizeof(blockstruct));
			temp1->block = (double *) malloc(blocksize * blocksize * sizeof(double));
			temp1->row = i;
			temp1->col = j;
			L[i][j] = temp1;
		
			blockstruct * temp2 =(blockstruct *) malloc(sizeof(blockstruct));
                        temp2->block = (double *) malloc(blocksize * blocksize * sizeof(double));
                        temp2->row = i;
                        temp2->col = j;
                        Amaster[i][j] = temp2;
		}
	}

	/*create random numbers for the blocks and assign them to their semetric counterparts*/
	double *temp = calloc(kblocks * kblocks * blocksize * blocksize,sizeof(double));

	/* Initialization timing start */
	tis = get_clock();
	printf("Initializing...\n");
	init_array(temp, blocksize, kblocks);
	
	/* Initialization timing end */
        tie = get_clock();
	
	/* Copy over the random positive definiite temp array into our A, L, and Amaster structures block by block */
	#pragma omp parallel for num_threads(nthreads) schedule(static,(8)) shared(temp) collapse(4)
	for(i = 0; i < kblocks; i++) {
		for(j = 0; j < kblocks; j++) {
			for(a = 0; a < blocksize; a++) {
				for(b = 0; b < blocksize; b++) {
					A[i][j] -> block[IDX(a, b, blocksize)] = temp[IDX(((i * blocksize) + a), ((j * blocksize) + b), (blocksize * kblocks))];
					L[i][j] -> block[IDX(a, b, blocksize)] = temp[IDX(((i * blocksize) + a), ((j * blocksize) + b), (blocksize * kblocks))];
					Amaster[i][j] -> block[IDX(a, b, blocksize)] = temp[IDX(((i * blocksize) + a), ((j * blocksize) + b), (blocksize * kblocks))];

				}	
			}
		}
	}


	/* blockCholesky factorization timing start */	
	tcs = get_clock();
	printf("Computing the block Cholesky Factorization of random %dX%d block, %dX%d blocksize matrix\n", kblocks, kblocks, blocksize, blocksize);
	int bs = blocksize*.1;
	if(bs == 0)
		bs = 1;

	j = 0;
	//# pragma omp parallel for schedule(dynamic) num_threads(nthreads)
	for (i = 0; i < kblocks; i++) {
		//printf("L11: iteration %d\n", i);
		calcL11(blocksize, A[i][j] -> block, L[i][j] -> block);
	
		//printf("L21: iteration %d\n", i);
		calcL21(i + 1, j, kblocks, blocksize, A, L);

		//printf("L22: iteration %d\n", i);
		calcL22(i + 1, j + 1, kblocks, blocksize, A, L);
		j++;
        }

	/* blockCholesky factorization timing end */
        tce = get_clock();
	
	/* Zero out upper triangle of L */
	for (i = 0; i < kblocks; i++) {
		for (j = 0; j < kblocks; j++) {
			if (j > i) {
				for (a = 0; a < blocksize; a++) {
					for (b = 0; b < blocksize; b++) {
			 			L[i][j] -> block[IDX(a, b, blocksize)] = 0.0;
					}
				}
			}
		}
	}
	
	/* Convert blockstruct to array */
	double *Alinear = calloc(blocksize * kblocks * blocksize * kblocks, sizeof(double));
	double *Llinear = calloc(blocksize * kblocks * blocksize * kblocks, sizeof(double));
	convertStructToPointer(Amaster, Alinear, blocksize, kblocks);
	convertStructToPointer(L, Llinear, blocksize, kblocks);

	/* Validation timing start */
	tvs = get_clock();
	printf("Validating block cholesky factorization...\n");
	badcount = validate(Alinear, Llinear, blocksize, kblocks, THRESH);
	cs160validate(Alinear, Llinear, (blocksize * kblocks), THRESH);

	/* Validation timing end */
	tve = get_clock();

	free(temp);
	free(Alinear);
	free(Llinear);

	for (i = 0; i < kblocks; i++) {
		for (j = 0; j < kblocks; j++) {
			free(A[i][j] -> block);
			free(L[i][j] -> block);
			free(Amaster[i][j] -> block);

			free(A[i][j]);
			free(L[i][j]);
			free(Amaster[i][j]);
		}
	}

	for (i = 0; i < kblocks; i++) {
		free(A[i]);
		free(L[i]);
		free(Amaster[i]);
	}

	free(A);
	free(L);
	free(Amaster);
}


/*
 * Convert a blockstruct to a single array pointer
 *
 * Parameters: A         - the blockstruct you want to convert
 *             linear    - the array to store the conversion in
 *             blocksize - the size of each row/col block in blockstruct
 *             kblocks   - the number of blocks in blockstruct row/col
 *
 */
void convertStructToPointer(blockstruct ***A, double *linear, int blocksize, int kblocks) {
	int i, j, a, b;
	int bs = blocksize * .1;
	if (bs == 0)
		bs = 1;
	# pragma omp parallel for collapse(4) schedule(dynamic, bs) num_threads(nthreads)
	for (i = 0; i < kblocks; i++) {
		for (j = 0; j < kblocks; j++) {
			for (a = 0; a < blocksize; a++) {
				for (b = 0; b < blocksize; b++) {
					linear[IDX(((i * blocksize) + a), ((j * blocksize) + b), blocksize * kblocks)] = A[i][j] -> block[IDX(a, b, blocksize)]; 
				}
			}
		}
	}
}


/*
 * Transpose a matrix L
 *
 * Parameters: L      - the matrix you want to transpose
 *             width  - the width of array L
 *             height - the height of array L
 *
 */
void transpose(double *L, double *LT, int width, int height) {
	int i, j;

	# pragma omp parallel for collapse(2) schedule(dynamic,height) num_threads(nthreads)
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			LT[IDX(j, i, height)] = L[IDX(i, j, width)];
		}
	}
}


/* 
 * Compute the serial Cholesky Decomposition of A 
 * 
 * Parameters: L - NxN result matrix, Lower Triangular L*L^T = A
 *             A - NxN symmetric, positive definite matrix A
 *             N - size of matrices;
 *
 */
void cholesky(double *L, double *A, int N) {
        int i, j, k;
        bzero(L, N * N * sizeof(double));
        double temp;
	int blocksize = (int) N * 0.1;
	if( blocksize == 0)
		blocksize =1;
	# pragma omp parallel for private(temp, i, j, k) schedule(dynamic, blocksize) num_threads(nthreads)
        for (i = 0; i < N; i++) {
                for (j = 0; j < (i + 1); j++) {
                        temp = 0;
                        /* Inner product of ith row of L, jth row of L */
                        for (k = 0; k < j; k++)
                                temp += L[IDX(i, k, N)] * L[IDX(j, k, N)];
                        if (i == j)
                                L[IDX(i, j, N)] = sqrt(A[IDX(i, i, N)] - temp);
                        else {
                                L[IDX(i, j, N)] = (A[IDX(i, j, N)] - temp)/ L[IDX(j, j, N)];
                        }
                }
        }
}


/* 
 * Validate that A ~= L*L^T 
 * 
 * Parameters: A      - NxN symmetric positive definite matrix
 *             L      - NxN lower triangular matrix such that L*L^T ~= A
 *             N      - size of matrices
 *             thresh - threshold considered to be zero (1e-14 good number)
 *
 * Returns # of elements where the residual abs(A(i,j) - LL^T(i,j)) > thresh
 *
 */
int validate(double *A, double *L, int blocksize, int kblocks, double thresh) {
	int N = blocksize * kblocks;
        double *R = malloc(N * N * sizeof(double));

	/* Multiply L with LT to get a result R, which should equal A, 
           L is lower triangular */
        multT(R, L, N, 1);

	/*printf("MATRIX A:\n");
	printMatrix(A, N);
	printf("MATRIX L:\n");
	printMatrix(L, N);
	printf("MATRIX R: \n");
	printMatrix(R, N);*/
	
        int badcount = 0;
        int i, j;
        double rdiff; /* relative difference */
	# pragma omp parallel for collapse(2) schedule(dynamic, blocksize) num_threads(nthreads)
        for (i = 0 ; i < N; i++) {
                for(j = 0; j < N; j ++) {
                        rdiff = fabs((R[((N * i) + j)] - A[((N * i) + j)]) / A[((N * i) + j)]);
                        if (rdiff > thresh) {
                                printf("(%d,%d):R(i,j):%f,A(i,j):%f (delta: %f)\n", i, j, R[((N * i) + j)], A[((N * i) + j)], rdiff);
                                
				badcount++;
                        }
                }
        }

	free(R);

	return badcount;
}


/*
 * Calculate L11 in the block cholesky algorithm
 * by doing the serial cholesky calculation
 *
 * Parameters: blocksize - the dimension of each block,
 *             A         - the block along the diagonal
 *
 */
void calcL11(int blocksize, double *A, double *L) {
	cholesky(L, A, blocksize);
}


/*
 * Calculate L21 in the block cholesky algorithm
 *
 * Parameters: i         - the row index for start of A21
 *             j         - the col index for the start of A21
 *             kblocks   - the last index for A21
 *             blocksize - the dimensions of each block
 *             A         - the symmetric positive definite matrix
 *             L         - the cholesky factorization matrix
 *
 */
void calcL21(int i, int j, int kblocks, int blocksize, blockstruct ***A, blockstruct ***L) {
	int k, a, b, c;
	int count = 0;

	double total;
	double *A21, *L21;
	double *L11 = L[i - 1][j] -> block;
	double *L11T = malloc(blocksize * blocksize * sizeof(double));

	/* Compute L11 transpose from L11 */
	transpose(L11, L11T, blocksize, blocksize);

	/* Solve for L21 by backsubstitution using A21 and L11T */
	int bs = blocksize * 0.1;
	if(bs == 0)
		bs = 1;
	# pragma omp parallel for private(total,k,a,b,c) schedule(dynamic, bs) num_threads(nthreads)
	for (k = i; k < kblocks; k++) {
		A21 = A[k][j] -> block;
		L21 = L[k][j] -> block;
		for (a = 0; a < blocksize; a++) {
			count = 0;
			for (b = 0; b < blocksize; b++) {
				total = A21[IDX(a, b, blocksize)];
				for (c = 0; c < count; c++) {
					total -= L21[IDX(a, (b - count + c), blocksize)] * L11T[IDX((b - count + c), b, blocksize)];
				}
				L21[IDX(a, b, blocksize)] = total / L11T[IDX(b, b, blocksize)];
				count++;
			}
		}
	}

	free(L11T);
}


/*
 * Calculate L22 in the block cholesky algorithm
 *
 * Parameters: i         - the row index for the start of A22
 *             j         - the col index for the start of A22
 *             kblocks   - the last index for A22
 *             blocksize - the dimensions of each block
 *             A         - the symmetric positive definite matrix
 *             L         - the cholesky factorization matrix
 *
 */
void calcL22(int i, int j, int kblocks, int blocksize, blockstruct ***A, blockstruct ***L) {
	int k, l;
	int x, y;
	int a, b;
	double *A22;
	double *temp;
	double *L21 = malloc((blocksize * (kblocks - i) * blocksize) * sizeof(double));
	double *L21T = malloc((blocksize * (kblocks - i) * blocksize) * sizeof(double));
	double *result = malloc(((blocksize * kblocks) - (i * blocksize)) * ((blocksize * kblocks) - (i * blocksize)) * sizeof(double));

	// Put blocks of L into L21 column vector
	int w = 0;
	int bs = blocksize * 0.1;
	if(bs == 0)
		bs = 1;
	# pragma omp parallel for private(k) schedule(dynamic, bs) num_threads(nthreads)
	for (k = i; k < kblocks; k++) {
		temp = L[k][j - 1] -> block;
		for (x = 0; x < blocksize; x++) {
			for (y = 0; y < blocksize; y++) {
				L21[IDX(((w * blocksize) + x), y, blocksize)] = temp[IDX(x, y, blocksize)];
			}
		}
		w++;
	}
	
	/* Compute L21 transpose from L21 */
	//transpose(L21, L21T, blocksize, (kblocks - i) * blocksize);

	/* If square do multT, if column do multTNonSquare */
	int width = blocksize;
	int height = (kblocks - i) * blocksize;
	if (height == width)
		multT(result, L21, height, 0);
	else 
		multTNonSquare(result, L21, L21, width, height, 0);

	/* Update A22 by solving A22 - (L21 * L21T), where * is elementwise matrix multiplication 
 	   (L21 * L21T is stored in result, so compute A22 - result */

	//int bs = blocksize*blocksize;
	# pragma omp parallel for private(k) schedule(dynamic, bs) num_threads(nthreads)
	for (k = i; k < kblocks; k++) {
		for (l = j; l < kblocks; l++) {
			y = 0;
			A22 = A[k][l] -> block;
			for (a = 0; a < blocksize; a++) {
				for (b = 0; b < blocksize; b++) {
					A22[IDX(a, b, blocksize)] = A22[IDX(a, b, blocksize)] - result[IDX((((k - i) * blocksize) + a), (((l - j) * blocksize) + b), ((kblocks * blocksize) - (i * blocksize)))];
				}
			}
		}
	}

	free(L21);
	free(L21T);
	free(result);
}

/*
 * Main program to compute the block cholesky factorization
 *
 * Parameters: argc - the number of arguments from command line
 *             argv - array of arguments from command line
 *
 */
int main(int argc, char* argv[]) {
	if (argc > 4 || argc < 4) {
		printf("usage: blockCholesky <K> <blksize>\n");
		return -1;
	}
	
	int kblocks = atoi(argv[1]);
	int blocksize = atoi(argv[2]);
	nthreads = atoi(argv[3]);
	blockCholesky(kblocks, blocksize);

	printf("Elpased time:\n\tinitialization=%e\n", tie - tis);
	printf("\tblockcholesky=%e\n", tce - tcs);
	printf("\tvalidate=%e\n", tve - tvs);

	if (badcount == 0)
		printf("solution validates\n");
	else
		printf("solution is invalid, %d elements above threshold\n", badcount);
	
	return 0;
}

