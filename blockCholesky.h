#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>

typedef struct{

	double * block;
	int row;
	int col;

}blockstruct;

void blackCholesky(int kblocks, int blocksize);
int init_array(double *A, int blocksize, int kblocks);
int multT(double *A, double *B, int N, int lowerT);
int multTNonSquare(double *result, double *A, double *B, int width, int height, int lowerT);
void calcL11(int blocksize, double *A, double *L);
void calcL21(int i, int j, int kblocks, int blocksize, blockstruct ***A, blockstruct ***L);
void calcL22(int i, int j, int kblocks, int blocksize, blockstruct ***A, blockstruct ***L);
void transpose(double *L, double *LT, int width, int height);
void cholesky(double *L, double *A, int N);
int validate(double *A, double *L, int blocksize, int kblocks, double thresh);
void printMatrix(double *A, int N);
void convertStructToPointer(blockstruct ***A, double *linear, int blocksize, int kblocks);
double get_clock();
