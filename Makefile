default: blockCholesky cholesky blockCholeskyMP

cholesky: cholesky.c cs160validate.c
	gcc -g -Wall -fopenmp -o cholesky -O3 cholesky.c cs160validate.c -lm

blockCholesky: blockCholesky.c blockCholesky.h cs160validate.o
	cc -o blockCholesky -O3 blockCholesky.c cs160validate.c -lm

cs160validate.o: cs160validate.c cs160validate.h
	cc -c -O3 cs160validate.c

blockCholeskyMP: cs160validate.o blockCholeskyMP.c
	gcc -g -Wall -fopenmp -o blockCholeskyMP blockCholeskyMP.c cs160validate.c -O3 -lm

clean:
	-/bin/rm blockCholesky
	-/bin/rm cholesky
	-/bin/rm *.o
	-/bin/rm blockCholeskyMP

