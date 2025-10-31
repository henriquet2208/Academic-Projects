#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <omp.h>

#define L 512
#define M 512
#define N 512
#define DEFAULT_NUM_THREADS 4

int A[L][M];
int B[M][N];
int C[L][N];
int expected[L][N];
double sequential_time;

#define MIN_RAND -10
#define MAX_RAND 10

void seq();
void par_row(int num_threads);
void par_block(int num_threads);

void calc(int l,int n);
void fill(int* matrix, int height,int width);
void print(int* matrix,int height,int width);
void assert(int C[L][N],int expected[L][N]);
void c_clean();
void setup();

int main(int argc, char *argv[])
{
    srand(time(NULL));
    int num_threads = DEFAULT_NUM_THREADS;
    if(argc < 2){
        printf("Number of threads was not specified. Will use default value: %d\n",DEFAULT_NUM_THREADS);
    }else{
        num_threads = atoi(argv[1]);
    }
    printf("Working with %d threads to multiply two matrices: A{%d,%d}*B{%d,%d} = C{%d,%d}\n", num_threads, L, M, M, N, L, N);

    setup();

    printf("Thread working on rows... ");
    double begin = omp_get_wtime();
    par_row(num_threads);
    double end = omp_get_wtime();
    double per_row_time = end - begin;
    printf("done.\n");
    assert(C, expected);

    c_clean();

    printf("Thread working on block (collapse)... ");
    begin = omp_get_wtime();
    par_block(num_threads);
    end = omp_get_wtime();
    double per_block_time = end - begin;
    printf("done.\n");
    assert(C, expected);

    printf("\n- ==== Performance ==== -\n");
    printf("Sequential time:     %fs\n", sequential_time);
    printf("Parallel rows time:  %fs\n", per_row_time);
    printf("Parallel block time: %fs\n", per_block_time);
}

void par_row(int num_threads) {
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            calc(l, n);  // Calculate cell [l, n]
        }
    }
}

void par_block(int num_threads) {
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            calc(l, n);  // Calculate the cell [l, n]
        }
    }
}

void seq() {
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int m = 0; m < M; m++) {
                sum += A[l][m] * B[m][n];
            }
            C[l][n] = sum;
        }
    }
}

void calc(int l, int n) {
    int sum = 0;
    for (int m = 0; m < M; m++) {
        sum += A[l][m] * B[m][n];
    }
    C[l][n] = sum;
}

void fill(int* matrix, int height, int width) {
    for (int l = 0; l < height; l++) {
        for (int n = 0; n < width; n++) {
            *((matrix + l * width) + n) = MIN_RAND + rand() % (MAX_RAND - MIN_RAND + 1);
        }
    }
}

void print(int* matrix, int height, int width) {
    for (int l = 0; l < height; l++) {
        printf("[");
        for (int n = 0; n < width; n++) {
            printf(" %5d", *((matrix + l * width) + n));
        }
        printf(" ]\n");
    }
}

void assert(int C[L][N], int expected[L][N]) {
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            if (C[l][n] != expected[l][n]) {
                printf("Wrong value at position [%d,%d], expected %d, but got %d instead\n", l, n, expected[l][n], C[l][n]);
                exit(-1);
            }
        }
    }
}

void c_clean() {
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            C[l][n] = 0;
        }
    }
}

void setup() {
    fill((int *)A, L, M);
    fill((int *)B, M, N);
    clock_t begin = clock();
    seq();
    clock_t end = clock();
    sequential_time = (double)(end - begin) / CLOCKS_PER_SEC;

    for (int l = 0; l < L; l++) {
        for (int n = 0; n < N; n++) {
            expected[l][n] = C[l][n];
            C[l][n] = 0;
        }
    }
}
