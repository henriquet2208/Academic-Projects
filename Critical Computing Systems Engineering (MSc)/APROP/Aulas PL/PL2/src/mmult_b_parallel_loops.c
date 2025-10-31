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
void par_row(int num_threads, int scheduling_choice);
void par_block(int num_threads, int scheduling_choice);

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

    // Parallel execution with static scheduling
    printf("\nThread working on rows with Static scheduling... ");
    double begin = omp_get_wtime();
    par_row(num_threads, 0);  // Static scheduling
    double end = omp_get_wtime();
    double per_row_time_static = end - begin;
    printf("done.\n");
    assert(C, expected);
    c_clean();

    printf("Thread working on block (collapse) with Static scheduling... ");
    begin = omp_get_wtime();
    par_block(num_threads, 0);  // Static scheduling
    end = omp_get_wtime();
    double per_block_time_static = end - begin;
    printf("done.\n");
    assert(C, expected);
    c_clean();

    // Parallel execution with dynamic scheduling
    printf("\nThread working on rows with Dynamic scheduling... ");
    begin = omp_get_wtime();
    par_row(num_threads, 1);  // Dynamic scheduling
    end = omp_get_wtime();
    double per_row_time_dynamic = end - begin;
    printf("done.\n");
    assert(C, expected);
    c_clean();

    printf("Thread working on block (collapse) with Dynamic scheduling... ");
    begin = omp_get_wtime();
    par_block(num_threads, 1);  // Dynamic scheduling
    end = omp_get_wtime();
    double per_block_time_dynamic = end - begin;
    printf("done.\n");
    assert(C, expected);
    c_clean();

    // Parallel execution with guided scheduling
    printf("\nThread working on rows with Guided scheduling... ");
    begin = omp_get_wtime();
    par_row(num_threads, 2);  // Guided scheduling
    end = omp_get_wtime();
    double per_row_time_guided = end - begin;
    printf("done.\n");
    assert(C, expected);
    c_clean();

    printf("Thread working on block (collapse) with Guided scheduling... ");
    begin = omp_get_wtime();
    par_block(num_threads, 2);  // Guided scheduling
    end = omp_get_wtime();
    double per_block_time_guided = end - begin;
    printf("done.\n");
    assert(C, expected);

    printf("\n- ==== Performance ==== -\n");
    printf("Sequential time:     %fs\n", sequential_time);
    printf("Parallel rows time (Static):  %fs\n", per_row_time_static);
    printf("Parallel block time (Static): %fs\n", per_block_time_static);
    printf("Parallel rows time (Dynamic):  %fs\n", per_row_time_dynamic);
    printf("Parallel block time (Dynamic): %fs\n", per_block_time_dynamic);
    printf("Parallel rows time (Guided):  %fs\n", per_row_time_guided);
    printf("Parallel block time (Guided): %fs\n", per_block_time_guided);
}

void par_row(int num_threads, int scheduling_choice) {
    if (scheduling_choice == 0) {
        printf("\nUsing Static scheduling...\n");
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                calc(l, n);  // Calculate cell [l, n]
            }
        }
    } else if (scheduling_choice == 1) {
        printf("\nUsing Dynamic scheduling...\n");
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(dynamic, 10)
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                calc(l, n);  // Calculate cell [l, n]
            }
        }
    } else if (scheduling_choice == 2) {
        printf("\nUsing Guided scheduling...\n");
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(guided)
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                calc(l, n);  // Calculate cell [l, n]
            }
        }
    }
}

void par_block(int num_threads, int scheduling_choice) {
    if (scheduling_choice == 0) {
        printf("\nUsing Static scheduling...\n");
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                calc(l, n);  // Calculate the cell [l, n]
            }
        }
    } else if (scheduling_choice == 1) {
        printf("\nUsing Dynamic scheduling...\n");
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(dynamic, 10)
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                calc(l, n);  // Calculate the cell [l, n]
            }
        }
    } else if (scheduling_choice == 2) {
        printf("\nUsing Guided scheduling...\n");
        #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(guided)
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                calc(l, n);  // Calculate the cell [l, n]
            }
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
