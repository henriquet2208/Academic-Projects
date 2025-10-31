#include <stdio.h>
#include <omp.h>
#include <assert.h>
#include <float.h>

#include "lib/helper.h"

double initial[N][N];
double expected[N][N];
double M[N][N];


void reset(double matrix[N][N]) {
    copy(matrix, initial);
}


void calculate(double matrix[N][N], int i, int j) {
    matrix[i][j] = (matrix[i][j - 1] + matrix[i - 1][j] + matrix[i][j + 1] + matrix[i + 1][j]) / 4.0;
}


/*
    Esta é a versão não-paralelizada
    da execução: uma única thread
    percorre toda a matriz, ponto a
    ponto, e processa cada um deles. 
*/
void sequential(double matrix[N][N], int matrix_size) {
    int size = matrix_size - 1;

    for (int i = 1; i < size; i++) {
        for (int j = 1; j < size; j++) {
            calculate(matrix, i, j);
        }
    }
}


/* 
    Esta é a versão em que todos os
    pontos da matriz são uma task em
    separado.
*/
void taskBasedMatrixUpdate(double matrix[N][N], int matrix_size) {
    int size = matrix_size - 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 1; i < size; i++) {
                for (int j = 1; j < size; j++) {
                    #pragma omp task depend(in: matrix[i][j-1], matrix[i-1][j]) \
                                     depend(in: matrix[i][j+1], matrix[i+1][j]) \
                                     depend(out: matrix[i][j])
                    {
                        calculate(matrix, i, j);
                    }
                }
            }
        }
    }
}

/*
    Esta é a versão em que a
    matriz é dividida em blocos,
    e cada task assume um bloco.
*/
void blockBasedMatrixUpdate(double matrix[N][N], int matrix_size, int block_size) {
    int size = matrix_size - 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int block_r = 1; block_r < size; block_r += block_size) {
                for (int block_c = 1; block_c < size; block_c += block_size) {
                    #pragma omp task shared(matrix) depend(in: matrix[block_r][block_c - block_size]) \
                                                    depend(in: matrix[block_r - block_size][block_c]) \
                                                    depend(in: matrix[block_r][block_c + block_size]) \
                                                    depend(in: matrix[block_r + block_size][block_c]) \
                                                    depend(out: matrix[block_r][block_c])
                    {
                        for (int i = block_r; i < (block_r + block_size < size ? block_r + block_size : size); i++) {
                            for (int j = block_c; j < (block_c + block_size < size ? block_c + block_size : size); j++) {
                                calculate(matrix, i, j);
                            }
                        }
                    }
                }
            }
        }
    }
}


void do_five_runs(char *file_name, int run, int num_threads, int block_size, int matrix_size){
    double begin;
    double end;

    double seq_times[5];
    double task_times[5];
    double block_times[5];

    omp_set_num_threads(num_threads);

    for (int i = 0; i < 5; i++) {  
        reset(M);
        begin = omp_get_wtime();
        sequential(M, matrix_size);
        end = omp_get_wtime();
        seq_times[i] = (end - begin);
        
        reset(M);
        begin = omp_get_wtime();
        taskBasedMatrixUpdate(M, matrix_size);
        end = omp_get_wtime();
        task_times[i] = (end - begin);
        compare(M, expected, matrix_size);
        
        reset(M);
        begin = omp_get_wtime();
        blockBasedMatrixUpdate(M, matrix_size, block_size);
        end = omp_get_wtime();
        block_times[i] = (end - begin);
        compare(M, expected, matrix_size);

        write_to_csv(file_name, run + i, seq_times[i], task_times[i], block_times[i], num_threads, matrix_size, block_size);
        printf("Run %d: Sequential: %.6f, Task-based: %.6f, Block-based: %.6f, Threads: %d, N: %d, BS: %d\n",
               run + i, seq_times[i], task_times[i], block_times[i], num_threads, matrix_size, block_size);
    }

    printf("\nAverage for Runs %d to %d:\n", run, run + 4);
    printf("Average Sequential Time: %.6f\n", average(seq_times, 5));
    printf("Average Task-based Time: %.6f\n", average(task_times, 5));
    printf("Average Block-based Time: %.6f\n\n", average(block_times, 5));
}


/*
    O setup só precisa ser 
    executado uma única vez.
    Não faz sentido executá-lo
    dentro do loop para medir
    o tempo sequencial porque 
    nós não queremos mudar os
    valores das matrizes inicial
    e expected, só a da matriz
    intermediária.
*/
void setup(double matrix[N][N]) {   
    fill(initial);
    copy(matrix, initial);
    sequential(matrix, N);
    copy(expected, matrix);
}


int main(int argc, char *argv[]) {

    char *file_name = argv[1]; 
    int run = 0;
    
    /* valores padrão */
    int default_matrix_size = N / 5;
    int default_block_size = N / 100;
    int default_threads = DEFAULT_NUM_THREADS;

    /* deltas */
    int delta_matrix_size = N / 5;
    int delta_block_size = default_block_size / 5; 
    int delta_threads = DEFAULT_NUM_THREADS;

    /* valores máximos - */
    int max_threads = DEFAULT_NUM_THREADS * 4;
    int max_matrix_size = N;
    int max_block_size = default_block_size;

    setup(M);

    /*
        cada um dos loops a seguir
        muda apenas uma variável
        por vez. As demais variáveis
        ficam no valor padrão
    */   

    int block_size = 0;
    while(block_size < max_block_size){
        block_size += delta_block_size;
        do_five_runs(file_name,
            run,
            default_threads,
            block_size,             // <-- aqui mudamos apenas o tamanho do bloco
            default_matrix_size
        );
        run += 5;
    }

    int matrix_size = 0;
    while(matrix_size < max_matrix_size){
        matrix_size += delta_matrix_size;
        do_five_runs(file_name,
            run,
            default_threads,
            default_block_size, 
            matrix_size            // <-- aqui mudamos apenas o tamanho da matriz
        );
        run += 5;
    }

    int threads = 0;
    while(threads < max_threads){
        threads += delta_threads;
        do_five_runs(file_name,
            run,
            threads,                // <-- aqui mudamos apenas o número de threads
            default_block_size,
            default_matrix_size
        );
        run += 5;
    } 

    printf("\n%d runs completed. Detailed results saved to '%s'.\n", run, file_name);

    return 0;
}

    
