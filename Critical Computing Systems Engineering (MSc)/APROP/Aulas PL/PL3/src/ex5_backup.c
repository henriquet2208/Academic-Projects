#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <assert.h>
#include <float.h>

#define DEFAULT_NUM_THREADS 4
#define N 1000 // This will change dynamically in main
#define BS 100 // This will change dynamically in main
#define NUM_RUNS 70

#define MIN_RAND -10
#define MAX_RAND 10

double initial[N][N], M[N][N], expected[N][N];
double seq_time; // Declare seq_time globally

void fill(double matrix[N][N], int height, int width);
void assertResult(double M[N][N], double expected[N][N]);
void c_clean();
void setup();
void seq();
void write_to_csv(double seq_times[], double task_times[], double block_times[], int num_threads[], int ns[], int bss[], int run_num);

void fill(double matrix[N][N], int height, int width) {
    for (int l = 0; l < height; l++) {
        for (int n = 0; n < width; n++) {
            matrix[l][n] = MIN_RAND + rand() % (MAX_RAND - MIN_RAND + 1);
        }
    }
}

void assertResult(double M[N][N], double expected[N][N]) {
    for (int l = 0; l < N; l++) {
        for (int n = 0; n < N; n++) {
            if (M[l][n] != expected[l][n]) {
                printf("Wrong value at position [%d,%d], expected %.2f, but got %.2f instead\n", l, n, expected[l][n], M[l][n]);
                exit(-1);
            }
        }
    }
}

void c_clean() {
    for (int l = 0; l < N; l++) {
        for (int n = 0; n < N; n++) {
            M[l][n] = initial[l][n]; // Copying the initial matrix
        }
    }
}

void setup() {
    fill(initial, N, N);
    c_clean(); // Ensure M is initialized from the initial matrix
    double begin = omp_get_wtime();
    seq();
    double end = omp_get_wtime();
    seq_time = end - begin; // Set the global seq_time variable

    // Store the expected result after seq()
    for (int l = 0; l < N; l++) {
        for (int n = 0; n < N; n++) {
            expected[l][n] = M[l][n]; // Store expected result from M
        }
    }

    c_clean(); // Reset M for parallel computations
}

void seq() {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            M[i][j] = (M[i][j - 1] + M[i - 1][j] + M[i][j + 1] + M[i + 1][j]) / 4.0;
        }
    }
}

void taskBasedMatrixUpdate(double M[N][N]) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    #pragma omp task depend(in: M[i][j-1], M[i-1][j]) \
                                     depend(in: M[i][j+1], M[i+1][j]) \
                                     depend(out: M[i][j])
                    {
                        M[i][j] = (M[i][j - 1] + M[i - 1][j] + M[i][j + 1] + M[i + 1][j]) / 4.0;
                    }
                }
            }
        }
    }
}

void blockBasedMatrixUpdate(double M[N][N]) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int block_r = 1; block_r < N - 1; block_r += BS) {
                for (int block_c = 1; block_c < N -  1; block_c += BS) {
                    #pragma omp task shared(M) depend(in: M[block_r][block_c-BS], M[block_r-BS][block_c], M[block_r][block_c+BS], M[block_r+BS][block_c]) \
                                                 depend(out: M[block_r][block_c])
                    {
                        for (int i = block_r; i < (block_r + BS < N - 1 ? block_r + BS : N - 1); i++) {
                            for (int j = block_c; j < (block_c + BS < N - 1 ? block_c + BS : N - 1); j++) {
                                {
                                    M[i][j] = (M[i][j - 1] + M[i - 1][j] + M[i][j + 1] + M[i + 1][j]) / 4.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void write_to_csv(double seq_times[], double task_times[], double block_times[], int num_threads[], int ns[], int bss[], int run_num) {
    FILE *f;

    // Open the file in write mode for the first run to create the header
    if (run_num == 0) {
        f = fopen("results_llvm.csv", "w"); // Create a new file or overwrite existing
        if (f == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }
        fprintf(f, "Run,Sequential Time,Task-based Time,Block-based Time,Num Threads,N,BS\n");
    } else {
        f = fopen("results_llvm.csv", "a"); // Append mode for subsequent runs
        if (f == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }
    }

    // Write the results of the current run
    fprintf(f, "%d,%f,%f,%f,%d,%d,%d\n", run_num + 1, seq_times[run_num], task_times[run_num], block_times[run_num], num_threads[run_num], ns[run_num], bss[run_num]);

    fclose(f);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    double seq_times[NUM_RUNS], task_times[NUM_RUNS], block_times[NUM_RUNS];
    int num_threads[NUM_RUNS], ns[NUM_RUNS], bss[NUM_RUNS];

    int current_threads;
    int current_bs;
    int current_n;

    for (int run = 0; run < NUM_RUNS; run++) {
        // Set the parameters based on the current iteration
        if (run < 5) {
            current_threads = 4;
            current_bs = 100;
            current_n = 1000;
        } else if (run < 10) {
            current_threads = 8;
            current_bs = 100;
            current_n = 1000;
        } else if (run < 15) {
            current_threads = 16;
            current_bs = 100;
            current_n = 1000;
        } else if (run < 20) {
            current_threads = 32;
            current_bs = 100;
            current_n = 1000;
        //para cima o numero de threads aumentou o dobro desde 4 até 32
        } else if (run < 25) {
            current_bs = 50;
            current_threads = 4;
            current_n = 1000;
        } else if (run < 30) {
            current_bs = 100;
            current_threads = 4;
            current_n = 1000;
        } else if (run < 35) {
            current_bs = 250;
            current_threads = 4;
            current_n = 1000;
        } else if (run < 40) {
            current_bs = 500;
            current_threads = 4;
            current_n = 1000;
        } else if (run < 45) {
            current_threads = 4;
            current_n = 1000;
            current_bs = 1000;
        //para cima o block sized aumentou desde 50 até 1000
        } else if (run < 50) {
            current_n = 1000;
            current_threads = 4;
            current_bs = 100;
        } else if (run < 55) {
            current_n = 1500;
            current_threads = 4;
            current_bs = 100;
        } else if (run < 60) {
            current_n = 2000;
            current_threads = 4;
            current_bs = 100;
        } else if (run < 65) {
            current_n = 2500;
            current_threads = 4;
            current_bs = 100;
        } else if (run < 70) {
            current_n = 3000;
            current_threads = 4;
            current_bs = 100;
        }
        //para cima o tamanho da matriz N aumentou desde 1000 até 3000

        omp_set_num_threads(current_threads);
        num_threads[run] = current_threads;
        ns[run] = current_n;
        bss[run] = current_bs;
        
        // Setup, task-based, and block-based computations with time measurements
        setup();
        
        double begin = omp_get_wtime();
        taskBasedMatrixUpdate(M);
        double end = omp_get_wtime();
        double task_time = end - begin;
        task_times[run] = task_time;
        assertResult(M, expected); // Verify task-based results

        c_clean(); // Reset the matrix before the next test

        begin = omp_get_wtime();
        blockBasedMatrixUpdate(M);
        end = omp_get_wtime();

        double block_time = end - begin;
        block_times[run] = block_time;
        assertResult(M, expected); // Verify block-based results

        // Store sequential time (same for each run)
        seq_times[run] = seq_time;

        // Print results for the current iteration
        printf("Run %d: Sequential: %.6f, Task-based: %.6f, Block-based: %.6f, Threads: %d, N: %d, BS: %d\n",
               run + 1, seq_times[run], task_times[run], block_times[run], num_threads[run], ns[run], bss[run]);

        // Write results to CSV file
        write_to_csv(seq_times, task_times, block_times, num_threads, ns, bss, run);

        // Calculate and print average times at specific intervals
        if ((run + 1) % 5 == 0) {
            double avg_seq_time = 0.0, avg_task_time = 0.0, avg_block_time = 0.0;
            int start = run - 4; // Start index for the last 10 runs

            for (int i = start; i <= run; i++) {
                avg_seq_time += seq_times[i];
                avg_task_time += task_times[i];
                avg_block_time += block_times[i];
            }

            avg_seq_time /= 5;
            avg_task_time /= 5;
            avg_block_time /= 5;

            printf("\nAverage for Runs %d to %d:\n", start + 1, run + 1);
            printf("Average Sequential Time: %.6f\n", avg_seq_time);
            printf("Average Task-based Time: %.6f\n", avg_task_time);
            printf("Average Block-based Time: %.6f\n\n", avg_block_time);
        }
    }

    printf("\nAll 70 runs completed. Detailed results saved to 'results_llvm.csv'.\n");

    return 0;
}

    
