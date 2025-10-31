#ifndef _APP_HELPER_H_
#define _APP_HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN_RAND -10
#define MAX_RAND 10

#define DEFAULT_NUM_THREADS 4
#define N 5000

void fill(double matrix[N][N]) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = MIN_RAND + rand() % (MAX_RAND - MIN_RAND + 1);
            // printf("[%d, %d]: %.2f\n", i, j, matrix[i][j]);
        }
    }
}


void compare(double matrix[N][N], double expected[N][N], int matrix_size) {
    int size = matrix_size - 1;

    for (int l = 0; l < size; l++) {
        for (int n = 0; n < size; n++) {
            if (matrix[l][n] != expected[l][n]) {
                printf("Wrong value at position [%d,%d], expected %.2f, but got %.2f instead\n", l, n, expected[l][n], matrix[l][n]);
                exit(-1);
            }
        }
    }
}


void copy(double destination[N][N], double origin[N][N]) {
    for (int l = 0; l < N; l++) {
        for (int n = 0; n < N; n++) {
            destination[l][n] = origin[l][n];
        }
    }
}


double average(double times[], int size){
    double avg = 0;
    int div = (double) size;

    for(int i = 0; i < size; i++){
        avg += times[i] / div;
    }
    return avg;
}


void write_to_csv(char *file_name, int run, double seq_time, double task_time, double block_time, int num_threads, int ns, int bss) {
    FILE *f;

    // Open the file in write mode for the first run to create the header
    if (run == 0) {
        f = fopen(file_name, "w"); // Create a new file or overwrite existing
        if (f == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }
        fprintf(f, "Run,Sequential Time,Task-based Time,Block-based Time,Num Threads,N,BS\n");
    } else {
        f = fopen(file_name, "a"); // Append mode for subsequent runs
        if (f == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }
    }

    // Write the results of the current run
    fprintf(f, "%d,%f,%f,%f,%d,%d,%d\n", run + 1, seq_time, task_time, block_time, num_threads, ns, bss);
    fclose(f);
}

#endif