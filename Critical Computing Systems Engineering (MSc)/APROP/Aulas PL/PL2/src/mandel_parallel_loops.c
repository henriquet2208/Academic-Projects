#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define DEFAULT_NUM_THREADS 4
#define NPOINTS 1000
#define MAXITER 1000

typedef struct result_t {
   double area;
   double error;
} result_t;

struct d_complex {
   double r;
   double i;
};

struct d_complex c;
int numoutside = 0;

void testpoint(struct d_complex);
int par_testpoint(struct d_complex);
result_t seq_mandel();
result_t par_mandel(int num_threads);

int main(int argc, char *argv[]) {
    srand(time(NULL));
    int num_threads = DEFAULT_NUM_THREADS;
    if(argc < 2){
        printf("Number of threads was not specified. Will use default value: %d\n", DEFAULT_NUM_THREADS);
    } else {
        num_threads = atoi(argv[1]);
    }

    result_t expected;

    // Sequential Mandelbrot calculation
    printf("Sequential Mandelbrot... ");
    double begin = omp_get_wtime();
    expected = seq_mandel();
    double end = omp_get_wtime();
    double seq_time = end - begin;
    printf("done.\n");
    printf("[SEQ] Area of Mandelbrot set = %12.8f +/- %12.8f (outside: %d)\n", expected.area, expected.error, numoutside);

    // Resetting values
    int expected_num_outside = numoutside;
    numoutside = 0;

    // Parallel Mandelbrot calculation
    printf("\nParallel Mandelbrot... ");
    double start = omp_get_wtime();
    result_t par_res = par_mandel(num_threads);
    double end_parallel = omp_get_wtime();
    double par_time = end_parallel - start;
    printf("done.\n");
    int par_num_outside = numoutside;
    printf("[PAR] Area of Mandelbrot set = %12.8f +/- %12.8f (outside: %d)\n", par_res.area, par_res.error, par_num_outside);

    // Performance comparison
    printf("\n- ==== Performance ==== -\n");
    printf("Sequential time: %fs\n", seq_time);
    printf("Parallel   time: %fs\n", par_time);  

    // Validate results
    if (expected.area != par_res.area || expected.error != par_res.error || expected_num_outside != par_num_outside) {
        printf("\n! Assert failed!\n");
        printf("Sequential:\n");
        printf("\tArea of Mandelbrot set = %12.8f +/- %12.8f (outside: %d)\n", expected.area, expected.error, expected_num_outside);
        printf("Parallel:\n");
        printf("\tArea of Mandelbrot set = %12.8f +/- %12.8f (outside: %d)\n", par_res.area, par_res.error, par_num_outside);
    }
}

/* Parallel Mandelbrot implementation using OpenMP */
result_t par_mandel(int num_threads) {
    double area, error, eps = 1.0e-5;
    int local_numoutside = 0;

    #pragma omp parallel for collapse (2) reduction(+:local_numoutside) num_threads(num_threads)
    for (int i = 0; i < NPOINTS; i++) {
        for (int j = 0; j < NPOINTS; j++) {
            struct d_complex c;
            c.r = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + eps;
            c.i = 1.125 * (double)(j) / (double)(NPOINTS) + eps;
            local_numoutside += par_testpoint(c);
        }
    }

    // Update shared numoutside variable
    numoutside = local_numoutside;

    // Calculate area of set and error estimate
    area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
    error = area / (double)NPOINTS;

    result_t result = {area, error};
    return result;
}

/* Sequential Mandelbrot implementation */
result_t seq_mandel() {
    double area, error, eps = 1.0e-5;

    for (int i = 0; i < NPOINTS; i++) {
        for (int j = 0; j < NPOINTS; j++) {
            struct d_complex c;
            c.r = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + eps;
            c.i = 1.125 * (double)(j) / (double)(NPOINTS) + eps;
            testpoint(c);
        }
    }

    // Calculate area of set and error estimate
    area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
    error = area / (double)NPOINTS;

    result_t result = {area, error};
    return result;
}

/* Check if a point is in the Mandelbrot set in parallel */
int par_testpoint(struct d_complex c) {
    struct d_complex z;
    int iter;
    double temp;

    z = c;
    for (iter = 0; iter < MAXITER; iter++) {
        temp = (z.r * z.r) - (z.i * z.i) + c.r;
        z.i = 2 * z.r * z.i + c.i;
        z.r = temp;
        if ((z.r * z.r + z.i * z.i) > 4.0) {
            return 1;  // Point is outside the Mandelbrot set
        }
    }
    return 0;  // Point is inside the Mandelbrot set
}

/* Check if a point is in the Mandelbrot set in sequential */
void testpoint(struct d_complex c) {
    struct d_complex z;
    int iter;
    double temp;

    z = c;
    for (iter = 0; iter < MAXITER; iter++) {
        temp = (z.r * z.r) - (z.i * z.i) + c.r;
        z.i = 2 * z.r * z.i + c.i;
        z.r = temp;
        if ((z.r * z.r + z.i * z.i) > 4.0) {
            numoutside++;
            break;
        }
    }
}
