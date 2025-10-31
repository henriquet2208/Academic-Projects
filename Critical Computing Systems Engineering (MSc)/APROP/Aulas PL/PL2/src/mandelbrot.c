/*
 * Copyright 2023 Instituto Superior de Engenharia do Porto
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
**  PROGRAM: Mandelbrot area (solution)
**
**  PURPOSE: Program to compute the area of a  Mandelbrot set.
**           The correct answer should be around 1.510659.
**
**  USAGE:   Program runs without input ... just run the executable
**
**  ADDITIONAL EXERCISES:  Experiment with the schedule clause to fix 
**               the load imbalance.   Experiment with atomic vs. critical vs.
**               reduction for numoutside.
**            
**  HISTORY: Written:  (Mark Bull, August 2011).
**
**           Changed "comples" to "d_comples" to avoid collsion with 
**           math.h complex type.   Fixed data environment errors
**          (Tim Mattson, September 2011)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define DEFAULT_NUM_THREADS 4

#define NPOINTS 1000
#define MAXITER 1000

typedef struct result_t{
   double area;
   double error;
}result_t;

struct d_complex{
   double r;
   double i;
};

struct d_complex c;
int numoutside = 0;

void testpoint(struct d_complex);
int par_testpoint(struct d_complex);
result_t seq_mandel();
result_t par_mandel(int num_threads);

int main(int argc, char *argv[])
{
    srand(time(NULL));
    int num_threads = DEFAULT_NUM_THREADS;
    if(argc < 2){
        printf("Number of threads was not specified. Will use default value: %d\n",DEFAULT_NUM_THREADS);
    }else{
        num_threads = atoi(argv[1]);
    }

    result_t expected;
    
    printf("Sequential Mandelbrot... ");
    clock_t begin = clock();
    expected = seq_mandel();
    clock_t end = clock();
    double seq_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("done.\n");
    printf("[SEQ]Area of Mandlebrot set = %12.8f +/- %12.8f (outside: %d)\n",expected.area,expected.error,numoutside);

    //resetting values
    int expected_num_outside = numoutside;
    numoutside = 0;
    
    printf("\nParallel Mandelbrot... ");
    double inicio = omp_get_wtime();
    result_t par_res = par_mandel(num_threads);
    double final = omp_get_wtime();
    double par_time = final - inicio;
    printf("done.\n");
    int par_num_outside = numoutside;
    printf("[PAR]Area of Mandlebrot set = %12.8f +/- %12.8f (outside: %d)\n",par_res.area,par_res.error,par_num_outside);

    printf("\n- ==== Performance ==== -\n");
    printf("Sequential time: %fs\n",seq_time);
    printf("Parallel   time: %fs\n",par_time);  

    if (expected.area != par_res.area 
        || expected.error != par_res.error
        || expected_num_outside != par_num_outside){
        printf("\n!Assert failed!\n");
        printf("Sequential:\n");
        printf("\tArea of Mandlebrot set = %12.8f +/- %12.8f (outside: %d)\n",expected.area,expected.error,expected_num_outside);
        printf("Parallel:\n");
        printf("\tArea of Mandlebrot set = %12.8f +/- %12.8f (outside: %d)\n",par_res.area,par_res.error,par_num_outside);
    }
}

/* YOUR IMPLEMENTATIONS HERE! */
result_t par_mandel(int num_threads){
    double area, error, eps  = 1.0e-5;

    printf("\n\n");
    #pragma omp parallel num_threads(num_threads)
    {
        int delta = NPOINTS/num_threads;
        int id = omp_get_thread_num();
        int outsiders = 0;
        struct d_complex c;

        /*
            The computation is yielding correct results
            and it appears to really be going parallel,
            but the computed times are not better than
            the sequential version.

            I don't really understand why, since the
            threads run isolated ad without mutexes to
            lock the execution. Maybe the FPU is
            stalling the execution? Or maybe I'm missing
            something here? 
        */

        double start = omp_get_wtime();

        for (int i = id*delta; i < (id+1)*delta; i++) {
            for (int j = 0; j < NPOINTS; j++) {
                c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
                c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
                outsiders += par_testpoint(c);
            }
        }

        double finish = omp_get_wtime();

        double thread_time = finish - start;
        printf("thread %d:\t%f seconds.\n", id, thread_time);

        #pragma omp critical
        numoutside += outsiders;
    }

    // Calculate area of set and error estimate and output the results
    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;

    printf("\n");

    result_t result = {area,error};
    return result;
}


result_t seq_mandel(){
    int i, j;
    double area, error, eps  = 1.0e-5;

    for (i=0; i<NPOINTS; i++) {
        for (j=0; j<NPOINTS; j++) {
            c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
            c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
            testpoint(c);
        }
    }

    // Calculate area of set and error estimate and output the results
    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;

    result_t result = {area,error};
    return result;
}

void testpoint(struct d_complex c){
    // Does the iteration z=z*z+c, until |z| > 2 when point is known to be outside set
    // If loop count reaches MAXITER, point is considered to be inside the set

    struct d_complex z;
    int iter;
    double temp;

    z=c;
    for (iter=0; iter<MAXITER; iter++){
        temp = (z.r*z.r)-(z.i*z.i)+c.r;
        z.i = z.r*z.i*2+c.i;
        z.r = temp;
        if ((z.r*z.r+z.i*z.i)>4.0) {
            numoutside++;
            break;
        }
    }

}


int par_testpoint(struct d_complex c){
    // Does the iteration z=z*z+c, until |z| > 2 when point is known to be outside set
    // If loop count reaches MAXITER, point is considered to be inside the set

    struct d_complex z;
    int iter;
    double temp;

    z=c;
    for (iter=0; iter<MAXITER; iter++){
        temp = (z.r*z.r)-(z.i*z.i)+c.r;
        z.i = z.r*z.i*2+c.i;
        z.r = temp;
        if ((z.r*z.r+z.i*z.i)>4.0) {
            return 1;
            break;
        }
    }

    return 0;
}