/*
 * Copyright 2022 Instituto Superior de Engenharia do Porto
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
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define DEFAULT_NUM_THREADS 4

#define STR_HELPER(v) #v
#define STR(v) STR_HELPER(v)

#define ARR_LEN 5000000 // The length of the array, increase to have more execution time
#define MAX_PRINT 30
#define FIRST_VALUES " (FIRST " STR(MAX_PRINT)")"
#define MIN_VAL -ARR_LEN*10 // Minimum value of the array elements
#define MAX_VAL ARR_LEN*10 // Maximum value of the array elements



//Quick sort versions
void quick_sort_seq(int array[], int low, int high);
int split(int array[], int low, int high);
void swap_element(int* x, int* y);

void quick_sort_par(int array[], int low, int high, int num_threads);


/**
 * Utility Functions
*/
int generate_random(int min, int max);
void copy(int from[], int to[], int size);
void assert(int expected[], int result[], int size);
void print_array(int array[], int size, const char* version);

/**
 *  The main function. 
 *  Algorithm:
 * 	1. Generate array with random values: 'original' array
 *  2. Copy 'original' array to another one named 'array'
 *  3. Run sequential version of quicksort with 'array'
 *  4. Copy result in 'array' to the 'expected' array
 *  5. Again copy 'original' array to the one named 'array'
 *  6. Run parallel version of quicksort with 'array'
 *  7. Compare results via the assert function: 'expected' vs 'array'
 * */
//using global arrays to easily allow lenghts >=1M (heap instead of stack!)
int array[ARR_LEN];
int original[ARR_LEN];
int expected[ARR_LEN];
int main(int argc, char* argv[]) {
	int i;

	/* Use current time as the seed for random generator */
	srand(time(0));
	int num_threads = DEFAULT_NUM_THREADS;
    if(argc < 2){
        printf("Number of threads was not specified. Will use default value: %d\n",DEFAULT_NUM_THREADS);
    }else{
        num_threads = atoi(argv[1]);
    }

	/* Generate the elements of the array randomly */
	for (i = 0; i < ARR_LEN; i++){
		original[i] = generate_random(MIN_VAL, MAX_VAL);
	}
	printf("\n");
	copy(original,array,ARR_LEN);

	printf("Sequential Sorting...");
	/* Call the quick_sort function to sort the array */
	clock_t begin = clock();
	quick_sort_seq(array, 0, ARR_LEN - 1);
	clock_t end = clock();
    double seq_time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf(" done!\n");
	copy(array,expected,ARR_LEN);

	//copy original to array again
	copy(original,array,ARR_LEN);

	//you can print an array using the "print" function, as below
	//print_array(array,ARR_LEN,"Version X");
	//note that it will only print the first 'MAX_PRINT' elements if the array is too large
	printf("Parallel Sorting...");
	begin = clock();
	quick_sort_par(array, 0, ARR_LEN - 1, num_threads);
    end = clock();
    double par_a_time =  (double)(end - begin) / CLOCKS_PER_SEC;
	printf(" done!\n");
	//then invoke the following code to validate your version
	assert(expected,array,ARR_LEN);


	printf("\n- ==== Performance ==== -\n");
    printf("Sequential time: %fs\n",seq_time);
    printf("Parallel time: %fs\n",par_a_time);
	return 0;
}


/************ Parallel versions of Quicksort algorithm *****************/
/**
 * Parallel Version
 **/
void quick_sort_par(int array[], int low, int high, int num_threads) {

	quick_sort_seq(array,low,high); //replace with your code!
}



/************ Seq. Quicksort algorithm *****************/
/* The main function to perform the sort operation */
void quick_sort_seq(int array[], int low, int high) {

	/* The main process of the function */
	if (low < high) {
		int m;
		
		/* Specify the middle of the array by calling the split function */
		m = split(array, low, high);
		
		/* Call the quick_sort function to sort the antecedent array */
		quick_sort_seq(array, low, m - 1);
		/* Call the quick_sort function to sort the subsequent array */
		quick_sort_seq(array, m + 1, high);
	}
}

/* This function takes last element as index, places the index element at
its correct position in sorted array, and places all smaller (smaller than
the index) to left of the index and all greater elements to right of the
index */
int split(int array[], int low, int high) {
	int i, index = array[high];
	int m = low - 1;

	/* The main process of the function */
	for (i = low; i <= high - 1; i++)
		if (array[i] < index) {
			m++;				
			/* Call the swap_element function to swap the elements */
			swap_element(&array[m], &array[i]);
		}

	/* Call the swap_element function to swap the elements */
	swap_element(&array[m + 1], &array[high]);
	return m + 1;
}

/* Swap two elements */
void swap_element(int* x, int* y) {

	/* The main process of the function */
	int temp;
	temp = *x;
	*x = *y;
	*y = temp;
}


/************ Util. Functions *****************/

/* Generate a random number */
int generate_random(int min, int max) {
	/* Generate a random number between min and max */
	return (rand() % (max - min + 1)) + min;
}

void copy(int from[], int to[], int size) {
	for(int i = 0; i < size;i++){
		to[i] = from[i];
	}
}

void assert(int expected[], int result[], int size) {
	for(int i = 0; i < size;i++){
		if(expected[i] != result[i]){
			printf("[ERROR]Bad result in position %d: expect %d, but %d was found.\n",i,expected[i], result[i]);
			exit(1);
		}
	}
}

void print_array(int array[], int size, const char* version) {
	
	printf("%s%s:",version, size> MAX_PRINT?FIRST_VALUES:"");
	int limit = size > MAX_PRINT? MAX_PRINT: size;
	for (int i = 0; i < limit; i++){
		printf(" %d",array[i]);
	}
	printf("\n");
}