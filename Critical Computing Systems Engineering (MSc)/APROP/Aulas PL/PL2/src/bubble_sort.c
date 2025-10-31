#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

#define DEFAULT_NUM_THREADS 4
#define ARR_LEN 25000		 // The length of the array
#define MIN_VAL 0			 // Minimum value of the array elements
#define MAX_VAL ARR_LEN * 10 // Maximum value of the array elements

int generate_random(int min, int max);
void copy(int from[], int to[], int size);
void assert(int expected[], int result[], int size);
void print(int x[], int size);

void sort_seq(int x[], int size){
	int i, changes = 1;
	while (changes)
	{
		changes = 0;
		for (i = 0; i < ARR_LEN-1; i = i + 1)
		{
			if (x[i] > x[i + 1])
			{
				int tmp;
				tmp = x[i];
				x[i] = x[i + 1];
				x[i + 1] = tmp;
				++changes;
			}
		}
	}
}

void sort_par(int x[], int size, int num_threads){
	/*
		Executar este código removendo todos
		os #pragmas do omp torna tudo sequencial,
		mas ainda assim se resolve muito mais
		rápido que o algoritmo mostrado na função
		sort_seq(). É apenas uma curiosidade.

		Executar o código com os #pragmas como 
		mostrado abaixo mostrou-se ainda mais rápido.
	*/
	int changes, buff;
	int i;
	do {
		changes = 0;
		#pragma omp parallel for private(i, buff)
		for(i = 0; i < size; i += 2){
			if(x[i] <= x[i+1]) continue;
			buff = x[i+1];
			x[i+1] = x[i];
			x[i] = buff;
			if(!changes) changes = 1;
		}

		#pragma omp parallel for private(i, buff)
		for(i = 1; i < (size - 1); i += 2){
			if(x[i] <= x[i+1]) continue;
			buff = x[i+1];
			x[i+1] = x[i];
			x[i] = buff;
			if(!changes) changes = 1;
		}
	} while(changes);
}


//using globals to allow large arrays
int expected[ARR_LEN];
int original[ARR_LEN];
int x[ARR_LEN];
int main(int argc, char *argv[])
{
	srand(time(0));
	int num_threads = DEFAULT_NUM_THREADS;
    if(argc < 2){
        printf("Number of threads was not specified. Will use default value: %d\n",DEFAULT_NUM_THREADS);
    }else{
        num_threads = atoi(argv[1]);
    }
    printf("Working with %d threads to sort an array of size %d\n", num_threads, ARR_LEN);
	for (int i = 0; i < ARR_LEN; ++i){
		original[i] = generate_random(MIN_VAL, MAX_VAL);
	}

	copy(original,x,ARR_LEN);
	printf("Sequential... ");
    clock_t begin = clock();
	sort_seq(x,ARR_LEN);
	clock_t end = clock();
    double seq_time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("done.\n");
	copy(x,expected,ARR_LEN);

	
	copy(original,x,ARR_LEN);
	printf("Parallel... ");
    begin = clock();
    sort_par(x, ARR_LEN,num_threads);
    end = clock();
    double par_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("done.\n");

    assert(x,expected,ARR_LEN);

    printf("\n- ==== Performance ==== -\n");
    printf("Sequential time: %fs\n",seq_time);
    printf("Parallel   time: %fs\n",par_time);
	
	return 0;
}

/************ Util. Functions *****************/

/* Generate a random number */
int generate_random(int min, int max)
{
	/* Generate a random number between min and max */
	return (rand() % (max - min + 1)) + min;
}

void copy(int from[], int to[], int size)
{
	for (int i = 0; i < size; i++)
	{
		to[i] = from[i];
	}
}

void assert(int expected[], int result[], int size)
{
	for (int i = 0; i < size; i++)
	{
		if (expected[i] != result[i])
		{
			printf("[ERROR]Bad result in position %d: expect %d, but %d was found.\n", i, expected[i], result[i]);
			exit(1);
		}
	}
}

void print(int x[], int size)
{
	for (int i = 0; i < size; i++)
	{
		printf(" %d", x[i]);
	}
	printf("\n");
}