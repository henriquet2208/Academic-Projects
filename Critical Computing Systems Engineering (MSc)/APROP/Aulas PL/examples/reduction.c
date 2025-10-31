#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 1000000
#define MAX_RAND 10


void populate(int *numbers){
    srand(time(NULL));
    #pragma omp parallel for
    for(int i = 0; i < SIZE; i++){
        numbers[i] = (int) rand()%(MAX_RAND + 1);
    }
    printf("populated the array with random integers.\n");
}


void main(int argc, char argv[]){
    int numbers[SIZE];
    int sum = 0;

    populate(numbers);

    #pragma omp parallel num_threads(4) reduction(+:sum)
    {
        int id = omp_get_thread_num();
        int local_max = 0;
        #pragma omp for
        for(int i = 0; i < SIZE; i++){
            if(numbers[i] <= local_max) continue;
            local_max = numbers[i];
        }
        printf("local maximum (thread %d) is: %d\n", id, local_max);
        sum += local_max;
    }

    printf("The sum is: %d\n", sum);
}