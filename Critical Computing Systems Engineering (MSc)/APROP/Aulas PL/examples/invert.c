#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_RAND 1000

#define NUM_THREADS 4

// helper /////////////////////////////////////////////////////////////////////

void show(char *prefix, int *numbers, int size){
    printf("%s: [ ", prefix);
    for(int i = 0; i < size; i++){
        printf("%d ", numbers[i]);
    }
    printf("]\n");
}


void populate(int *numbers, int size){
    srand(time(NULL));
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < size; i++){
        numbers[i] = (int) rand()%(MAX_RAND + 1);
    }
    printf("Array has been populated with random integers.\n");
}


void validate(int argc, char *argv[], int *size){
    if(argc != 2){
        printf("usage: ./invert <size>\n\n");
        return;
    }
    int length = atoi(argv[1]);
    if((length <= 0) || (length % NUM_THREADS)){
        printf("invalid array size: %d. Must be >= 0 and multiple of %d\n\n", length, NUM_THREADS);
        return;
    }
    *size = length;
}

/////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]){
    int size = 0;

    validate(argc, argv, &size);
    if(!size) return 0;

    printf("size: %d\n", size);
    
    int *array = malloc(size * sizeof(int));
    
    populate(array, size);
    show("\noriginal", array, size);

    int last = (size - 1);
    int half = (size >> 1);

    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < half; i++){
        int buffer = array[i];
        array[i] = array[last-i];
        array[last-i] = buffer; 
    }

    show("inverted", array, size);
    printf("\n");
    free(array);
}


