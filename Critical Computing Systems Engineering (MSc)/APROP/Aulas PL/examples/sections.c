#include <stdio.h>
#include <omp.h>

void main(int argc, char argv[]){
    #pragma omp parallel
    {
        int threads = omp_get_num_threads();
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("[%d/%d]\tLuke, I am your father.\n", omp_get_thread_num(), threads);
            }

            #pragma omp section
            {
                printf("[%d/%d]\tNooooooo...\n", omp_get_thread_num(), threads);
            }
        }
        printf("This is a star wars thread.\n");
        #pragma omp single
        {
            printf("and these quotes above are from movie #5.\n");
        }
    }
}