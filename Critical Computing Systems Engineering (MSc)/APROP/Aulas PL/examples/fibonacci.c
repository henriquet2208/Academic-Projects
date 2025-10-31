#include <stdio.h>
#include <stdlib.h>

int fib(int n) {
  int x, y;
  if (n < 2) return n;

  #pragma omp task shared(x)
  x = fib(n-1);

  #pragma omp task shared(y)
  y = fib(n-2);

  #pragma omp taskwait              // <-- sinaliza que devemos esperar todas as tasks criadas anteriormente terminarem para só então continuar
  return x + y;
}

int main(int argc, char *argv[]){
    
    if(argc != 2){
        printf("Usage: %s <number>\n", argv[0]);
        return 0;
    }

    int i = atoi(argv[1]);
    int result = 0;

    #pragma omp parallel default(none) shared(i, result)
    {
        #pragma omp single
        {
            result = fib(i);
        }
    }

    printf("\nfib(%d) = %d\n\n", i, result);
    return 0;
}