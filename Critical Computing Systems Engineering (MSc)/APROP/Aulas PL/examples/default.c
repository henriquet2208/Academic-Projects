#include <stdio.h>
#include <omp.h>

void main(int argc, char argv[]){

    int a = 5, b = 10;

    #pragma omp parallel default(none) shared(a)
    {
        /*
            b precisa ser re-declarado
            aqui dentro porque só a
            variável a está marcada como
            partilhada pelo #pragma.

            Na prática, b não existe dentro
            deste espaço a menos que nós o
            criemos de novo. Qualquer valor
            que ele assumir aqui será
            descartado quando as threads
            acabarem.
        */
        int b = 5;
        int c = a + b;
        printf("c is: %d\n", c);
    }
}