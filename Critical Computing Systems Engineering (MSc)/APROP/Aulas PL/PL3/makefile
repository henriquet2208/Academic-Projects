BIN= init mmult mandel_seq mandel_par mandel

NUM_THREADS = 4
all: $(BIN)

init:
	mkdir -p bin
	
mmult: src/mmult.c
	gcc -Wall src/mmult.c -o bin/mmult -fopenmp 

mmult_run: mmult
	bin/mmult $(NUM_THREADS)

mandel: src/mandelbrot.c
	gcc -Wall src/mandelbrot.c -o bin/mandel -fopenmp -lm 

mandel_run: mandel
	bin/mandel $(NUM_THREADS)

quicksort: src/quick_sort.c
	gcc -Wall src/quick_sort.c -o bin/quicksort -fopenmp 

quicksort_run: quicksort
	bin/quicksort $(NUM_THREADS)

bubblesort: src/bubble_sort.c
	gcc -Wall src/bubble_sort.c -o bin/bubblesort -fopenmp 

bubblesort_run: bubblesort
	bin/bubblesort $(NUM_THREADS)

ex5_gcc: src/ex5.c
	gcc -Wall src/ex5.c -o bin/ex5_gcc -fopenmp 

ex5_llvm: src/ex5.c
	clang src/ex5.c -o bin/ex5_llvm -fopenmp 

ex5_gcc_run: ex5_gcc
	bin/ex5_gcc results_gcc.csv

ex5_llvm_run: ex5_llvm
	bin/ex5_llvm results_llvm.csv