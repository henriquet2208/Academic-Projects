BIN= init mmult mandel_seq mandel_par mandel

NUM_THREADS = 4
all: $(BIN)

init:
	mkdir -p bin

################################################## MMULT CODES ##########################################################

mmult: src/mmult.c
	gcc -Wall src/mmult.c -o bin/mmult -fopenmp 

mmult_run: mmult
	bin/mmult $(NUM_THREADS)

mmult_parallel_loops: src/mmult_parallel_loops.c
	gcc -Wall src/mmult_parallel_loops.c -o bin/mmult_parallel_loops -fopenmp 

mmult_parallel_loops_run: mmult_parallel_loops
	bin/mmult_parallel_loops $(NUM_THREADS)

mmult_b_parallel_loops: src/mmult_b_parallel_loops.c
	gcc -Wall src/mmult_b_parallel_loops.c -o bin/mmult_b_parallel_loops -fopenmp 

mmult_b_parallel_loops_run: mmult_b_parallel_loops
	bin/mmult_b_parallel_loops $(NUM_THREADS)


################################################## MANDEL CODES ##########################################################

mandel_parallel_loops: src/mandel_parallel_loops.c
	gcc -Wall src/mandel_parallel_loops.c -o bin/mandel_parallel_loops -fopenmp 

mandel_parallel_loops_run: mandel_parallel_loops
	bin/mandel_parallel_loops $(NUM_THREADS)

mandel_b_parallel_loops: src/mandel_b_parallel_loops.c
	gcc -Wall src/mandel_b_parallel_loops.c -o bin/mandel_b_parallel_loops -fopenmp 

mandel_b_parallel_loops_run: mandel_b_parallel_loops
	bin/mandel_b_parallel_loops $(NUM_THREADS)	

mandel: src/mandelbrot.c
	gcc -Wall src/mandelbrot.c -o bin/mandel -fopenmp -lm 

mandel_run: mandel
	bin/mandel $(NUM_THREADS)

###################################################### OTHER CODES ####################################################

quicksort: src/quick_sort.c
	gcc -Wall src/quick_sort.c -o bin/quicksort -fopenmp 

quicksort_run: quicksort
	bin/quicksort $(NUM_THREADS)

bubblesort: src/bubble_sort.c
	gcc -Wall src/bubble_sort.c -o bin/bubblesort -fopenmp 

bubblesort_run: bubblesort
	bin/quicksort $(NUM_THREADS)