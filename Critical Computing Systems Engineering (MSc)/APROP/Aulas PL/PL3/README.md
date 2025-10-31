# Requirements

GCC compiler and openmp library (should be included in gcc).

An IDE for programming is recommended (we suggest: VSCode) 

# Initial setup

Run the following command
```
make init
```
This will simply to create a "bin" folder

# Build
To build the exercises:

```
make mmult
```
or
```
make mandel
```
or
```
make quicksort
```

# Running
To run the exercises:

```
bin/mmult [num_threads]
```
or
```
bin/mandel [num_threads]
```
or
```
bin/quicksort [num_threads]
```
Note that num_threads is optional and will default to the defined DEFAULT_NUM_THREADS

# Build and Run in the same command
All of the previous targetshave an extra target that allows one to build and run in the same command, by simply adding the "_run" prefix of the make target. For instance, to build and immediatelly run the mmult example, simply use:

```
make mmult_run [NUM_THREADS=<num_threads>]
```
This will build mmult with "make mmult" and then run the program with "bin/mmult $NUM_THREADS"

Note that NUM_THREADS=<num_threads> is optional, and will default to the value in the environment variable defined in the Makefile. More specifically, if you want to use 8 threads and use this command:
```
make quicksort_run NUM_THREADS=8
```
If you don't want to specify the number of threads and use the default value (which in principle will be 4), then just use:
```
make quicksort_run
```


