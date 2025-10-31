use rand::Rng;
use nalgebra::DMatrix;
use std::sync::{Arc, Mutex};
use std::thread;

fn generate_random_matrix(rows: usize, cols: usize) -> DMatrix<i32> {
    let mut rng = rand::thread_rng();
    let data: Vec<i32> = (0..rows * cols).map(|_| rng.gen_range(0..10)).collect();
    DMatrix::from_row_slice(rows, cols, &data)
}

fn multiply_matrices_parallel(a: &DMatrix<i32>, b: &DMatrix<i32>) -> DMatrix<i32> {
    let rows_a = a.nrows();
    let cols_b = b.ncols();
    let cols_a = a.ncols();

    // Ensure matrix dimensions are valid for multiplication
    assert_eq!(cols_a, b.nrows(), "Matrix dimensions do not match for multiplication.");

    // Shared result matrix, protected by a Mutex
    let result: Arc<Mutex<DMatrix<i32>>> = Arc::new(Mutex::new(DMatrix::zeros(rows_a, cols_b)));

    let mut handles = vec![];

    for i in 0..rows_a {
        // Clone the row to avoid borrowing issues
        let row = a.row(i).into_owned(); // `into_owned` ensures an independent copy
        let b_clone = b.clone(); // Clone matrix B for thread safety
        let result_clone: Arc<Mutex<DMatrix<i32>>> = Arc::clone(&result);

        // Spawn a thread to compute the i-th row of the result
        let handle = thread::spawn(move || {
            let mut local_row = vec![0; cols_b];

            for j in 0..cols_b {
                local_row[j] = (0..cols_a).map(|k| row[k] * b_clone[(k, j)]).sum();
            }

            // Write the computed row into the shared result matrix
            let mut result_lock = result_clone.lock().unwrap();
            for j in 0..cols_b {
                result_lock[(i, j)] = local_row[j];
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().unwrap();
    }

    // Return the result matrix
    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate random dimensions for matrices A and B within a range of 1 to 10
    let rows_a = rng.gen_range(1..=10);
    let cols_a = rng.gen_range(1..=10);
    let rows_b = cols_a; // To ensure matrix multiplication is valid, set rows_b = cols_a
    let cols_b = rng.gen_range(1..=10);

    println!("Matrix A dimensions: {}x{}", rows_a, cols_a);
    println!("Matrix B dimensions: {}x{}", rows_b, cols_b);

    // Generate two random matrices of the specified dimensions
    let matrix_a = generate_random_matrix(rows_a, cols_a);
    let matrix_b = generate_random_matrix(rows_b, cols_b);

    println!("\nMatrix A:");
    println!("{}", matrix_a);

    println!("\nMatrix B:");
    println!("{}", matrix_b);

    // Perform matrix multiplication using threads
    let result = multiply_matrices_parallel(&matrix_a, &matrix_b);

    println!("\nResult of A x B:");
    println!("{}", result);
}
