use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;

// Function to generate a random matrix with given dimensions
fn generate_random_matrix(rows: usize, cols: usize) -> Vec<Vec<i32>> {
    let mut rng = rand::thread_rng();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(0..10)).collect())
        .collect()
}

// Function to multiply matrices using threads
fn multiply_matrices_concurrently(
    a: Arc<Vec<Vec<i32>>>,
    b: Arc<Vec<Vec<i32>>>,
    result: Arc<Mutex<Vec<Vec<i32>>>>,
) {
    let rows_a = a.len();
    let cols_b = b[0].len();
    let num_threads = rows_a.min(4); // Use up to 4 threads or fewer if rows_a < 4
    let chunk_size = (rows_a + num_threads - 1) / num_threads;

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let a = Arc::clone(&a);
        let b = Arc::clone(&b);
        let result = Arc::clone(&result);

        let handle = thread::spawn(move || {
            let start_row = thread_id * chunk_size;
            let end_row = ((thread_id + 1) * chunk_size).min(rows_a);

            for i in start_row..end_row {
                for j in 0..cols_b {
                    let mut sum = 0;
                    for k in 0..a[0].len() {
                        sum += a[i][k] * b[k][j];
                    }
                    let mut result_lock = result.lock().unwrap();
                    result_lock[i][j] = sum;
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate random dimensions for matrices A and B
    let rows_a = rng.gen_range(1..=10);
    let cols_a = rng.gen_range(1..=10);
    let rows_b = cols_a; // Ensures valid matrix multiplication
    let cols_b = rng.gen_range(1..=10);

    println!("Matrix A dimensions: {}x{}", rows_a, cols_a);
    println!("Matrix B dimensions: {}x{}", rows_b, cols_b);

    // Generate two random matrices
    let matrix_a = generate_random_matrix(rows_a, cols_a);
    let matrix_b = generate_random_matrix(rows_b, cols_b);

    println!("\nMatrix A:");
    for row in &matrix_a {
        println!("{:?}", row);
    }

    println!("\nMatrix B:");
    for row in &matrix_b {
        println!("{:?}", row);
    }

    // Prepare shared data structures for threading
    let a = Arc::new(matrix_a);
    let b = Arc::new(matrix_b);
    let result = Arc::new(Mutex::new(vec![vec![0; cols_b]; rows_a]));

    // Perform matrix multiplication with concurrency
    multiply_matrices_concurrently(a, b, result.clone());

    // Display the result
    let result = result.lock().unwrap();
    println!("\nResult of A x B:");
    for row in &*result {
        println!("{:?}", row);
    }
}
