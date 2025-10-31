use rand::Rng;
use nalgebra::DMatrix;

fn generate_random_matrix(rows: usize, cols: usize) -> DMatrix<i32> {
    let mut rng = rand::thread_rng();
    let data: Vec<i32> = (0..rows * cols).map(|_| rng.gen_range(0..10)).collect();
    DMatrix::from_row_slice(rows, cols, &data)
}

fn multiply_matrices(a: &DMatrix<i32>, b: &DMatrix<i32>) -> DMatrix<i32> {
    a * b // Using the overloaded operator to multiply two DMatrix instances
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

    // Perform matrix multiplication
    let result = multiply_matrices(&matrix_a, &matrix_b);

    println!("\nResult of A x B:");
    println!("{}", result);
}
