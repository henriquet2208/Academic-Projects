use rand::Rng;

fn generate_random_matrix(rows: usize, cols: usize) -> Vec<Vec<i32>> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![vec![0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            matrix[i][j] = rng.gen_range(0..10); // Random values between 0 and 9
        }
    }
    matrix
}

fn multiply_matrices(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let cols_b = b[0].len();

    // Result matrix with dimensions rows_a x cols_b
    let mut result = vec![vec![0; cols_b]; rows_a];
    
    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
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
    for row in &matrix_a {
        println!("{:?}", row);
    }

    println!("\nMatrix B:");
    for row in &matrix_b {
        println!("{:?}", row);
    }

    // Perform matrix multiplication
    let result = multiply_matrices(&matrix_a, &matrix_b);

    println!("\nResult of A x B:");
    for row in &result {
        println!("{:?}", row);
    }
}
