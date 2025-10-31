use rand::Rng;

// Function to calculate the maximum element in a vector
fn max(vec: &Vec<i32>) -> Option<i32> {
    vec.iter().cloned().max()
}

// Function to calculate the minimum element in a vector
fn min(vec: &Vec<i32>) -> Option<i32> {
    vec.iter().cloned().min()
}

// Function to calculate the average of elements in a vector
fn average(vec: &Vec<i32>) -> Option<f64> {
    if vec.is_empty() {
        None
    } else {
        Some(vec.iter().sum::<i32>() as f64 / vec.len() as f64)
    }
}

// Function to calculate the median, assuming the vector is already sorted
fn median(vec: &Vec<i32>) -> Option<f64> {
    if vec.is_empty() {
        None
    } else if !vec.windows(2).all(|w| w[0] <= w[1]) {
        panic!("Vector is not sorted!");
    } else {
        let len = vec.len();
        Some(if len % 2 == 0 {
            (vec[len / 2 - 1] + vec[len / 2]) as f64 / 2.0
        } else {
            vec[len / 2] as f64
        })
    }
}

// Function to calculate the median after sorting a copy of the vector if not already sorted
fn median_unsorted(vec: &Vec<i32>) -> Option<f64> {
    if vec.windows(2).all(|w| w[0] <= w[1]) {
        median(vec)
    } else {
        let mut sorted_vec = vec.clone();
        sorted_vec.sort();
        median(&sorted_vec)
    }
}

// Function to add elements of two vectors to create a new vector
fn vector_addition(v1: &Vec<i32>, v2: &Vec<i32>) -> Option<Vec<i32>> {
    if v1.len() != v2.len() {
        return None;
    }
    Some(v1.iter().zip(v2.iter()).map(|(x, y)| x + y).collect())
}

// Quick Sort implementation
fn quick_sort(vec: &mut [i32]) {
    if vec.len() <= 1 {
        return;
    }

    let pivot_index = vec.len() / 2;
    vec.swap(pivot_index, vec.len() - 1);
    let pivot = vec[vec.len() - 1];

    let mut i = 0;
    for j in 0..vec.len() - 1 {
        if vec[j] < pivot {
            vec.swap(i, j);
            i += 1;
        }
    }
    vec.swap(i, vec.len() - 1);

    quick_sort(&mut vec[0..i]);
    quick_sort(&mut vec[i + 1..]);
}

// Bubble Sort implementation
fn bubble_sort(vec: &mut [i32]) {
    let len = vec.len();
    for i in 0..len {
        for j in 0..len - 1 - i {
            if vec[j] > vec[j + 1] {
                vec.swap(j, j + 1);
            }
        }
    }
}

fn main() {
    // Generate a random vector with 10 elements between 1 and 100
    let mut rng = rand::thread_rng();
    let vec: Vec<i32> = (0..10).map(|_| rng.gen_range(1..101)).collect();
    println!("Random Vector: {:?}", vec);

    // Test max, min, and average functions
    println!("Max: {}", max(&vec).unwrap());
    println!("Min: {}", min(&vec).unwrap());
    println!("Average: {:.2}", average(&vec).unwrap());

    // Test median function on an already sorted vector
    let mut sorted_vec = vec.clone();
    sorted_vec.sort();
    println!("Median (sorted): {:.2}", median(&sorted_vec).unwrap());

    // Test median_unsorted function on an unsorted vector
    println!("Median (unsorted): {:.2}", median_unsorted(&vec).unwrap());

    // Test quick sort
    let mut quick_sort_vec = vec.clone();
    quick_sort(&mut quick_sort_vec);
    println!("Quick Sorted Vector: {:?}", quick_sort_vec);

    // Test bubble sort
    let mut bubble_sort_vec = vec.clone();
    bubble_sort(&mut bubble_sort_vec);
    println!("Bubble Sorted Vector: {:?}", bubble_sort_vec);

    // Generate two random vectors with 3 elements each for vector addition
    let v1: Vec<i32> = (0..3).map(|_| rng.gen_range(1..101)).collect();
    let v2: Vec<i32> = (0..3).map(|_| rng.gen_range(1..101)).collect();
    println!("\nRandom Vectors 1 and 2 for Addition: {:?} and {:?}", v1, v2);

    // Test vector addition function
    match vector_addition(&v1, &v2) {
        Some(v3) => println!("Vector Addition Result: {:?}", v3),
        None => println!("Vectors have different lengths, cannot add."),
    }
}

