use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;

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

// Function to calculate the median
fn median(vec: &Vec<i32>) -> Option<f64> {
    if vec.is_empty() {
        None
    } else {
        let len = vec.len();
        Some(if len % 2 == 0 {
            (vec[len / 2 - 1] + vec[len / 2]) as f64 / 2.0
        } else {
            vec[len / 2] as f64
        })
    }
}

// Function to calculate the median on an unsorted vector
fn median_unsorted(vec: &Vec<i32>) -> Option<f64> {
    let mut sorted_vec = vec.clone();
    sorted_vec.sort();
    median(&sorted_vec)
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
    let mut rng = rand::thread_rng();

    // Generate a random vector with 10 elements between 1 and 100
    let vec: Arc<Vec<i32>> = Arc::new((0..10).map(|_| rng.gen_range(1..101)).collect());
    println!("Random Vector: {:?}", vec);

    // Shared mutable storage for computation results
    let results = Arc::new(Mutex::new(Vec::new()));

    // Spawn threads to calculate max, min, average, and median
    let vec_clone = Arc::clone(&vec);
    let results_clone = Arc::clone(&results);
    let handle1 = thread::spawn(move || {
        let max_val = max(&vec_clone);
        results_clone.lock().unwrap().push(format!("Max: {:?}", max_val.unwrap()));
    });

    let vec_clone = Arc::clone(&vec);
    let results_clone = Arc::clone(&results);
    let handle2 = thread::spawn(move || {
        let min_val = min(&vec_clone);
        results_clone.lock().unwrap().push(format!("Min: {:?}", min_val.unwrap()));
    });

    let vec_clone = Arc::clone(&vec);
    let results_clone = Arc::clone(&results);
    let handle3 = thread::spawn(move || {
        let avg = average(&vec_clone);
        results_clone.lock().unwrap().push(format!("Average: {:.2}", avg.unwrap()));
    });

    let vec_clone = Arc::clone(&vec);
    let results_clone = Arc::clone(&results);
    let handle4 = thread::spawn(move || {
        let med = median_unsorted(&vec_clone);
        results_clone.lock().unwrap().push(format!("Median (unsorted): {:.2}", med.unwrap()));
    });

    // Sort the vector concurrently
    let vec_clone = Arc::clone(&vec);
    let handle5 = thread::spawn(move || {
        let mut quick_sort_vec = vec_clone.to_vec();
        quick_sort(&mut quick_sort_vec);
        println!("Quick Sorted Vector: {:?}", quick_sort_vec);
    });

    let vec_clone = Arc::clone(&vec);
    let handle6 = thread::spawn(move || {
        let mut bubble_sort_vec = vec_clone.to_vec();
        bubble_sort(&mut bubble_sort_vec);
        println!("Bubble Sorted Vector: {:?}", bubble_sort_vec);
    });

    // Wait for all threads to complete
    handle1.join().unwrap();
    handle2.join().unwrap();
    handle3.join().unwrap();
    handle4.join().unwrap();
    handle5.join().unwrap();
    handle6.join().unwrap();

    // Print results
    for result in results.lock().unwrap().iter() {
        println!("{}", result);
    }

    // Generate two random vectors for vector addition
    let v1: Arc<Vec<i32>> = Arc::new((0..3).map(|_| rng.gen_range(1..101)).collect());
    let v2: Arc<Vec<i32>> = Arc::new((0..3).map(|_| rng.gen_range(1..101)).collect());
    println!("\nRandom Vectors 1 and 2 for Addition: {:?} and {:?}", v1, v2);

    // Compute vector addition
    let v1_clone = Arc::clone(&v1);
    let v2_clone = Arc::clone(&v2);
    let results_clone = Arc::clone(&results);
    let handle_add = thread::spawn(move || {
        match vector_addition(&v1_clone, &v2_clone) {
            Some(v3) => results_clone.lock().unwrap().push(format!("Vector Addition Result: {:?}", v3)),
            None => results_clone.lock().unwrap().push("Vectors have different lengths, cannot add.".to_string()),
        }
    });

    handle_add.join().unwrap();

    // Print final vector addition result
    for result in results.lock().unwrap().iter() {
        println!("{}", result);
    }
}
