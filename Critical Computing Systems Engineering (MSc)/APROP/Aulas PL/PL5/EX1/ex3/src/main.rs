use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// Function to compute the mode of a list using threads
fn mode_parallel(list: &[i32]) -> Option<Vec<i32>> {
    let num_threads = 4; // Number of threads to use
    let chunk_size = (list.len() + num_threads - 1) / num_threads;

    // Shared HashMap to store the combined frequency counts
    let frequency_map = Arc::new(Mutex::new(HashMap::new()));

    let mut handles = vec![];

    // Divide the work among threads
    for chunk in list.chunks(chunk_size) {
        let freq_map_clone = Arc::clone(&frequency_map);

        // Create a copy of the chunk for thread safety
        let chunk_owned: Vec<i32> = chunk.to_vec();

        let handle = thread::spawn(move || {
            let mut local_map = HashMap::new();

            // Count frequencies in the local chunk
            for &num in &chunk_owned {
                *local_map.entry(num).or_insert(0) += 1;
            }

            // Merge local_map into the shared frequency_map
            let mut freq_map = freq_map_clone.lock().unwrap();
            for (num, count) in local_map {
                *freq_map.entry(num).or_insert(0) += count;
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().unwrap();
    }

    // Find the mode(s)
    let freq_map = frequency_map.lock().unwrap();
    let max_value = *freq_map.values().max().unwrap_or(&0);

    let modes: Vec<i32> = freq_map
        .iter()
        .filter(|&(_, &count)| count == max_value)
        .map(|(&num, _)| num)
        .collect();

    // Return None if all numbers are unique, otherwise return the modes
    if modes.len() == list.len() {
        None // No mode if each element is unique
    } else {
        Some(modes)
    }
}

fn main() {
    // Generate a random vector with 10 elements between 1 and 100
    let mut rng = rand::thread_rng();
    let vec: Vec<i32> = (0..10).map(|_| rng.gen_range(1..101)).collect();
    println!("\nRandom Vector: {:?}", vec);

    // Compute the mode using the parallel implementation
    match mode_parallel(&vec) {
        Some(modes) => {
            if modes.len() == 1 {
                println!("Mode: {}\n", modes[0]);
            } else {
                println!("Modes: {:?}\n", modes);
            }
        }
        None => println!("Mode: None (all elements are unique)\n"),
    }
}
