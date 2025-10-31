use rand::Rng;
use std::collections::HashMap;

fn mode(list: &[i32]) -> Option<Vec<i32>> {
    let mut map = HashMap::new();

    // Count occurrences of each number
    for &num in list {
        let count = map.entry(num).or_insert(0);
        *count += 1;
    }

    // Find the maximum frequency
    let max_value = *map.values().max().unwrap_or(&0);

    // Collect all numbers with the maximum frequency
    let modes: Vec<i32> = map.iter()
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
    
    match mode(&vec) {
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
