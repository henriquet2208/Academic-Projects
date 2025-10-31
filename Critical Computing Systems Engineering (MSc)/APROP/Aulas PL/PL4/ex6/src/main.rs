fn sum_numbers_1_to_100() -> i32 {
    (1..=100).sum() // Using an iterator over the range 1 to 100
}

fn sum_squares_lower_than(upper_bound: i32) -> i32 {
    (1..).map(|x| x * x) // Generate squares of numbers starting from 1
        .take_while(|&x| x < upper_bound) // Take squares less than the upper_bound
        .sum() // Sum the squares
}

fn main() {
    // Task a: Sum of all numbers from 1 to 100
    let sum_1_to_100 = sum_numbers_1_to_100();
    println!("The sum of all numbers from 1 to 100 is: {}", sum_1_to_100);

    // Task b: Sum of all squares lower than an upper_bound value
    let upper_bound = 50; // Example upper bound
    let sum_of_squares = sum_squares_lower_than(upper_bound);
    println!("The sum of all squares lower than {} is: {}", upper_bound, sum_of_squares);
}