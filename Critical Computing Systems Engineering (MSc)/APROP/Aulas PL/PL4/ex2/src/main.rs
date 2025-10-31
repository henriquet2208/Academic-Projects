use rand::Rng;
use std::fmt;

// Define the Complex struct
#[derive(Debug)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

// Implement custom formatting for Complex to display in the desired format
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.imag < 0.0 {
            write!(f, "{:.2} - {:.2}i", self.real, -self.imag)
        } else {
            write!(f, "{:.2} + {:.2}i", self.real, self.imag)
        }
    }
}

impl Complex {
    // Initialize a new complex number
    pub fn new(real: f64, imag: f64) -> Complex {
        Complex { real, imag }
    }

    // Add two complex numbers
    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }

    // Subtract two complex numbers
    pub fn subtract(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }

    // Multiply two complex numbers
    pub fn multiply(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    // Divide two complex numbers
    pub fn divide(&self, other: &Complex) -> Option<Complex> {
        let denominator = other.real * other.real + other.imag * other.imag;
        if denominator == 0.0 {
            None // Avoid division by zero
        } else {
            Some(Complex {
                real: (self.real * other.real + self.imag * other.imag) / denominator,
                imag: (self.imag * other.real - self.real * other.imag) / denominator,
            })
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate random complex numbers with six decimal places
    let c1 = Complex::new(
        rng.gen_range(-10.0..10.0),
        rng.gen_range(-10.0..10.0),
    );
    let c2 = Complex::new(
        rng.gen_range(-10.0..10.0),
        rng.gen_range(-10.0..10.0),
    );

    println!("\n");
    println!("Random Complex Number 1: {}", c1);
    println!("Random Complex Number 2: {}", c2);
    println!("\n");

    // Test addition
    let result_add = c1.add(&c2);
    println!("Addition: {}", result_add);

    // Test subtraction
    let result_subtract = c1.subtract(&c2);
    println!("Subtraction: {}", result_subtract);

    // Test multiplication
    let result_multiply = c1.multiply(&c2);
    println!("Multiplication: {}", result_multiply);

    // Test division
    match c1.divide(&c2) {
        Some(result_divide) => println!("Division: {}\n", result_divide),
        None => println!("Division by zero error!"),
    }
}
