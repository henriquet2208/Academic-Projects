use std::time::Instant;

const NPOINTS: usize = 1000;
const MAXITER: usize = 1000;

struct Result {
    area: f64,
    error: f64,
}

#[derive(Debug, Clone)] // Implement Clone for Complex
struct Complex {
    r: f64,
    i: f64,
}

struct Mandelbrot {
    num_outside: usize,
}

impl Mandelbrot {
    fn new() -> Self {
        Mandelbrot { num_outside: 0 }
    }

    fn test_point(&mut self, c: Complex) {
        let mut z = c.clone(); // Clone c to avoid moving it
        for _ in 0..MAXITER {
            let temp = z.r * z.r - z.i * z.i + c.r;
            z.i = 2.0 * z.r * z.i + c.i;
            z.r = temp;
            if (z.r * z.r + z.i * z.i) > 4.0 {
                self.num_outside += 1;
                return; // Exit early if outside the set
            }
        }
    }

    fn seq_mandel(&mut self) -> Result {
        let eps = 1.0e-5;
        for i in 0..NPOINTS {
            for j in 0..NPOINTS {
                let c = Complex {
                    r: -2.0 + 2.5 * (i as f64) / (NPOINTS as f64) + eps,
                    i: 1.125 * (j as f64) / (NPOINTS as f64) + eps,
                };
                self.test_point(c);
            }
        }

        // Calculate area of the set and error estimate
        let area = 2.0 * 2.5 * 1.125 * (NPOINTS * NPOINTS - self.num_outside) as f64 / (NPOINTS * NPOINTS) as f64;
        let error = area / (NPOINTS as f64);

        Result { area, error }
    }
}

fn main() {
    let mut mandelbrot = Mandelbrot::new();
    
    println!("\nSequential Mandelbrot... ");
    let start = Instant::now();
    let result = mandelbrot.seq_mandel();
    let duration = start.elapsed();
    
    println!("\nDone.\n");
    println!("[SEQ] Area of Mandelbrot set = {:12.8} +/- {:12.8} (outside: {})", result.area, result.error, mandelbrot.num_outside);
    println!("Time taken: {:?}", duration);
}