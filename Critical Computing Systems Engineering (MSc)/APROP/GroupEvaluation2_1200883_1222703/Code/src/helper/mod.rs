use std::time::Duration;
use crate::constants::MANDEL_RUNS;

#[derive(Copy, Clone)]
pub struct Complex {
    r: f64,
    i: f64,
}

// the core mandelbrot test is made here
pub fn test_point(i: usize, j: usize, nsize: usize) -> f64 {
    let c = Complex {
        r: -2.0 + 2.5 * (i as f64) / (nsize as f64),
        i: 1.125 * (j as f64) / (nsize as f64),
    };
    let mut z = c;
    for _ in 0..MANDEL_RUNS {
        let temp = z.r * z.r - z.i * z.i + c.r;
        z.i = 2.0 * z.r * z.i + c.i;
        z.r = temp;
        if z.r * z.r + z.i * z.i > 4.0 {
            return 1.0;
        }
    }
    return 0.0;
}


// Function to calculate statistics
pub fn calculate_stats(times: &[Duration]) -> (Duration, Duration, Duration, f64) {
    
    let total: Duration = times.iter().copied().sum();
    let average = total / (times.len() as u32);
    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();

    let stddev = {
        let mean = average.as_secs_f64();
        let variance = times.iter().map(|t| {
            let diff = t.as_secs_f64() - mean;
            diff * diff
        }).sum::<f64>() / times.len() as f64;
        variance.sqrt()
    };
    
    (min, max, average, stddev)
}