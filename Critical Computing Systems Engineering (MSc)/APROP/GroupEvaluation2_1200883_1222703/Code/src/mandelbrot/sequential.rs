use std::time::Duration;
use std::time::Instant;
use crate::helper::test_point;
use crate::constants::SHOW_RESULTS;

pub fn mandelbrot_sequential(nsize: usize) -> Duration {
    let start = Instant::now();
    let mut outsiders = 0.0;

    for i in 0..nsize {
        for j in 0..nsize {
            outsiders += test_point(i, j, nsize);
        }
    }

    if SHOW_RESULTS {
        let nsquared = (nsize*nsize) as f64;
        let area = 2.0*2.5*1.125*((nsquared - outsiders) / nsquared);
        let error = area / (nsize as f64);
    
        println!("SEQ\tarea: {area}, error: {error}");
    }

    start.elapsed()
}





