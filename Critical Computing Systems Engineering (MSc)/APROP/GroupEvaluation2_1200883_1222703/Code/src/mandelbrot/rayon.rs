use std::time::Duration;
use std::time::Instant;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::helper::test_point;
use crate::constants::SHOW_RESULTS;

pub fn mandelbrot_rayon(nsize: usize, block_size: usize, num_threads: usize) -> Duration {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap()
    ;

    let start = Instant::now();

    let nblocks = nsize / block_size;
    let outsiders: i32 = pool.install(|| {
        (0..nblocks).into_par_iter().map(|bi| {
            let mut out= 0.0;
            for bj in 0..nblocks {
                for i in bi * block_size..(bi + 1) * block_size {
                    for j in bj * block_size..(bj + 1) * block_size {
                        out += test_point(i, j, nsize);
                    }
                }
            }
            out as i32
        }).sum()
    });

    if SHOW_RESULTS {
        let nsquared = (nsize*nsize) as f64;
        let area = 2.0*2.5*1.125*((nsquared - (outsiders as f64)) / nsquared);
        let error = area / (nsize as f64);
    
        println!("RAY\tarea: {area}, error: {error}y");
    }

    start.elapsed()
}