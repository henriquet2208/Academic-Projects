use std::time::Duration;
use std::time::Instant;
use threadpool::ThreadPool;

use std::sync::mpsc::channel;
use crate::helper::test_point;
use crate::constants::SHOW_RESULTS;


pub fn mandelbrot_threadpool(nsize: usize, block_size: usize, num_threads: usize) -> Duration {
    let pool = ThreadPool::new(num_threads);
    let (tx, rx) = channel();
    let start = Instant::now();
    let mut outsiders = 0.0;

    let nblocks = nsize / block_size;
    for bi in 0..nblocks {
        let mut out = 0.0;
        let tx = tx.clone();
        pool.execute(move || {
            for bj in 0..nblocks {
                for i in bi * block_size..(bi + 1) * block_size {
                    for j in bj * block_size..(bj + 1) * block_size {
                        out += test_point(i, j, nsize);
                    }
                }
            }
            tx.send(out).unwrap();
        });
    }

    drop(tx);

    if SHOW_RESULTS {
        for out in rx {
            outsiders += out;
        }
    
        let nsquared = (nsize*nsize) as f64;
        let area = 2.0*2.5*1.125*((nsquared - outsiders) / nsquared);
        let error = area / (nsize as f64);
    
        println!("POO\tarea: {area}, error: {error}y");
    } else {
        for _ in rx {}
    }

    start.elapsed()
}