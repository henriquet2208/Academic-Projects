use threadpool::ThreadPool;
use std::sync::mpsc::channel;

#[derive(Copy, Clone)]
struct Complex {
    r: f64,
    i: f64,
}

const NPOINTS: u32 = 1000;
const MAXITER: u32 = 1000;
const EPS: f64 = 1e-5;
const NUM_THREADS: usize = 8;

fn main() {
    println!("Mandelbrot!");

    let pool = ThreadPool::new(NUM_THREADS);
    let (tx, rx) = channel();
    let mut num_outside = 0;

    for i in 0..NPOINTS {
        let tx = tx.clone();
        pool.execute(move || {
            let mut local_count = 0;
            for j in 0..NPOINTS {
                let c = Complex {
                    r: -2.0 + 2.5 * (i as f64) / (NPOINTS as f64) + EPS,
                    i: 1.125 * (j as f64) / (NPOINTS as f64) + EPS,
                };
                local_count += test_point(c);
            }
            tx.send(local_count).unwrap();
        });
    }

    drop(tx);

    for received in rx {
        num_outside += received;
    }

    let np = NPOINTS as f64;
    let size = np * np;
    let area = 2.0 * 2.5 * 1.125 * ((size - num_outside as f64) / size);
    let error = area / NPOINTS as f64;

    println!(
        "Area of Mandelbrot set = {:12.8} +/- {:12.8}\n",
        area, error
    );
}

fn test_point(c: Complex) -> i32 {
    let mut z = c.clone();
    for _ in 0..MAXITER {
        let temp = (z.r * z.r) - (z.i * z.i) + c.r;
        z.i = z.r * z.i * 2.0 + c.i;
        z.r = temp;
        if z.r * z.r + z.i * z.i > 4.0 {
            return 1;
        }
    }
    0
}
