mod rayon;
mod threadpool;
mod sequential;

pub use rayon::mandelbrot_rayon;
pub use threadpool::mandelbrot_threadpool;
pub use sequential::mandelbrot_sequential;