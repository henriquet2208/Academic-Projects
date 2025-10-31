mod mandelbrot;
mod helper;
mod constants;

use crate::constants::RUNS;
use crate::helper::calculate_stats;
use crate::mandelbrot::mandelbrot_rayon;
use crate::mandelbrot::mandelbrot_threadpool;
use crate::mandelbrot::mandelbrot_sequential;

use std::fs::File;
use csv;


fn main() {
    let results_file = File::create("results.csv").expect("Could not create results.csv");
    let averages_file = File::create("averages.csv").expect("Could not create averages.csv");

    let mut results_writer = csv::Writer::from_writer(results_file);
    let mut averages_writer = csv::Writer::from_writer(averages_file);

    results_writer.write_record(&[
        "Run",
        "N",
        "Block Size",
        "Num Threads",
        "Sequential Time",
        "Threadpool Time",
        "Rayon Time",
    ]).expect("Could not write results_write header");

   averages_writer.write_record(&[
        "N",
        "Block Size",
        "Num Threads",
        "Sequential Time",
        "Threadpool Time",
        "Rayon Time",
        "Sequential Deviation",
        "Threadpool Deviation",
        "Rayon Deviation",
    ]).expect("Could not write averages_write header");

    let configurations = vec![
        (1000, vec![50, 100, 200], vec![2, 4, 8, 16]),
        (2000, vec![100, 200, 400], vec![2, 4, 8, 16]),
    ];

    let mut run_id = 1;

    for (nsize, block_sizes, thread_counts) in configurations {

        /*
            For sequential execution only Nsize
            matters, so there's no reason to run
            it inside the loops below! They only
            change block_size and num_threads.
            
            So first we run the sequential alone.
        */
        let mut seq_times = Vec::new();
        for _ in 0..RUNS {
            let seq_time = mandelbrot_sequential(nsize);
            seq_times.push(seq_time);
        }

        /*
            Now that the sequential is done, we
            can run the threadpool and rayon
            versions of the mandelbrot function.
        */

        for block_size in block_sizes {
            for num_threads in thread_counts.clone() {
                
                let mut rayon_times = Vec::new();
                let mut threadpool_times = Vec::new();

                for i in 0..RUNS {
                    /*
                        As we already have the results for
                        the sequential execution, we can
                        just echo them here in seq_time.
                    */
                    let seq_time = seq_times[i as usize];
                    let rayon_time = mandelbrot_rayon(nsize, block_size, num_threads);
                    let threadpool_time = mandelbrot_threadpool(nsize, block_size, num_threads);

                    rayon_times.push(rayon_time);
                    threadpool_times.push(threadpool_time);

                    println!(
                        "Run {}\t[ Nsize: {}, Block Size: {}, Threads: {} ] \tSequential: {:?}\tThreadpool: {:?}\tRayon: {:?}",
                        run_id, nsize, block_size, num_threads, seq_time, threadpool_time, rayon_time
                    );

                    results_writer.write_record(&[
                        run_id.to_string(),
                        nsize.to_string(),
                        block_size.to_string(),
                        num_threads.to_string(),
                        format!("{:?}", seq_time),
                        format!("{:?}", threadpool_time),
                        format!("{:?}", rayon_time),
                    ]).expect("Could not write results data");

                    run_id += 1;
                }

                // Calculate statistics after every 10 runs
                let (seq_min, seq_max, seq_avg, seq_std_dev) = calculate_stats(&seq_times);
                let (ray_min, ray_max, ray_avg, ray_std_dev) = calculate_stats(&rayon_times);
                let (tp_min, tp_max, tp_avg, tp_std_dev) = calculate_stats(&threadpool_times);

                println!("\nSequential: Min: {:?}, Max: {:?}, Avg: {:?}, Std Dev: {:.4}", seq_min, seq_max, seq_avg, seq_std_dev);
                println!("Threadpool: Min: {:?}, Max: {:?}, Avg: {:?}, Std Dev: {:.4}", tp_min, tp_max, tp_avg, tp_std_dev);
                println!("Rayon:      Min: {:?}, Max: {:?}, Avg: {:?}, Std Dev: {:.4}\n", ray_min, ray_max, ray_avg, ray_std_dev);

                averages_writer.write_record(&[
                    nsize.to_string(),
                    block_size.to_string(),
                    num_threads.to_string(),
                    format!("{:?}", seq_avg),
                    format!("{:?}", tp_avg),
                    format!("{:?}", ray_avg),
                    format!("{:.4}", seq_std_dev),
                    format!("{:.4}", tp_std_dev),
                    format!("{:.4}", ray_std_dev),
                ]).expect("Could not write averages data");

                results_writer.flush().expect("Could not flush results_writer");
                averages_writer.flush().expect("Could not flush results_writer");
            }
        }
    }

    // writer.flush().expect("Could not flush writer");
    println!("Results saved to results.csv");
}
