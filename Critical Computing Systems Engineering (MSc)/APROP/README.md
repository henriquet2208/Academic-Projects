# APROP — Concurrency & Parallel Programming (Class Materials)

This directory collects class materials, lab exercises, and group evaluations for **APROP (Advanced Programming Paradigms)** focused on **concurrency and parallelism** in C/OpenMP and Rust.

---

## 👥 Author of the Practical Classes Individual Exercises
- Henrique Teixeira

---

## 👥 Authors of the Group Evaluations
- Henrique Teixeira
- Alexander Paschoaletto 

--- 

## 🔧 Prerequisites

### System
- Linux or macOS recommended (Windows with WSL works)
- `make`, `gcc`, `clang`, `git`

### C / OpenMP (GroupEvaluation1 + PL1–PL3)
- GCC with OpenMP: `sudo apt install build-essential`
- LLVM/Clang (optional): `sudo apt install clang`
- Python 3 + plotting: `pip install matplotlib pandas`

### Rust (GroupEvaluation2 + PL4–PL5)
- Rust toolchain: `curl https://sh.rustup.rs -sSf | sh`
- Ensure `cargo` is in your PATH: `rustc --version && cargo --version`

---

## ▶️ Quick Start

### 1) Practical Lectures (PL1–PL5)
- PDFs are under `APROP/Aulas PL/PLx`.
- Some PL folders include small code exercises with a `README.md` and/or simple `make` targets.

### 2) Group Evaluations (1 and 2)
- The **requirements** for each group project are included in the **same PDF files** as the smaller exercises.
Each group evaluation corresponds to a specific exercise described in the respective Practical Class (Aulas PL) PDF, located under:

   - APROP/Aulas PL/GroupEvaluation1/Project Requirements (Exercice 6) for Group Evaluation 1.

   - APROP/Aulas PL/GroupEvaluation2/Project Requirements (Exercice 4) for Group Evaluation 2.

### 2) Group Evaluation 1 — C/OpenMP
```bash
cd APROP/GroupEvaluation1_1200883_1222703/GroupEvaluation1

# Build (creates ./bin)
make

# Example targets (selected)
make mmult_run            # matrix-multiply with OpenMP (uses NUM_THREADS from makefile)
make ex5_gcc_run          # build/run ex5 with GCC, writes results_gcc.csv
make ex5_llvm_run         # build/run ex5 with Clang/LLVM, writes results_llvm.csv

# Plotting (requires matplotlib + pandas)
python3 plot_results_gcc.py
python3 plot_results_llvm.py
```

#### ⚙️ What it does

- Performs a matrix update (or computation) for different configurations:
   - NUM_THREADS (e.g., 4, 8, 16, 32)
   - N (matrix size, e.g., 1000, 2000, 2500, 3000)
   - BS (block size, e.g., 50, 100, 200, 500)
- Uses OpenMP to parallelize the computation.
- Runs multiple iterations per configuration to compute average times.
- Writes results to CSV files (results_gcc.csv and results_llvm.csv), depending on compiler used.
- At the end of each group of tests, the program prints and stores average runtimes per configuration.

**Notes**
- The `makefile` defines several targets (e.g., `mmult`, `mandel`, `bubblesort`, `ex5_gcc`, `ex5_llvm`) and matching `*_run` targets.
- Adjust thread counts via the `NUM_THREADS` variable in the `makefile` or via program arguments when applicable.
- Output CSVs (e.g., `results_gcc.csv`, `results_llvm.csv`) are consumed by the provided plot scripts.

### 3) Group Evaluation 2 — Rust (Mandelbrot benchmark)
This project compares **sequential**, **thread‑pool**, and **Rayon** implementations over different **N**, **block sizes**, and **thread counts**.

```bash
cd APROP/GroupEvaluation2_1200883_1222703/Code

# Run benchmarks (release mode recommended)
cargo run

# Plot averages across thread counts for each N and block size
python3 plot_averages.py
```

#### ⚙️ **What it does**
- Implements three versions of the same workload:
   - Sequential (sequential.rs) — baseline nested loops.
   - Thread-pool (threadpool.rs) — uses the threadpool crate with a fixed pool size.
   - Rayon (rayon.rs) — uses rayon parallel iterators with a configurable pool.
- Sweeps multiple configurations (as defined in src/main.rs):
- Example sets:
   - N ∈ {1000, 2000}
   - Block Size ∈ {50, 100, 200} (for N=1000) and {100, 200, 400} (for N=2000)
   - Threads ∈ {2, 4, 8, 16}
- Repeats each configuration RUNS times (see src/constants/mod.rs) to capture stable stats.
- Records per-run results to results.csv and per-configuration averages & std. dev. to averages.csv.

---

## 📊 Outputs
- **GroupEvaluation1**: `results_gcc.csv`, `results_llvm.csv`, plots saved/shown by the `plot_results_*.py` scripts.
- **GroupEvaluation2**: `results.csv` (per‑run), `averages.csv` (per‑config), interactive plots from `plot_averages.py`.

---

## ✅ Final Remarks
These projects provided hands-on experience with parallel computing concepts across two languages (C/OpenMP and Rust) and paradigms.
They highlight how design choices — granularity, scheduling, and thread management — directly influence performance, scalability, and maintainability.
