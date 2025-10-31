import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('results_gcc.csv')

# Ensure that the dataframe is structured correctly
print(df.head())

# Rename columns to avoid key errors
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()  # Strip spaces and lowercase the names

# Extracting the necessary data for 3D plots
num_runs = df['run'].values
seq_times = df['sequential_time'].values
task_times = df['task-based_time'].values
block_times = df['block-based_time'].values
num_threads = df['num_threads'].values
block_sizes = df['bs'].values  # Renamed to match
N_values = df['n'].values  # Renamed to match

# 3D plot for changing threads
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(num_runs, num_threads, seq_times, label='Sequential Time', color='r')
ax1.plot(num_runs, num_threads, seq_times, color='r', alpha=0.3)  # Add lines
ax1.scatter(num_runs, num_threads, task_times, label='Task-Based Time', color='g')
ax1.plot(num_runs, num_threads, task_times, color='g', alpha=0.3)  # Add lines
ax1.scatter(num_runs, num_threads, block_times, label='Block-Based Time', color='b')
ax1.plot(num_runs, num_threads, block_times, color='b', alpha=0.3)  # Add lines

ax1.set_xlabel('Run Number')
ax1.set_ylabel('Number of Threads')
ax1.set_zlabel('Execution Time (s)')
ax1.set_title('Execution Times vs. Number of Threads')
ax1.legend()

# 3D plot for changing block sizes
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(num_runs, block_sizes, seq_times, label='Sequential Time', color='r')
ax2.plot(num_runs, block_sizes, seq_times, color='r', alpha=0.3)  # Add lines
ax2.scatter(num_runs, block_sizes, task_times, label='Task-Based Time', color='g')
ax2.plot(num_runs, block_sizes, task_times, color='g', alpha=0.3)  # Add lines
ax2.scatter(num_runs, block_sizes, block_times, label='Block-Based Time', color='b')
ax2.plot(num_runs, block_sizes, block_times, color='b', alpha=0.3)  # Add lines

ax2.set_xlabel('Run Number')
ax2.set_ylabel('Block Size')
ax2.set_zlabel('Execution Time (s)')
ax2.set_title('Execution Times vs. Block Size')
ax2.legend()

# 3D plot for changing N values
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(num_runs, N_values, seq_times, label='Sequential Time', color='r')
ax3.plot(num_runs, N_values, seq_times, color='r', alpha=0.3)  # Add lines
ax3.scatter(num_runs, N_values, task_times, label='Task-Based Time', color='g')
ax3.plot(num_runs, N_values, task_times, color='g', alpha=0.3)  # Add lines
ax3.scatter(num_runs, N_values, block_times, label='Block-Based Time', color='b')
ax3.plot(num_runs, N_values, block_times, color='b', alpha=0.3)  # Add lines

ax3.set_xlabel('Run Number')
ax3.set_ylabel('Matrix Size (N)')
ax3.set_zlabel('Execution Time (s)')
ax3.set_title('Execution Times vs. Matrix Size (N)')
ax3.legend()

# Show all plots
plt.show()
