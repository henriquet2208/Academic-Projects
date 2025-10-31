import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('averages.csv')

df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower() 

def convert_to_seconds(time_str):
    if 'ms' in time_str:
        return float(time_str.replace('ms', '')) * 1e-3  
    elif 's' in time_str:
        return float(time_str.replace('s', ''))  
    else:
        return float(time_str)  

df['sequential_time'] = df['sequential_time'].apply(convert_to_seconds)
df['threadpool_time'] = df['threadpool_time'].apply(convert_to_seconds)
df['rayon_time'] = df['rayon_time'].apply(convert_to_seconds)

bs_filters1 = [50, 100, 200]
bs_filters2 = [100, 200, 400]

for bs_filter in bs_filters1:
    filtered_df = df[(df['n'] == 1000) & (df['block_size'] == bs_filter)]
    
    num_threads = filtered_df['num_threads']
    seq_times = filtered_df['sequential_time']
    threadpool_times = filtered_df['threadpool_time']
    rayon_times = filtered_df['rayon_time']

    plt.figure(figsize=(10, 6))
    plt.plot(num_threads, seq_times, label='Sequential Time', marker='o', color='red')
    plt.plot(num_threads, threadpool_times, label='Threadpool Time', marker='o', color='green')
    plt.plot(num_threads, rayon_times, label='Rayon Time', marker='o', color='blue')

    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.title(f'Execution Times vs. Number of Threads (BS={bs_filter}, N=1000)', fontsize=14)

    plt.legend(fontsize=10)
    plt.grid(True)

for bs_filter in bs_filters2:
    filtered_df = df[(df['n'] == 2000) & (df['block_size'] == bs_filter)]
    
    num_threads = filtered_df['num_threads']
    seq_times = filtered_df['sequential_time']
    threadpool_times = filtered_df['threadpool_time']
    rayon_times = filtered_df['rayon_time']

    plt.figure(figsize=(10, 6))
    plt.plot(num_threads, seq_times, label='Sequential Time', marker='o', color='red')
    plt.plot(num_threads, threadpool_times, label='Threadpool Time', marker='o', color='green')
    plt.plot(num_threads, rayon_times, label='Rayon Time', marker='o', color='blue')

    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.title(f'Execution Times vs. Number of Threads (BS={bs_filter}, N=2000)', fontsize=14)

    plt.legend(fontsize=10)
    plt.grid(True)

plt.show()