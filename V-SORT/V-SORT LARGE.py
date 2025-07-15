import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial
import psutil
import os
from tabulate import tabulate

process = psutil.Process(os.getpid())

def v_sort(arr, max_value=None):
    arr = np.asarray(arr)
    if max_value is None:
        max_value = np.max(arr)
    all_indices = np.arange(max_value + 1, dtype=arr.dtype)
    counts = np.bincount(arr, minlength=max_value + 1)
    return np.repeat(all_indices, counts)

def merge_sort(arr):
    return np.sort(arr, kind='mergesort')

def quick_sort(arr):
    return np.sort(arr, kind='quicksort')

def python_sort(arr):
    return sorted(arr)

def measure_memory_usage():
    return process.memory_info().rss / (1024 * 1024)

def benchmark_large_datasets():
    algorithms = {
        'V-SORT': v_sort,
        'Merge Sort': merge_sort,
        'Quick Sort': quick_sort,
        'Python Sort': python_sort
    }
    
    large_datasets = [
        {
            'name': "100M Small Integers (range 0-999)",
            'size': 100_000_000,
            'generator': lambda: np.random.randint(0, 1000, size=100_000_000),
            'max_val': 999
        },
        {
            'name': "50M Medium Integers (range 0-9999)",
            'size': 50_000_000,
            'generator': lambda: np.random.randint(0, 10000, size=50_000_000),
            'max_val': 9999
        },
        {
            'name': "20M Large Integers (range 0-999999)",
            'size': 20_000_000,
            'generator': lambda: np.random.randint(0, 1000000, size=20_000_000),
            'max_val': 999999
        },
        {
            'name': "10M Few Unique Values (10 values)",
            'size': 10_000_000,
            'generator': lambda: np.random.randint(0, 10, size=10_000_000),
            'max_val': 9
        },
        {
            'name': "5M Nearly Sorted (95% sorted)",
            'size': 5_000_000,
            'generator': lambda: np.sort(np.random.randint(0, 1000, size=5_000_000)) + 
                                 np.random.randint(0, 5, size=5_000_000) * 
                                 np.random.choice([0, 1], size=5_000_000, p=[0.95, 0.05]),
            'max_val': 1004
        }
    ]
    
    results = {}
    
    for dataset in large_datasets:
        print(f"\n\n===== Testing: {dataset['name']} =====")
        dataset_name = dataset['name']
        results[dataset_name] = {}
        
        print(f"Generating dataset of {dataset['size']:,} elements...")
        data = dataset['generator']()
        
        for alg_name, sort_func in algorithms.items():
            print(f"\nBenchmarking {alg_name}...")
            
            if alg_name == "V-SORT" and dataset['max_val'] > 10_000_000:
                print(f"  Skipping {alg_name} due to excessive memory requirements for range {dataset['max_val']}")
                results[dataset_name][alg_name] = {
                    'time': float('inf'),
                    'memory': float('inf')
                }
                continue
                
            if alg_name in ['Python Sort'] and dataset['size'] > 50_000_000:
                print(f"  Skipping {alg_name} due to excessive runtime for size {dataset['size']:,}")
                results[dataset_name][alg_name] = {
                    'time': float('inf'),
                    'memory': float('inf')
                }
                continue
            
            test_data = np.copy(data)
            mem_before = measure_memory_usage()
            start = time.time()
            
            if alg_name == "V-SORT":
                sorted_arr = sort_func(test_data, max_value=dataset['max_val'])
            else:
                sorted_arr = sort_func(test_data)
                
            end = time.time()
            mem_after = measure_memory_usage()
            
            runtime = end - start
            mem_usage = mem_after - mem_before
            
            print(f"  Time: {runtime:.4f} seconds")
            print(f"  Additional memory: {mem_usage:.2f} MB")
            
            results[dataset_name][alg_name] = {
                'time': runtime,
                'memory': mem_usage
            }
            
            del test_data
            del sorted_arr
    
    return results

def create_results_table(results):
    datasets = list(results.keys())
    algorithms = list(results[datasets[0]].keys())
    
    time_headers = ["Dataset"] + [f"{alg} Time (s)" for alg in algorithms]
    time_data = []
    
    for dataset in datasets:
        row = [dataset]
        for alg in algorithms:
            time_val = results[dataset][alg]['time']
            if time_val == float('inf'):
                row.append("N/A")
            else:
                row.append(f"{time_val:.4f}")
        time_data.append(row)
    
    time_table = tabulate(time_data, headers=time_headers, tablefmt="grid")
    
    fastest_alg = {}
    for dataset in datasets:
        valid_times = {alg: results[dataset][alg]['time'] for alg in algorithms 
                      if results[dataset][alg]['time'] != float('inf')}
        if valid_times:
            fastest = min(valid_times.items(), key=lambda x: x[1])
            fastest_alg[dataset] = fastest[0]
    
    speedup_data = []
    for dataset in datasets:
        if dataset in fastest_alg:
            baseline = results[dataset][fastest_alg[dataset]]['time']
            row = [dataset]
            for alg in algorithms:
                time_val = results[dataset][alg]['time']
                if time_val == float('inf'):
                    row.append("N/A")
                else:
                    speedup = time_val / baseline
                    row.append(f"{speedup:.2f}x")
            speedup_data.append(row)
    
    speedup_headers = ["Dataset"] + [f"{alg} (rel. speed)" for alg in algorithms]
    speedup_table = tabulate(speedup_data, headers=speedup_headers, tablefmt="grid")
    
    return time_table, speedup_table, fastest_alg

print("Running large dataset benchmarks...")
benchmark_results = benchmark_large_datasets()

time_table, speedup_table, fastest_algorithms = create_results_table(benchmark_results)

print("\n=== BENCHMARK RESULTS (TIME IN SECONDS) ===\n")
print(time_table)

print("\n=== RELATIVE PERFORMANCE (TIMES SLOWER THAN FASTEST) ===\n")
print(speedup_table)

print("\n=== FASTEST ALGORITHM FOR EACH DATASET ===")
for dataset, algorithm in fastest_algorithms.items():
    time = benchmark_results[dataset][algorithm]['time']
    print(f"{dataset}: {algorithm} ({time:.4f}s)")
