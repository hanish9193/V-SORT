import numpy as np
import time
import matplotlib.pyplot as plt
import torch

# Import V-SORT DirectIndexSort class from your code
class DirectIndexSort:
    @staticmethod
    def numpy_sort(arr, max_value=None, return_unique=False):
        # Convert to numpy array if needed
        arr = np.asarray(arr)
        
        # Determine max value if not provided
        if max_value is None:
            max_value = np.max(arr)
        
        # Create array of all possible indices
        all_indices = np.arange(max_value + 1, dtype=arr.dtype)
        
        if return_unique:
            # Create mask of which indices exist in input array
            mask = np.isin(all_indices, arr)
            
            # Apply mask to get sorted result
            return all_indices[mask]
        else:
            # Count occurrences of each value
            counts = np.bincount(arr, minlength=max_value + 1)
            
            # Use repeat to construct the fully sorted array with duplicates
            return np.repeat(all_indices, counts)

def torch_sort(arr):
    """PyTorch sorting implementation for benchmarking"""
    t_arr = torch.tensor(arr)
    sorted_t, _ = torch.sort(t_arr)
    return sorted_t.numpy()

def run_comprehensive_benchmark(sizes, max_val=1000, n_trials=3):
    """
    Benchmark the Direct Index Mapping Sort against all standard sorting methods.
    """
    methods = {
        'Python sorted()': lambda x: sorted(x),
        'NumPy sort()': lambda x: np.sort(x),
        'PyTorch sort()': lambda x: torch_sort(x),
        'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    results = {method: [] for method in methods}
    
    for size in sizes:
        print(f"\nBenchmarking array size: {size}")
        
        for method_name, sort_func in methods.items():
            times = []
            
            for _ in range(n_trials):
                # Generate random test array
                test_arr = np.random.randint(0, max_val + 1, size=size)
                
                # Time the sorting operation
                start = time.time()
                sorted_arr = sort_func(test_arr)
                end = time.time()
                
                times.append(end - start)
            
            # Calculate average time
            avg_time = sum(times) / len(times)
            results[method_name].append(avg_time)
            print(f"  {method_name}: {avg_time:.6f} seconds")
    
    return results

def plot_benchmark_results(sizes, results):
    """Plot benchmark results with comparison to theoretical complexity bounds."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'PyTorch sort()': 'purple',
        'V-SORT (Direct Index)': 'red'
    }
    
    for method, times in results.items():
        plt.plot(sizes, times, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('V-SORT Performance Benchmark', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add theoretical complexity reference lines
    x = np.array(sizes, dtype=float)
    plt.plot(x, x * np.log2(x) * 1e-8, '--', color='gray', alpha=0.5, label='O(n log n)')
    plt.plot(x, x * 1e-7, '--', color='gray', alpha=0.5, label='O(n)')
    
    # Add a flatter line representing close to O(1) behavior
    plt.plot(x, np.ones_like(x) * 1e-3, '--', color='gray', alpha=0.5, label='O(1)')
    
    plt.tight_layout()
    plt.savefig('vsort_benchmark_comparison.png')
    plt.close()
    
    # Create a second plot focusing just on ratio to better show the difference
    plt.figure(figsize=(12, 8))
    
    baseline = results['NumPy sort()']
    for method, times in results.items():
        if method != 'NumPy sort()':
            ratio = [t1/t2 for t1, t2 in zip(times, baseline)]
            plt.plot(sizes, ratio, marker='o', linewidth=2, 
                     label=f'{method} vs NumPy sort() ratio', 
                     color=colors.get(method, 'black'))
    
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='NumPy sort() baseline')
    plt.xscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time Ratio (compared to NumPy sort)', fontsize=12)
    plt.title('V-SORT Relative Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_relative_performance.png')

# Run benchmarks with different array sizes
array_sizes = [10000, 100000, 1000000, 5000000, 10000000]
max_value = 1000  # Fixed bounded range

print("Running comprehensive benchmarks...")
results = run_comprehensive_benchmark(array_sizes, max_value)
plot_benchmark_results(array_sizes, results)
print("\nBenchmark completed and plots saved.")
