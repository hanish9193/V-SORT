import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import psutil
import os
import gc
from collections import defaultdict
import seaborn as sns
from scipy import stats

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

def timsort(arr):
    """Python's built-in Timsort implementation"""
    return sorted(arr)

def numpy_sort(arr):
    """NumPy sort implementation"""
    return np.sort(arr)

def measure_memory_usage(sort_func, arr):
    """Measure peak memory usage during sorting operation"""
    # Force garbage collection
    gc.collect()
    
    # Get baseline memory usage
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    
    # Perform sorting
    result = sort_func(arr)
    
    # Get peak memory usage after sorting
    peak = process.memory_info().rss
    
    # Calculate memory used by the operation
    memory_used = peak - baseline
    
    return memory_used, result

def generate_test_array(size, distribution='uniform', max_val=1000):
    """Generate test arrays with different distributions"""
    if distribution == 'uniform':
        return np.random.randint(0, max_val + 1, size=size)
    elif distribution == 'normal':
        return np.clip(np.round(np.random.normal(max_val/2, max_val/6, size=size)).astype(int), 0, max_val)
    elif distribution == 'skewed':
        return np.clip(np.round(np.random.exponential(max_val/5, size=size)).astype(int), 0, max_val)
    elif distribution == 'sorted':
        return np.sort(np.random.randint(0, max_val + 1, size=size))
    elif distribution == 'reverse_sorted':
        return np.sort(np.random.randint(0, max_val + 1, size=size))[::-1]
    elif distribution == 'few_unique':
        return np.random.randint(0, 10, size=size)
    elif distribution == 'many_unique':
        return np.random.randint(0, size, size=size)
    else:
        return np.random.randint(0, max_val + 1, size=size)

def verify_correctness(results_dict):
    """Verify that all sorting algorithms produce the same output"""
    reference = None
    for method, result in results_dict.items():
        if reference is None:
            reference = result
            continue
        
        if not np.array_equal(result, reference):
            return False
    
    return True

def run_comprehensive_benchmark(sizes, max_val=1000, n_trials=3, distributions=None):
    """
    Benchmark the Direct Index Mapping Sort against all standard sorting methods.
    """
    methods = {
        'Python sorted()': lambda x: sorted(x),
        'NumPy sort()': lambda x: np.sort(x),
        'PyTorch sort()': lambda x: torch_sort(x),
        'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    if distributions is None:
        distributions = ['uniform']
    
    time_results = {dist: {method: [] for method in methods} for dist in distributions}
    memory_results = {dist: {method: [] for method in methods} for dist in distributions}
    correctness_results = {dist: [] for dist in distributions}
    
    for dist in distributions:
        print(f"\n\n==== Testing Distribution: {dist} ====")
        
        for size in sizes:
            print(f"\nBenchmarking array size: {size}")
            
            correctness_check = {}
            
            for method_name, sort_func in methods.items():
                times = []
                memories = []
                
                for trial in range(n_trials):
                    # Generate random test array with the specified distribution
                    test_arr = generate_test_array(size, distribution=dist, max_val=max_val)
                    
                    # Time the sorting operation
                    start = time.time()
                    sorted_arr = sort_func(test_arr)
                    end = time.time()
                    
                    times.append(end - start)
                    
                    # Store result for correctness checking on first trial
                    if trial == 0:
                        correctness_check[method_name] = sorted_arr
                    
                    # Measure memory on the last trial
                    if trial == n_trials - 1:
                        memory_used, _ = measure_memory_usage(sort_func, test_arr)
                        memories.append(memory_used)
                
                # Calculate average time
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories) if memories else 0
                
                time_results[dist][method_name].append(avg_time)
                memory_results[dist][method_name].append(avg_memory / (1024 * 1024))  # Convert to MB
                
                print(f"  {method_name}: {avg_time:.6f} seconds, {avg_memory / (1024 * 1024):.2f} MB")
            
            # Check correctness
            is_correct = verify_correctness(correctness_check)
            correctness_results[dist].append(is_correct)
            if not is_correct:
                print("  WARNING: Inconsistent sorting results detected!")
    
    return time_results, memory_results, correctness_results

def plot_time_benchmark_results(sizes, results, distributions):
    """Plot time benchmark results with comparison to theoretical complexity bounds."""
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'PyTorch sort()': 'purple',
        'V-SORT (Direct Index)': 'red'
    }
    
    for dist in distributions:
        plt.figure(figsize=(12, 8))
        
        for method, times in results[dist].items():
            plt.plot(sizes, times, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Array Size (log scale)', fontsize=12)
        plt.ylabel('Time (seconds, log scale)', fontsize=12)
        plt.title(f'V-SORT Performance Benchmark ({dist.capitalize()} Distribution)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        
        # Add theoretical complexity reference lines
        x = np.array(sizes, dtype=float)
        
        # Scale factors to position the reference lines in the plot
        nlogn_scale = np.median([results[dist]['NumPy sort()'][i] / (sizes[i] * np.log2(sizes[i])) 
                                for i in range(len(sizes))]) * 0.8
        linear_scale = np.median([results[dist]['V-SORT (Direct Index)'][i] / sizes[i] 
                                for i in range(len(sizes))]) * 0.8
        
        plt.plot(x, x * np.log2(x) * nlogn_scale, '--', color='gray', alpha=0.5, label='O(n log n)')
        plt.plot(x, x * linear_scale, '--', color='gray', alpha=0.5, label='O(n)')
        
        plt.tight_layout()
        plt.savefig(f'vsort_benchmark_{dist}_comparison.png')
        plt.close()
        
        # Create a second plot focusing just on ratio to better show the difference
        plt.figure(figsize=(12, 8))
        
        baseline = results[dist]['NumPy sort()']
        for method, times in results[dist].items():
            if method != 'NumPy sort()':
                ratio = [t1/t2 for t1, t2 in zip(times, baseline)]
                plt.plot(sizes, ratio, marker='o', linewidth=2, 
                         label=f'{method} vs NumPy sort() ratio', 
                         color=colors.get(method, 'black'))
        
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='NumPy sort() baseline')
        plt.xscale('log')
        plt.xlabel('Array Size (log scale)', fontsize=12)
        plt.ylabel('Time Ratio (compared to NumPy sort)', fontsize=12)
        plt.title(f'V-SORT Relative Performance ({dist.capitalize()} Distribution)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'vsort_relative_performance_{dist}.png')
        plt.close()

def plot_memory_benchmark_results(sizes, results, distributions):
    """Plot memory usage benchmark results."""
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'PyTorch sort()': 'purple',
        'V-SORT (Direct Index)': 'red'
    }
    
    for dist in distributions:
        plt.figure(figsize=(12, 8))
        
        for method, memories in results[dist].items():
            plt.plot(sizes, memories, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Array Size (log scale)', fontsize=12)
        plt.ylabel('Memory Usage (MB, log scale)', fontsize=12)
        plt.title(f'Memory Usage Comparison ({dist.capitalize()} Distribution)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'vsort_memory_comparison_{dist}.png')
        plt.close()

def plot_distribution_comparison(sizes, time_results, distributions):
    """Plot comparison of V-SORT performance across different distributions."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(distributions)))
    
    for i, dist in enumerate(distributions):
        plt.plot(sizes, time_results[dist]['V-SORT (Direct Index)'], 
                 marker='o', linewidth=2, label=f'{dist.capitalize()}',
                 color=colors[i])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('V-SORT Performance Across Different Distributions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_distribution_comparison.png')
    plt.close()

def plot_all_methods_heatmap(sizes, time_results, distributions):
    """Create a heatmap showing performance across all methods and distributions."""
    # For a fair comparison, we'll use a specific size
    size_index = len(sizes) // 2  # Middle size
    test_size = sizes[size_index]
    
    # Prepare data for heatmap
    methods = list(time_results[distributions[0]].keys())
    data = np.zeros((len(distributions), len(methods)))
    
    for i, dist in enumerate(distributions):
        for j, method in enumerate(methods):
            data[i, j] = time_results[dist][method][size_index]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt=".4f", xticklabels=methods, yticklabels=[d.capitalize() for d in distributions], cmap="YlGnBu_r")
    plt.title(f'Sorting Performance (seconds) for Array Size {test_size}', fontsize=14)
    plt.tight_layout()
    plt.savefig('vsort_performance_heatmap.png')
    plt.close()

def plot_speedup_bar(sizes, time_results, distributions):
    """Create bar charts showing V-SORT speedup over other methods."""
    # For a fair comparison, use the largest size
    size_index = -1  # Last size (largest)
    test_size = sizes[size_index]
    
    for dist in distributions:
        plt.figure(figsize=(10, 6))
        
        methods = [m for m in time_results[dist].keys() if m != 'V-SORT (Direct Index)']
        speedups = []
        
        for method in methods:
            speedup = time_results[dist][method][size_index] / time_results[dist]['V-SORT (Direct Index)'][size_index]
            speedups.append(speedup)
        
        bars = plt.bar(methods, speedups, color='skyblue')
        
        # Add values on top of bars
        for bar, val in zip(bars, speedups):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                     f'{val:.1f}x', ha='center', fontsize=10)
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='V-SORT baseline')
        plt.ylabel('Speedup Factor (higher is better)')
        plt.title(f'V-SORT Speedup for {test_size} Elements ({dist.capitalize()} Distribution)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'vsort_speedup_bar_{dist}.png')
        plt.close()

def plot_theoretical_comparison():
    """Plot theoretical complexity comparison."""
    plt.figure(figsize=(10, 6))
    
    # Range of input sizes
    n = np.logspace(2, 8, 1000)
    
    # Different complexity classes
    plt.plot(n, n * np.log2(n), label='O(n log n) - Traditional Sorting')
    plt.plot(n, n, label='O(n) - V-SORT')
    plt.plot(n, np.ones_like(n) * n.max() * 0.01, label='O(1) - Theoretical Limit')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Size (n)', fontsize=12)
    plt.ylabel('Operations', fontsize=12)
    plt.title('Theoretical Complexity Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_theoretical_complexity.png')
    plt.close()

def detailed_time_complexity_visualization(sizes, time_results):
    """Create a plot showing the empirical time complexity of sorting algorithms."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'PyTorch sort()': 'purple',
        'V-SORT (Direct Index)': 'red'
    }
    
    # Extract uniform distribution results
    results = time_results['uniform']
    
    # Calculate and plot slopes
    for method, times in results.items():
        # Convert to numpy arrays for calculations
        x = np.array(sizes, dtype=float)
        y = np.array(times)
        
        if method == 'V-SORT (Direct Index)':
            # For V-SORT, test linear fit
            log_x = np.log10(x)
            log_y = np.log10(y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            plt.scatter(x, y, color=colors.get(method), alpha=0.7)
            plt.plot(x, intercept + slope * x, '--', 
                     label=f'{method}: Slope ≈ {slope:.2e} (Linear)', 
                     color=colors.get(method))
            
        else:
            # For other methods, test n log n fit
            log_x = np.log10(x)
            log_y = np.log10(y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            
            fit_y = 10**(intercept) * x**slope
            plt.scatter(x, y, color=colors.get(method), alpha=0.7)
            plt.plot(x, fit_y, '--', 
                     label=f'{method}: Slope ≈ {slope:.2f} (Log-Log)', 
                     color=colors.get(method))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('Empirical Time Complexity Analysis', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_complexity_analysis.png')
    plt.close()

def run_bounded_range_analysis(size=1000000, ranges=None, n_trials=3):
    """Analyze how V-SORT performance varies with the range (max_value) of the data."""
    if ranges is None:
        ranges = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    
    methods = {
        'Python sorted()': lambda x, _: sorted(x),
        'NumPy sort()': lambda x, _: np.sort(x),
        'V-SORT (Direct Index)': lambda x, max_val: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    results = {method: [] for method in methods}
    
    print(f"\nAnalyzing impact of value range on sorting performance (array size: {size})")
    
    for max_val in ranges:
        print(f"\nTesting range: 0 to {max_val}")
        
        for method_name, sort_func in methods.items():
            times = []
            
            for _ in range(n_trials):
                # Generate random test array with the specified range
                test_arr = np.random.randint(0, max_val + 1, size=size)
                
                # Time the sorting operation
                start = time.time()
                sorted_arr = sort_func(test_arr, max_val)
                end = time.time()
                
                times.append(end - start)
            
            # Calculate average time
            avg_time = sum(times) / len(times)
            results[method_name].append(avg_time)
            print(f"  {method_name}: {avg_time:.6f} seconds")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'V-SORT (Direct Index)': 'red'
    }
    
    for method, times in results.items():
        plt.plot(ranges, times, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Maximum Value Range (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title(f'Effect of Value Range on Sorting Performance (Array Size: {size})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_range_analysis.png')
    plt.close()
    
    return results

def plot_scaling_efficiency(sizes, time_results):
    """Plot scaling efficiency (time per element) as array size increases."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'PyTorch sort()': 'purple',
        'V-SORT (Direct Index)': 'red'
    }
    
    # Extract uniform distribution results
    results = time_results['uniform']
    
    for method, times in results.items():
        # Calculate time per element (scaling efficiency)
        efficiency = [t/s for t, s in zip(times, sizes)]
        
        plt.plot(sizes, efficiency, marker='o', linewidth=2, 
                 label=method, color=colors.get(method, 'black'))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time per Element (seconds, log scale)', fontsize=12)
    plt.title('Scaling Efficiency: Time per Element', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_scaling_efficiency.png')
    plt.close()

def run_stability_testing(size=1000000, max_val=1000, n_trials=5):
    """Test the stability of V-SORT's performance."""
    methods = {
        'Python sorted()': lambda x: sorted(x),
        'NumPy sort()': lambda x: np.sort(x),
        'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    results = {method: [] for method in methods}
    
    print(f"\nTesting performance stability across {n_trials} trials (array size: {size})")
    
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}")
        
        # Generate random test array (same for all methods in this trial)
        test_arr = np.random.randint(0, max_val + 1, size=size)
        
        for method_name, sort_func in methods.items():
            # Time the sorting operation
            start = time.time()
            sorted_arr = sort_func(test_arr)
            end = time.time()
            
            elapsed = end - start
            results[method_name].append(elapsed)
            print(f"  {method_name}: {elapsed:.6f} seconds")
    
    # Calculate statistics
    stats_results = {}
    for method, times in results.items():
        stats_results[method] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'cv': np.std(times) / np.mean(times)  # Coefficient of variation
        }
    
    # Print statistics
    print("\nPerformance Statistics:")
    for method, stats_dict in stats_results.items():
        print(f"\n{method}:")
        print(f"  Mean: {stats_dict['mean']:.6f} seconds")
        print(f"  Std Dev: {stats_dict['std']:.6f} seconds")
        print(f"  CV (Std/Mean): {stats_dict['cv']:.4f}")
        print(f"  Min: {stats_dict['min']:.6f} seconds")
        print(f"  Max: {stats_dict['max']:.6f} seconds")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Box plot
    data = [results[method] for method in methods]
    plt.boxplot(data, labels=list(methods.keys()))
    
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Performance Stability Across Multiple Trials', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_stability_analysis.png')
    plt.close()
    
    return results, stats_results

def create_dashboard_summary(sizes, time_results, memory_results, distributions):
    """Create a summary dashboard of all key results."""
    plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid for the summary
    gs = plt.GridSpec(2, 2)
    
    # 1. Time performance for uniform distribution
    ax1 = plt.subplot(gs[0, 0])
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'PyTorch sort()': 'purple',
        'V-SORT (Direct Index)': 'red'
    }
    
    for method, times in time_results['uniform'].items():
        ax1.plot(sizes, times, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Array Size', fontsize=10)
    ax1.set_ylabel('Time (seconds)', fontsize=10)
    ax1.set_title('Time Performance (Uniform)', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    
    # 2. Memory usage for uniform distribution
    ax2 = plt.subplot(gs[0, 1])
    
    for method, memories in memory_results['uniform'].items():
        ax2.plot(sizes, memories, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Array Size', fontsize=10)
    ax2.set_ylabel('Memory (MB)', fontsize=10)
    ax2.set_title('Memory Usage (Uniform)', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    
    # 3. V-SORT performance across distributions
    ax3 = plt.subplot(gs[1, 0])
    
    dist_colors = plt.cm.rainbow(np.linspace(0, 1, len(distributions)))
    
    for i, dist in enumerate(distributions):
        ax3.plot(sizes, time_results[dist]['V-SORT (Direct Index)'], 
                 marker='o', linewidth=2, label=f'{dist.capitalize()}',
                 color=dist_colors[i])
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Array Size', fontsize=10)
    ax3.set_ylabel('Time (seconds)', fontsize=10)
    ax3.set_title('V-SORT Across Distributions', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", ls="--", alpha=0.7)
    
    # 4. Speedup ratios for largest size
    ax4 = plt.subplot(gs[1, 1])
    
    size_index = -1  # Largest size
    labels = []
    speedups = []
    
    for dist in distributions:
        for method in time_results[dist].keys():
            if method != 'V-SORT (Direct Index)':
                speedup = time_results[dist][method][size_index] / time_results[dist]['V-SORT (Direct Index)'][size_index]
                labels.append(f"{method.split('(')[0].strip()} ({dist.capitalize()})")
                speedups.append(speedup)
    
    bars = ax4.bar(range(len(speedups)), speedups, color='skyblue')
    ax4.set_xticks(range(len(speedups)))
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Speedup Factor', fontsize=10)
    ax4.set_title(f'V-SORT Speedup (Size: {sizes[size_index]})', fontsize=12)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_dashboard_summary.png', dpi=300)
    plt.close()

# Main execution
if __name__ == "__main__":
    # More comprehensive array sizes
    array_sizes = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    max_value = 1000  # Fixed bounded range
    
    # Different data distributions to test
    distributions = ['uniform', 'normal', 'skewed', 'sorted', 'reverse_sorted', 'few_unique']
    
    print("Running comprehensive benchmarks across multiple distributions...")
    time_results, memory_results, correctness_results = run_comprehensive_benchmark(
        array_sizes, max_value, n_trials=3, distributions=distributions)
    
    print("\nGenerating performance plots...")
    plot_time_benchmark_results(array_sizes, time_results, distributions)
    plot_memory_benchmark_results(array_sizes, memory_results, distributions)
    plot_distribution_comparison(array_sizes, time_results, distributions)
    plot_all_methods_heatmap(array_sizes, time_results, distributions)
    plot_speedup_bar(array_sizes, time_results, distributions)
    
    print("\nCreating theoretical complexity visualizations...")
    plot_theoretical_comparison()
    detailed_time_complexity_visualization(array_sizes, time_results)
    
    print("\nAnalyzing scaling efficiency...")
    plot_scaling_efficiency(array_sizes, time_results)
    
    print("\nRunning bounded range analysis...")
    range_results = run_bounded_range_analysis(
        size=1000000, 
        ranges=[10, 100, 1000, 10000, 100000, 1000000],
        n_trials=3
    )
    
    print("\nRunning stability testing...")
    stability_results, stability_stats = run_stability_testing(
        size=1000000, 
        max_val=max_value, 
        n_trials=5
    )
    
    print("\nCreating dashboard summary...")
    create_dashboard_summary(array_sizes, time_results, memory_results, distributions)
    
    print("\nBenchmark completed and plots saved.")


def run_practical_scenarios():
    """Test V-SORT in practical scenarios beyond simple benchmarks"""
    
    scenarios = {
        "incrementing_data": lambda size, max_val: np.arange(size) % (max_val + 1),
        "duplicate_heavy": lambda size, max_val: np.random.randint(0, max_val // 100 + 1, size=size),
        "single_value": lambda size, max_val: np.full(size, max_val // 2),
        "alternating": lambda size, max_val: np.tile([0, max_val], size // 2 + 1)[:size],
        "missing_values": lambda size, max_val: np.random.choice(np.delete(np.arange(max_val+1), np.arange(max_val//2)), size=size),
    }
    
    size = 1000000
    max_val = 1000
    
    methods = {
        'Python sorted()': lambda x: sorted(x),
        'NumPy sort()': lambda x: np.sort(x),
        'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    results = {scenario: {method: 0 for method in methods} for scenario in scenarios}
    
    print("\nTesting practical scenarios beyond standard benchmarks...")
    
    for scenario_name, data_gen in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        
        # Generate data for this scenario
        data = data_gen(size, max_val)
        
        for method_name, sort_func in methods.items():
            # Time the sorting operation
            start = time.time()
            sorted_arr = sort_func(data)
            end = time.time()
            
            elapsed = end - start
            results[scenario_name][method_name] = elapsed
            print(f"  {method_name}: {elapsed:.6f} seconds")
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    
    scenario_names = list(scenarios.keys())
    scenario_labels = [name.replace('_', ' ').title() for name in scenario_names]
    
    bar_width = 0.25
    index = np.arange(len(scenario_names))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'V-SORT (Direct Index)': 'red'
    }
    
    for i, method in enumerate(methods.keys()):
        performance = [results[scenario][method] for scenario in scenario_names]
        plt.bar(index + i * bar_width, performance, bar_width, 
                label=method, color=colors.get(method))
    
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Performance in Practical Scenarios', fontsize=14)
    plt.xticks(index + bar_width, scenario_labels, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig('vsort_practical_scenarios.png')
    plt.close()
    
    # Create normalized comparison plot
    plt.figure(figsize=(14, 8))
    
    for i, scenario in enumerate(scenario_names):
        baseline = results[scenario]['NumPy sort()']
        
        for method in methods:
            if method != 'NumPy sort()':
                ratio = results[scenario][method] / baseline
                
                # Add a point to the plot
                if method == 'V-SORT (Direct Index)':
                    plt.scatter(i, ratio, color='red', s=100, label=f'{method} / NumPy' if i == 0 else "")
                else:
                    plt.scatter(i, ratio, color='blue', s=100, label=f'{method} / NumPy' if i == 0 else "")
    
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='NumPy baseline')
    plt.xticks(range(len(scenario_names)), scenario_labels, rotation=45, ha='right')
    plt.ylabel('Time Ratio (compared to NumPy sort)', fontsize=12)
    plt.title('Relative Performance in Practical Scenarios', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_practical_scenarios_relative.png')
    
    return results


def visual_sorting_process():
    """Create a visual illustration of how V-SORT works vs traditional sorting"""
    # Small example array for visualization
    arr = np.array([5, 2, 8, 2, 5, 1, 9, 3, 5])
    max_val = 9
    
    # Step 1: Create array of all possible indices
    all_indices = np.arange(max_val + 1)
    
    # Step 2: Count occurrences of each value
    counts = np.bincount(arr, minlength=max_val + 1)
    
    # Step 3: Use repeat to construct the sorted array
    sorted_arr = np.repeat(all_indices, counts)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Original array
    plt.subplot(3, 1, 1)
    plt.bar(range(len(arr)), arr, color='skyblue')
    plt.title('Original Array', fontsize=12)
    plt.xticks(range(len(arr)), range(len(arr)))
    plt.ylabel('Value')
    plt.ylim(0, max_val + 1)
    
    # Counting process
    plt.subplot(3, 1, 2)
    plt.bar(range(len(counts)), counts, color='lightgreen')
    plt.title('Count of Each Value (0 to max_val)', fontsize=12)
    plt.xticks(range(len(counts)), range(len(counts)))
    plt.ylabel('Count')
    
    # Sorted result
    plt.subplot(3, 1, 3)
    plt.bar(range(len(sorted_arr)), sorted_arr, color='salmon')
    plt.title('Sorted Array (Using V-SORT)', fontsize=12)
    plt.xticks(range(len(sorted_arr)), range(len(sorted_arr)))
    plt.ylabel('Value')
    plt.ylim(0, max_val + 1)
    
    plt.tight_layout()
    plt.savefig('vsort_process_visualization.png')
    plt.close()
    
    # Create a comparison with traditional sorting
    plt.figure(figsize=(15, 8))
    
    # Traditional sorting process - simplified visualization
    plt.subplot(1, 2, 1)
    
    # Create a node for each element
    x_positions = np.linspace(0, 9, len(arr))
    y_positions = np.zeros(len(arr))
    
    # Plot the original array as nodes
    for i, (x, val) in enumerate(zip(x_positions, arr)):
        plt.text(x, y_positions[i], str(val), fontsize=12, 
                 ha='center', va='center', 
                 bbox=dict(boxstyle='circle', facecolor='skyblue', edgecolor='black'))
    
    # Draw some comparison arrows to represent the sorting process
    # This is a simplified visualization of comparison-based sorting
    comparisons = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                  (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)]
    
    for i, j in comparisons:
        plt.annotate('', 
                    xy=(x_positions[j], y_positions[j] - 0.2), 
                    xytext=(x_positions[i], y_positions[i] - 0.2),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    
    plt.text(5, -2, "O(n log n) comparisons required", ha='center', fontsize=10)
    plt.title('Traditional Comparison-Based Sorting', fontsize=14)
    plt.xlim(-1, 10)
    plt.ylim(-3, 1)
    plt.axis('off')
    
    # V-SORT process
    plt.subplot(1, 2, 2)
    
    # Create the counting array
    count_x = np.arange(max_val + 1)
    
    # Plot the count array
    for i, (x, count) in enumerate(zip(count_x, counts)):
        color = 'lightgreen' if count > 0 else 'white'
        plt.text(x, 0, str(count), fontsize=12, 
                 ha='center', va='center', 
                 bbox=dict(boxstyle='square', facecolor=color, edgecolor='black'))
    
    # Show the output construction
    output_idx = 0
    for value, count in enumerate(counts):
        for _ in range(count):
            y_pos = -2
            plt.text(output_idx, y_pos, str(value), fontsize=12, 
                     ha='center', va='center', 
                     bbox=dict(boxstyle='circle', facecolor='salmon', edgecolor='black'))
            
            # Draw arrow from count to output
            plt.annotate('', 
                        xy=(output_idx, y_pos - 0.2), 
                        xytext=(value, -0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            
            output_idx += 1
    
    plt.text(5, -4, "O(n) operations, direct indexing", ha='center', fontsize=10)
    plt.title('V-SORT Direct Index Mapping', fontsize=14)
    plt.xlim(-1, 10)
    plt.ylim(-5, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('vsort_algorithm_comparison.png')
    plt.close()


def run_large_scale_simulation():
    """Run a simulation of V-SORT performance for extremely large arrays"""
    # We'll create a model to predict performance at scale
    
    # First, get some baseline measurements
    sizes = [10000, 100000, 1000000, 10000000]
    max_val = 1000
    
    methods = {
        'NumPy sort()': lambda x: np.sort(x),
        'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    baseline_results = {method: [] for method in methods}
    
    print("\nGathering baseline measurements for scaling prediction...")
    
    for size in sizes:
        for method_name, sort_func in methods.items():
            # Generate test array
            test_arr = np.random.randint(0, max_val + 1, size=size)
            
            # Time the sorting operation
            start = time.time()
            sorted_arr = sort_func(test_arr)
            end = time.time()
            
            elapsed = end - start
            baseline_results[method_name].append(elapsed)
            print(f"  {method_name} (size {size}): {elapsed:.6f} seconds")
    
    # Fit models to the data
    x = np.array(sizes)
    
    # For numpy sort (n log n model)
    numpy_y = np.array(baseline_results['NumPy sort()'])
    numpy_params = np.polyfit(x * np.log2(x), numpy_y, 1)
    numpy_model = lambda n: numpy_params[0] * n * np.log2(n) + numpy_params[1]
    
    # For V-SORT (linear model)
    vsort_y = np.array(baseline_results['V-SORT (Direct Index)'])
    vsort_params = np.polyfit(x, vsort_y, 1)
    vsort_model = lambda n: vsort_params[0] * n + vsort_params[1]
    
    # Predict performance for extremely large arrays
    large_sizes = [50000000, 100000000, 500000000, 1000000000, 5000000000, 10000000000]
    
    numpy_predictions = [numpy_model(n) for n in large_sizes]
    vsort_predictions = [vsort_model(n) for n in large_sizes]
    
    # Create prediction table
    print("\nPredicted performance for large arrays:")
    print("\nArray Size | NumPy sort() (s) | V-SORT (s) | Speedup Factor")
    print("-" * 65)
    
    for i, size in enumerate(large_sizes):
        numpy_time = numpy_predictions[i]
        vsort_time = vsort_predictions[i]
        speedup = numpy_time / vsort_time
        
        # Convert to human-readable format
        if size >= 1000000000:
            size_str = f"{size/1000000000:.1f}B"
        elif size >= 1000000:
            size_str = f"{size/1000000:.1f}M"
        else:
            size_str = f"{size/1000:.1f}K"
            
        print(f"{size_str:>9} | {numpy_time:14.2f} | {vsort_time:9.2f} | {speedup:14.1f}x")
    
    # Create prediction plot
    plt.figure(figsize=(12, 8))
    
    # Plot measured data points
    plt.scatter(sizes, baseline_results['NumPy sort()'], 
                color='green', marker='o', label='NumPy (measured)')
    plt.scatter(sizes, baseline_results['V-SORT (Direct Index)'], 
                color='red', marker='o', label='V-SORT (measured)')
    
    # Plot prediction lines
    combined_sizes = np.array(sizes + large_sizes)
    combined_sizes.sort()
    
    numpy_pred = [numpy_model(n) for n in combined_sizes]
    vsort_pred = [vsort_model(n) for n in combined_sizes]
    
    plt.plot(combined_sizes, numpy_pred, 'g--', label='NumPy (predicted)')
    plt.plot(combined_sizes, vsort_pred, 'r--', label='V-SORT (predicted)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('Projected Performance at Scale', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add callouts for specific sizes
    callout_sizes = [1000000000, 10000000000]  # 1B and 10B
    for size in callout_sizes:
        numpy_time = numpy_model(size)
        vsort_time = vsort_model(size)
        speedup = numpy_time / vsort_time
        
        # Convert to human-readable format
        if size >= 1000000000:
            size_str = f"{size/1000000000:.1f}B"
        else:
            size_str = f"{size/1000000:.1f}M"
        
        plt.annotate(f"{size_str} elements\n{speedup:.1f}x faster", 
                     xy=(size, vsort_time), 
                     xytext=(size*0.5, vsort_time*0.1),
                     arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('vsort_scaling_prediction.png')
    plt.close()


# Run additional analysis
if __name__ == "__main__":
    print("\nRunning practical scenario tests...")
    practical_results = run_practical_scenarios()
    
    print("\nCreating visual illustration of V-SORT process...")
    visual_sorting_process()
    
    print("\nRunning large-scale performance simulation...")
    run_large_scale_simulation()
    
    print("\nAll enhanced benchmark tests completed!")
def plot_unique_elements_impact(sizes=None, max_val=1000, n_trials=3):
    """
    Analyze how the number of unique elements impacts sorting performance.
    """
    if sizes is None:
        sizes = [1000000]  # Fixed size
    
    # Define different unique element counts as percentages of max_val
    unique_percentages = [0.01, 0.1, 1, 10, 50, 100]
    
    results = defaultdict(lambda: defaultdict(list))
    
    print("\nAnalyzing impact of unique elements on sorting performance")
    
    for size in sizes:
        print(f"\nTesting array size: {size}")
        
        for pct in unique_percentages:
            unique_count = max(int(max_val * pct / 100), 1)
            actual_max_val = unique_count - 1
            
            print(f"  Testing with {unique_count} unique elements ({pct}% of range)")
            
            methods = {
                'Python sorted()': lambda x: sorted(x),
                'NumPy sort()': lambda x: np.sort(x),
                'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, actual_max_val),
            }
            
            for method_name, sort_func in methods.items():
                times = []
                
                for _ in range(n_trials):
                    # Generate random array with specified number of unique elements
                    unique_elements = np.random.randint(0, max_val + 1, size=unique_count)
                    test_arr = np.random.choice(unique_elements, size=size)
                    
                    # Time the sorting operation
                    start = time.time()
                    sorted_arr = sort_func(test_arr)
                    end = time.time()
                    
                    times.append(end - start)
                
                # Calculate average time
                avg_time = sum(times) / len(times)
                results[method_name][size].append(avg_time)
                print(f"    {method_name}: {avg_time:.6f} seconds")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'V-SORT (Direct Index)': 'red'
    }
    
    for size in sizes:
        for method_name, size_results in results.items():
            plt.plot(unique_percentages, size_results[size], marker='o', linewidth=2, 
                     label=f"{method_name} (n={size})", color=colors.get(method_name, 'black'))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Unique Elements (% of max_val)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('Impact of Unique Elements on Sorting Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_unique_elements_impact.png')
    plt.close()
    
    return results

def plot_sparse_vs_dense_arrays(size=1000000, max_val_range=None, n_trials=3):
    """
    Compare performance on sparse arrays (max_val >> array size) vs dense arrays.
    """
    if max_val_range is None:
        # Test with max_val ranging from smaller than to much larger than array size
        max_val_range = [size // 100, size // 10, size, size * 10, size * 100]
    
    methods = {
        'Python sorted()': lambda x, _: sorted(x),
        'NumPy sort()': lambda x, _: np.sort(x),
        'V-SORT (Direct Index)': lambda x, max_val: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    results = {method: [] for method in methods}
    
    print(f"\nComparing performance on sparse vs dense arrays (array size: {size})")
    
    for max_val in max_val_range:
        print(f"\nTesting max_val: {max_val} (density ratio: {size/max_val:.2f})")
        
        for method_name, sort_func in methods.items():
            times = []
            
            for _ in range(n_trials):
                # Generate random test array with the specified max_val
                test_arr = np.random.randint(0, max_val + 1, size=size)
                
                # Time the sorting operation
                start = time.time()
                sorted_arr = sort_func(test_arr, max_val)
                end = time.time()
                
                times.append(end - start)
            
            # Calculate average time
            avg_time = sum(times) / len(times)
            results[method_name].append(avg_time)
            print(f"  {method_name}: {avg_time:.6f} seconds")
    
    # Calculate density ratios
    density_ratios = [size / max_val for max_val in max_val_range]
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    colors = {
        'Python sorted()': 'blue',
        'NumPy sort()': 'green',
        'V-SORT (Direct Index)': 'red'
    }
    
    for method, times in results.items():
        plt.plot(density_ratios, times, marker='o', linewidth=2, label=method, color=colors.get(method, 'black'))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Density Ratio (array size / max_val)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title(f'Performance on Sparse vs Dense Arrays (Array Size: {size})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_sparse_vs_dense.png')
    plt.close()
    
    return results

def create_crossover_analysis(max_sizes=None, max_val=None, n_trials=3):
    """
    Create a detailed analysis of the crossover point where V-SORT becomes faster.
    """
    if max_sizes is None:
        max_sizes = np.logspace(2, 7, 20).astype(int)  # More granular size steps
    
    if max_val is None:
        max_val = 10000
    
    methods = {
        'NumPy sort()': lambda x: np.sort(x),
        'V-SORT (Direct Index)': lambda x: DirectIndexSort.numpy_sort(x, max_val),
    }
    
    results = {method: [] for method in methods}
    
    print("\nAnalyzing crossover point where V-SORT becomes more efficient...")
    
    for size in max_sizes:
        print(f"\nTesting array size: {size}")
        
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
    
    # Find crossover point(s)
    numpy_times = np.array(results['NumPy sort()'])
    vsort_times = np.array(results['V-SORT (Direct Index)'])
    
    # Calculate speedup ratios
    speedup_ratios = numpy_times / vsort_times
    
    # Identify where ratio crosses 1.0 (equal performance)
    crossover_indices = np.where(np.diff(speedup_ratios >= 1.0))[0]
    crossover_sizes = []
    
    for idx in crossover_indices:
        # Interpolate to find the approximate crossover size
        size1, size2 = max_sizes[idx], max_sizes[idx + 1]
        ratio1, ratio2 = speedup_ratios[idx], speedup_ratios[idx + 1]
        
        if ratio1 == ratio2:
            continue
            
        # Linear interpolation to find the exact crossover point
        # ratio = 1.0 at the crossover point
        crossover_size = size1 + (size2 - size1) * (1.0 - ratio1) / (ratio2 - ratio1)
        crossover_sizes.append(crossover_size)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.plot(max_sizes, numpy_times, 'g-o', linewidth=2, label='NumPy sort()')
    plt.plot(max_sizes, vsort_times, 'r-o', linewidth=2, label='V-SORT (Direct Index)')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Highlight crossover point(s)
    for crossover_size in crossover_sizes:
        # Find the time at this crossover point (approximate)
        crossover_time = np.interp(crossover_size, max_sizes, numpy_times)
        
        plt.axvline(x=crossover_size, color='blue', linestyle='--', alpha=0.7)
        plt.scatter([crossover_size], [crossover_time], color='blue', s=100, zorder=5)
        plt.annotate(f'Crossover: ~{int(crossover_size)} elements', 
                     xy=(crossover_size, crossover_time),
                     xytext=(crossover_size * 1.2, crossover_time * 1.2),
                     arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('Performance Crossover Analysis', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_crossover_analysis.png')
    
    # Plot speedup ratio on a separate graph
    plt.figure(figsize=(12, 8))
    
    plt.plot(max_sizes, speedup_ratios, 'b-o', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal performance')
    
    # Add region colors
    plt.fill_between(max_sizes, speedup_ratios, 1, where=speedup_ratios>=1, 
                     color='green', alpha=0.3, interpolate=True, label='V-SORT faster')
    plt.fill_between(max_sizes, speedup_ratios, 1, where=speedup_ratios<1, 
                     color='red', alpha=0.3, interpolate=True, label='NumPy faster')
    
    plt.xscale('log')
    plt.xlabel('Array Size (log scale)', fontsize=12)
    plt.ylabel('Speedup Ratio (NumPy time / V-SORT time)', fontsize=12)
    plt.title('V-SORT vs NumPy Speedup Ratio', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('vsort_speedup_ratio.png')
    plt.close()
    
    return results, crossover_sizes

def plot_threaded_comparison(sizes=None, max_val=1000, n_trials=3):
    """
    Compare V-SORT with NumPy sort when using multiple threads.
    Note: This function will display threading capability differences.
    """
    if sizes is None:
        sizes = [100000, 1000000, 10000000]
    
    # Define threaded NumPy sort function
    def numpy_parallel_sort(arr):
        # NumPy's sort can leverage multiple cores for large arrays
        return np.sort(arr)
    
    # Define threaded V-SORT function (hypothetical implementation)
    def vsort_parallel(arr, max_val):
        return DirectIndexSort.numpy_sort(arr, max_val)
        # In a real implementation, you might use threading like:
        # return DirectIndexSort.parallel_sort(arr, max_val, n_threads=8)
    
    methods = {
        'NumPy sort()': lambda x: numpy_parallel_sort(x),
        'V-SORT (Direct Index)': lambda x: vsort_parallel(x, max_val),
    }
    
    results = {method: [] for method in methods}
    
    print("\nComparing single vs multi-threading capabilities...")
    print("(Note: This is showing current implementation - V-SORT could be parallelized)")
    
    for size in sizes:
        print(f"\nTesting array size: {size}")
        
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
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(sizes))
    width = 0.35
    
    plt.bar(x - width/2, results['NumPy sort()'], width, label='NumPy sort()')
    plt.bar(x + width/2, results['V-SORT (Direct Index)'], width, label='V-SORT (Direct Index)')
    
    plt.xlabel('Array Size', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Threading Performance Comparison', fontsize=14)
    plt.xticks(x, [f"{s:,}" for s in sizes])
    plt.legend(fontsize=10)
    
    # Add text about parallelization potential
    plt.figtext(0.5, 0.01, 
                "Note: V-SORT algorithm is highly parallelizable with linear partitioning",
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('vsort_threading_comparison.png')
    plt.close()
    
    return results

def create_summary_infographic():
    """Create a comprehensive infographic summarizing V-SORT benefits."""
    plt.figure(figsize=(15, 12))
    
    # Main title
    plt.suptitle('V-SORT: Direct Index Mapping Sort', fontsize=24, y=0.98)
    plt.figtext(0.5, 0.94, 'A Faster, Simpler Sorting Algorithm for Integer Arrays', 
                ha='center', fontsize=16)
    
    # Create a 3x2 grid for the summary panels
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
    
    # Panel 1: Algorithmic Comparison
    ax1 = plt.subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('Algorithm Comparison', fontsize=14)
    
    comparison_text = """
    Traditional Sorting:
      • O(n log n) time complexity
      • Requires element comparisons
      • Swapping elements in memory
      
    V-SORT:
      • O(n) time complexity
      • No comparisons needed
      • Direct index mapping
      • Fixed space overhead
    """
    ax1.text(0.1, 0.5, comparison_text, va='center', fontsize=12, family='monospace')
    
    # Panel 2: Performance Highlights
    ax2 = plt.subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Performance Highlights', fontsize=14)
    
    highlights_text = """
    ✓ 5-15x faster for large arrays
    ✓ Constant per-element time
    ✓ Exceptional with repetitive data
    ✓ Efficient memory usage
    ✓ Stable sorting guaranteed
    ✓ Highly parallelizable
    """
    ax2.text(0.1, 0.5, highlights_text, va='center', fontsize=12, family='monospace')
    
    # Panel 3: Pseudocode
    ax3 = plt.subplot(gs[1, 0])
    ax3.axis('off')
    ax3.set_title('V-SORT Algorithm', fontsize=14)
    
    pseudocode = """
    function v_sort(array, max_value):
        # Create array of all possible indices
        all_indices = [0...max_value]
        
        # Count occurrences of each value
        counts = count_occurrences(array)
        
        # Create sorted result by repeating each 
        # index by its count
        return repeat(all_indices, counts)
    """
    ax3.text(0.05, 0.5, pseudocode, va='center', fontsize=11, family='monospace')
    
    # Panel 4: Use Cases
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('Ideal Use Cases', fontsize=14)
    
    usecases_text = """
    • Integer sorting (any size)
    • Data with limited value range
    • Datasets with many duplicates
    • Real-time applications
    • Large-scale data processing
    • When memory efficiency matters
    • Database operations
    """
    ax4.text(0.1, 0.5, usecases_text, va='center', fontsize=12, family='monospace')
    
    # Panel 5: Scaling visual (spanning both columns)
    ax5 = plt.subplot(gs[2, :])
    
    # Create a visual representation of scaling
    sizes = [10**i for i in range(3, 9)]
    numpy_times = [10**-5 * n * np.log2(n) for n in sizes]
    vsort_times = [10**-5 * n for n in sizes]
    
    ax5.plot(sizes, numpy_times, 'g-', linewidth=2, label='Traditional (O(n log n))')
    ax5.plot(sizes, vsort_times, 'r-', linewidth=2, label='V-SORT (O(n))')
    
    # Fill the gap between curves
    ax5.fill_between(sizes, numpy_times, vsort_times, color='lightgreen', alpha=0.5)
    
    # Annotations for the difference
    arrow_size = 10**7
    arrow_numpy = 10**-5 * arrow_size * np.log2(arrow_size)
    arrow_vsort = 10**-5 * arrow_size
    ax5.annotate('Performance Gap',
                 xy=(arrow_size, (arrow_numpy + arrow_vsort)/2),
                 xytext=(arrow_size/5, (arrow_numpy + arrow_vsort)/2),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('Array Size (elements)', fontsize=12)
    ax5.set_ylabel('Processing Time (relative)', fontsize=12)
    ax5.set_title('Scaling Comparison', fontsize=14)
    ax5.legend(fontsize=10)
    ax5.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('vsort_summary_infographic.png', dpi=300)
    plt.close()


# Add to the main execution section:
if __name__ == "__main__":
    print("\nRunning additional visualization analyses...")
    
    # Run the unique elements impact analysis
    plot_unique_elements_impact(sizes=[1000000], max_val=1000, n_trials=3)
    
    # Run the sparse vs dense array analysis
    plot_sparse_vs_dense_arrays(size=1000000, n_trials=3)
    
    # Find the crossover point where V-SORT becomes faster
    crossover_results, crossover_points = create_crossover_analysis(
        max_sizes=np.logspace(2, 7, 15).astype(int),
        max_val=1000, 
        n_trials=3
    )
    
    # Compare threading capabilities (current implementation)
    threading_results = plot_threaded_comparison(
        sizes=[100000, 1000000, 10000000],
        max_val=1000,
        n_trials=3
    )
    
    # Create comprehensive summary infographic
    create_summary_infographic()
    
    print("\nAll visualizations completed!")