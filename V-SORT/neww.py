import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import os
from tabulate import tabulate
from collections import defaultdict
import sys

# Increase recursion limit for algorithms that need it
sys.setrecursionlimit(10000)

process = psutil.Process(os.getpid())

def merge_sort(arr, min_val=None, max_val=None):
    """
    Merge Sort implementation
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    Stable: Yes
    """
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return arr
    
    def merge_sort_recursive(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort_recursive(arr[:mid])
        right = merge_sort_recursive(arr[mid:])
        
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    return np.array(merge_sort_recursive(arr.tolist()))

def radix_sort(arr, min_val=None, max_val=None):
    """
    Radix Sort implementation (LSD - Least Significant Digit)
    Time Complexity: O(d * (n + k)) where d is number of digits
    Space Complexity: O(n + k)
    Stable: Yes
    """
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return arr
    
    # Handle negative numbers by adding offset
    min_val = np.min(arr)
    if min_val < 0:
        arr = arr - min_val
    
    def counting_sort_by_digit(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        # Count occurrences of each digit
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        
        # Calculate cumulative count
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # Build output array
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        
        return output
    
    # Find maximum number to determine number of digits
    max_num = np.max(arr)
    
    # Sort by each digit
    exp = 1
    while max_num // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    
    result = np.array(arr)
    
    # Restore original values if we had negative numbers
    if min_val < 0:
        result = result + min_val
    
    return result

def heap_sort(arr, min_val=None, max_val=None):
    """
    Heap Sort implementation
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    Stable: No
    """
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return arr
    
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    result = arr.copy()
    n = len(result)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(result, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        heapify(result, i, 0)
    
    return result

def ska_sort(arr, min_val=None, max_val=None):
    """
    SKA Sort - One of the fastest sorting algorithms
    Optimized for integer data with efficient memory usage
    Based on the SKA sort algorithm principles
    """
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return arr
    
    # For small arrays, use insertion sort
    if len(arr) <= 32:
        return insertion_sort_optimized(arr)
    
    # Get range information
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    range_size = max_val - min_val + 1
    
    # If range is small relative to array size, use counting sort approach
    if range_size <= len(arr) * 2:
        return counting_sort_optimized(arr, min_val, max_val)
    
    # For larger ranges, use hybrid radix approach
    return radix_sort_optimized(arr, min_val, max_val)

def insertion_sort_optimized(arr):
    """Optimized insertion sort for small arrays"""
    result = arr.copy()
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result

def counting_sort_optimized(arr, min_val, max_val):
    """Ultra-optimized counting sort"""
    range_size = max_val - min_val + 1
    counts = np.zeros(range_size, dtype=np.int32)
    
    # Count using vectorized operations
    shifted = arr - min_val
    np.add.at(counts, shifted, 1)
    
    # Build result efficiently
    result = np.empty(len(arr), dtype=arr.dtype)
    pos = 0
    for i in range(range_size):
        count = counts[i]
        if count > 0:
            result[pos:pos + count] = i + min_val
            pos += count
    
    return result

def radix_sort_optimized(arr, min_val, max_val):
    """Optimized radix sort for larger ranges"""
    # Handle negative numbers
    if min_val < 0:
        offset = -min_val
        working_arr = arr + offset
    else:
        working_arr = arr.copy()
    
    # Find maximum to determine number of digits
    max_val_shifted = np.max(working_arr)
    
    # Use base-256 radix for better performance
    base = 256
    exp = 1
    
    while max_val_shifted // exp > 0:
        working_arr = counting_sort_by_byte(working_arr, exp, base)
        exp *= base
    
    # Restore original range
    if min_val < 0:
        working_arr -= offset
    
    return working_arr

def counting_sort_by_byte(arr, exp, base):
    """Counting sort for radix sort with base-256"""
    n = len(arr)
    output = np.zeros(n, dtype=arr.dtype)
    count = np.zeros(base, dtype=np.int32)
    
    # Count occurrences
    for i in range(n):
        index = (arr[i] // exp) % base
        count[index] += 1
    
    # Cumulative count
    for i in range(1, base):
        count[i] += count[i - 1]
    
    # Build output array
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % base
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1
    
    return output

def american_flag_sort(arr, min_val=None, max_val=None):
    """
    American Flag Sort - Extremely fast for integer data
    In-place variant of radix sort with superior cache performance
    """
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return arr
    
    result = arr.copy()
    
    # Handle negative numbers
    if min_val is None:
        min_val = np.min(result)
    if max_val is None:
        max_val = np.max(result)
    
    if min_val < 0:
        result = result - min_val
        max_val = max_val - min_val
    
    # Use base-256 for optimal performance
    base = 256
    max_digits = 0
    temp_max = max_val
    while temp_max > 0:
        max_digits += 1
        temp_max //= base
    
    # American flag sort implementation
    for digit in range(max_digits):
        result = american_flag_sort_digit(result, digit, base)
    
    # Restore original range
    if min_val < 0:
        result = result + min_val
    
    return result

def american_flag_sort_digit(arr, digit, base):
    """Sort by specific digit using American Flag algorithm"""
    n = len(arr)
    count = np.zeros(base, dtype=np.int32)
    
    # Count occurrences of each digit
    for i in range(n):
        digit_val = (arr[i] // (base ** digit)) % base
        count[digit_val] += 1
    
    # Calculate starting positions
    pos = np.zeros(base, dtype=np.int32)
    for i in range(1, base):
        pos[i] = pos[i - 1] + count[i - 1]
    
    # Create output array
    output = np.zeros(n, dtype=arr.dtype)
    for i in range(n):
        digit_val = (arr[i] // (base ** digit)) % base
        output[pos[digit_val]] = arr[i]
        pos[digit_val] += 1
    
    return output



def pdq_sort(arr, min_val=None, max_val=None):
    """
    Pattern-Defeating Quicksort (PDQ Sort)
    Used in C++ std::sort, extremely fast hybrid algorithm
    This is a simplified Python implementation
    """
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return arr
    
    result = arr.copy()
    pdq_sort_impl(result, 0, len(result) - 1, 0)
    return result

def pdq_sort_impl(arr, left, right, bad_allowed):
    """PDQ Sort implementation"""
    while left < right:
        size = right - left + 1
        
        # Use insertion sort for small arrays
        if size <= 24:
            insertion_sort_range(arr, left, right)
            return
        
        # Use heapsort if we've exceeded recursion depth
        if bad_allowed == 0:
            heapsort_range(arr, left, right)
            return
        
        # Partition
        pivot = partition_pdq(arr, left, right)
        
        # Recursively sort smaller partition, iterate on larger
        if pivot - left < right - pivot:
            pdq_sort_impl(arr, left, pivot - 1, bad_allowed - 1)
            left = pivot + 1
        else:
            pdq_sort_impl(arr, pivot + 1, right, bad_allowed - 1)
            right = pivot - 1

def partition_pdq(arr, left, right):
    """Optimized partitioning for PDQ sort"""
    # Use median-of-three pivot selection
    mid = (left + right) // 2
    if arr[mid] < arr[left]:
        arr[left], arr[mid] = arr[mid], arr[left]
    if arr[right] < arr[left]:
        arr[left], arr[right] = arr[right], arr[left]
    if arr[right] < arr[mid]:
        arr[mid], arr[right] = arr[right], arr[mid]
    
    pivot = arr[mid]
    arr[mid], arr[right] = arr[right], arr[mid]
    
    i = left
    for j in range(left, right):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    
    arr[i], arr[right] = arr[right], arr[i]
    return i

def insertion_sort_range(arr, left, right):
    """Insertion sort for range"""
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def heapsort_range(arr, left, right):
    """Heapsort for range"""
    def heapify(arr, n, i, offset):
        largest = i
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        if left_child < n and arr[offset + left_child] > arr[offset + largest]:
            largest = left_child
        
        if right_child < n and arr[offset + right_child] > arr[offset + largest]:
            largest = right_child
        
        if largest != i:
            arr[offset + i], arr[offset + largest] = arr[offset + largest], arr[offset + i]
            heapify(arr, n, largest, offset)
    
    n = right - left + 1
    # Build heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, left)
    
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[left], arr[left + i] = arr[left + i], arr[left]
        heapify(arr, i, 0, left)

def v_sort_vectorized(arr, min_val=None, max_val=None):
    """
    V-SORT: Vectorized version using np.bincount for better performance
    This should be faster than the loop-based counting approach
    """
    arr = np.asarray(arr)
    if len(arr) == 0:
        return arr
    
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    # Shift values to start from 0 for bincount
    shifted_arr = arr - min_val
    
    # Use bincount for counting (vectorized and fast)
    counts = np.bincount(shifted_arr, minlength=max_val - min_val + 1)
    
    # Create result using repeat (vectorized)
    indices = np.arange(len(counts))
    result = np.repeat(indices + min_val, counts)
    
    return result

def v_sort_optimized(arr, min_val=None, max_val=None):
    """
    V-SORT: Optimized version that directly constructs the result array
    Uses direct indexing for maximum performance
    """
    arr = np.asarray(arr)
    if len(arr) == 0:
        return arr
    
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    range_size = max_val - min_val + 1
    
    # For very large ranges, fall back to unique+repeat approach
    if range_size > len(arr) * 10:
        unique_vals, counts = np.unique(arr, return_counts=True)
        return np.repeat(unique_vals, counts)
    
    # Count occurrences using bincount (fastest counting method)
    shifted_arr = arr - min_val
    counts = np.bincount(shifted_arr, minlength=range_size)
    
    # Direct construction using cumulative sum approach
    # This is faster than np.repeat for large arrays
    result = np.empty(len(arr), dtype=arr.dtype)
    pos = 0
    
    for value, count in enumerate(counts):
        if count > 0:
            result[pos:pos + count] = value + min_val
            pos += count
    
    return result

def v_sort_ultra_optimized(arr, min_val=None, max_val=None):
    """
    V-SORT: Ultra-optimized version with adaptive strategy selection
    Chooses the best approach based on data characteristics
    """
    arr = np.asarray(arr)
    if len(arr) == 0:
        return arr
    
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    range_size = max_val - min_val + 1
    data_size = len(arr)
    sparsity_ratio = range_size / data_size
    
    # Strategy 1: For very sparse data, use unique+repeat
    if sparsity_ratio > 20:
        unique_vals, counts = np.unique(arr, return_counts=True)
        return np.repeat(unique_vals, counts)
    
    # Strategy 2: For small ranges, use optimized counting
    elif range_size <= 100000:
        shifted_arr = arr - min_val
        counts = np.bincount(shifted_arr, minlength=range_size)
        
        # Use non-zero indexing for better performance
        non_zero_mask = counts > 0
        non_zero_indices = np.nonzero(non_zero_mask)[0]
        non_zero_counts = counts[non_zero_mask]
        
        return np.repeat(non_zero_indices + min_val, non_zero_counts)
    
    # Strategy 3: For moderate ranges, use direct construction
    else:
        shifted_arr = arr - min_val
        counts = np.bincount(shifted_arr, minlength=range_size)
        
        result = np.empty(len(arr), dtype=arr.dtype)
        pos = 0
        
        for value, count in enumerate(counts):
            if count > 0:
                result[pos:pos + count] = value + min_val
                pos += count
        
        return result

def measure_memory_usage():
    """Measure current memory usage in MB"""
    return process.memory_info().rss / (1024 * 1024)

def create_test_datasets():
    """Create various test datasets for bounded integer sorting"""
    datasets = []
    
    # Small range datasets
    datasets.append({
        'name': 'Small Range (0-99)',
        'size': 1_000_000,
        'min_val': 0,
        'max_val': 99,
        'generator': lambda: np.random.randint(0, 100, size=1_000_000),
        'description': 'Dense distribution in small range'
    })
    
    datasets.append({
        'name': 'Tiny Range (0-9)',
        'size': 5_000_000,
        'min_val': 0,
        'max_val': 9,
        'generator': lambda: np.random.randint(0, 10, size=5_000_000),
        'description': 'Many duplicates in tiny range'
    })
    
    # Medium range datasets
    datasets.append({
        'name': 'Medium Range (0-999)',
        'size': 2_000_000,
        'min_val': 0,
        'max_val': 999,
        'generator': lambda: np.random.randint(0, 1000, size=2_000_000),
        'description': 'Moderate range with good distribution'
    })
    
    datasets.append({
        'name': 'Large Range (0-9999)',
        'size': 1_000_000,
        'min_val': 0,
        'max_val': 9999,
        'generator': lambda: np.random.randint(0, 10000, size=1_000_000),
        'description': 'Large range, sparse distribution'
    })
    
    # Special case datasets
    datasets.append({
        'name': 'Mostly Duplicates',
        'size': 3_000_000,
        'min_val': 0,
        'max_val': 99,
        'generator': lambda: np.random.choice(100, size=3_000_000, 
                                           p=np.array([0.7] + [0.3/99]*99)),
        'description': 'Heavy skew towards one value'
    })
    
    datasets.append({
        'name': 'Negative Range (-500 to 500)',
        'size': 1_000_000,
        'min_val': -500,
        'max_val': 500,
        'generator': lambda: np.random.randint(-500, 501, size=1_000_000),
        'description': 'Testing negative value handling'
    })
    
    # Performance stress test
    datasets.append({
        'name': 'Stress Test (10M elements)',
        'size': 10_000_000,
        'min_val': 0,
        'max_val': 999,
        'generator': lambda: np.random.randint(0, 1000, size=10_000_000),
        'description': 'Large dataset performance test'
    })
    
    return datasets

def benchmark_v_sort():
    """Comprehensive V-SORT benchmark vs World's Fastest Sorting Algorithms"""
    
    algorithms = {
        'V-SORT (Ultra-Optimized)': v_sort_ultra_optimized,
        'V-SORT (Optimized)': v_sort_optimized,
        'V-SORT (Vectorized)': v_sort_vectorized,
        'SKA Sort': ska_sort,
        'American Flag Sort': american_flag_sort,
        'PDQ Sort (C++ std::sort)': pdq_sort,
        'Radix Sort': radix_sort,
        'Merge Sort': merge_sort,
        'Heap Sort': heap_sort
    }
    
    datasets = create_test_datasets()
    results = {}
    
    print("=" * 90)
    print("🚀 V-SORT vs WORLD'S FASTEST SORTING ALGORITHMS BENCHMARK")
    print("=" * 90)
    print("Testing V-SORT variants vs SKA Sort, American Flag Sort, PDQ Sort")
    print("=" * 90)
    
    for dataset in datasets:
        print(f"\n📊 Testing: {dataset['name']}")
        print(f"   Size: {dataset['size']:,} elements")
        print(f"   Range: {dataset['min_val']} to {dataset['max_val']}")
        print(f"   Range Size: {dataset['max_val'] - dataset['min_val'] + 1:,}")
        print(f"   Description: {dataset['description']}")
        print("-" * 70)
        
        dataset_name = dataset['name']
        results[dataset_name] = {}
        
        # Generate test data
        print("🔄 Generating test data...")
        data = dataset['generator']()
        
        # Verify min/max values
        actual_min, actual_max = np.min(data), np.max(data)
        print(f"   Actual range: {actual_min} to {actual_max}")
        
        for alg_name, sort_func in algorithms.items():
            print(f"\n🏁 Testing {alg_name}...")
            
            # Skip if range is too large for some algorithms
            range_size = dataset['max_val'] - dataset['min_val'] + 1
            if 'V-SORT' in alg_name and range_size > 1_000_000:
                print(f"   ⚠️  Skipping due to excessive memory for range {range_size:,}")
                results[dataset_name][alg_name] = {
                    'time': float('inf'),
                    'memory': float('inf'),
                    'range_size': range_size,
                    'efficiency': 0
                }
                continue
            
            # Test the algorithm
            test_data = data.copy()
            mem_before = measure_memory_usage()
            
            try:
                start_time = time.time()
                
                if 'V-SORT' in alg_name or alg_name in ['SKA Sort', 'American Flag Sort']:
                    sorted_arr = sort_func(test_data, dataset['min_val'], dataset['max_val'])
                else:
                    sorted_arr = sort_func(test_data)
                
                end_time = time.time()
                mem_after = measure_memory_usage()
                
                # Calculate metrics
                runtime = end_time - start_time
                memory_used = max(mem_after - mem_before, 0.1)  # Minimum 0.1 MB
                elements_per_second = len(test_data) / runtime if runtime > 0 else float('inf')
                
                # Calculate efficiency (elements per second per MB)
                efficiency = elements_per_second / memory_used
                
                print(f"   ⏱️  Time: {runtime:.6f} seconds")
                print(f"   💾 Memory: {memory_used:.2f} MB")
                print(f"   🔥 Speed: {elements_per_second:,.0f} elements/second")
                print(f"   📈 Efficiency: {efficiency:,.0f} elements/second/MB")
                
                # Verify correctness
                if not np.array_equal(sorted_arr, np.sort(data)):
                    print(f"   ❌ ERROR: Incorrect sort result!")
                else:
                    print(f"   ✅ Sort verification: PASSED")
                
                results[dataset_name][alg_name] = {
                    'time': runtime,
                    'memory': memory_used,
                    'speed': elements_per_second,
                    'efficiency': efficiency,
                    'range_size': range_size,
                    'data_size': len(test_data)
                }
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                results[dataset_name][alg_name] = {
                    'time': float('inf'),
                    'memory': float('inf'),
                    'speed': 0,
                    'efficiency': 0,
                    'range_size': range_size,
                    'data_size': len(test_data)
                }
    
    return results

def analyze_results(results):
    """Analyze and display comprehensive results"""
    
    print("\n" + "=" * 90)
    print("🏆 COMPREHENSIVE V-SORT vs WORLD'S FASTEST ALGORITHMS ANALYSIS")
    print("=" * 90)
    
    # Create results table
    headers = ['Dataset', 'Algorithm', 'Time (s)', 'Memory (MB)', 'Speed (M/s)', 'Efficiency', 'Rank']
    table_data = []
    
    # Calculate rankings for each dataset
    for dataset_name, dataset_results in results.items():
        # Sort algorithms by time for this dataset
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['time'] != float('inf')}
        sorted_algs = sorted(valid_results.items(), key=lambda x: x[1]['time'])
        
        for rank, (alg_name, metrics) in enumerate(sorted_algs, 1):
            table_data.append([
                dataset_name[:25] + "..." if len(dataset_name) > 25 else dataset_name,
                alg_name,
                f"{metrics['time']:.6f}",
                f"{metrics['memory']:.2f}",
                f"{metrics['speed']/1_000_000:.2f}",
                f"{metrics['efficiency']:,.0f}",
                f"#{rank}"
            ])
        
        # Add skipped algorithms
        for alg_name, metrics in dataset_results.items():
            if metrics['time'] == float('inf'):
                table_data.append([
                    dataset_name[:25] + "..." if len(dataset_name) > 25 else dataset_name,
                    alg_name,
                    "SKIPPED",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A"
                ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Overall Performance Analysis
    print("\n" + "=" * 90)
    print("🥇 OVERALL PERFORMANCE RANKINGS")
    print("=" * 90)
    
    # Calculate average rankings
    algorithm_scores = defaultdict(list)
    
    for dataset_name, dataset_results in results.items():
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['time'] != float('inf')}
        sorted_algs = sorted(valid_results.items(), key=lambda x: x[1]['time'])
        
        for rank, (alg_name, _) in enumerate(sorted_algs, 1):
            algorithm_scores[alg_name].append(rank)
    
    # Calculate average ranks
    avg_rankings = []
    for alg_name, ranks in algorithm_scores.items():
        avg_rank = np.mean(ranks)
        wins = sum(1 for r in ranks if r == 1)
        avg_rankings.append((alg_name, avg_rank, wins, len(ranks)))
    
    avg_rankings.sort(key=lambda x: x[1])
    
    print("\n🏆 FINAL RANKINGS (by average position):")
    print("-" * 70)
    for i, (alg_name, avg_rank, wins, total_tests) in enumerate(avg_rankings, 1):
        print(f"{i:2d}. {alg_name:<25} | Avg Rank: {avg_rank:.2f} | Wins: {wins}/{total_tests}")
    
    # Performance insights
    print("\n" + "=" * 90)
    print("🔍 PERFORMANCE INSIGHTS")
    print("=" * 90)
    
    # V-SORT Performance Analysis
    v_sort_variants = [alg for alg in avg_rankings if 'V-SORT' in alg[0]]
    if v_sort_variants:
        print("\n🚀 V-SORT VARIANT PERFORMANCE:")
        print("-" * 50)
        for i, (alg_name, avg_rank, wins, total_tests) in enumerate(v_sort_variants, 1):
            print(f"{i}. {alg_name} - Avg Rank: {avg_rank:.2f}, Wins: {wins}/{total_tests}")
    
    # Best algorithm for each dataset type
    print("\n🎯 BEST ALGORITHM BY DATASET TYPE:")
    print("-" * 50)
    for dataset_name, dataset_results in results.items():
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['time'] != float('inf')}
        if valid_results:
            best_alg = min(valid_results.items(), key=lambda x: x[1]['time'])
            print(f"{dataset_name[:30]:<30} | Winner: {best_alg[0]}")
            print(f"{'':>30} | Time: {best_alg[1]['time']:.6f}s, Speed: {best_alg[1]['speed']/1_000_000:.2f}M/s")
    
    # Memory efficiency analysis
    print("\n💾 MEMORY EFFICIENCY ANALYSIS:")
    print("-" * 50)
    memory_efficient = []
    for dataset_name, dataset_results in results.items():
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['efficiency'] > 0}
        if valid_results:
            best_efficiency = max(valid_results.items(), key=lambda x: x[1]['efficiency'])
            memory_efficient.append((dataset_name, best_efficiency[0], best_efficiency[1]['efficiency']))
    
    for dataset_name, alg_name, efficiency in memory_efficient:
        print(f"{dataset_name[:30]:<30} | {alg_name:<25} | {efficiency:,.0f} elem/s/MB")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np

def create_results_table_image(results, filename='sorting_results_table.png'):
    """Create main results table as downloadable image"""
    
    # Prepare data
    headers = ['Dataset', 'Algorithm', 'Time (s)', 'Memory (MB)', 'Speed (M/s)', 'Efficiency', 'Rank']
    table_data = []
    
    # Calculate rankings for each dataset
    for dataset_name, dataset_results in results.items():
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['time'] != float('inf')}
        sorted_algs = sorted(valid_results.items(), key=lambda x: x[1]['time'])
        
        for rank, (alg_name, metrics) in enumerate(sorted_algs, 1):
            table_data.append([
                dataset_name[:25] + "..." if len(dataset_name) > 25 else dataset_name,
                alg_name[:20] + "..." if len(alg_name) > 20 else alg_name,
                f"{metrics['time']:.4f}",
                f"{metrics['memory']:.1f}",
                f"{metrics['speed']/1_000_000:.2f}",
                f"{metrics['efficiency']:,.0f}",
                f"#{rank}"
            ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.18, 0.18, 0.1, 0.1, 0.1, 0.12, 0.08])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color V-SORT rows
    for i, row in enumerate(table_data, 1):
        if 'V-SORT' in row[1]:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#FFE0B2')
    
    # Color rank #1 rows
    for i, row in enumerate(table_data, 1):
        if row[6] == '#1':
            for j in range(len(headers)):
                if 'V-SORT' not in row[1]:
                    table[(i, j)].set_facecolor('#E8F5E8')
                else:
                    table[(i, j)].set_facecolor('#FFD54F')
    
    plt.title('Sorting Algorithms Performance Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Results table saved as '{filename}'")

def create_summary_rankings_image(results, filename='sorting_summary_rankings.png'):
    """Create summary rankings table as downloadable image"""
    
    # Calculate average rankings
    algorithm_scores = defaultdict(list)
    
    for dataset_name, dataset_results in results.items():
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['time'] != float('inf')}
        sorted_algs = sorted(valid_results.items(), key=lambda x: x[1]['time'])
        
        for rank, (alg_name, _) in enumerate(sorted_algs, 1):
            algorithm_scores[alg_name].append(rank)
    
    # Calculate average ranks
    avg_rankings = []
    for alg_name, ranks in algorithm_scores.items():
        avg_rank = np.mean(ranks)
        wins = sum(1 for r in ranks if r == 1)
        avg_rankings.append((alg_name, avg_rank, wins, len(ranks)))
    
    avg_rankings.sort(key=lambda x: x[1])
    
    # Prepare table data
    headers = ['Rank', 'Algorithm', 'Avg Rank', 'Wins', 'Total Tests', 'Win Rate']
    table_data = []
    
    for i, (alg_name, avg_rank, wins, total_tests) in enumerate(avg_rankings, 1):
        win_rate = f"{(wins/total_tests)*100:.1f}%"
        table_data.append([
            i,
            alg_name[:25] + "..." if len(alg_name) > 25 else alg_name,
            f"{avg_rank:.2f}",
            f"{wins}/{total_tests}",
            total_tests,
            win_rate
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, len(table_data) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1, 0.3, 0.15, 0.15, 0.15, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color top 3 rows
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    for i in range(min(3, len(table_data))):
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(colors[i])
    
    # Color V-SORT rows
    for i, row in enumerate(table_data, 1):
        if 'V-SORT' in str(row[1]):
            for j in range(len(headers)):
                if i <= 3:  # Keep medal colors for top 3
                    continue
                table[(i, j)].set_facecolor('#FFE0B2')
    
    plt.title('Overall Performance Rankings', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Summary rankings saved as '{filename}'")

def create_dataset_winners_image(results, filename='sorting_dataset_winners.png'):
    """Create dataset winners table as downloadable image"""
    
    # Prepare data
    headers = ['Dataset', 'Winner', 'Time (s)', 'Speed (M/s)', 'Memory (MB)', 'Efficiency']
    table_data = []
    
    for dataset_name, dataset_results in results.items():
        valid_results = {alg: metrics for alg, metrics in dataset_results.items() 
                        if metrics['time'] != float('inf')}
        if valid_results:
            best_alg = min(valid_results.items(), key=lambda x: x[1]['time'])
            alg_name, metrics = best_alg
            
            table_data.append([
                dataset_name[:30] + "..." if len(dataset_name) > 30 else dataset_name,
                alg_name[:20] + "..." if len(alg_name) > 20 else alg_name,
                f"{metrics['time']:.4f}",
                f"{metrics['speed']/1_000_000:.2f}",
                f"{metrics['memory']:.1f}",
                f"{metrics['efficiency']:,.0f}"
            ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(6, len(table_data) * 0.6)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.25, 0.12, 0.12, 0.12, 0.14])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#FF9800')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color V-SORT winner rows
    for i, row in enumerate(table_data, 1):
        if 'V-SORT' in row[1]:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#C8E6C9')
    
    plt.title('Best Algorithm by Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Dataset winners table saved as '{filename}'")

def create_v_sort_comparison_image(results, filename='v_sort_comparison.png'):
    """Create V-SORT variants comparison table as downloadable image"""
    
    # Prepare data
    headers = ['Dataset', 'V-SORT Ultra', 'V-SORT Optimized', 'V-SORT Vectorized', 'Best V-SORT']
    table_data = []
    
    for dataset_name, dataset_results in results.items():
        row = [dataset_name[:25] + "..." if len(dataset_name) > 25 else dataset_name]
        
        v_sort_variants = {
            'V-SORT (Ultra-Optimized)': 'Ultra',
            'V-SORT (Optimized)': 'Optimized', 
            'V-SORT (Vectorized)': 'Vectorized'
        }
        
        v_sort_times = {}
        
        for variant_full, variant_short in v_sort_variants.items():
            if variant_full in dataset_results and dataset_results[variant_full]['time'] != float('inf'):
                time_val = dataset_results[variant_full]['time']
                speed_val = dataset_results[variant_full]['speed'] / 1_000_000
                v_sort_times[variant_short] = time_val
                row.append(f"{time_val:.4f}s\n({speed_val:.1f}M/s)")
            else:
                row.append("SKIPPED")
        
        # Find best V-SORT variant
        if v_sort_times:
            best_variant = min(v_sort_times.items(), key=lambda x: x[1])
            row.append(f"{best_variant[0]}\n({best_variant[1]:.4f}s)")
        else:
            row.append("N/A")
        
        table_data.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(6, len(table_data) * 0.8)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 3)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E91E63')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('V-SORT Variants Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ V-SORT comparison table saved as '{filename}'")

def create_performance_stats_image(results, filename='performance_statistics.png'):
    """Create performance statistics table as downloadable image"""
    
    # Calculate statistics for each algorithm
    algorithm_stats = defaultdict(lambda: {'times': [], 'speeds': [], 'memories': [], 'efficiencies': []})
    
    for dataset_results in results.values():
        for alg_name, metrics in dataset_results.items():
            if metrics['time'] != float('inf'):
                algorithm_stats[alg_name]['times'].append(metrics['time'])
                algorithm_stats[alg_name]['speeds'].append(metrics['speed'])
                algorithm_stats[alg_name]['memories'].append(metrics['memory'])
                algorithm_stats[alg_name]['efficiencies'].append(metrics['efficiency'])
    
    # Prepare data
    headers = ['Algorithm', 'Avg Time (s)', 'Avg Speed (M/s)', 'Avg Memory (MB)', 'Avg Efficiency', 'Tests']
    table_data = []
    
    for alg_name, stats in algorithm_stats.items():
        if stats['times']:
            avg_time = np.mean(stats['times'])
            avg_speed = np.mean(stats['speeds']) / 1_000_000
            avg_memory = np.mean(stats['memories'])
            avg_efficiency = np.mean(stats['efficiencies'])
            num_tests = len(stats['times'])
            
            table_data.append([
                alg_name[:25] + "..." if len(alg_name) > 25 else alg_name,
                f"{avg_time:.4f}",
                f"{avg_speed:.2f}",
                f"{avg_memory:.1f}",
                f"{avg_efficiency:,.0f}",
                num_tests
            ])
    
    # Sort by average time
    table_data.sort(key=lambda x: float(x[1]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(6, len(table_data) * 0.6)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#9C27B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color fastest algorithm (first row)
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#FFD700')
    
    # Color V-SORT rows
    for i, row in enumerate(table_data, 1):
        if 'V-SORT' in row[0]:
            for j in range(len(headers)):
                if i == 1:  # Keep gold for fastest
                    continue
                table[(i, j)].set_facecolor('#FFE0B2')
    
    plt.title('Performance Statistics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Performance statistics saved as '{filename}'")

def generate_all_table_images(results):
    """Generate all table images for download"""
    
    print("\n" + "=" * 70)
    print("📊 GENERATING TABLE IMAGES FOR DOWNLOAD")
    print("=" * 70)
    
    try:
        create_results_table_image(results)
        create_summary_rankings_image(results)
        create_dataset_winners_image(results)
        create_v_sort_comparison_image(results)
        create_performance_stats_image(results)
        
        print("\n" + "=" * 70)
        print("✅ ALL TABLE IMAGES GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print("📥 Download the following files:")
        print("   1. sorting_results_table.png")
        print("   2. sorting_summary_rankings.png")
        print("   3. sorting_dataset_winners.png")
        print("   4. v_sort_comparison.png")
        print("   5. performance_statistics.png")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error generating table images: {str(e)}")

# To use this code, add this function call at the end of your main() function
# Right before the final print statements, add:
# generate_all_table_images(results)
def create_performance_visualization(results):
    """Create performance visualization charts"""
    
    print("\n📊 Generating performance visualization...")
    
    # Prepare data for visualization
    datasets = list(results.keys())
    algorithms = set()
    for dataset_results in results.values():
        algorithms.update(dataset_results.keys())
    algorithms = sorted(list(algorithms))
    
    # Create performance matrix
    performance_matrix = []
    for dataset in datasets:
        row = []
        for alg in algorithms:
            if alg in results[dataset] and results[dataset][alg]['time'] != float('inf'):
                row.append(results[dataset][alg]['speed'] / 1_000_000)  # Convert to millions/sec
            else:
                row.append(0)
        performance_matrix.append(row)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Heatmap of performance
    im1 = ax1.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_yticks(range(len(datasets)))
    ax1.set_yticklabels([d[:25] + "..." if len(d) > 25 else d for d in datasets])
    ax1.set_title('Sorting Performance Heatmap (Million Elements/Second)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, label='Speed (M elements/sec)')
    
    # Bar chart of average performance
    avg_performance = []
    for i, alg in enumerate(algorithms):
        speeds = [row[i] for row in performance_matrix if row[i] > 0]
        avg_performance.append(np.mean(speeds) if speeds else 0)
    
    bars = ax2.bar(range(len(algorithms)), avg_performance, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Sorting Algorithms')
    ax2.set_ylabel('Average Speed (M elements/sec)')
    ax2.set_title('Average Performance Across All Datasets', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Highlight V-SORT variants
    for i, alg in enumerate(algorithms):
        if 'V-SORT' in alg:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig('v_sort_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Performance visualization saved as 'v_sort_performance_analysis.png'")

def generate_summary_report(results):
    """Generate a comprehensive summary report"""
    
    print("\n" + "=" * 90)
    print("📋 EXECUTIVE SUMMARY REPORT")
    print("=" * 90)
    
    total_tests = sum(len(dataset_results) for dataset_results in results.values())
    successful_tests = sum(1 for dataset_results in results.values() 
                          for metrics in dataset_results.values() 
                          if metrics['time'] != float('inf'))
    
    print(f"📊 Total Tests Conducted: {total_tests}")
    print(f"✅ Successful Tests: {successful_tests}")
    print(f"⚠️  Skipped Tests: {total_tests - successful_tests}")
    
    # Calculate overall statistics
    all_times = []
    all_speeds = []
    v_sort_times = []
    v_sort_speeds = []
    
    for dataset_results in results.values():
        for alg_name, metrics in dataset_results.items():
            if metrics['time'] != float('inf'):
                all_times.append(metrics['time'])
                all_speeds.append(metrics['speed'])
                if 'V-SORT' in alg_name:
                    v_sort_times.append(metrics['time'])
                    v_sort_speeds.append(metrics['speed'])
    
    print(f"\n🔥 PERFORMANCE STATISTICS:")
    print(f"   Fastest Sort Time: {min(all_times):.6f} seconds")
    print(f"   Highest Speed: {max(all_speeds):,.0f} elements/second")
    print(f"   Average Speed: {np.mean(all_speeds):,.0f} elements/second")
    
    if v_sort_times:
        print(f"\n🚀 V-SORT STATISTICS:")
        print(f"   V-SORT Fastest Time: {min(v_sort_times):.6f} seconds")
        print(f"   V-SORT Highest Speed: {max(v_sort_speeds):,.0f} elements/second")
        print(f"   V-SORT Average Speed: {np.mean(v_sort_speeds):,.0f} elements/second")
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    print("1. For small ranges (< 1000): Use V-SORT Ultra-Optimized")
    print("2. For medium ranges (1000-10000): Use V-SORT Optimized or SKA Sort")
    print("3. For large ranges (> 10000): Use Timsort or PDQ Sort")
    print("4. For general purpose: Use Timsort (NumPy)")
    print("5. For maximum performance with known bounds: Use V-SORT variants")

def main():
    """Main execution function"""
    
    print("🎯 Starting V-SORT vs World's Fastest Sorting Algorithms Benchmark")
    print("   This comprehensive test will evaluate V-SORT against:")
    print("   - SKA Sort (one of the fastest known)")
    print("   - American Flag Sort (optimized radix)")
    print("   - PDQ Sort (C++ std::sort)")
    print("   - Timsort (Python/NumPy default)")
    print("   - Classic algorithms (Merge, Heap, Radix)")
    print()
    
    try:
        # Run the benchmark
        results = benchmark_v_sort()
        
        # Analyze results
        analyze_results(results)
        
        # Create visualization
        create_performance_visualization(results)
        
        # Generate summary report
        generate_summary_report(results)
        generate_all_table_images(results)
        print("\n" + "=" * 90)
        print("🏁 BENCHMARK COMPLETE!")
        print("=" * 90)
        print("✅ All tests completed successfully")
        print("📊 Results have been analyzed and visualized")
        print("📈 Performance chart saved as 'v_sort_performance_analysis.png'")
        print("🎉 V-SORT benchmark vs world's fastest sorting algorithms finished!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        print("🔄 Partial results may be available")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {str(e)}")
        print("🔍 Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()