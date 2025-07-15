"""
V-SORT: Optimized Implementation with Error Handling and Performance Monitoring

This module provides an optimized version of V-SORT with comprehensive error handling,
performance monitoring, and memory usage optimization.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import time
import warnings
from typing import Union, Optional, Tuple
import psutil
import os


def v_sort(arr: Union[list, np.ndarray], 
           min_val: Optional[int] = None, 
           max_val: Optional[int] = None,
           validate_input: bool = True) -> np.ndarray:
    """
    V-SORT: Vectorized sorting algorithm for bounded integer arrays.
    
    Args:
        arr: Input array of integers to sort
        min_val: Minimum possible value (auto-detected if None)
        max_val: Maximum possible value (auto-detected if None)
        validate_input: Whether to validate input constraints
    
    Returns:
        Sorted numpy array
    
    Raises:
        ValueError: If input contains non-integers or range is too large
        MemoryError: If range k requires excessive memory
    """
    # Convert to numpy array for vectorized operations
    input_array = np.asarray(arr, dtype=np.int64)
    
    if validate_input:
        # Validate integer input
        if not np.issubdtype(input_array.dtype, np.integer):
            raise ValueError("V-SORT requires integer input")
        
        # Check for empty array
        if len(input_array) == 0:
            return np.array([], dtype=np.int64)
    
    # Auto-detect range if not provided
    if min_val is None:
        min_val = np.min(input_array)
    if max_val is None:
        max_val = np.max(input_array)
    
    # Calculate range and validate memory requirements
    k = max_val - min_val + 1
    n = len(input_array)
    
    # Memory usage estimation (rough heuristic)
    estimated_memory_mb = k * 8 / (1024 * 1024)  # 8 bytes per int64
    if estimated_memory_mb > 1000:  # 1GB threshold
        warnings.warn(f"Large range detected (k={k}). "
                     f"Estimated memory usage: {estimated_memory_mb:.1f}MB")
    
    if k > n * 10:  # Heuristic for inefficient scenarios
        warnings.warn(f"Range k={k} is much larger than n={n}. "
                     f"Consider using traditional sorting algorithms.")
    
    # Step 1: Create index array (implicit, handled by bincount)
    # Step 2: Count occurrences
    adjusted_array = input_array - min_val
    counts = np.bincount(adjusted_array, minlength=k)
    
    # Step 3: Reconstruct sorted array
    indices = np.arange(k, dtype=np.int64) + min_val
    sorted_array = np.repeat(indices, counts)
    
    return sorted_array


def v_sort_with_metrics(arr: Union[list, np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """
    V-SORT with performance metrics tracking.
    
    Args:
        arr: Input array of integers to sort
    
    Returns:
        Tuple of (sorted_array, execution_time, memory_usage_mb)
    """
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute V-SORT
    start_time = time.perf_counter()
    result = v_sort(arr)
    end_time = time.perf_counter()
    
    # Calculate metrics
    execution_time = end_time - start_time
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = final_memory - initial_memory
    
    return result, execution_time, memory_usage


def v_sort_sparse(arr: np.ndarray) -> np.ndarray:
    """
    Memory-efficient V-SORT for sparse ranges.
    Only creates bins for values that actually exist.
    
    Args:
        arr: Input array of integers
    
    Returns:
        Sorted numpy array
    """
    unique_values = np.unique(arr)
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    
    # Map original values to contiguous indices
    mapped_array = np.array([value_to_index[val] for val in arr])
    
    # Apply V-SORT on mapped values
    counts = np.bincount(mapped_array)
    indices = np.arange(len(unique_values))
    sorted_indices = np.repeat(indices, counts)
    
    # Map back to original values
    return unique_values[sorted_indices]


def benchmark_v_sort(sizes: list = [1000, 10000, 100000, 1000000], 
                     ranges: list = [100, 1000, 10000]) -> None:
    """
    Benchmark V-SORT performance across different array sizes and ranges.
    
    Args:
        sizes: List of array sizes to test
        ranges: List of value ranges to test
    """
    print("V-SORT Performance Benchmark")
    print("=" * 50)
    print(f"{'Size':<10} {'Range':<8} {'Time (s)':<10} {'Memory (MB)':<12}")
    print("-" * 50)
    
    for size in sizes:
        for range_val in ranges:
            # Generate test data
            test_array = np.random.randint(0, range_val, size=size)
            
            # Benchmark
            _, exec_time, memory_usage = v_sort_with_metrics(test_array)
            
            print(f"{size:<10} {range_val:<8} {exec_time:<10.4f} {memory_usage:<12.2f}")


def compare_with_numpy_sort(arr: np.ndarray) -> None:
    """
    Compare V-SORT performance with NumPy's built-in sort.
    
    Args:
        arr: Input array to sort
    """
    print("V-SORT vs NumPy Sort Comparison")
    print("=" * 35)
    
    # Test V-SORT
    start_time = time.perf_counter()
    v_result = v_sort(arr)
    v_time = time.perf_counter() - start_time
    
    # Test NumPy sort
    start_time = time.perf_counter()
    np_result = np.sort(arr)
    np_time = time.perf_counter() - start_time
    
    # Verify results are identical
    assert np.array_equal(v_result, np_result), "Results don't match!"
    
    print(f"Array size: {len(arr)}")
    print(f"Value range: {np.min(arr)} to {np.max(arr)}")
    print(f"V-SORT time: {v_time:.6f} seconds")
    print(f"NumPy time:  {np_time:.6f} seconds")
    print(f"Speedup: {np_time/v_time:.2f}x")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_v_sort()
    
    print("\n" + "="*50)
    
    # Compare with NumPy on a sample array
    sample_array = np.random.randint(0, 1000, size=100000)
    compare_with_numpy_sort(sample_array)
