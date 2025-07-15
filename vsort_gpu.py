"""
V-SORT: GPU-Accelerated Implementation

GPU-accelerated version of V-SORT using CuPy for CUDA-enabled systems.
Provides significant performance improvements for large datasets.

Requirements:
- CuPy library (pip install cupy)
- CUDA-compatible GPU

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import time
from typing import Union, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. GPU acceleration disabled.")


def v_sort_gpu(arr: Union[list, np.ndarray], 
               return_gpu: bool = False) -> Union[np.ndarray, 'cp.ndarray']:
    """
    GPU-accelerated V-SORT implementation using CuPy.
    
    Args:
        arr: Input integer array
        return_gpu: If True, returns CuPy array (stays on GPU)
    
    Returns:
        Sorted array (CPU numpy array by default, GPU CuPy array if return_gpu=True)
    
    Raises:
        ImportError: If CuPy is not available
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration. Install with: pip install cupy")
    
    # Transfer data to GPU
    gpu_array = cp.asarray(arr, dtype=cp.int32)
    
    # Find range on GPU
    min_val = cp.min(gpu_array)
    max_val = cp.max(gpu_array)
    
    # GPU-accelerated bincount
    adjusted_array = gpu_array - min_val
    counts = cp.bincount(adjusted_array)
    
    # Reconstruct on GPU
    k = max_val - min_val + 1
    indices = cp.arange(k) + min_val
    sorted_gpu = cp.repeat(indices, counts)
    
    # Return result
    if return_gpu:
        return sorted_gpu
    else:
        return cp.asnumpy(sorted_gpu)


def v_sort_gpu_chunked(arr: Union[list, np.ndarray], 
                       chunk_size: int = 10000000) -> np.ndarray:
    """
    GPU V-SORT for extremely large arrays using chunking strategy.
    
    Args:
        arr: Input array
        chunk_size: Size of chunks to process (default: 10M elements)
    
    Returns:
        Sorted numpy array
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration")
    
    arr = np.asarray(arr)
    
    if len(arr) <= chunk_size:
        return v_sort_gpu(arr)
    
    # Find global range
    global_min = np.min(arr)
    global_max = np.max(arr)
    
    # Process chunks on GPU
    sorted_chunks = []
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i+chunk_size]
        
        # Process chunk on GPU
        gpu_chunk = cp.asarray(chunk, dtype=cp.int32)
        adjusted_chunk = gpu_chunk - global_min
        counts = cp.bincount(adjusted_chunk, minlength=global_max-global_min+1)
        
        # Accumulate counts
        if i == 0:
            total_counts = counts
        else:
            total_counts += counts
    
    # Final reconstruction
    k = global_max - global_min + 1
    indices = cp.arange(k) + global_min
    sorted_gpu = cp.repeat(indices, total_counts)
    
    return cp.asnumpy(sorted_gpu)


def benchmark_cpu_vs_gpu(sizes: list = [100000, 1000000, 10000000],
                        ranges: list = [1000, 10000, 100000]) -> None:
    """
    Benchmark CPU vs GPU V-SORT performance.
    
    Args:
        sizes: List of array sizes to test
        ranges: List of value ranges to test
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available. Cannot run GPU benchmarks.")
        return
    
    print("CPU vs GPU V-SORT Performance Comparison")
    print("=" * 60)
    print(f"{'Size':<10} {'Range':<8} {'CPU Time':<10} {'GPU Time':<10} {'Speedup':<8}")
    print("-" * 60)
    
    # Import CPU version
    from v_sort_optimized import v_sort
    
    for size in sizes:
        for range_val in ranges:
            # Generate test data
            test_array = np.random.randint(0, range_val, size=size)
            
            # Benchmark CPU version
            start_time = time.perf_counter()
            cpu_result = v_sort(test_array)
            cpu_time = time.perf_counter() - start_time
            
            # Benchmark GPU version
            start_time = time.perf_counter()
            gpu_result = v_sort_gpu(test_array)
            gpu_time = time.perf_counter() - start_time
            
            # Calculate speedup
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            # Verify results match
            assert np.array_equal(cpu_result, gpu_result), "CPU and GPU results don't match!"
            
            print(f"{size:<10} {range_val:<8} {cpu_time:<10.4f} {gpu_time:<10.4f} {speedup:<8.2f}x")


def gpu_memory_benchmark(arr: np.ndarray) -> dict:
    """
    Benchmark GPU memory usage for V-SORT.
    
    Args:
        arr: Input array
        
    Returns:
        Dictionary with memory usage statistics
    """
    if not CUPY_AVAILABLE:
        return {"error": "CuPy not available"}
    
    # Get initial GPU memory
    mempool = cp.get_default_memory_pool()
    initial_used = mempool.used_bytes()
    
    # Execute GPU V-SORT
    start_time = time.perf_counter()
    result = v_sort_gpu(arr, return_gpu=True)
    end_time = time.perf_counter()
    
    # Get peak memory usage
    peak_used = mempool.used_bytes()
    
    # Clean up
    del result
    mempool.free_all_blocks()
    
    return {
        "execution_time": end_time - start_time,
        "peak_memory_mb": (peak_used - initial_used) / 1024 / 1024,
        "array_size": len(arr),
        "value_range": np.max(arr) - np.min(arr) + 1
    }


class GPUVSortContext:
    """
    Context manager for GPU V-SORT operations with automatic memory management.
    """
    
    def __init__(self):
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for GPU operations")
        self.mempool = cp.get_default_memory_pool()
    
    def __enter__(self):
        self.initial_memory = self.mempool.used_bytes()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mempool.free_all_blocks()
    
    def sort(self, arr: Union[list, np.ndarray]) -> np.ndarray:
        """Sort array using GPU V-SORT within the context."""
        return v_sort_gpu(arr)


def demo_gpu_vsort():
    """
    Demonstration of GPU V-SORT capabilities.
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available. Please install with: pip install cupy")
        return
    
    print("GPU V-SORT Demonstration")
    print("=" * 30)
    
    # Test basic functionality
    print("\n1. Basic GPU V-SORT:")
    test_array = np.random.randint(0, 100, size=10000)
    gpu_result = v_sort_gpu(test_array)
    cpu_result = np.sort(test_array)
    
    print(f"Arrays match: {np.array_equal(gpu_result, cpu_result)}")
    
    # Test context manager
    print("\n2. Using GPU Context Manager:")
    with GPUVSortContext() as gpu_ctx:
        large_array = np.random.randint(0, 1000, size=100000)
        sorted_array = gpu_ctx.sort(large_array)
        print(f"Sorted {len(large_array)} elements successfully")
    
    # Memory benchmark
    print("\n3. Memory Usage Benchmark:")
    benchmark_array = np.random.randint(0, 10000, size=1000000)
    memory_stats = gpu_memory_benchmark(benchmark_array)
    print(f"Execution time: {memory_stats['execution_time']:.4f} seconds")
    print(f"Peak memory usage: {memory_stats['peak_memory_mb']:.2f} MB")


if __name__ == "__main__":
    demo_gpu_vsort()
    
    if CUPY_AVAILABLE:
        print("\n" + "="*60)
        benchmark_cpu_vs_gpu()
