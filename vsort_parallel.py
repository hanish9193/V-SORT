"""
V-SORT: Parallel Processing Implementation

Parallel processing strategies for V-SORT including multi-array processing
and chunked processing for extremely large datasets.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union, Optional, Tuple
import threading
from queue import Queue


def v_sort_basic(arr: Union[list, np.ndarray]) -> np.ndarray:
    """Basic V-SORT implementation for use in parallel processing."""
    input_array = np.asarray(arr, dtype=np.int64)
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    
    adjusted_array = input_array - min_val
    counts = np.bincount(adjusted_array)
    indices = np.arange(len(counts)) + min_val
    
    return np.repeat(indices, counts)


def parallel_v_sort(arrays: List[Union[list, np.ndarray]], 
                   num_processes: Optional[int] = None) -> List[np.ndarray]:
    """
    Parallel V-SORT processing for multiple arrays.
    
    Args:
        arrays: List of arrays to sort
        num_processes: Number of processes to use (default: CPU count)
    
    Returns:
        List of sorted arrays
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    with Pool(processes=num_processes) as pool:
        return pool.map(v_sort_basic, arrays)


def parallel_v_sort_concurrent(arrays: List[Union[list, np.ndarray]], 
                              max_workers: Optional[int] = None) -> List[np.ndarray]:
    """
    Parallel V-SORT using concurrent.futures for better control.
    
    Args:
        arrays: List of arrays to sort
        max_workers: Maximum number of worker processes
    
    Returns:
        List of sorted arrays in original order
    """
    if max_workers is None:
        max_workers = cpu_count()
    
    results = [None] * len(arrays)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(v_sort_basic, arr): i 
            for i, arr in enumerate(arrays)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    
    return results


def chunked_v_sort(arr: np.ndarray, 
                  chunk_size: int = 1000000,
                  num_processes: Optional[int] = None) -> np.ndarray:
    """
    V-SORT for extremely large arrays using chunking strategy.
    
    Args:
        arr: Input array
        chunk_size: Size of chunks to process
        num_processes: Number of processes for parallel processing
    
    Returns:
        Sorted numpy array
    """
    if len(arr) <= chunk_size:
        return v_sort_basic(arr)
    
    if num_processes is None:
        num_processes = cpu_count()
    
    # Find global range
    global_min = np.min(arr)
    global_max = np.max(arr)
    k = global_max - global_min + 1
    
    # Create chunks
    chunks = [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]
    
    # Process chunks in parallel to get count arrays
    def process_chunk(chunk):
        adjusted_chunk = chunk - global_min
        return np.bincount(adjusted_chunk, minlength=k)
    
    # Parallel processing of chunks
    with Pool(processes=num_processes) as pool:
        chunk_counts = pool.map(process_chunk, chunks)
    
    # Combine all counts
    total_counts = np.sum(chunk_counts, axis=0)
    
    # Reconstruct sorted array
    indices = np.arange(k) + global_min
    return np.repeat(indices, total_counts)


def streaming_v_sort(arr: np.ndarray, 
                    chunk_size: int = 1000000,
                    num_threads: int = 4) -> np.ndarray:
    """
    Streaming V-SORT for memory-efficient processing of large arrays.
    
    Args:
        arr: Input array
        chunk_size: Size of chunks to process
        num_threads: Number of threads for parallel processing
    
    Returns:
        Sorted numpy array
    """
    if len(arr) <= chunk_size:
        return v_sort_basic(arr)
    
    # Find global range
    global_min = np.min(arr)
    global_max = np.max(arr)
    k = global_max - global_min + 1
    
    # Initialize global counts
    total_counts = np.zeros(k, dtype=np.int64)
    
    # Thread worker function
    def worker(chunk_queue: Queue, result_queue: Queue):
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            
            adjusted_chunk = chunk - global_min
            counts = np.bincount(adjusted_chunk, minlength=k)
            result_queue.put(counts)
            chunk_queue.task_done()
    
    # Create queues and start threads
    chunk_queue = Queue()
    result_queue = Queue()
    
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(chunk_queue, result_queue))
        t.start()
        threads.append(t)
    
    # Add chunks to queue
    num_chunks = 0
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i+chunk_size]
        chunk_queue.put(chunk)
        num_chunks += 1
    
    # Collect results
    for _ in range(num_chunks):
        chunk_counts = result_queue.get()
        total_counts += chunk_counts
    
    # Stop threads
    for _ in range(num_threads):
        chunk_queue.put(None)
    
    for t in threads:
        t.join()
    
    # Reconstruct sorted array
    indices = np.arange(k) + global_min
    return np.repeat(indices, total_counts)


def adaptive_parallel_v_sort(arr: np.ndarray, 
                           target_chunk_size: int = 1000000) -> np.ndarray:
    """
    Adaptive parallel V-SORT that automatically determines optimal processing strategy.
    
    Args:
        arr: Input array
        target_chunk_size: Target size for chunks
    
    Returns:
        Sorted numpy array
    """
    n = len(arr)
    num_cores = cpu_count()
    
    # Decision logic for processing strategy
    if n <= target_chunk_size:
        # Small array: use basic V-SORT
        return v_sort_basic(arr)
    elif n <= target_chunk_size * num_cores:
        # Medium array: use simple chunking
        return chunked_v_sort(arr, chunk_size=n // num_cores)
    else:
        # Large array: use streaming approach
        return streaming_v_sort(arr, chunk_size=target_chunk_size)


def benchmark_parallel_strategies(array_sizes: List[int] = [1000000, 5000000, 10000000],
                                value_range: int = 10000) -> None:
    """
    Benchmark different parallel V-SORT strategies.
    
    Args:
        array_sizes: List of array sizes to test
        value_range: Range of values (0 to value_range-1)
    """
    print("Parallel V-SORT Strategy Benchmark")
    print("=" * 50)
    print(f"{'Size':<10} {'Basic':<10} {'Chunked':<10} {'Streaming':<10} {'Adaptive':<10}")
    print("-" * 50)
    
    for size in array_sizes:
        # Generate test data
        test_array = np.random.randint(0, value_range, size=size)
        
        # Test basic V-SORT
        start_time = time.perf_counter()
        basic_result = v_sort_basic(test_array)
        basic_time = time.perf_counter() - start_time
        
        # Test chunked V-SORT
        start_time = time.perf_counter()
        chunked_result = chunked_v_sort(test_array)
        chunked_time = time.perf_counter() - start_time
        
        # Test streaming V-SORT
        start_time = time.perf_counter()
        streaming_result = streaming_v_sort(test_array)
        streaming_time = time.perf_counter() - start_time
        
        # Test adaptive V-SORT
        start_time = time.perf_counter()
        adaptive_result = adaptive_parallel_v_sort(test_array)
        adaptive_time = time.perf_counter() - start_time
        
        # Verify all results are identical
        assert np.array_equal(basic_result, chunked_result), "Chunked result mismatch"
        assert np.array_equal(basic_result, streaming_result), "Streaming result mismatch"
        assert np