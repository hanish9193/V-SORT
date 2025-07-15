"""
V-SORT: Vectorized Sorting Algorithm for Bounded Integer Arrays

Basic implementation of the V-SORT algorithm demonstrating the core three-step process:
1. Direct Index Mapping
2. Occurrence Counting  
3. Array Reconstruction

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from typing import Union, Optional


def v_sort_basic(arr: Union[list, np.ndarray]) -> np.ndarray:
    """
    Basic V-SORT implementation for bounded integer arrays.
    
    Args:
        arr: Input array of integers to sort
    
    Returns:
        Sorted numpy array
    
    Example:
        >>> input_array = [3, 1, 3, 0, 2, 1]
        >>> sorted_array = v_sort_basic(input_array)
        >>> print(sorted_array)
        [0 1 1 2 3 3]
    """
    # Convert to numpy array for vectorized operations
    input_array = np.asarray(arr, dtype=np.int64)
    
    # Find the range of values
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    
    # Step 1: Direct Index Mapping (implicit in bincount)
    # Create histogram structure for range [min_val, max_val]
    
    # Step 2: Occurrence Counting
    # Adjust array so minimum value becomes 0 (required for bincount)
    adjusted_array = input_array - min_val
    counts = np.bincount(adjusted_array)
    
    # Step 3: Array Reconstruction
    # Create indices array and restore original values
    indices = np.arange(len(counts)) + min_val
    sorted_array = np.repeat(indices, counts)
    
    return sorted_array


def v_sort_demo():
    """
    Demonstration of V-SORT algorithm with example datasets.
    """
    print("V-SORT Algorithm Demonstration")
    print("=" * 40)
    
    # Example 1: Simple array
    print("\nExample 1: Simple Array")
    arr1 = [3, 1, 3, 0, 2, 1]
    print(f"Input:  {arr1}")
    print(f"Output: {v_sort_basic(arr1)}")
    
    # Example 2: Array with negative values
    print("\nExample 2: Array with Negative Values")
    arr2 = [-2, 5, -1, 3, -2, 0, 5]
    print(f"Input:  {arr2}")
    print(f"Output: {v_sort_basic(arr2)}")
    
    # Example 3: Large array with small range
    print("\nExample 3: Large Array with Small Range")
    arr3 = np.random.randint(0, 10, size=100000)
    sorted_arr3 = v_sort_basic(arr3)
    print(f"Input:  Array of {len(arr3)} integers in range [0, 9]")
    print(f"Output: Sorted array (first 10 elements): {sorted_arr3[:10]}")
    print(f"        Is sorted: {np.array_equal(sorted_arr3, np.sort(arr3))}")


if __name__ == "__main__":
    v_sort_demo()
