import numpy as np

def zeroOrOne(arr, threshold=0.5):
    """
    Applies a threshold to a numpy array. Returns a binary array
    where elements greater than the threshold are 1, and others are 0.

    Parameters:
        arr (numpy.ndarray): Input array with floats between 0 and 1.
        threshold (float): The threshold value.

    Returns:
        numpy.ndarray: Binary array with the same shape as the input array.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    # Apply the threshold
    result = np.where(arr > threshold, 1, 0)
    return result