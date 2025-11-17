"""
Solution decoding from y-vector to Grid.

This module provides functions to decode the solved y vector (one-hot encoding
of pixel colors) back into a 2D grid of colors.

Given:
  - y: solved one-hot vector, shape (H*W, C) or flat (H*W*C,)
  - H, W, C: grid dimensions and color count

Returns:
  - Grid: shape (H, W) with integer colors in [0, C-1]

This implements the inverse of the one-hot encoding from math spec section 1.1:
  y(Z)_{(p,c)} = 1 iff Z(p) = c

Decoding: Z(p) = argmax_c y[p, c]
"""

import numpy as np

from src.core.grid_types import Grid


def y_to_grid(
    y: np.ndarray,
    H: int,
    W: int,
    C: int
) -> Grid:
    """
    Decode a solved y vector into a grid.

    This is the inverse operation of one-hot encoding from the math spec.
    For each pixel p, we find the color c where y[p,c] is maximum (should be 1).

    Args:
        y: numpy array either:
           - shape (H*W, C): one-hot per pixel (row), or
           - flat of length H*W*C: will be reshaped to (H*W, C)
        H: output grid height.
        W: output grid width.
        C: number of colors.

    Returns:
        grid: numpy array of shape (H, W) with integer color values in [0, C-1].

    Raises:
        ValueError: if y dimensions don't match H, W, C

    Example:
        >>> # 2x2 grid, 3 colors, all pixels are color 1
        >>> y = np.array([[0,1,0], [0,1,0], [0,1,0], [0,1,0]])
        >>> grid = y_to_grid(y, H=2, W=2, C=3)
        >>> grid
        array([[1, 1],
               [1, 1]])
    """
    # 1. Validate and normalize shape to (H*W, C)
    num_pixels = H * W

    if y.ndim == 1:
        # Flat y of length H*W*C
        if y.size != num_pixels * C:
            raise ValueError(
                f"Flat y length {y.size} does not match H*W*C = {num_pixels * C}"
            )
        y2 = y.reshape(num_pixels, C)
    elif y.ndim == 2:
        # Already 2D, check shape
        if y.shape != (num_pixels, C):
            raise ValueError(
                f"y shape {y.shape} does not match (H*W, C) = ({num_pixels}, {C})"
            )
        y2 = y
    else:
        raise ValueError(f"y must be 1D or 2D, got ndim={y.ndim}")

    # 2. Argmax per pixel to find color
    # For each row (pixel), find the column (color) with maximum value
    # Shape: (num_pixels,), entries in [0..C-1]
    color_indices = np.argmax(y2, axis=1)

    # 3. Reshape to grid
    grid = color_indices.reshape(H, W).astype(int)

    return grid


def y_flat_to_grid(
    y_flat: np.ndarray,
    H: int,
    W: int,
    C: int
) -> Grid:
    """
    Convenience wrapper for flat y of length H*W*C.

    This is just a thin wrapper around y_to_grid that explicitly documents
    the flat input case.

    Args:
        y_flat: 1D numpy array of length H*W*C.
        H: grid height
        W: grid width
        C: number of colors

    Returns:
        grid: shape (H, W), same as y_to_grid.

    Example:
        >>> # 2x2 grid, 3 colors, flattened
        >>> y_flat = np.array([0,1,0, 0,0,1, 1,0,0, 0,1,0])
        >>> grid = y_flat_to_grid(y_flat, H=2, W=2, C=3)
        >>> grid
        array([[1, 2],
               [0, 1]])
    """
    return y_to_grid(y_flat, H, W, C)


if __name__ == "__main__":
    # Simple self-test
    print("Testing decoding.py with minimal example...")
    print("=" * 70)

    # Test 1: 2x2 grid, 3 colors, diagonal pattern
    H, W, C = 2, 2, 3
    num_pixels = H * W

    y = np.zeros((num_pixels, C), dtype=int)
    # p0 -> color 0
    y[0, 0] = 1
    # p1 -> color 1
    y[1, 1] = 1
    # p2 -> color 2
    y[2, 2] = 1
    # p3 -> color 1
    y[3, 1] = 1

    grid = y_to_grid(y, H, W, C)
    expected = np.array([[0, 1], [2, 1]], dtype=int)

    print(f"Input y shape: {y.shape}")
    print(f"Output grid:\n{grid}")
    print(f"Expected:\n{expected}")

    assert np.array_equal(grid, expected), f"Grid mismatch"
    print("✓ Test passed")

    print("\n" + "=" * 70)
    print("✓ decoding.py self-test passed.")
