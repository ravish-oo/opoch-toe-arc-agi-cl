"""
Core grid types and utilities for ARC-AGI constraint solver.

This module defines the fundamental Grid representation and pixel indexing
functions that align with the math kernel specification (section 1.1).

Grid: always shape (H, W), dtype=int
Pixels: indexed as (row, col) tuples or as flat indices in [0, H*W-1]
"""

import numpy as np
from typing import TypeAlias, Tuple


# Type aliases as per math kernel spec (section 1.1)
Grid: TypeAlias = np.ndarray  # shape: (H, W), dtype: int, values in {0, ..., C-1}
Pixel: TypeAlias = Tuple[int, int]  # (row, col) in {0, ..., H-1} x {0, ..., W-1}


def pixel_index(r: int, c: int, width: int) -> int:
    """
    Map (row r, col c) to a flat index idx in [0, H*W-1], row-major.

    This is used for indexing into the one-hot encoding vector y ∈ {0,1}^(N×C)
    where N = H·W (math kernel section 1.1).

    Formula: idx = r * width + c

    Args:
        r: Row index (0-based)
        c: Column index (0-based)
        width: Grid width W

    Returns:
        Flat pixel index in [0, H*W-1]

    Raises:
        ValueError: If r or c is negative
    """
    if r < 0 or c < 0:
        raise ValueError(f"Pixel coordinates must be non-negative, got r={r}, c={c}")

    return r * width + c


def index_to_pixel(idx: int, width: int) -> Tuple[int, int]:
    """
    Inverse of pixel_index: given flat idx and width, return (row, col).

    This recovers spatial coordinates from the flat one-hot encoding index.

    Formula: r = idx // width, c = idx % width

    Args:
        idx: Flat pixel index
        width: Grid width W

    Returns:
        (row, col) tuple

    Raises:
        ValueError: If idx is negative
    """
    if idx < 0:
        raise ValueError(f"Index must be non-negative, got idx={idx}")

    r = idx // width
    c = idx % width

    return (r, c)


def print_grid(grid: Grid) -> None:
    """
    Print a small ASCII representation of the grid for debugging.

    Each row is printed on its own line with space-separated integer values.
    This is purely for human inspection during development.

    Args:
        grid: Grid to print (must be 2D numpy array)

    Raises:
        AssertionError: If grid is not 2-dimensional

    Example:
        >>> grid = np.array([[0, 1], [2, 3]], dtype=int)
        >>> print_grid(grid)
        0 1
        2 3
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    for row in grid:
        print(' '.join(str(int(val)) for val in row))


if __name__ == "__main__":
    # Self-test: verify index mapping roundtrip
    import numpy as np

    grid = np.array([[0, 1], [2, 3]], dtype=int)
    print("Grid:")
    print_grid(grid)

    width = grid.shape[1]
    for r in range(grid.shape[0]):
        for c in range(width):
            idx = pixel_index(r, c, width)
            r2, c2 = index_to_pixel(idx, width)
            assert (r, c) == (r2, c2), f"Roundtrip failed at {(r,c)}"

    print("Index roundtrip test passed.")
