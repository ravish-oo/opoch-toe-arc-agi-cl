"""
Line features and neighborhood hashes for ARC grids.

This module implements the line features and local pattern hashes from the
math kernel spec (section 1.2.6-7):
  - Line features: row/col flags for "contains non-zero"
  - Local pattern hashes: 3×3 (or general radius) neighborhood hashes

All features are deterministic and non-learned.
"""

from typing import Dict, Tuple
import numpy as np

from src.core.grid_types import Grid, Pixel


def row_nonzero_flags(grid: Grid) -> Dict[int, bool]:
    """
    For each row r in [0, H-1], return True if any cell in that row != 0.

    This implements the "row contains any non-zero" feature from the math kernel
    spec (section 1.2.6).

    Args:
        grid: Input grid (H, W)

    Returns:
        Dictionary mapping row index -> bool (True if row has any non-zero value)

    Example:
        >>> grid = np.array([[0, 0], [1, 0], [0, 0]], dtype=int)
        >>> row_nonzero_flags(grid)
        {0: False, 1: True, 2: False}
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    H, W = grid.shape

    # Use numpy to check if any value in each row is non-zero
    row_has_nonzero = (grid != 0).any(axis=1)

    # Convert to dict
    result = {r: bool(row_has_nonzero[r]) for r in range(H)}

    return result


def col_nonzero_flags(grid: Grid) -> Dict[int, bool]:
    """
    For each column c in [0, W-1], return True if any cell in that column != 0.

    This implements the "col contains any non-zero" feature from the math kernel
    spec (section 1.2.6).

    Args:
        grid: Input grid (H, W)

    Returns:
        Dictionary mapping column index -> bool (True if col has any non-zero value)

    Example:
        >>> grid = np.array([[0, 1], [0, 0], [0, 2]], dtype=int)
        >>> col_nonzero_flags(grid)
        {0: False, 1: True}
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    H, W = grid.shape

    # Use numpy to check if any value in each column is non-zero
    col_has_nonzero = (grid != 0).any(axis=0)

    # Convert to dict
    result = {c: bool(col_has_nonzero[c]) for c in range(W)}

    return result


def neighborhood_hashes(grid: Grid, radius: int = 1) -> Dict[Pixel, int]:
    """
    For each pixel (r,c) in the grid, compute a 'hash' of its local neighborhood.

    Neighborhood definition:
      - Neighborhood size = (2*radius + 1) x (2*radius + 1)
      - Centered at (r, c)
      - If (r+dr, c+dc) is out of bounds, use sentinel value -1 at that position

    This implements the "3×3 neighborhood hash" feature from the math kernel
    spec (section 1.2.7). With radius=1, this gives 3×3 neighborhoods.

    Procedure:
      - Build a small 2D patch for each pixel with these rules
      - Flatten the patch row-major into a 1D list of ints
      - Convert to a tuple and pass it to Python's built-in hash(...) to get an int

    Args:
        grid: Input grid (H, W)
        radius: Neighborhood radius (default 1 for 3×3)

    Returns:
        Dictionary mapping (r,c) -> hash_value (int)

    Example:
        >>> grid = np.array([[1, 1], [1, 1]], dtype=int)
        >>> hashes = neighborhood_hashes(grid, radius=1)
        >>> len(hashes)
        4
        >>> # All interior pixels with same pattern have same hash (if pattern identical)
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    H, W = grid.shape
    K = 2 * radius + 1  # Neighborhood size

    result = {}

    for r in range(H):
        for c in range(W):
            # Build neighborhood patch
            patch = []

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr = r + dr
                    cc = c + dc

                    # Check bounds
                    if 0 <= rr < H and 0 <= cc < W:
                        patch.append(int(grid[rr, cc]))
                    else:
                        # Out of bounds: use sentinel
                        patch.append(-1)

            # Convert to tuple and hash
            patch_tuple = tuple(patch)
            hash_value = hash(patch_tuple)

            result[(r, c)] = hash_value

    return result


if __name__ == "__main__":
    # Small test grid with repeated local patterns
    grid = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    row_flags = row_nonzero_flags(grid)
    col_flags = col_nonzero_flags(grid)
    print("\nRow nonzero flags:", row_flags)
    print("Col nonzero flags:", col_flags)

    hashes = neighborhood_hashes(grid, radius=1)
    print("\nNeighborhood hashes (radius=1) for a few pixels:")

    # Check pixels with same 3x3 pattern
    # The 2x2 block of 1's at (0,1), (0,2), (1,1), (1,2)
    # Let's check (1,1) and (1,2) - they should have different hashes
    # because their neighborhoods differ
    # Actually let's just show hashes for corners to demonstrate
    print(f"  hash(0, 0) =", hashes[(0, 0)])
    print(f"  hash(0, 3) =", hashes[(0, 3)])
    print(f"  hash(3, 0) =", hashes[(3, 0)])
    print(f"  hash(3, 3) =", hashes[(3, 3)])

    # Show that identical local patterns get same hash
    # The center of the 1-block at (1,1) and (1,2) might differ
    # Let's verify by checking all hashes
    print(f"\nTotal unique hash values: {len(set(hashes.values()))}")
    print(f"Total pixels: {len(hashes)}")
