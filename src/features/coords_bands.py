"""
Coordinate, bands, and border features for ARC grids.

This module implements the coordinate-based features from the math kernel spec
(section 1.2 - Feature dictionary φ(p)):
  1. Coordinates: row, col
  2. Residues / periodic indices: r mod k, c mod k for k ∈ {2,3,4,5}
  3. Bands & frames: row-band, col-band, border flag

All features are deterministic and non-learned.
"""

from typing import Dict, Tuple
import numpy as np

from src.core.grid_types import Grid, Pixel


def coord_features(grid: Grid) -> Dict[Pixel, Dict]:
    """
    For each pixel (r,c) in the grid, return coordinate features:
      - "row": int (0-based row index)
      - "col": int (0-based column index)
      - "row_mod": {2: r % 2, 3: r % 3, 4: r % 4, 5: r % 5}
      - "col_mod": {2: c % 2, 3: c % 3, 4: c % 4, 5: c % 5}

    This implements the coordinate and residue features from math kernel section 1.2.

    Args:
        grid: Input grid (H, W)

    Returns:
        Dictionary mapping (r,c) -> feature dict

    Example:
        >>> grid = np.array([[0, 1], [2, 3]], dtype=int)
        >>> features = coord_features(grid)
        >>> features[(0, 0)]
        {'row': 0, 'col': 0, 'row_mod': {2: 0, 3: 0, 4: 0, 5: 0},
         'col_mod': {2: 0, 3: 0, 4: 0, 5: 0}}
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    H, W = grid.shape
    features = {}

    for r in range(H):
        for c in range(W):
            features[(r, c)] = {
                "row": r,
                "col": c,
                "row_mod": {
                    2: r % 2,
                    3: r % 3,
                    4: r % 4,
                    5: r % 5
                },
                "col_mod": {
                    2: c % 2,
                    3: c % 3,
                    4: c % 4,
                    5: c % 5
                }
            }

    return features


def row_band_labels(H: int) -> Dict[int, str]:
    """
    Assign each row index r in [0, H-1] to a band: 'top', 'middle', or 'bottom'.

    Convention (fixed, deterministic):
      - If H == 1:
          row 0 -> 'middle'
      - If H == 2:
          row 0 -> 'top'
          row 1 -> 'bottom'
      - If H >= 3:
          row 0         -> 'top'
          row H-1       -> 'bottom'
          all rows 1..H-2 -> 'middle'

    Args:
        H: Grid height

    Returns:
        Dictionary mapping row index -> band label

    Example:
        >>> row_band_labels(4)
        {0: 'top', 1: 'middle', 2: 'middle', 3: 'bottom'}
    """
    if H == 1:
        return {0: 'middle'}
    elif H == 2:
        return {0: 'top', 1: 'bottom'}
    else:  # H >= 3
        labels = {0: 'top', H - 1: 'bottom'}
        for r in range(1, H - 1):
            labels[r] = 'middle'
        return labels


def col_band_labels(W: int) -> Dict[int, str]:
    """
    Assign each column index c in [0, W-1] to a band: 'left', 'middle', or 'right'.

    Convention (fixed, deterministic):
      - If W == 1:
          col 0 -> 'middle'
      - If W == 2:
          col 0 -> 'left'
          col 1 -> 'right'
      - If W >= 3:
          col 0         -> 'left'
          col W-1       -> 'right'
          all cols 1..W-2 -> 'middle'

    Args:
        W: Grid width

    Returns:
        Dictionary mapping column index -> band label

    Example:
        >>> col_band_labels(5)
        {0: 'left', 1: 'middle', 2: 'middle', 3: 'middle', 4: 'right'}
    """
    if W == 1:
        return {0: 'middle'}
    elif W == 2:
        return {0: 'left', 1: 'right'}
    else:  # W >= 3
        labels = {0: 'left', W - 1: 'right'}
        for c in range(1, W - 1):
            labels[c] = 'middle'
        return labels


def border_mask(grid: Grid) -> np.ndarray:
    """
    Return a boolean mask of shape (H, W) where True indicates
    that the pixel is on the outer border of the grid.

    A pixel is on the border if:
      r == 0 or r == H-1 or c == 0 or c == W-1

    Args:
        grid: Input grid (H, W)

    Returns:
        Boolean numpy array of shape (H, W)

    Example:
        >>> grid = np.zeros((3, 3), dtype=int)
        >>> mask = border_mask(grid)
        >>> mask.astype(int)
        array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    H, W = grid.shape
    mask = np.zeros((H, W), dtype=bool)

    # Set borders to True
    if H > 0:
        mask[0, :] = True    # top row
        mask[H-1, :] = True  # bottom row

    if W > 0:
        mask[:, 0] = True    # left column
        mask[:, W-1] = True  # right column

    return mask


if __name__ == "__main__":
    # Simple smoke test on a toy 4x5 grid
    import numpy as np

    grid = np.arange(20, dtype=int).reshape(4, 5)
    print("Grid:")
    from src.core.grid_types import print_grid
    print_grid(grid)

    H, W = grid.shape

    cf = coord_features(grid)
    print("\nFeatures for a few pixels:")
    for (r, c) in [(0, 0), (0, W-1), (H-1, 0), (H-1, W-1)]:
        print(f"  (r={r}, c={c}):", cf[(r, c)])

    row_bands = row_band_labels(H)
    col_bands = col_band_labels(W)
    print("\nRow bands:", row_bands)
    print("Col bands:", col_bands)

    bm = border_mask(grid)
    print("\nBorder mask (True on border pixels):")
    print(bm.astype(int))  # print as 0/1 for clarity
