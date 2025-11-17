"""
y-vector indexing helpers for the constraint system.

This module provides canonical mappings between:
  - Pixel coordinates (r, c) ↔ flat pixel index p_idx (0..N-1)
  - Pixel/color (p_idx, color) ↔ y-vector index y_idx (0..N*C-1)

Conventions (matching math kernel spec section 1.1):
  - Grid shape: (H, W)
  - Pixel ordering: row-major, p_idx = r * W + c
  - y-vector layout: y_idx = p_idx * C + color
  - All indices are 0-based (Python convention)

This is pure indexing math with no dependencies on constraints or solver.
"""

from typing import Tuple


def flatten_index(r: int, c: int, W: int) -> int:
    """
    Convert row/col coordinates to a flat pixel index (0 .. N-1).

    Uses row-major ordering: all columns of row 0, then all columns of row 1, etc.

    Args:
        r: row index, 0 <= r < H
        c: col index, 0 <= c < W
        W: grid width

    Returns:
        p_idx: flat pixel index, p_idx = r * W + c

    Example:
        >>> # 3x4 grid, pixel at row 1, col 2
        >>> flatten_index(1, 2, 4)
        6
    """
    return r * W + c


def unflatten_index(p_idx: int, W: int) -> Tuple[int, int]:
    """
    Convert a flat pixel index back to (row, col).

    This is the inverse of flatten_index.

    Args:
        p_idx: flat pixel index, 0 <= p_idx < N
        W: grid width

    Returns:
        (r, c): row and column indices, consistent with flatten_index

    Example:
        >>> # 3x4 grid, pixel index 6
        >>> unflatten_index(6, 4)
        (1, 2)
    """
    r = p_idx // W
    c = p_idx % W
    return (r, c)


def y_index(p_idx: int, color: int, C: int) -> int:
    """
    Convert a (pixel index, color index) pair to a y-vector index (0 .. N*C-1).

    The y-vector is structured as:
      [p0_c0, p0_c1, ..., p0_cC-1, p1_c0, p1_c1, ..., p1_cC-1, ...]

    All colors for pixel 0, then all colors for pixel 1, etc.

    Args:
        p_idx: flat pixel index, 0 <= p_idx < N
        color: color index, 0 <= color < C
        C: number of colors

    Returns:
        y_idx: index in the y vector, y_idx = p_idx * C + color

    Example:
        >>> # 12 pixels, 5 colors, pixel 3 with color 2
        >>> y_index(3, 2, 5)
        17
    """
    return p_idx * C + color


def y_index_to_pc(y_idx: int, C: int, W: int) -> Tuple[int, int]:
    """
    Convert a y-vector index back to (pixel index, color index).

    This is the inverse of y_index.

    Args:
        y_idx: index in y, 0 <= y_idx < N*C
        C: number of colors
        W: grid width (for potential future (r,c) recovery via unflatten_index)

    Returns:
        (p_idx, color) pair, such that:
            y_idx == p_idx * C + color

    Note:
        The actual (r,c) coordinates can be recovered by calling:
            unflatten_index(p_idx, W)

    Example:
        >>> # y-vector index 17 with 5 colors
        >>> y_index_to_pc(17, 5, 4)
        (3, 2)
    """
    p_idx = y_idx // C
    color = y_idx % C
    return (p_idx, color)


if __name__ == "__main__":
    # Sanity checks for indexing roundtrips
    H, W, C = 3, 4, 5

    print("Testing pixel index roundtrip...")
    # Check pixel index roundtrip: (r,c) -> p_idx -> (r,c)
    for r in range(H):
        for c in range(W):
            p = flatten_index(r, c, W)
            rr, cc = unflatten_index(p, W)
            assert (rr, cc) == (r, c), f"Roundtrip (r,c)->p->(r,c) failed at {(r,c)}"

    print("  ✓ Pixel index roundtrip passed")

    print("Testing y-index roundtrip...")
    # Check y-index roundtrip: (p_idx, color) -> y_idx -> (p_idx, color)
    N = H * W
    for p_idx in range(N):
        for color in range(C):
            y_idx = y_index(p_idx, color, C)
            p_back, color_back = y_index_to_pc(y_idx, C, W)
            assert p_back == p_idx, f"p_idx mismatch: {p_idx} != {p_back}"
            assert color_back == color, f"color mismatch: {color} != {color_back}"

    print("  ✓ y-index roundtrip passed")

    # Additional sanity: check y_idx range
    print("Testing y-index bounds...")
    max_y_idx = y_index(N - 1, C - 1, C)
    expected_max = N * C - 1
    assert max_y_idx == expected_max, f"Max y_idx mismatch: {max_y_idx} != {expected_max}"
    print(f"  ✓ y-index range: 0 to {max_y_idx} (N*C-1 = {expected_max})")

    # Test specific example from docstrings
    print("Testing docstring examples...")
    assert flatten_index(1, 2, 4) == 6
    assert unflatten_index(6, 4) == (1, 2)
    assert y_index(3, 2, 5) == 17
    assert y_index_to_pc(17, 5, 4) == (3, 2)
    print("  ✓ Docstring examples passed")

    print("\n✓ indexing.py sanity checks passed.")
