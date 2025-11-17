"""
Smoke tests for solution decoding (y → Grid).

This test verifies that y_to_grid and y_flat_to_grid correctly decode
one-hot y vectors into grids with known pixel colors.

No ARC integration; just pure decoding sanity checks.
"""

import numpy as np

from src.core.grid_types import Grid
from src.solver.decoding import y_to_grid, y_flat_to_grid


def test_y_to_grid_2x2():
    """
    Test y_to_grid with a 2x2 grid, 3 colors.

    Pixel colors (flattened row-major):
      - p0 (0,0) -> color 0
      - p1 (0,1) -> color 1
      - p2 (1,0) -> color 2
      - p3 (1,1) -> color 1

    Expected grid:
      [[0, 1],
       [2, 1]]
    """
    print("\n" + "=" * 70)
    print("TEST: y_to_grid with 2x2 grid, 3 colors")
    print("=" * 70)

    H, W, C = 2, 2, 3
    num_pixels = H * W

    # Build one-hot y for each pixel
    y = np.zeros((num_pixels, C), dtype=int)
    y[0, 0] = 1  # p0 -> color 0
    y[1, 1] = 1  # p1 -> color 1
    y[2, 2] = 1  # p2 -> color 2
    y[3, 1] = 1  # p3 -> color 1

    print(f"  Input y shape: {y.shape}")
    print(f"  Input y:\n{y}")

    grid = y_to_grid(y, H, W, C)

    expected = np.array([[0, 1],
                         [2, 1]], dtype=int)

    print(f"  Output grid:\n{grid}")
    print(f"  Expected:\n{expected}")

    # Verify shape
    assert grid.shape == (H, W), \
        f"Grid shape mismatch: {grid.shape} != ({H}, {W})"

    # Verify contents
    assert np.array_equal(grid, expected), \
        f"Grid mismatch:\nGot:\n{grid}\nExpected:\n{expected}"

    print("  ✓ test_y_to_grid_2x2: PASSED")


def test_y_flat_to_grid_2x2():
    """
    Test y_flat_to_grid with flattened y (1D array).

    Pixel colors:
      - p0 (0,0) -> color 2
      - p1 (0,1) -> color 0
      - p2 (1,0) -> color 1
      - p3 (1,1) -> color 2

    Expected grid:
      [[2, 0],
       [1, 2]]
    """
    print("\n" + "=" * 70)
    print("TEST: y_flat_to_grid with flat 1D y")
    print("=" * 70)

    H, W, C = 2, 2, 3
    num_pixels = H * W

    # Build one-hot y as 2D first
    y2 = np.zeros((num_pixels, C), dtype=int)
    y2[0, 2] = 1  # p0 -> color 2
    y2[1, 0] = 1  # p1 -> color 0
    y2[2, 1] = 1  # p2 -> color 1
    y2[3, 2] = 1  # p3 -> color 2

    # Flatten to 1D
    y_flat = y2.reshape(-1)

    print(f"  Input y_flat shape: {y_flat.shape}")
    print(f"  Input y_flat: {y_flat}")

    grid = y_flat_to_grid(y_flat, H, W, C)

    expected = np.array([[2, 0],
                         [1, 2]], dtype=int)

    print(f"  Output grid:\n{grid}")
    print(f"  Expected:\n{expected}")

    # Verify shape
    assert grid.shape == (H, W), \
        f"Grid shape mismatch: {grid.shape} != ({H}, {W})"

    # Verify contents
    assert np.array_equal(grid, expected), \
        f"Grid mismatch:\nGot:\n{grid}\nExpected:\n{expected}"

    print("  ✓ test_y_flat_to_grid_2x2: PASSED")


def test_y_to_grid_with_floats():
    """
    Test y_to_grid with float values (simulating solver output).

    Solver might return values like 0.999 instead of 1.0.
    Argmax should still work correctly.
    """
    print("\n" + "=" * 70)
    print("TEST: y_to_grid with float values (solver-like)")
    print("=" * 70)

    H, W, C = 2, 2, 3
    num_pixels = H * W

    # Simulate solver output with small noise
    y = np.array([
        [0.001, 0.998, 0.001],  # p0 -> color 1 (nearly one-hot)
        [0.002, 0.001, 0.997],  # p1 -> color 2
        [0.999, 0.0005, 0.0005],  # p2 -> color 0
        [0.001, 0.001, 0.998]   # p3 -> color 2
    ], dtype=float)

    print(f"  Input y (float):\n{y}")

    grid = y_to_grid(y, H, W, C)

    expected = np.array([[1, 2],
                         [0, 2]], dtype=int)

    print(f"  Output grid:\n{grid}")
    print(f"  Expected:\n{expected}")

    assert grid.shape == (H, W)
    assert np.array_equal(grid, expected), \
        f"Grid mismatch with float input"

    print("  ✓ test_y_to_grid_with_floats: PASSED")


def test_invalid_shape():
    """
    Test that invalid y shapes raise ValueError.
    """
    print("\n" + "=" * 70)
    print("TEST: Invalid y shape detection")
    print("=" * 70)

    H, W, C = 2, 2, 3

    # Wrong flat length
    y_wrong = np.zeros(10)  # Should be 2*2*3 = 12
    try:
        grid = y_to_grid(y_wrong, H, W, C)
        raise AssertionError("Should have raised ValueError for wrong flat length")
    except ValueError as e:
        print(f"  ✓ Caught expected ValueError: {e}")

    # Wrong 2D shape
    y_wrong_2d = np.zeros((5, 3))  # Should be (4, 3)
    try:
        grid = y_to_grid(y_wrong_2d, H, W, C)
        raise AssertionError("Should have raised ValueError for wrong 2D shape")
    except ValueError as e:
        print(f"  ✓ Caught expected ValueError: {e}")

    print("  ✓ test_invalid_shape: PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DECODING SMOKE TEST SUITE")
    print("=" * 70)

    # Run all tests
    test_y_to_grid_2x2()
    test_y_flat_to_grid_2x2()
    test_y_to_grid_with_floats()
    test_invalid_shape()

    print("\n" + "=" * 70)
    print("✓ ALL DECODING TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - 2D y decoding: ✓")
    print("  - Flat 1D y decoding: ✓")
    print("  - Float value handling: ✓")
    print("  - Invalid shape detection: ✓")
    print("\nDecoding module is ready for integration with solver")
    print()
