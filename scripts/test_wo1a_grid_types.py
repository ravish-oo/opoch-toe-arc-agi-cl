#!/usr/bin/env python3
"""
Comprehensive test script for WO1a - grid_types.py

Tests beyond the built-in self-test to catch edge cases and validate
alignment with math kernel requirements.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.grid_types import Grid, Pixel, pixel_index, index_to_pixel, print_grid


def test_pixel_index_basic():
    """Test basic pixel_index calculations"""
    print("Testing pixel_index basic cases...")

    # 2x2 grid
    assert pixel_index(0, 0, 2) == 0, "Top-left should be 0"
    assert pixel_index(0, 1, 2) == 1, "Top-right should be 1"
    assert pixel_index(1, 0, 2) == 2, "Bottom-left should be 2"
    assert pixel_index(1, 1, 2) == 3, "Bottom-right should be 3"

    # 3x4 grid (H=3, W=4)
    assert pixel_index(0, 0, 4) == 0
    assert pixel_index(0, 3, 4) == 3
    assert pixel_index(1, 0, 4) == 4
    assert pixel_index(2, 3, 4) == 11  # Last pixel: 2*4 + 3 = 11

    print("  ✓ Basic cases pass")


def test_index_to_pixel_basic():
    """Test basic index_to_pixel calculations"""
    print("Testing index_to_pixel basic cases...")

    # 2x2 grid
    assert index_to_pixel(0, 2) == (0, 0)
    assert index_to_pixel(1, 2) == (0, 1)
    assert index_to_pixel(2, 2) == (1, 0)
    assert index_to_pixel(3, 2) == (1, 1)

    # 3x4 grid
    assert index_to_pixel(0, 4) == (0, 0)
    assert index_to_pixel(3, 4) == (0, 3)
    assert index_to_pixel(4, 4) == (1, 0)
    assert index_to_pixel(11, 4) == (2, 3)

    print("  ✓ Basic cases pass")


def test_roundtrip_various_sizes():
    """Test roundtrip for various grid sizes (math kernel compliance)"""
    print("Testing roundtrip for various grid sizes...")

    test_sizes = [
        (1, 1),   # Minimal
        (2, 2),   # Square small
        (3, 4),   # Rectangular
        (10, 10), # Square medium
        (5, 20),  # Wide
        (20, 5),  # Tall
        (30, 30), # Large (ARC tasks are max 30x30)
    ]

    for H, W in test_sizes:
        for r in range(H):
            for c in range(W):
                idx = pixel_index(r, c, W)
                r2, c2 = index_to_pixel(idx, W)
                assert (r, c) == (r2, c2), f"Roundtrip failed for ({H},{W}) at ({r},{c})"

        # Also verify index bounds
        expected_max_idx = H * W - 1
        actual_max_idx = pixel_index(H-1, W-1, W)
        assert actual_max_idx == expected_max_idx, \
            f"Max index mismatch for {H}x{W}: expected {expected_max_idx}, got {actual_max_idx}"

    print(f"  ✓ Roundtrip passed for {len(test_sizes)} different grid sizes")


def test_error_handling():
    """Test that error cases are properly caught"""
    print("Testing error handling...")

    # Test pixel_index with negative row
    try:
        pixel_index(-1, 0, 5)
        assert False, "Should have raised ValueError for negative row"
    except ValueError as e:
        assert "non-negative" in str(e)

    # Test pixel_index with negative col
    try:
        pixel_index(0, -1, 5)
        assert False, "Should have raised ValueError for negative col"
    except ValueError as e:
        assert "non-negative" in str(e)

    # Test index_to_pixel with negative index
    try:
        index_to_pixel(-1, 5)
        assert False, "Should have raised ValueError for negative index"
    except ValueError as e:
        assert "non-negative" in str(e)

    print("  ✓ Error handling works correctly")


def test_print_grid_output():
    """Test print_grid produces correct format"""
    print("Testing print_grid output format...")

    # Capture output
    import io
    from contextlib import redirect_stdout

    grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)

    f = io.StringIO()
    with redirect_stdout(f):
        print_grid(grid)
    output = f.getvalue()

    expected_lines = ["0 1 2", "3 4 5"]
    actual_lines = output.strip().split('\n')

    assert actual_lines == expected_lines, \
        f"Print output mismatch:\nExpected: {expected_lines}\nActual: {actual_lines}"

    print("  ✓ Print format correct")


def test_print_grid_assertions():
    """Test print_grid catches invalid inputs"""
    print("Testing print_grid assertions...")

    # 1D array should fail
    try:
        print_grid(np.array([1, 2, 3]))
        assert False, "Should have raised AssertionError for 1D array"
    except AssertionError as e:
        assert "2D" in str(e)

    # 3D array should fail
    try:
        print_grid(np.array([[[1]]]))
        assert False, "Should have raised AssertionError for 3D array"
    except AssertionError as e:
        assert "2D" in str(e)

    print("  ✓ Dimension assertions work")


def test_type_alias_usage():
    """Verify type aliases are properly defined"""
    print("Testing type aliases...")

    # Grid should be ndarray type
    grid: Grid = np.array([[1, 2], [3, 4]], dtype=int)
    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.dtype('int64') or grid.dtype == np.dtype('int32')

    # Pixel should be tuple
    pixel: Pixel = (0, 1)
    assert isinstance(pixel, tuple)
    assert len(pixel) == 2

    print("  ✓ Type aliases correctly defined")


def test_math_kernel_compliance():
    """
    Verify compliance with math kernel section 1.1:
    - Grid: height H, width W, palette C = {0,...,C-1}
    - Pixels Ω = {1,...,N} with N=H·W (in code: 0-indexed so [0, N-1])
    - One-hot encoding foundation: y ∈ {0,1}^{NC}
    """
    print("Testing math kernel compliance...")

    # Test case: 3x4 grid with palette {0,1,2,3,4}
    H, W = 3, 4
    C = 5
    grid = np.random.randint(0, C, size=(H, W), dtype=int)

    # Verify N = H * W
    N = H * W
    assert N == 12

    # Verify all pixels can be indexed in [0, N-1]
    max_idx = pixel_index(H-1, W-1, W)
    assert max_idx == N - 1, f"Max index should be {N-1}, got {max_idx}"

    # Verify one-hot space dimensionality: NC = 12 * 5 = 60
    NC_dim = N * C
    assert NC_dim == 60

    # Verify every pixel maps to unique index
    indices = set()
    for r in range(H):
        for c in range(W):
            idx = pixel_index(r, c, W)
            assert idx not in indices, f"Duplicate index {idx} for pixel ({r},{c})"
            assert 0 <= idx < N, f"Index {idx} out of bounds [0, {N-1}]"
            indices.add(idx)

    assert len(indices) == N, f"Should have {N} unique indices, got {len(indices)}"

    print("  ✓ Math kernel section 1.1 compliance verified")


def main():
    print("=" * 60)
    print("WO1a Comprehensive Test Suite - grid_types.py")
    print("=" * 60)
    print()

    tests = [
        test_pixel_index_basic,
        test_index_to_pixel_basic,
        test_roundtrip_various_sizes,
        test_error_handling,
        test_print_grid_output,
        test_print_grid_assertions,
        test_type_alias_usage,
        test_math_kernel_compliance,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print()
    print("=" * 60)
    print(f"✅ ALL {len(tests)} TEST SUITES PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
