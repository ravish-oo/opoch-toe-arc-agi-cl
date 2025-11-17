#!/usr/bin/env python3
"""
Comprehensive review test for WO-M4.2: Solution decoding (y → Grid).

This test verifies:
  1. All WO implementation steps from WO are present
  2. Shape validation works (1D and 2D)
  3. Argmax logic is correct
  4. Reshape logic is correct
  5. y_flat_to_grid is thin wrapper (no duplication)
  6. Error handling works properly
  7. Float handling (solver-like output)
  8. Integration with M4.1 solver
  9. No TODOs, stubs, or simplified implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.solver.decoding import y_to_grid, y_flat_to_grid
from src.constraints.builder import ConstraintBuilder
from src.solver.lp_solver import solve_constraints_for_grid


def test_implementation_steps():
    """Test that all WO implementation steps are present."""
    print("\nTest: All WO implementation steps present")
    print("-" * 70)

    # Read source to verify steps
    decoding_file = project_root / "src/solver/decoding.py"
    source = decoding_file.read_text()

    # Step 1: Validate shape (both 1D and 2D)
    assert "if y.ndim == 1:" in source, "Step 1a: 1D shape validation missing"
    assert "elif y.ndim == 2:" in source, "Step 1b: 2D shape validation missing"
    assert "raise ValueError" in source, "Step 1c: ValueError for invalid shapes missing"
    print("  ✓ Step 1: Validate shape (1D and 2D)")

    # Step 2: Argmax per pixel
    assert "np.argmax" in source, "Step 2: Argmax missing"
    assert "axis=1" in source, "Step 2: Row-wise argmax missing"
    print("  ✓ Step 2: Argmax per pixel (row-wise)")

    # Step 3: Reshape to grid
    assert "reshape(H, W)" in source, "Step 3: Reshape to (H,W) missing"
    assert ".astype(int)" in source, "Step 3: Integer conversion missing"
    print("  ✓ Step 3: Reshape to (H, W) grid")

    # Step 4: Return
    assert "return grid" in source, "Step 4: Return missing"
    print("  ✓ Step 4: Return grid")

    # Step 5: y_flat_to_grid is thin wrapper
    assert "return y_to_grid(y_flat, H, W, C)" in source, \
        "Step 5: y_flat_to_grid should be thin wrapper"
    print("  ✓ Step 5: y_flat_to_grid is thin wrapper (no duplication)")

    print("  ✓ All WO implementation steps verified")


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    decoding_file = project_root / "src/solver/decoding.py"
    source = decoding_file.read_text()

    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError"]

    for marker in markers:
        assert marker not in source, \
            f"Found '{marker}' in decoding.py"

    print("  ✓ No TODOs, stubs, or markers found")


def test_shape_validation_2d():
    """Test shape validation with 2D input."""
    print("\nTest: Shape validation (2D input)")
    print("-" * 70)

    H, W, C = 3, 3, 4
    num_pixels = H * W

    # Valid 2D shape
    y = np.zeros((num_pixels, C))
    y[:, 0] = 1  # All pixels color 0
    grid = y_to_grid(y, H, W, C)
    assert grid.shape == (H, W), f"Expected shape ({H},{W}), got {grid.shape}"
    print("  ✓ Valid 2D shape works")

    # Invalid 2D shape (wrong num_pixels)
    y_wrong = np.zeros((num_pixels + 1, C))
    try:
        grid = y_to_grid(y_wrong, H, W, C)
        raise AssertionError("Should have raised ValueError for wrong num_pixels")
    except ValueError as e:
        print(f"  ✓ Caught expected ValueError: {e}")

    # Invalid 2D shape (wrong C)
    y_wrong = np.zeros((num_pixels, C + 1))
    try:
        grid = y_to_grid(y_wrong, H, W, C)
        raise AssertionError("Should have raised ValueError for wrong C")
    except ValueError as e:
        print(f"  ✓ Caught expected ValueError: {e}")

    print("  ✓ 2D shape validation works correctly")


def test_shape_validation_1d():
    """Test shape validation with 1D flat input."""
    print("\nTest: Shape validation (1D flat input)")
    print("-" * 70)

    H, W, C = 2, 3, 4
    num_pixels = H * W

    # Valid flat shape
    y_flat = np.zeros(num_pixels * C)
    y_flat[0] = 1  # First pixel, color 0
    grid = y_to_grid(y_flat, H, W, C)
    assert grid.shape == (H, W), f"Expected shape ({H},{W}), got {grid.shape}"
    print("  ✓ Valid flat 1D shape works")

    # Invalid flat length
    y_wrong = np.zeros(num_pixels * C + 1)
    try:
        grid = y_to_grid(y_wrong, H, W, C)
        raise AssertionError("Should have raised ValueError for wrong flat length")
    except ValueError as e:
        print(f"  ✓ Caught expected ValueError: {e}")

    print("  ✓ 1D shape validation works correctly")


def test_argmax_logic():
    """Test that argmax logic picks correct color."""
    print("\nTest: Argmax logic")
    print("-" * 70)

    H, W, C = 2, 2, 5
    num_pixels = H * W

    # Each pixel has a different max color
    y = np.zeros((num_pixels, C))
    y[0, 2] = 1  # p0 -> color 2
    y[1, 4] = 1  # p1 -> color 4
    y[2, 0] = 1  # p2 -> color 0
    y[3, 3] = 1  # p3 -> color 3

    grid = y_to_grid(y, H, W, C)
    expected = np.array([[2, 4], [0, 3]], dtype=int)

    assert np.array_equal(grid, expected), \
        f"Argmax failed:\nGot:\n{grid}\nExpected:\n{expected}"

    print(f"  Grid:\n{grid}")
    print("  ✓ Argmax picks correct color for each pixel")


def test_argmax_with_floats():
    """Test argmax with float values (solver-like output)."""
    print("\nTest: Argmax with float values")
    print("-" * 70)

    H, W, C = 2, 2, 3
    num_pixels = H * W

    # Simulate solver output with near-one-hot values
    y = np.array([
        [0.001, 0.997, 0.002],  # p0 -> color 1
        [0.998, 0.001, 0.001],  # p1 -> color 0
        [0.002, 0.003, 0.995],  # p2 -> color 2
        [0.001, 0.998, 0.001]   # p3 -> color 1
    ], dtype=float)

    grid = y_to_grid(y, H, W, C)
    expected = np.array([[1, 0], [2, 1]], dtype=int)

    assert np.array_equal(grid, expected), \
        f"Float argmax failed:\nGot:\n{grid}\nExpected:\n{expected}"

    print(f"  Grid:\n{grid}")
    print("  ✓ Argmax handles float values correctly")


def test_reshape_logic():
    """Test that reshape produces correct (H, W) grid."""
    print("\nTest: Reshape logic")
    print("-" * 70)

    # Non-square grid to test row-major ordering
    H, W, C = 3, 4, 2
    num_pixels = H * W

    # Create pattern: row 0 all color 0, row 1 all color 1, row 2 all color 0
    y = np.zeros((num_pixels, C))
    for p in range(num_pixels):
        r = p // W
        color = r % C
        y[p, color] = 1

    grid = y_to_grid(y, H, W, C)

    expected = np.array([
        [0, 0, 0, 0],  # row 0
        [1, 1, 1, 1],  # row 1
        [0, 0, 0, 0]   # row 2
    ], dtype=int)

    assert grid.shape == (H, W), f"Shape mismatch: {grid.shape} != ({H},{W})"
    assert np.array_equal(grid, expected), \
        f"Reshape failed:\nGot:\n{grid}\nExpected:\n{expected}"

    print(f"  Grid shape: {grid.shape}")
    print(f"  Grid:\n{grid}")
    print("  ✓ Reshape produces correct (H, W) row-major grid")


def test_y_flat_to_grid_is_wrapper():
    """Test that y_flat_to_grid is just a thin wrapper."""
    print("\nTest: y_flat_to_grid is thin wrapper")
    print("-" * 70)

    H, W, C = 2, 2, 3
    num_pixels = H * W

    # Same y in both 2D and flat forms
    y_2d = np.zeros((num_pixels, C))
    y_2d[0, 1] = 1
    y_2d[1, 2] = 1
    y_2d[2, 0] = 1
    y_2d[3, 1] = 1

    y_flat = y_2d.reshape(-1)

    grid_from_2d = y_to_grid(y_2d, H, W, C)
    grid_from_flat = y_flat_to_grid(y_flat, H, W, C)

    assert np.array_equal(grid_from_2d, grid_from_flat), \
        f"y_flat_to_grid doesn't match y_to_grid:\n{grid_from_2d}\nvs\n{grid_from_flat}"

    print(f"  Grid from 2D:\n{grid_from_2d}")
    print(f"  Grid from flat:\n{grid_from_flat}")
    print("  ✓ y_flat_to_grid produces same result as y_to_grid")


def test_integration_with_solver():
    """Test decoding works with M4.1 solver output."""
    print("\nTest: Integration with M4.1 solver")
    print("-" * 70)

    # Build simple constraint: 2 pixels, 3 colors
    # p0 = color 1, p1 = color 2
    builder = ConstraintBuilder()
    builder.fix_pixel_color(0, 1, C=3)
    builder.fix_pixel_color(1, 2, C=3)

    # Solve
    y_sol = solve_constraints_for_grid(builder, num_pixels=2, num_colors=3)

    # Decode to 1x2 grid
    grid = y_to_grid(y_sol, H=1, W=2, C=3)

    expected = np.array([[1, 2]], dtype=int)

    assert grid.shape == (1, 2)
    assert np.array_equal(grid, expected), \
        f"Integration failed:\nGot:\n{grid}\nExpected:\n{expected}"

    print(f"  Solver output y shape: {y_sol.shape}")
    print(f"  Decoded grid: {grid}")
    print(f"  Expected: {expected}")
    print("  ✓ Decoding works with M4.1 solver output")


def test_single_pixel():
    """Test edge case: single pixel grid."""
    print("\nTest: Single pixel grid")
    print("-" * 70)

    H, W, C = 1, 1, 5

    # Single pixel, color 3
    y = np.zeros((1, C))
    y[0, 3] = 1

    grid = y_to_grid(y, H, W, C)
    expected = np.array([[3]], dtype=int)

    assert grid.shape == (H, W)
    assert np.array_equal(grid, expected)

    print(f"  Grid: {grid}")
    print("  ✓ Single pixel works correctly")


def test_large_grid():
    """Test with larger grid (10x10, 10 colors)."""
    print("\nTest: Large grid (10x10, 10 colors)")
    print("-" * 70)

    H, W, C = 10, 10, 10
    num_pixels = H * W

    # Create checkerboard-like pattern
    y = np.zeros((num_pixels, C))
    for p in range(num_pixels):
        r = p // W
        c = p % W
        color = (r + c) % C
        y[p, color] = 1

    grid = y_to_grid(y, H, W, C)

    # Verify shape
    assert grid.shape == (H, W)

    # Verify pattern
    for r in range(H):
        for c in range(W):
            expected_color = (r + c) % C
            assert grid[r, c] == expected_color, \
                f"Pixel ({r},{c}) should be color {expected_color}, got {grid[r,c]}"

    print(f"  Grid shape: {grid.shape}")
    print(f"  Sample corner:\n{grid[:3, :3]}")
    print("  ✓ Large grid works correctly")


def test_wo_test_cases():
    """Test that WO-specified test cases are present and working."""
    print("\nTest: WO-specified test cases")
    print("-" * 70)

    # WO Test 1: 2x2, 3 colors, pattern [0,1,2,1]
    H, W, C = 2, 2, 3
    y1 = np.zeros((4, C))
    y1[0, 0] = 1
    y1[1, 1] = 1
    y1[2, 2] = 1
    y1[3, 1] = 1
    grid1 = y_to_grid(y1, H, W, C)
    expected1 = np.array([[0, 1], [2, 1]])
    assert np.array_equal(grid1, expected1)
    print("  ✓ WO Test 1 (test_y_to_grid_2x2) works")

    # WO Test 2: 2x2, 3 colors, flat, pattern [2,0,1,2]
    y2_2d = np.zeros((4, C))
    y2_2d[0, 2] = 1
    y2_2d[1, 0] = 1
    y2_2d[2, 1] = 1
    y2_2d[3, 2] = 1
    y2_flat = y2_2d.reshape(-1)
    grid2 = y_flat_to_grid(y2_flat, H, W, C)
    expected2 = np.array([[2, 0], [1, 2]])
    assert np.array_equal(grid2, expected2)
    print("  ✓ WO Test 2 (test_y_flat_to_grid_2x2) works")

    print("  ✓ Both WO-specified test cases verified")


def test_error_messages():
    """Test that error messages are informative."""
    print("\nTest: Error messages are informative")
    print("-" * 70)

    H, W, C = 2, 2, 3

    # Wrong flat length
    y_wrong = np.zeros(10)
    try:
        y_to_grid(y_wrong, H, W, C)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "does not match H*W*C" in error_msg, \
            f"Error message should mention H*W*C: {error_msg}"
        print(f"  ✓ Flat length error mentions H*W*C: {error_msg}")

    # Wrong 2D shape
    y_wrong_2d = np.zeros((5, 3))
    try:
        y_to_grid(y_wrong_2d, H, W, C)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "does not match" in error_msg, \
            f"Error message should mention mismatch: {error_msg}"
        print(f"  ✓ 2D shape error mentions mismatch: {error_msg}")

    # Invalid ndim
    y_3d = np.zeros((2, 2, 3))
    try:
        y_to_grid(y_3d, H, W, C)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "ndim" in error_msg or "1D or 2D" in error_msg, \
            f"Error message should mention ndim: {error_msg}"
        print(f"  ✓ Invalid ndim error mentions dimension: {error_msg}")

    print("  ✓ All error messages are informative")


def test_code_organization():
    """Test code organization and quality."""
    print("\nTest: Code organization")
    print("-" * 70)

    decoding_file = project_root / "src/solver/decoding.py"
    source = decoding_file.read_text()

    # Check has module docstring
    assert '"""' in source[:100], "Module should have docstring"
    print("  ✓ Module has docstring")

    # Check both functions have docstrings
    assert "def y_to_grid" in source
    assert 'Args:' in source, "y_to_grid should document Args"
    assert 'Returns:' in source, "y_to_grid should document Returns"
    print("  ✓ Functions have docstrings with Args/Returns")

    # Check self-test present
    assert 'if __name__ == "__main__":' in source
    print("  ✓ Self-test present")

    # Check imports are minimal (only numpy + own modules)
    import_lines = [line for line in source.split('\n') if line.startswith('import ') or line.startswith('from ')]
    external_imports = [line for line in import_lines if 'numpy' not in line and 'src.' not in line]
    # Filter out empty/whitespace
    external_imports = [line for line in external_imports if line.strip() and not line.startswith('#')]
    assert len(external_imports) == 0, \
        f"Should only import numpy + own modules, found: {external_imports}"
    print("  ✓ Only imports numpy + own modules")

    print("  ✓ Code organization is clean")


def main():
    print("=" * 70)
    print("WO-M4.2 COMPREHENSIVE REVIEW TEST")
    print("Testing solution decoding (y → Grid)")
    print("=" * 70)

    try:
        # Core implementation
        test_implementation_steps()
        test_no_todos_stubs()
        test_code_organization()

        # Shape validation
        test_shape_validation_2d()
        test_shape_validation_1d()

        # Core logic
        test_argmax_logic()
        test_argmax_with_floats()
        test_reshape_logic()
        test_y_flat_to_grid_is_wrapper()

        # Integration & edge cases
        test_integration_with_solver()
        test_single_pixel()
        test_large_grid()

        # WO compliance
        test_wo_test_cases()
        test_error_messages()

        print("\n" + "=" * 70)
        print("✅ WO-M4.2 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ All WO implementation steps - VERIFIED")
        print("  ✓ No TODOs or stubs - VERIFIED")
        print("  ✓ Code organization - CLEAN")
        print()
        print("  ✓ Shape validation - ALL WORKING")
        print("    - 2D input (H*W, C)")
        print("    - 1D flat input (H*W*C)")
        print("    - Error detection for invalid shapes")
        print()
        print("  ✓ Core logic - CORRECT")
        print("    - Argmax picks correct color")
        print("    - Handles float values (solver-like)")
        print("    - Reshape produces (H, W) row-major grid")
        print("    - y_flat_to_grid is thin wrapper")
        print()
        print("  ✓ Integration & edge cases - EXCELLENT")
        print("    - Works with M4.1 solver output")
        print("    - Single pixel grids")
        print("    - Large grids (10x10)")
        print()
        print("  ✓ WO compliance - 100%")
        print("    - Both WO test cases verified")
        print("    - Error messages informative")
        print()
        print("WO-M4.2 IMPLEMENTATION READY FOR PRODUCTION")
        print("=" * 70)
        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
