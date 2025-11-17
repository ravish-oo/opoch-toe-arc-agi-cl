"""
Smoke test for LP/ILP solver wrapper.

This test verifies that solve_constraints_for_grid works correctly
with a minimal artificial constraint system (no real ARC data).

Test scenario:
  - 2 pixels: p0, p1
  - 3 colors: c0, c1, c2
  - Constraints:
    1. Pixel 0 must be color 1
    2. Pixel 1 must have same color as pixel 0
  - Expected solution: both pixels are color 1
"""

import numpy as np

from src.constraints.builder import ConstraintBuilder
from src.constraints.indexing import y_index
from src.solver.lp_solver import solve_constraints_for_grid, InfeasibleModelError


def build_simple_test_constraints() -> tuple[ConstraintBuilder, int, int]:
    """
    Build a simple test constraint system:
      - 2 pixels, 3 colors
      - p0 = color 1
      - p1 = p0 (same color)

    Returns:
        (builder, num_pixels, num_colors)
    """
    num_pixels = 2
    num_colors = 3
    builder = ConstraintBuilder()

    # Constraint 1: p0 must be color 1
    # This means: y[p0, 1] = 1, and y[p0, 0] = 0, y[p0, 2] = 0
    p0_idx = 0
    color1 = 1

    # Fix y[p0, 1] = 1
    idx_p0_c1 = y_index(p0_idx, color1, num_colors)
    builder.add_eq(indices=[idx_p0_c1], coeffs=[1.0], rhs=1.0)

    # Fix y[p0, 0] = 0
    idx_p0_c0 = y_index(p0_idx, 0, num_colors)
    builder.add_eq(indices=[idx_p0_c0], coeffs=[1.0], rhs=0.0)

    # Fix y[p0, 2] = 0
    idx_p0_c2 = y_index(p0_idx, 2, num_colors)
    builder.add_eq(indices=[idx_p0_c2], coeffs=[1.0], rhs=0.0)

    # Constraint 2: p1 has same color as p0
    # For each color c: y[p1, c] - y[p0, c] = 0
    p1_idx = 1
    for c in range(num_colors):
        idx_p1_c = y_index(p1_idx, c, num_colors)
        idx_p0_c = y_index(p0_idx, c, num_colors)
        builder.add_eq(
            indices=[idx_p1_c, idx_p0_c],
            coeffs=[1.0, -1.0],
            rhs=0.0
        )

    return builder, num_pixels, num_colors


def test_simple_ilp():
    """
    Test basic ILP solving with simple constraints.

    Expected:
      - Both pixels should be color 1
      - Solution: [[0, 1, 0], [0, 1, 0]]
    """
    print("\n" + "=" * 70)
    print("TEST: Simple ILP (2 pixels, 3 colors)")
    print("=" * 70)

    builder, num_pixels, num_colors = build_simple_test_constraints()

    print(f"  Constraints: {len(builder.constraints)}")
    print(f"  Pixels: {num_pixels}")
    print(f"  Colors: {num_colors}")

    # Solve
    y_sol = solve_constraints_for_grid(
        builder,
        num_pixels,
        num_colors,
        objective="min_sum"
    )

    print(f"  Solution shape: {y_sol.shape}")
    print(f"  Solution:\n{y_sol}")

    # Verify shape
    assert y_sol.shape == (num_pixels, num_colors), \
        f"Expected shape ({num_pixels}, {num_colors}), got {y_sol.shape}"

    # Verify pixel 0 is color 1: [0, 1, 0]
    expected_p0 = np.array([0, 1, 0])
    assert np.array_equal(y_sol[0], expected_p0), \
        f"Pixel 0 expected {expected_p0}, got {y_sol[0]}"

    # Verify pixel 1 is same as pixel 0
    assert np.array_equal(y_sol[1], y_sol[0]), \
        f"Pixel 1 should match pixel 0, got {y_sol[1]} vs {y_sol[0]}"

    print("  ✓ test_simple_ilp: PASSED")


def test_infeasible_constraints():
    """
    Test that infeasible constraints are detected.

    Scenario:
      - 1 pixel, 2 colors
      - Constraint 1: y[0, 0] = 1 (pixel 0 is color 0)
      - Constraint 2: y[0, 1] = 1 (pixel 0 is color 1)
      - This contradicts one-hot constraint → infeasible
    """
    print("\n" + "=" * 70)
    print("TEST: Infeasible constraints detection")
    print("=" * 70)

    num_pixels = 1
    num_colors = 2
    builder = ConstraintBuilder()

    # Pixel 0 = color 0
    idx_c0 = y_index(0, 0, num_colors)
    builder.add_eq(indices=[idx_c0], coeffs=[1.0], rhs=1.0)

    # Pixel 0 = color 1 (contradicts above!)
    idx_c1 = y_index(0, 1, num_colors)
    builder.add_eq(indices=[idx_c1], coeffs=[1.0], rhs=1.0)

    # Should raise InfeasibleModelError
    try:
        y_sol = solve_constraints_for_grid(builder, num_pixels, num_colors)
        raise AssertionError("Expected InfeasibleModelError, but solver succeeded!")
    except InfeasibleModelError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("  ✓ test_infeasible_constraints: PASSED")


def test_zero_objective():
    """
    Test solving with zero objective (feasibility only).

    Same constraints as test_simple_ilp, but with objective="none".
    """
    print("\n" + "=" * 70)
    print("TEST: Zero objective (feasibility only)")
    print("=" * 70)

    builder, num_pixels, num_colors = build_simple_test_constraints()

    # Solve with zero objective
    y_sol = solve_constraints_for_grid(
        builder,
        num_pixels,
        num_colors,
        objective="none"
    )

    # Same verification as test_simple_ilp
    expected_p0 = np.array([0, 1, 0])
    assert np.array_equal(y_sol[0], expected_p0), \
        f"Pixel 0 expected {expected_p0}, got {y_sol[0]}"

    assert np.array_equal(y_sol[1], y_sol[0]), \
        f"Pixel 1 should match pixel 0"

    print("  ✓ test_zero_objective: PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LP SOLVER SMOKE TEST SUITE")
    print("=" * 70)

    # Run all tests
    test_simple_ilp()
    test_infeasible_constraints()
    test_zero_objective()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - Basic ILP solving: ✓")
    print("  - Infeasibility detection: ✓")
    print("  - Zero objective mode: ✓")
    print("\nLP solver wrapper is ready for integration with kernel.py")
    print()
