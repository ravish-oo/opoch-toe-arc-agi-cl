#!/usr/bin/env python3
"""
Comprehensive review test for WO-M4.1: LP/ILP solver wrapper.

This test verifies:
  1. All 9 implementation steps from WO are present
  2. Solver correctly handles various constraint types
  3. Error handling works properly
  4. Integration with ConstraintBuilder
  5. Solution quality and correctness
  6. No TODOs, stubs, or simplified implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.constraints.builder import ConstraintBuilder
from src.constraints.indexing import y_index
from src.solver.lp_solver import solve_constraints_for_grid, InfeasibleModelError


def test_implementation_steps():
    """Test that all 9 WO implementation steps are present."""
    print("\nTest: All 9 WO implementation steps present")
    print("-" * 70)

    # Read source to verify steps
    lp_solver_file = project_root / "src/solver/lp_solver.py"
    source = lp_solver_file.read_text()

    # Step 1: Create model
    assert "pulp.LpProblem" in source, "Step 1: Create LpProblem missing"
    print("  ✓ Step 1: Create LpProblem")

    # Step 2: Create binary variables
    assert "pulp.LpVariable" in source, "Step 2: Create variables missing"
    assert "LpBinary" in source, "Step 2: Binary variables missing"
    print("  ✓ Step 2: Create binary variables y[p][c]")

    # Step 3: Add builder constraints
    assert "builder.constraints" in source, "Step 3: Builder constraints missing"
    assert "y_index_to_pc" in source, "Step 3: Index mapping missing"
    print("  ✓ Step 3: Add builder constraints")

    # Step 4: One-hot constraints
    assert "sum(y[p][c] for c in range(num_colors)) == 1" in source, \
        "Step 4: One-hot constraints missing"
    print("  ✓ Step 4: Add one-hot constraints")

    # Step 5: Objective
    assert 'if objective == "min_sum"' in source, "Step 5: Objective missing"
    print("  ✓ Step 5: Set objective")

    # Step 6: Solve
    assert "prob.solve" in source, "Step 6: Solve missing"
    assert "PULP_CBC_CMD" in source, "Step 6: CBC solver missing"
    print("  ✓ Step 6: Solve with CBC")

    # Step 7: Extract solution
    assert "np.zeros((num_pixels, num_colors)" in source, "Step 7: Array creation missing"
    assert "pulp.value(y[p][c])" in source, "Step 7: Extract values missing"
    print("  ✓ Step 7: Extract solution")

    # Step 8: Sanity check
    assert "y_sol.sum(axis=1)" in source, "Step 8: Sanity check missing"
    assert "AssertionError" in source or "raise" in source, "Step 8: Check missing"
    print("  ✓ Step 8: Sanity check (one-hot)")

    # Step 9: Return
    assert "return y_sol" in source, "Step 9: Return missing"
    print("  ✓ Step 9: Return numpy array")

    print("  ✓ All 9 implementation steps verified")


def test_fix_constraint():
    """Test solving with fix_pixel_color constraint."""
    print("\nTest: Fix constraint (y[p,c] = 1)")
    print("-" * 70)

    builder = ConstraintBuilder()
    builder.fix_pixel_color(0, 2, C=5)  # Pixel 0 → color 2

    y_sol = solve_constraints_for_grid(builder, num_pixels=1, num_colors=5)

    expected = np.array([[0, 0, 1, 0, 0]])
    assert np.array_equal(y_sol, expected), \
        f"Expected {expected}, got {y_sol}"

    print(f"  Solution: {y_sol}")
    print("  ✓ Fix constraint works correctly")


def test_tie_constraint():
    """Test solving with tie_pixel_colors constraint."""
    print("\nTest: Tie constraint (y[p1,c] = y[p2,c])")
    print("-" * 70)

    builder = ConstraintBuilder()

    # Pixel 0 = color 1
    builder.fix_pixel_color(0, 1, C=3)

    # Pixel 1 = pixel 0 (tie)
    builder.tie_pixel_colors(0, 1, C=3)

    y_sol = solve_constraints_for_grid(builder, num_pixels=2, num_colors=3)

    # Both should be color 1
    expected = np.array([[0, 1, 0], [0, 1, 0]])
    assert np.array_equal(y_sol, expected), \
        f"Expected {expected}, got {y_sol}"

    print(f"  Solution:\n{y_sol}")
    print("  ✓ Tie constraint works correctly")


def test_forbid_constraint():
    """Test solving with forbid_pixel_color constraint."""
    print("\nTest: Forbid constraint (y[p,c] = 0)")
    print("-" * 70)

    builder = ConstraintBuilder()

    # 1 pixel, 3 colors
    # Forbid color 0 and color 2 → must be color 1
    builder.forbid_pixel_color(0, 0, C=3)
    builder.forbid_pixel_color(0, 2, C=3)

    y_sol = solve_constraints_for_grid(builder, num_pixels=1, num_colors=3)

    expected = np.array([[0, 1, 0]])
    assert np.array_equal(y_sol, expected), \
        f"Expected {expected}, got {y_sol}"

    print(f"  Solution: {y_sol}")
    print("  ✓ Forbid constraint works correctly")


def test_one_hot_enforcement():
    """Test that one-hot constraints are enforced."""
    print("\nTest: One-hot enforcement")
    print("-" * 70)

    # Empty builder (no constraints except one-hot)
    builder = ConstraintBuilder()

    y_sol = solve_constraints_for_grid(builder, num_pixels=3, num_colors=4)

    # Each pixel should sum to exactly 1
    row_sums = y_sol.sum(axis=1)
    assert np.all(row_sums == 1), \
        f"One-hot violated: row sums = {row_sums}"

    print(f"  Solution shape: {y_sol.shape}")
    print(f"  Row sums: {row_sums}")
    print("  ✓ One-hot constraints enforced")


def test_infeasible_detection():
    """Test that infeasible models are detected."""
    print("\nTest: Infeasible model detection")
    print("-" * 70)

    builder = ConstraintBuilder()

    # Contradictory: pixel 0 = color 0 AND color 1
    builder.fix_pixel_color(0, 0, C=2)
    builder.fix_pixel_color(0, 1, C=2)

    try:
        y_sol = solve_constraints_for_grid(builder, num_pixels=1, num_colors=2)
        raise AssertionError("Should have raised InfeasibleModelError")
    except InfeasibleModelError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("  ✓ Infeasible models detected correctly")


def test_objective_modes():
    """Test both objective modes (min_sum and none)."""
    print("\nTest: Objective modes")
    print("-" * 70)

    builder = ConstraintBuilder()
    builder.fix_pixel_color(0, 1, C=3)

    # Test min_sum
    y_sol1 = solve_constraints_for_grid(
        builder, num_pixels=1, num_colors=3, objective="min_sum"
    )
    expected = np.array([[0, 1, 0]])
    assert np.array_equal(y_sol1, expected)
    print("  ✓ objective='min_sum' works")

    # Test none
    y_sol2 = solve_constraints_for_grid(
        builder, num_pixels=1, num_colors=3, objective="none"
    )
    assert np.array_equal(y_sol2, expected)
    print("  ✓ objective='none' works")

    print("  ✓ Both objective modes work correctly")


def test_invalid_objective():
    """Test that invalid objective raises error."""
    print("\nTest: Invalid objective")
    print("-" * 70)

    builder = ConstraintBuilder()
    builder.fix_pixel_color(0, 0, C=2)

    try:
        y_sol = solve_constraints_for_grid(
            builder, num_pixels=1, num_colors=2, objective="invalid"
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("  ✓ Invalid objective raises ValueError")


def test_multiple_pixels():
    """Test solving with multiple pixels and constraints."""
    print("\nTest: Multiple pixels with mixed constraints")
    print("-" * 70)

    builder = ConstraintBuilder()

    # 4 pixels, 3 colors
    # p0 = color 0
    builder.fix_pixel_color(0, 0, C=3)
    # p1 = color 1
    builder.fix_pixel_color(1, 1, C=3)
    # p2 = p0 (tie)
    builder.tie_pixel_colors(2, 0, C=3)
    # p3 forbid color 0, forbid color 1 → must be color 2
    builder.forbid_pixel_color(3, 0, C=3)
    builder.forbid_pixel_color(3, 1, C=3)

    y_sol = solve_constraints_for_grid(builder, num_pixels=4, num_colors=3)

    # Expected: p0=[1,0,0], p1=[0,1,0], p2=[1,0,0], p3=[0,0,1]
    expected = np.array([
        [1, 0, 0],  # p0
        [0, 1, 0],  # p1
        [1, 0, 0],  # p2 (same as p0)
        [0, 0, 1],  # p3
    ])
    assert np.array_equal(y_sol, expected), \
        f"Expected:\n{expected}\nGot:\n{y_sol}"

    print(f"  Solution:\n{y_sol}")
    print("  ✓ Multiple pixels with mixed constraints work correctly")


def test_solution_is_binary():
    """Test that solution contains only 0s and 1s."""
    print("\nTest: Solution is binary")
    print("-" * 70)

    builder = ConstraintBuilder()
    # Mix of constraints
    builder.fix_pixel_color(0, 1, C=5)
    builder.fix_pixel_color(1, 3, C=5)
    builder.tie_pixel_colors(2, 0, C=5)

    y_sol = solve_constraints_for_grid(builder, num_pixels=3, num_colors=5)

    # Check all values are 0 or 1
    unique_values = np.unique(y_sol)
    assert set(unique_values).issubset({0, 1}), \
        f"Solution contains non-binary values: {unique_values}"

    print(f"  Unique values in solution: {unique_values}")
    print("  ✓ Solution is binary (only 0s and 1s)")


def test_large_grid():
    """Test solving with larger grid."""
    print("\nTest: Large grid (100 pixels, 10 colors)")
    print("-" * 70)

    builder = ConstraintBuilder()

    # Fix every pixel to a specific color
    for p in range(100):
        color = p % 10  # Cycle through colors
        builder.fix_pixel_color(p, color, C=10)

    y_sol = solve_constraints_for_grid(builder, num_pixels=100, num_colors=10)

    # Verify shape
    assert y_sol.shape == (100, 10)

    # Verify each pixel has correct color
    for p in range(100):
        expected_color = p % 10
        assert y_sol[p, expected_color] == 1, \
            f"Pixel {p} should be color {expected_color}"

    print(f"  Solution shape: {y_sol.shape}")
    print(f"  All 100 pixels correctly assigned")
    print("  ✓ Large grid solves correctly")


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    lp_solver_file = project_root / "src/solver/lp_solver.py"
    source = lp_solver_file.read_text()

    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError"]

    for marker in markers:
        assert marker not in source, \
            f"Found '{marker}' in lp_solver.py"

    print("  ✓ No TODOs, stubs, or markers found")


def test_uses_pulp_only():
    """Test that implementation uses pulp (no custom solver)."""
    print("\nTest: Uses PuLP library (no custom solver)")
    print("-" * 70)

    lp_solver_file = project_root / "src/solver/lp_solver.py"
    source = lp_solver_file.read_text()

    # Check uses pulp
    assert "import pulp" in source, "Should import pulp"
    assert "pulp.LpProblem" in source, "Should use pulp.LpProblem"
    assert "pulp.LpVariable" in source, "Should use pulp.LpVariable"
    assert "PULP_CBC_CMD" in source, "Should use CBC solver"

    # Check NO custom solver keywords
    forbidden = ["simplex", "dual", "primal", "tableau", "basis"]
    for word in forbidden:
        assert word.lower() not in source.lower(), \
            f"Found custom solver keyword: {word}"

    print("  ✓ Uses PuLP library correctly")
    print("  ✓ No custom solver implementation")


def test_error_handling():
    """Test error handling is comprehensive."""
    print("\nTest: Error handling")
    print("-" * 70)

    # Test 1: Infeasible raises InfeasibleModelError
    builder1 = ConstraintBuilder()
    builder1.fix_pixel_color(0, 0, C=2)
    builder1.fix_pixel_color(0, 1, C=2)

    try:
        solve_constraints_for_grid(builder1, 1, 2)
        raise AssertionError("Should raise InfeasibleModelError")
    except InfeasibleModelError:
        print("  ✓ InfeasibleModelError raised correctly")

    # Test 2: Invalid objective raises ValueError
    builder2 = ConstraintBuilder()
    try:
        solve_constraints_for_grid(builder2, 1, 2, objective="bad")
        raise AssertionError("Should raise ValueError")
    except ValueError:
        print("  ✓ ValueError for invalid objective")

    print("  ✓ Error handling is comprehensive")


def main():
    print("=" * 70)
    print("WO-M4.1 COMPREHENSIVE REVIEW TEST")
    print("Testing LP/ILP solver wrapper")
    print("=" * 70)

    try:
        # Core implementation
        test_implementation_steps()
        test_uses_pulp_only()
        test_no_todos_stubs()

        # Constraint types
        test_fix_constraint()
        test_tie_constraint()
        test_forbid_constraint()

        # Solver behavior
        test_one_hot_enforcement()
        test_infeasible_detection()
        test_objective_modes()
        test_invalid_objective()
        test_error_handling()

        # Solution quality
        test_solution_is_binary()
        test_multiple_pixels()
        test_large_grid()

        print("\n" + "=" * 70)
        print("✅ WO-M4.1 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ All 9 WO implementation steps - VERIFIED")
        print("  ✓ Uses PuLP library (no custom solver) - VERIFIED")
        print("  ✓ No TODOs or stubs - VERIFIED")
        print()
        print("  ✓ Constraint types - ALL WORKING")
        print("    - Fix constraints (y[p,c] = 1)")
        print("    - Tie constraints (y[p1,c] = y[p2,c])")
        print("    - Forbid constraints (y[p,c] = 0)")
        print()
        print("  ✓ Solver behavior - CORRECT")
        print("    - One-hot enforcement")
        print("    - Infeasible detection")
        print("    - Objective modes (min_sum, none)")
        print("    - Error handling")
        print()
        print("  ✓ Solution quality - EXCELLENT")
        print("    - Binary solutions (only 0s and 1s)")
        print("    - Multiple pixels with mixed constraints")
        print("    - Large grids (100 pixels, 10 colors)")
        print()
        print("WO-M4.1 IMPLEMENTATION READY FOR PRODUCTION")
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
