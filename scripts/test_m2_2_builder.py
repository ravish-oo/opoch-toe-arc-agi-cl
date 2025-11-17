#!/usr/bin/env python3
"""
Comprehensive test for WO-M2.2 builder.py

Tests:
1. LinearConstraint and ConstraintBuilder correctness
2. Helper methods (tie, fix, forbid)
3. One-hot constraints
4. Manual verification of indices for tiny grids
5. Integration with indexing from M2.1
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.constraints.builder import (
    LinearConstraint,
    ConstraintBuilder,
    add_one_hot_constraints
)
from src.constraints.indexing import y_index


def test_linear_constraint_structure():
    """Test LinearConstraint dataclass"""
    print("Testing LinearConstraint structure...")

    lc = LinearConstraint(
        indices=[5, 10, 15],
        coeffs=[1.0, -2.0, 3.5],
        rhs=7.0
    )

    assert lc.indices == [5, 10, 15]
    assert lc.coeffs == [1.0, -2.0, 3.5]
    assert lc.rhs == 7.0

    print("  ✓ LinearConstraint structure correct")


def test_constraint_builder_add_eq():
    """Test ConstraintBuilder.add_eq method"""
    print("Testing ConstraintBuilder.add_eq...")

    builder = ConstraintBuilder()
    assert len(builder.constraints) == 0, "Should start empty"

    # Add a constraint
    builder.add_eq(indices=[1, 2], coeffs=[1.0, -1.0], rhs=0.0)
    assert len(builder.constraints) == 1

    lc = builder.constraints[0]
    assert lc.indices == [1, 2]
    assert lc.coeffs == [1.0, -1.0]
    assert lc.rhs == 0.0

    # Add another
    builder.add_eq(indices=[5], coeffs=[1.0], rhs=1.0)
    assert len(builder.constraints) == 2

    print("  ✓ add_eq works correctly")


def test_add_eq_assertion():
    """Test that add_eq asserts length mismatch"""
    print("Testing add_eq assertion on length mismatch...")

    builder = ConstraintBuilder()

    try:
        # This should raise AssertionError
        builder.add_eq(indices=[1, 2, 3], coeffs=[1.0, -1.0], rhs=0.0)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "same length" in str(e).lower()

    print("  ✓ add_eq correctly asserts length match")


def test_tie_pixel_colors_basic():
    """Test tie_pixel_colors creates C constraints"""
    print("Testing tie_pixel_colors basic behavior...")

    N, C = 4, 5
    builder = ConstraintBuilder()

    # Tie pixels 0 and 2
    builder.tie_pixel_colors(p_idx=0, q_idx=2, C=C)

    # Should create exactly C constraints
    assert len(builder.constraints) == C, \
        f"Expected {C} constraints, got {len(builder.constraints)}"

    # Check each constraint
    for c in range(C):
        lc = builder.constraints[c]

        # Should have 2 indices: y[p,c] and y[q,c]
        assert len(lc.indices) == 2
        assert len(lc.coeffs) == 2

        # Coefficients should be [1.0, -1.0]
        assert lc.coeffs == [1.0, -1.0]

        # RHS should be 0
        assert lc.rhs == 0.0

        # Verify indices are correct
        expected_p = y_index(0, c, C)
        expected_q = y_index(2, c, C)
        assert lc.indices == [expected_p, expected_q], \
            f"Color {c}: expected [{expected_p}, {expected_q}], got {lc.indices}"

    print(f"  ✓ Created {C} tie constraints with correct structure")


def test_tie_pixel_colors_manual_verification():
    """Manual verification of tie_pixel_colors indices"""
    print("Testing tie_pixel_colors with manual index verification...")

    # Tiny grid: H=1, W=2, C=3
    # N = 2 pixels: p0 and p1
    # y-vector has 6 elements:
    #   y[0] = y[p0,c0], y[1] = y[p0,c1], y[2] = y[p0,c2]
    #   y[3] = y[p1,c0], y[4] = y[p1,c1], y[5] = y[p1,c2]

    H, W, C = 1, 2, 3
    N = H * W

    builder = ConstraintBuilder()
    builder.tie_pixel_colors(p_idx=0, q_idx=1, C=C)

    # Should create 3 constraints:
    # c=0: y[0] - y[3] = 0  (both have color 0)
    # c=1: y[1] - y[4] = 0  (both have color 1)
    # c=2: y[2] - y[5] = 0  (both have color 2)

    expected_constraints = [
        ([0, 3], [1.0, -1.0], 0.0),  # c=0
        ([1, 4], [1.0, -1.0], 0.0),  # c=1
        ([2, 5], [1.0, -1.0], 0.0),  # c=2
    ]

    assert len(builder.constraints) == 3

    for i, (exp_indices, exp_coeffs, exp_rhs) in enumerate(expected_constraints):
        lc = builder.constraints[i]
        assert lc.indices == exp_indices, \
            f"Constraint {i}: expected indices {exp_indices}, got {lc.indices}"
        assert lc.coeffs == exp_coeffs
        assert lc.rhs == exp_rhs

    print("  ✓ Manual verification passed for H=1, W=2, C=3")


def test_fix_pixel_color_basic():
    """Test fix_pixel_color creates exactly 1 constraint"""
    print("Testing fix_pixel_color basic behavior...")

    N, C = 4, 10
    builder = ConstraintBuilder()

    # Fix pixel 2 to color 7
    builder.fix_pixel_color(p_idx=2, color=7, C=C)

    # Should create exactly 1 constraint
    assert len(builder.constraints) == 1, \
        f"Expected 1 constraint, got {len(builder.constraints)}"

    lc = builder.constraints[0]

    # Should have 1 index and 1 coefficient
    assert len(lc.indices) == 1
    assert len(lc.coeffs) == 1

    # Coefficient should be 1.0, rhs should be 1.0
    assert lc.coeffs[0] == 1.0
    assert lc.rhs == 1.0

    # Index should be y_index(2, 7, 10)
    expected_idx = y_index(2, 7, C)
    assert lc.indices[0] == expected_idx, \
        f"Expected index {expected_idx}, got {lc.indices[0]}"

    print(f"  ✓ Created 1 constraint: y[{expected_idx}] = 1")


def test_fix_pixel_color_does_not_overconstrain():
    """
    CRITICAL TEST: Verify fix_pixel_color does NOT set other colors to 0.

    This is the highest priority check - over-constraining would break solver.
    """
    print("Testing fix_pixel_color does NOT over-constrain...")

    N, C = 5, 10
    builder = ConstraintBuilder()

    # Fix pixel 3 to color 5
    builder.fix_pixel_color(p_idx=3, color=5, C=C)

    # CRITICAL: Should create exactly 1 constraint (not C constraints)
    assert len(builder.constraints) == 1, \
        f"OVER-CONSTRAINED: Expected 1 constraint, got {len(builder.constraints)}"

    lc = builder.constraints[0]

    # Should only set y[p,5] = 1
    expected_idx = y_index(3, 5, C)
    assert lc.indices == [expected_idx], \
        "Should only constrain the specified color"
    assert lc.coeffs == [1.0]
    assert lc.rhs == 1.0

    # Verify it did NOT add constraints for other colors
    # (i.e., no y[p,c!=5] = 0 constraints)
    for c in range(C):
        if c != 5:
            # No constraint should exist for other colors
            idx = y_index(3, c, C)
            # Check this index doesn't appear in any constraint
            # (we only have 1 constraint, and it's for color 5)
            pass

    print("  ✓ CRITICAL: fix_pixel_color correctly avoids over-constraining")


def test_forbid_pixel_color_basic():
    """Test forbid_pixel_color creates exactly 1 constraint"""
    print("Testing forbid_pixel_color basic behavior...")

    N, C = 4, 10
    builder = ConstraintBuilder()

    # Forbid pixel 1 from being color 3
    builder.forbid_pixel_color(p_idx=1, color=3, C=C)

    # Should create exactly 1 constraint
    assert len(builder.constraints) == 1

    lc = builder.constraints[0]

    # Should have 1 index and 1 coefficient
    assert len(lc.indices) == 1
    assert len(lc.coeffs) == 1

    # Coefficient should be 1.0, rhs should be 0.0
    assert lc.coeffs[0] == 1.0
    assert lc.rhs == 0.0

    # Index should be y_index(1, 3, 10)
    expected_idx = y_index(1, 3, C)
    assert lc.indices[0] == expected_idx

    print(f"  ✓ Created 1 constraint: y[{expected_idx}] = 0")


def test_one_hot_constraints_basic():
    """Test add_one_hot_constraints creates N constraints"""
    print("Testing add_one_hot_constraints basic behavior...")

    N, C = 6, 10
    builder = ConstraintBuilder()

    add_one_hot_constraints(builder, N, C)

    # Should create exactly N constraints (one per pixel)
    assert len(builder.constraints) == N, \
        f"Expected {N} constraints, got {len(builder.constraints)}"

    # Check structure of each constraint
    for p_idx in range(N):
        lc = builder.constraints[p_idx]

        # Should have C indices (one per color)
        assert len(lc.indices) == C, \
            f"Pixel {p_idx}: expected {C} indices, got {len(lc.indices)}"
        assert len(lc.coeffs) == C

        # All coefficients should be 1.0
        assert lc.coeffs == [1.0] * C

        # RHS should be 1.0
        assert lc.rhs == 1.0

        # Indices should be y_index(p_idx, c, C) for c in 0..C-1
        expected_indices = [y_index(p_idx, c, C) for c in range(C)]
        assert lc.indices == expected_indices, \
            f"Pixel {p_idx}: indices mismatch"

    print(f"  ✓ Created {N} one-hot constraints with correct structure")


def test_one_hot_constraints_manual_verification():
    """Manual verification of one-hot constraints for tiny grid"""
    print("Testing one-hot constraints with manual verification...")

    # Tiny grid: H=1, W=2, C=3
    # N = 2 pixels
    # y-vector: [y[p0,c0], y[p0,c1], y[p0,c2], y[p1,c0], y[p1,c1], y[p1,c2]]
    #           [y[0],     y[1],     y[2],     y[3],     y[4],     y[5]]

    H, W, C = 1, 2, 3
    N = H * W

    builder = ConstraintBuilder()
    add_one_hot_constraints(builder, N, C)

    # Should create 2 constraints:
    # p0: y[0] + y[1] + y[2] = 1
    # p1: y[3] + y[4] + y[5] = 1

    expected_constraints = [
        ([0, 1, 2], [1.0, 1.0, 1.0], 1.0),  # pixel 0
        ([3, 4, 5], [1.0, 1.0, 1.0], 1.0),  # pixel 1
    ]

    assert len(builder.constraints) == 2

    for i, (exp_indices, exp_coeffs, exp_rhs) in enumerate(expected_constraints):
        lc = builder.constraints[i]
        assert lc.indices == exp_indices, \
            f"Pixel {i}: expected indices {exp_indices}, got {lc.indices}"
        assert lc.coeffs == exp_coeffs
        assert lc.rhs == exp_rhs

    print("  ✓ Manual verification passed for H=1, W=2, C=3")


def test_combined_constraints():
    """Test combining multiple constraint types"""
    print("Testing combined constraints...")

    N, C = 3, 4
    builder = ConstraintBuilder()

    # Add one-hot constraints
    add_one_hot_constraints(builder, N, C)
    assert len(builder.constraints) == N

    # Tie two pixels
    builder.tie_pixel_colors(0, 1, C)
    assert len(builder.constraints) == N + C

    # Fix a pixel
    builder.fix_pixel_color(2, 3, C)
    assert len(builder.constraints) == N + C + 1

    # Forbid a color
    builder.forbid_pixel_color(0, 2, C)
    assert len(builder.constraints) == N + C + 1 + 1

    total_expected = N + C + 1 + 1
    assert len(builder.constraints) == total_expected

    print(f"  ✓ Combined {total_expected} constraints from different sources")


def test_different_grid_sizes():
    """Test with various grid sizes"""
    print("Testing with different grid sizes...")

    test_cases = [
        (1, 1, 3),   # Minimal 1x1, C=3
        (1, 10, 5),  # 1x10 strip
        (10, 1, 5),  # 10x1 column
        (5, 5, 10),  # Square
        (30, 30, 10),  # Max ARC size
    ]

    for (H, W, C) in test_cases:
        N = H * W
        builder = ConstraintBuilder()

        # Test one-hot
        add_one_hot_constraints(builder, N, C)
        assert len(builder.constraints) == N

        # Test tie
        builder2 = ConstraintBuilder()
        if N >= 2:
            builder2.tie_pixel_colors(0, N-1, C)
            assert len(builder2.constraints) == C

    print(f"  ✓ Tested {len(test_cases)} different grid sizes")


def test_y_vector_index_correctness():
    """Verify y-vector indices are in valid range"""
    print("Testing y-vector index bounds...")

    N, C = 10, 10  # 100-element y-vector
    builder = ConstraintBuilder()

    # Add various constraints
    add_one_hot_constraints(builder, N, C)
    builder.tie_pixel_colors(0, 5, C)
    builder.fix_pixel_color(3, 7, C)
    builder.forbid_pixel_color(8, 2, C)

    # Check all indices are in valid range [0, N*C-1]
    max_idx = N * C - 1

    for lc in builder.constraints:
        for idx in lc.indices:
            assert 0 <= idx <= max_idx, \
                f"Index {idx} out of range [0, {max_idx}]"

    print(f"  ✓ All indices in valid range [0, {max_idx}]")


def main():
    print("=" * 60)
    print("WO-M2.2 Comprehensive Test - builder.py")
    print("=" * 60)
    print()

    try:
        # Basic structure tests
        test_linear_constraint_structure()
        test_constraint_builder_add_eq()
        test_add_eq_assertion()

        # tie_pixel_colors tests
        test_tie_pixel_colors_basic()
        test_tie_pixel_colors_manual_verification()

        # fix_pixel_color tests
        test_fix_pixel_color_basic()
        test_fix_pixel_color_does_not_overconstrain()  # CRITICAL

        # forbid_pixel_color tests
        test_forbid_pixel_color_basic()

        # One-hot tests
        test_one_hot_constraints_basic()
        test_one_hot_constraints_manual_verification()

        # Integration tests
        test_combined_constraints()
        test_different_grid_sizes()
        test_y_vector_index_correctness()

        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
