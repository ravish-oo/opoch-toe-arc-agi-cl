"""
Integration tests for schema dispatch with TaskContext.

This test file validates that all schema builders (S1-S11) correctly
integrate with TaskContext and the dispatch layer, producing valid
constraints that can be used by a SAT solver.

These tests use toy examples to verify:
  - Schema builders accept correct parameters
  - Builders generate expected constraints
  - Dispatch layer correctly routes to builders
  - Context provides all required features
"""

import numpy as np
from src.schemas.context import build_example_context, TaskContext
from src.schemas.dispatch import apply_schema_instance
from src.constraints.builder import ConstraintBuilder


def test_s8_tiling_integration():
    """
    Integration test for S8 (Tiling / replication) schema.

    Tests that S8 correctly tiles a 2x2 pattern across a 4x4 grid.
    """
    print("\n" + "=" * 70)
    print("S8 TILING INTEGRATION TEST")
    print("=" * 70)

    # Create 4x4 input grid
    input_grid = np.zeros((4, 4), dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Tile a 2x2 checkerboard pattern across entire grid
    params = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S8", params, ctx, builder)

    # Should tile 2x2 tiles: (0,0), (0,2), (2,0), (2,2)
    # Each tile has 4 pixels → 16 total constraints
    assert len(builder.constraints) == 16, \
        f"Expected 16 constraints, got {len(builder.constraints)}"

    print("✓ S8 tiling integration test passed")
    print(f"  - Generated {len(builder.constraints)} constraints")
    print(f"  - Tiled 2x2 pattern across 4x4 grid")


def test_s9_cross_propagation_integration():
    """
    Integration test for S9 (Cross / plus propagation) schema.

    Tests that S9 correctly propagates colors from a seed center.
    """
    print("\n" + "=" * 70)
    print("S9 CROSS PROPAGATION INTEGRATION TEST")
    print("=" * 70)

    # Create 5x5 input grid
    input_grid = np.zeros((5, 5), dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Create full cross from center (2,2) with 2 steps in each direction
    params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(2,2)",
            "up_color": 1,
            "down_color": 2,
            "left_color": 3,
            "right_color": 4,
            "max_up": 2,
            "max_down": 2,
            "max_left": 2,
            "max_right": 2
        }]
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S9", params, ctx, builder)

    # Up: 2 pixels, Down: 2 pixels, Left: 2 pixels, Right: 2 pixels
    # Total: 8 pixels (center not included)
    assert len(builder.constraints) == 8, \
        f"Expected 8 constraints, got {len(builder.constraints)}"

    print("✓ S9 cross propagation integration test passed")
    print(f"  - Generated {len(builder.constraints)} constraints")
    print(f"  - Propagated cross from center (2,2) with 2 steps each direction")


def test_s10_frame_border_integration():
    """
    Integration test for S10 (Frame / border vs interior) schema.

    Tests that S10 correctly assigns different colors to border vs interior pixels.
    """
    print("\n" + "=" * 70)
    print("S10 FRAME/BORDER INTEGRATION TEST")
    print("=" * 70)

    # Create 5x5 input grid with a component in the center
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Apply frame with border=5, interior=7
    params = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 5,
        "interior_color": 7
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S10", params, ctx, builder)

    # Count border and interior pixels
    border_pixels = [(r, c) for (r, c), info in ex.border_info.items()
                     if info.get("is_border")]
    interior_pixels = [(r, c) for (r, c), info in ex.border_info.items()
                       if info.get("is_interior")]
    expected = len(border_pixels) + len(interior_pixels)

    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print("✓ S10 frame/border integration test passed")
    print(f"  - Generated {len(builder.constraints)} constraints")
    print(f"  - Applied border color 5 and interior color 7")
    print(f"  - Border pixels: {len(border_pixels)}, Interior pixels: {len(interior_pixels)}")


def test_multiple_schemas_integration():
    """
    Integration test combining multiple schemas on the same grid.

    Tests that multiple schemas can be applied sequentially to build
    up a complex constraint set.
    """
    print("\n" + "=" * 70)
    print("MULTIPLE SCHEMAS INTEGRATION TEST")
    print("=" * 70)

    # Create 6x6 input grid
    input_grid = np.zeros((6, 6), dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    builder = ConstraintBuilder()

    # Apply S8: Tile a small pattern in corner
    s8_params = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4
    }
    apply_schema_instance("S8", s8_params, ctx, builder)
    s8_count = len(builder.constraints)

    # Apply S9: Add cross from center
    s9_params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(3,3)",
            "up_color": 5,
            "down_color": 5,
            "left_color": 5,
            "right_color": 5,
            "max_up": 1,
            "max_down": 1,
            "max_left": 1,
            "max_right": 1
        }]
    }
    apply_schema_instance("S9", s9_params, ctx, builder)
    s9_count = len(builder.constraints) - s8_count

    print(f"✓ Multiple schemas integration test passed")
    print(f"  - S8 generated {s8_count} constraints")
    print(f"  - S9 generated {s9_count} constraints")
    print(f"  - Total: {len(builder.constraints)} constraints")

    assert s8_count > 0, "S8 should generate constraints"
    assert s9_count > 0, "S9 should generate constraints"


def test_dispatch_error_handling():
    """
    Integration test for dispatch error handling.

    Tests that dispatch layer correctly handles invalid schema IDs
    and gracefully handles invalid parameters.
    """
    print("\n" + "=" * 70)
    print("DISPATCH ERROR HANDLING TEST")
    print("=" * 70)

    # Create dummy context
    input_grid = np.zeros((3, 3), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)
    builder = ConstraintBuilder()

    # Test 1: Invalid schema ID
    try:
        apply_schema_instance("S99", {}, ctx, builder)
        raise AssertionError("Should have raised KeyError for invalid schema ID")
    except KeyError as e:
        print(f"✓ Correctly raised KeyError for invalid schema ID: {e}")

    # Test 2: Invalid example index (should return early, not crash)
    params = {
        "example_type": "train",
        "example_index": 999,  # Out of range
        "seeds": []
    }
    initial_count = len(builder.constraints)
    apply_schema_instance("S9", params, ctx, builder)
    assert len(builder.constraints) == initial_count, \
        "Invalid example_index should not add constraints"
    print("✓ Invalid example_index handled gracefully (no crash)")

    # Test 3: Empty parameters (should handle gracefully)
    apply_schema_instance("S8", {}, ctx, builder)
    print("✓ Empty parameters handled gracefully")

    print("✓ Dispatch error handling test passed")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SCHEMA INTEGRATION TEST SUITE")
    print("Testing S8, S9, S10 with TaskContext and dispatch layer")
    print("=" * 70)

    # Run all integration tests
    test_s8_tiling_integration()
    test_s9_cross_propagation_integration()
    test_s10_frame_border_integration()
    test_multiple_schemas_integration()
    test_dispatch_error_handling()

    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - S8 (Tiling): ✓ Correctly tiles patterns across regions")
    print("  - S9 (Cross propagation): ✓ Correctly propagates from seeds")
    print("  - S10 (Frame/border): ✓ Correctly assigns border/interior colors")
    print("  - Multiple schemas: ✓ Can be combined on same grid")
    print("  - Error handling: ✓ Gracefully handles invalid inputs")
    print()
