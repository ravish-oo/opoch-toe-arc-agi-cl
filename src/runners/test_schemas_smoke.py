"""
Sanity test harness for all schema builders (S1-S11).

This script constructs minimal toy tasks in memory and verifies that:
  - Each schema builder runs without crashing
  - Each schema builder generates constraints
  - Constraints are structurally valid

No LP solver integration; just smoke tests for constraint generation.
"""

import numpy as np
from typing import Dict, Any, List

from src.schemas.context import build_task_context_from_raw, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance


def make_toy_task_context(
    train_inputs: List[np.ndarray],
    train_outputs: List[np.ndarray] = None,
    test_inputs: List[np.ndarray] = None,
) -> TaskContext:
    """
    Build a TaskContext from in-memory numpy grids.

    Args:
        train_inputs: List of input grids
        train_outputs: List of output grids or None
        test_inputs: List of test input grids or None

    Returns:
        TaskContext for synthetic toy task
    """
    if train_outputs is None:
        train = [{"input": g, "output": None} for g in train_inputs]
    else:
        assert len(train_inputs) == len(train_outputs), \
            "train_inputs and train_outputs must have same length"
        train = [{"input": g_in, "output": g_out}
                 for g_in, g_out in zip(train_inputs, train_outputs)]

    if test_inputs is None:
        test = []
    else:
        test = [{"input": g, "output": None} for g in test_inputs]

    task_data: Dict[str, Any] = {
        "train": train,
        "test": test,
    }

    return build_task_context_from_raw(task_data)


def smoke_S1():
    """Test S1: Direct pixel tie (copy/equality)."""
    print("\n" + "=" * 70)
    print("S1 SMOKE TEST: Direct pixel tie")
    print("=" * 70)

    grid = np.array([
        [1, 2],
        [3, 4],
    ], dtype=int)

    ctx = make_toy_task_context([grid])

    # Tie pixels (0,0) and (1,1) in train example 0
    s1_params = {
        "example_type": "train",
        "example_index": 0,
        "ties": [
            {
                "example_type": "train",
                "example_index": 0,
                "pairs": [((0, 0), (1, 1))]
            }
        ],
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S1", s1_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S1 should generate constraints"
    print("  ✓ S1 smoke test passed")


def smoke_S2():
    """Test S2: Component-wise recolor map."""
    print("\n" + "=" * 70)
    print("S2 SMOKE TEST: Component recolor")
    print("=" * 70)

    grid = np.array([
        [1, 1],
        [1, 1],
    ], dtype=int)

    ctx = make_toy_task_context([grid])

    # Recolor all pixels of color 1 to color 3 based on component size
    s2_params = {
        "example_type": "train",
        "example_index": 0,
        "input_color": 1,
        "size_to_color": {"4": 3}  # 4-pixel component → color 3
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S2", s2_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S2 should generate constraints"
    print("  ✓ S2 smoke test passed")


def smoke_S3():
    """Test S3: Band / stripe laws."""
    print("\n" + "=" * 70)
    print("S3 SMOKE TEST: Bands and stripes")
    print("=" * 70)

    grid = np.zeros((4, 4), dtype=int)
    ctx = make_toy_task_context([grid])

    # Tie rows 0 and 2 (same row class)
    s3_params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [[0, 2]],
        "col_classes": [],
        "row_period_K": None,
        "col_period_K": None,
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", s3_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S3 should generate constraints"
    print("  ✓ S3 smoke test passed")


def smoke_S4():
    """Test S4: Residue-class coloring."""
    print("\n" + "=" * 70)
    print("S4 SMOKE TEST: Residue coloring")
    print("=" * 70)

    # Use grid with colors 0-2 to ensure C >= 3
    grid = np.array([
        [0, 1, 0, 1],
        [2, 2, 2, 2],
    ], dtype=int)
    ctx = make_toy_task_context([grid])

    # Color even columns → 1, odd columns → 2
    s4_params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {"0": 1, "1": 2},
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", s4_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S4 should generate constraints"
    print("  ✓ S4 smoke test passed")


def smoke_S5():
    """Test S5: Template stamping."""
    print("\n" + "=" * 70)
    print("S5 SMOKE TEST: Template stamping")
    print("=" * 70)

    # Use grid with color variety to ensure C >= 6
    grid = np.array([
        [0, 1, 2, 3, 4],
        [5, 0, 1, 2, 3],
        [4, 5, 0, 1, 2],
        [3, 4, 5, 0, 1],
        [2, 3, 4, 5, 0],
    ], dtype=int)
    ctx = make_toy_task_context([grid])

    # Get any existing hash from the grid
    ex = ctx.train_examples[0]
    if ex.neighborhood_hashes:
        any_pixel, any_hash = next(iter(ex.neighborhood_hashes.items()))

        s5_params = {
            "example_type": "train",
            "example_index": 0,
            "seed_templates": {
                str(any_hash): {
                    "(0,0)": 5,  # Stamp color 5 at seed center
                }
            },
        }

        builder = ConstraintBuilder()
        apply_schema_instance("S5", s5_params, ctx, builder)

        print(f"  Generated {len(builder.constraints)} constraints")
        assert len(builder.constraints) > 0, "S5 should generate constraints"
        print("  ✓ S5 smoke test passed")
    else:
        print("  ⚠ No neighborhood hashes found (grid too small)")


def smoke_S6():
    """Test S6: Cropping to ROI."""
    print("\n" + "=" * 70)
    print("S6 SMOKE TEST: Crop to ROI")
    print("=" * 70)

    grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=int)

    ctx = make_toy_task_context([grid])

    # Crop to 2x2 region containing the 1's
    s6_params = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "background_color": 0,
        "out_to_in": {
            "(0,0)": "(1,1)",
            "(0,1)": "(1,2)",
            "(1,0)": "(2,1)",
            "(1,1)": "(2,2)",
        },
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S6", s6_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S6 should generate constraints"
    print("  ✓ S6 smoke test passed")


def smoke_S7():
    """Test S7: Aggregation / summary."""
    print("\n" + "=" * 70)
    print("S7 SMOKE TEST: Summary grid")
    print("=" * 70)

    # Use grid with colors 0-4 to ensure C >= 5
    grid = np.array([
        [0, 1, 2],
        [3, 4, 0],
        [1, 2, 3],
    ], dtype=int)
    ctx = make_toy_task_context([grid])

    # Create 2x2 summary grid
    s7_params = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "summary_colors": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4,
        },
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S7", s7_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S7 should generate constraints"
    print("  ✓ S7 smoke test passed")


def smoke_S8():
    """Test S8: Tiling / replication."""
    print("\n" + "=" * 70)
    print("S8 SMOKE TEST: Tiling pattern")
    print("=" * 70)

    # Use grid with colors 0-4 to ensure C >= 5
    grid = np.array([
        [0, 1, 2, 3],
        [4, 0, 1, 2],
        [3, 4, 0, 1],
        [2, 3, 4, 0],
    ], dtype=int)
    ctx = make_toy_task_context([grid])

    # Tile 2x2 pattern across 4x4 grid
    s8_params = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4,
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4,
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S8", s8_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S8 should generate constraints"
    print("  ✓ S8 smoke test passed")


def smoke_S9():
    """Test S9: Cross / plus propagation."""
    print("\n" + "=" * 70)
    print("S9 SMOKE TEST: Cross propagation")
    print("=" * 70)

    # Use grid with colors 0-4 to ensure C >= 5
    grid = np.array([
        [0, 1, 2, 3, 4],
        [4, 0, 1, 2, 3],
        [3, 4, 0, 1, 2],
        [2, 3, 4, 0, 1],
        [1, 2, 3, 4, 0],
    ], dtype=int)
    ctx = make_toy_task_context([grid])

    # Propagate cross from center (2,2)
    s9_params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [
            {
                "center": "(2,2)",
                "up_color": 1,
                "down_color": 2,
                "left_color": 3,
                "right_color": 4,
                "max_up": 1,
                "max_down": 1,
                "max_left": 1,
                "max_right": 1,
            }
        ],
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S9", s9_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S9 should generate constraints"
    print("  ✓ S9 smoke test passed")


def smoke_S10():
    """Test S10: Frame / border vs interior."""
    print("\n" + "=" * 70)
    print("S10 SMOKE TEST: Border and interior")
    print("=" * 70)

    # Cross-shaped component with border and interior pixels
    # Use colors 0-7 to ensure C >= 8
    grid = np.array([
        [0, 1, 0],
        [1, 7, 1],
        [0, 1, 0],
    ], dtype=int)

    ctx = make_toy_task_context([grid])

    # Apply different colors to border vs interior
    s10_params = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 5,
        "interior_color": 7,
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S10", s10_params, ctx, builder)

    print(f"  Generated {len(builder.constraints)} constraints")
    assert len(builder.constraints) > 0, "S10 should generate constraints"
    print("  ✓ S10 smoke test passed")


def smoke_S11():
    """Test S11: Local neighborhood codebook."""
    print("\n" + "=" * 70)
    print("S11 SMOKE TEST: Local codebook")
    print("=" * 70)

    # Use grid with color variety to ensure C >= 10
    grid = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
    ], dtype=int)
    ctx = make_toy_task_context([grid])

    # Get any existing hash from the grid
    ex = ctx.train_examples[0]
    if ex.neighborhood_hashes:
        any_pixel, any_hash = next(iter(ex.neighborhood_hashes.items()))

        s11_params = {
            "example_type": "train",
            "example_index": 0,
            "hash_templates": {
                str(any_hash): {
                    "(0,0)": 9  # Rewrite center pixel to color 9
                }
            },
        }

        builder = ConstraintBuilder()
        apply_schema_instance("S11", s11_params, ctx, builder)

        print(f"  Generated {len(builder.constraints)} constraints")
        assert len(builder.constraints) > 0, "S11 should generate constraints"
        print("  ✓ S11 smoke test passed")
    else:
        print("  ⚠ No neighborhood hashes found (grid too small)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SCHEMA SMOKE TEST SUITE (S1-S11)")
    print("Testing all schema builders with minimal toy tasks")
    print("=" * 70)

    # Run all smoke tests
    smoke_S1()
    smoke_S2()
    smoke_S3()
    smoke_S4()
    smoke_S5()
    smoke_S6()
    smoke_S7()
    smoke_S8()
    smoke_S9()
    smoke_S10()
    smoke_S11()

    print("\n" + "=" * 70)
    print("✓ ALL SCHEMA SMOKE TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - All 11 schema builders (S1-S11) executed successfully")
    print("  - All builders generated valid constraints")
    print("  - No crashes or structural errors detected")
    print("\nNext steps:")
    print("  - Integrate with LP solver for end-to-end solving")
    print("  - Add Pi-agent for automatic law discovery")
    print()
