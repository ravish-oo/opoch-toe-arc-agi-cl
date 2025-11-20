"""
S7 schema builder: Aggregation / summary grids.

This implements the S7 schema from the math kernel spec (section 2):
    "Compress large region into small matrix: for each block/region,
     output a summary color (e.g., dominant color, unique non-zero color)."

S7 is NOT geometry-preserving: output shape is typically much smaller than input.
This builder applies pre-computed summary colors (from Pi-agent) as constraints.
"""

from typing import Dict, Any, Tuple
from ast import literal_eval

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S7_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S7 constraints: fix output pixels to precomputed summary colors.

    S7 creates a smaller summary grid where each output cell represents
    an aggregated region of the input (e.g., block, band, or semantic region).

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "output_height": int,
          "output_width": int,
          "summary_colors": {
            "(0,0)": 3,   # output (0,0) has summary color 3
            "(0,1)": 0,   # output (0,1) has summary color 0
            "(1,0)": 2,   # output (1,0) has summary color 2
            "(1,1)": 5    # output (1,1) has summary color 5
          }
        }

    Where:
        - Keys of summary_colors are stringified output coords "(r_out,c_out)"
        - Values are int colors (precomputed summaries from regions)
        - Unmapped output pixels are left unconstrained

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying summary colors and output dimensions
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # Create 2x2 summary grid with specific colors
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "output_height": 2,
        ...     "output_width": 2,
        ...     "summary_colors": {
        ...         "(0,0)": 1,
        ...         "(0,1)": 2,
        ...         "(1,0)": 3,
        ...         "(1,1)": 4
        ...     }
        ... }
        >>> build_S7_constraints(ctx, params, builder)
    """
    # 1. Select example
    example_type = schema_params.get("example_type", "train")
    example_index = schema_params.get("example_index", 0)

    if example_type == "train":
        if example_index >= len(task_context.train_examples):
            return  # Invalid index
        ex = task_context.train_examples[example_index]
    else:  # "test"
        if example_index >= len(task_context.test_examples):
            return  # Invalid index
        ex = task_context.test_examples[example_index]

    # 2. Get output dimensions and parameters
    output_H = int(schema_params.get("output_height", 0))
    output_W = int(schema_params.get("output_width", 0))
    C = task_context.C

    if output_H <= 0 or output_W <= 0:
        return  # Invalid output dimensions

    # 3. Parse summary_colors from string tuples to actual tuples
    raw_summaries = schema_params.get("summary_colors", {})
    summary_colors: Dict[Tuple[int, int], int] = {}

    for k_str, color in raw_summaries.items():
        try:
            # Parse output coords from key
            r_out, c_out = literal_eval(k_str)
            summary_colors[(r_out, c_out)] = int(color)
        except (ValueError, SyntaxError):
            # Skip malformed entries
            continue

    # 4. Apply summary colors to output pixels
    for (r_out, c_out), color in summary_colors.items():
        # Validate output coordinates are in bounds
        if not (0 <= r_out < output_H and 0 <= c_out < output_W):
            continue  # Skip out-of-bounds

        # Validate color is in palette
        if not (0 <= color < C):
            continue  # Skip invalid colors

        # Compute output pixel index (using OUTPUT dimensions)
        # IMPORTANT: S7 is NOT geometry-preserving!
        p_idx_out = r_out * output_W + c_out

        # Prefer this output pixel's color to the summary color (Tier 1: Structure, weight 100.0)
        builder.prefer_pixel_color(p_idx_out, color, weight=100.0)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S7 builder with toy example...")
    print("=" * 70)

    # Create a larger input grid (input doesn't matter for S7, only summary colors)
    input_grid = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ], dtype=int)

    # Output doesn't matter for context building
    output_grid = None

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Full 2x2 summary grid")
    print("-" * 70)
    print("Input grid (4x4) with 4 blocks of different colors")
    print("Creating 2x2 summary (one color per block)")

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "summary_colors": {
            "(0,0)": 1,  # Top-left block -> color 1
            "(0,1)": 2,  # Top-right block -> color 2
            "(1,0)": 3,  # Bottom-left block -> color 3
            "(1,1)": 4   # Bottom-right block -> color 4
        }
    }

    builder1 = ConstraintBuilder()
    build_S7_constraints(ctx, params1, builder1)

    # Should have 4 preferences (one per summary cell)
    expected1 = 4
    print(f"  Expected: {expected1} preferences (2x2 summary)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1, \
        f"Expected {expected1} preferences, got {len(builder1.preferences)}"

    # Inspect preferences
    print(f"\n  Sample preferences:")
    for i in range(min(4, len(builder1.preferences))):
        p_idx, color, weight = builder1.preferences[i]
        print(f"    Preference {i}: p_idx={p_idx}, color={color}, weight={weight}")

    print("\nTest 2: Partial summary (some cells unmapped)")
    print("-" * 70)
    print("Only define summary colors for some cells")

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 3,
        "output_width": 3,
        "summary_colors": {
            "(0,0)": 1,
            "(1,1)": 5,
            "(2,2)": 9
        }
    }

    builder2 = ConstraintBuilder()
    build_S7_constraints(ctx, params2, builder2)

    # Should have 3 preferences (only 3 cells defined)
    expected2 = 3
    print(f"  Expected: {expected2} preferences (3 cells mapped)")
    print(f"  Actual: {len(builder2.preferences)}")
    assert len(builder2.preferences) == expected2, \
        f"Expected {expected2} preferences, got {len(builder2.preferences)}"

    print("\nTest 3: Out-of-bounds summary coords (should be skipped)")
    print("-" * 70)

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "summary_colors": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(5,5)": 7,  # Out of bounds
            "(1,0)": 3,
            "(-1,-1)": 8  # Out of bounds
        }
    }

    builder3 = ConstraintBuilder()
    build_S7_constraints(ctx, params3, builder3)

    # Should have 3 preferences (2 out-of-bounds skipped)
    expected3 = 3
    print(f"  Expected: {expected3} preferences (2 out-of-bounds skipped)")
    print(f"  Actual: {len(builder3.preferences)}")
    assert len(builder3.preferences) == expected3, \
        f"Expected {expected3} preferences, got {len(builder3.preferences)}"

    print("\nTest 4: Empty summary colors")
    print("-" * 70)

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "summary_colors": {}  # No summaries
    }

    builder4 = ConstraintBuilder()
    build_S7_constraints(ctx, params4, builder4)

    expected4 = 0
    print(f"  Expected: {expected4} preferences (no summaries)")
    print(f"  Actual: {len(builder4.preferences)}")
    assert len(builder4.preferences) == expected4, \
        f"Expected {expected4} preferences, got {len(builder4.preferences)}"

    print("\nTest 5: Invalid colors (should be skipped)")
    print("-" * 70)

    params5 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "summary_colors": {
            "(0,0)": 1,
            "(0,1)": 100,  # Out of palette range (C=10)
            "(1,0)": 3,
            "(1,1)": -1    # Negative color
        }
    }

    builder5 = ConstraintBuilder()
    build_S7_constraints(ctx, params5, builder5)

    # Should have 2 preferences (2 invalid colors skipped)
    expected5 = 2
    print(f"  Expected: {expected5} preferences (2 invalid colors skipped)")
    print(f"  Actual: {len(builder5.preferences)}")
    assert len(builder5.preferences) == expected5, \
        f"Expected {expected5} preferences, got {len(builder5.preferences)}"

    print("\n" + "=" * 70)
    print("✓ S7 builder self-test passed.")
