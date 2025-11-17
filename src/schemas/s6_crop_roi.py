"""
S6 schema builder: Cropping to ROI / dominant object.

This implements the S6 schema from the math kernel spec (section 2):
    "Output is a cropped subgrid of the input corresponding to some selected
     component or band. Output pixels correspond 1-1 to pixels of selected box B."

S6 is NOT geometry-preserving: output shape can differ from input shape.
This builder applies pre-determined ROI mappings (from Pi-agent) as constraints.
"""

from typing import Dict, Any, Tuple
from ast import literal_eval

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Grid


def build_S6_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S6 constraints: fix output pixels to cropped input region.

    S6 creates a smaller output grid that is a crop of the input.
    Each output pixel either:
      - maps to a specific input pixel (reads its color), or
      - is set to background color.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "output_height": int,
          "output_width": int,
          "background_color": int,
          "out_to_in": {
            "(0,0)": "(2,3)",   # output (0,0) <- input (2,3)
            "(0,1)": "(2,4)",   # output (0,1) <- input (2,4)
            ...
          }
        }

    Where:
        - Keys of out_to_in are stringified output coords "(r_out,c_out)"
        - Values are stringified input coords "(r_in,c_in)"
        - Unmapped output pixels are set to background_color

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying ROI mapping and output dimensions
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # Crop central 2x2 square from 4x4 input
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "output_height": 2,
        ...     "output_width": 2,
        ...     "background_color": 0,
        ...     "out_to_in": {
        ...         "(0,0)": "(1,1)",
        ...         "(0,1)": "(1,2)",
        ...         "(1,0)": "(2,1)",
        ...         "(1,1)": "(2,2)"
        ...     }
        ... }
        >>> build_S6_constraints(ctx, params, builder)
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

    # 2. Get input grid and dimensions
    input_grid: Grid = ex.input_grid
    input_H = ex.input_H
    input_W = ex.input_W
    C = task_context.C

    # 3. Get output dimensions and parameters
    output_H = int(schema_params.get("output_height", 0))
    output_W = int(schema_params.get("output_width", 0))
    background = int(schema_params.get("background_color", 0))
    out_to_in_raw = schema_params.get("out_to_in", {})

    if output_H <= 0 or output_W <= 0:
        return  # Invalid output dimensions

    # Validate background color
    if not (0 <= background < C):
        background = 0  # Fallback to color 0

    # 4. Parse out_to_in mapping from string tuples to actual tuples
    out_to_in: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for k_str, v_str in out_to_in_raw.items():
        try:
            # Parse output coords from key
            r_out, c_out = literal_eval(k_str)
            # Parse input coords from value
            r_in, c_in = literal_eval(v_str)
            out_to_in[(r_out, c_out)] = (r_in, c_in)
        except (ValueError, SyntaxError):
            # Skip malformed entries
            continue

    # 5. For each output position, fix its color
    for r_out in range(output_H):
        for c_out in range(output_W):
            # Compute output pixel index (using OUTPUT dimensions)
            # IMPORTANT: S6 is NOT geometry-preserving!
            p_idx_out = r_out * output_W + c_out

            # Check if this output pixel is mapped to an input pixel
            key = (r_out, c_out)
            if key in out_to_in:
                r_in, c_in = out_to_in[key]

                # Validate input coordinates are in bounds
                if 0 <= r_in < input_H and 0 <= c_in < input_W:
                    # Read color from input grid
                    color = int(input_grid[r_in, c_in])
                else:
                    # Out of bounds: use background
                    color = background
            else:
                # Not mapped: use background
                color = background

            # Validate color is in palette
            if not (0 <= color < C):
                color = background

            # Fix this output pixel's color
            builder.fix_pixel_color(p_idx_out, color, C)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S6 builder with toy example...")
    print("=" * 70)

    # Create a 4x4 input grid with a pattern in the center
    input_grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    # Output doesn't matter for context building (we're testing crop)
    output_grid = None

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Crop central 2x2 square")
    print("-" * 70)
    print("Input grid (4x4):")
    print(input_grid)
    print("\nCropping to central 2x2 (rows 1-2, cols 1-2)")

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "background_color": 0,
        "out_to_in": {
            "(0,0)": "(1,1)",  # output (0,0) <- input (1,1) = 1
            "(0,1)": "(1,2)",  # output (0,1) <- input (1,2) = 2
            "(1,0)": "(2,1)",  # output (1,0) <- input (2,1) = 3
            "(1,1)": "(2,2)"   # output (1,1) <- input (2,2) = 4
        }
    }

    builder1 = ConstraintBuilder()
    build_S6_constraints(ctx, params1, builder1)

    # Should have 4 constraints (one per output pixel)
    expected1 = 4
    print(f"  Expected: {expected1} constraints (2x2 output)")
    print(f"  Actual: {len(builder1.constraints)}")
    assert len(builder1.constraints) == expected1, \
        f"Expected {expected1} constraints, got {len(builder1.constraints)}"

    # Inspect constraints to verify colors
    print(f"\n  Sample constraints:")
    for i in range(min(4, len(builder1.constraints))):
        c = builder1.constraints[i]
        print(f"    Constraint {i}: indices={c.indices}, coeffs={c.coeffs}, rhs={c.rhs}")

    print("\nTest 2: Crop with background pixels")
    print("-" * 70)
    print("Cropping 3x3 output, but only map some pixels")

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 3,
        "output_width": 3,
        "background_color": 9,
        "out_to_in": {
            "(1,1)": "(1,1)",  # Center pixel only
        }
    }

    builder2 = ConstraintBuilder()
    build_S6_constraints(ctx, params2, builder2)

    # Should have 9 constraints (3x3 output)
    # 8 pixels set to background (9), 1 pixel set to input color (1)
    expected2 = 9
    print(f"  Expected: {expected2} constraints (3x3 output)")
    print(f"  Actual: {len(builder2.constraints)}")
    assert len(builder2.constraints) == expected2, \
        f"Expected {expected2} constraints, got {len(builder2.constraints)}"

    print("\nTest 3: Out-of-bounds input mapping")
    print("-" * 70)
    print("Mapping to invalid input coords (should use background)")

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "background_color": 7,
        "out_to_in": {
            "(0,0)": "(10,10)",  # Out of bounds
            "(0,1)": "(1,1)",    # Valid
            "(1,0)": "(-1,-1)",  # Out of bounds
            "(1,1)": "(2,2)"     # Valid
        }
    }

    builder3 = ConstraintBuilder()
    build_S6_constraints(ctx, params3, builder3)

    expected3 = 4
    print(f"  Expected: {expected3} constraints (2x2 output)")
    print(f"  Actual: {len(builder3.constraints)}")
    assert len(builder3.constraints) == expected3, \
        f"Expected {expected3} constraints, got {len(builder3.constraints)}"

    print("\nTest 4: Empty mapping (all background)")
    print("-" * 70)

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "background_color": 5,
        "out_to_in": {}  # No mappings
    }

    builder4 = ConstraintBuilder()
    build_S6_constraints(ctx, params4, builder4)

    expected4 = 4
    print(f"  Expected: {expected4} constraints (all background)")
    print(f"  Actual: {len(builder4.constraints)}")
    assert len(builder4.constraints) == expected4, \
        f"Expected {expected4} constraints, got {len(builder4.constraints)}"

    print("\n" + "=" * 70)
    print("✓ S6 builder self-test passed.")
