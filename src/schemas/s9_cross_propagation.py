"""
S9 schema builder: Cross / plus propagation.

This implements the S9 schema from the math kernel spec (section 2):
    "Given cross-shaped patterns, propagate them along rows/cols at certain anchors.
     For each seed, paint spokes (up/down/left/right) with specified colors."

S9 is geometry-preserving: output has same shape as input.
This builder applies pre-determined cross seeds and propagation rules (from Pi-agent) as constraints.
"""

from typing import Dict, Any, List
from ast import literal_eval

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S9_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S9 constraints: propagate colors in cardinal directions from seeds.

    S9 creates cross/plus patterns by propagating colors along four
    cardinal directions (up, down, left, right) from specified seed centers.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "seeds": [
            {
              "center": "(r,c)",      # seed center position
              "up_color": int | null,     # color to propagate upward (null = skip)
              "down_color": int | null,   # color to propagate downward
              "left_color": int | null,   # color to propagate leftward
              "right_color": int | null,  # color to propagate rightward
              "max_up": int,          # max steps to propagate up
              "max_down": int,        # max steps to propagate down
              "max_left": int,        # max steps to propagate left
              "max_right": int        # max steps to propagate right
            },
            ...
          ]
        }

    Where:
        - Each seed defines a center and propagation in 4 directions
        - null color means "don't propagate in this direction"
        - Propagation stops at max_steps or grid boundary
        - Center pixel itself is NOT colored by this schema

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying seeds and propagation rules
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # Cross pattern from center (2,2) with different colors in each direction
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "seeds": [{
        ...         "center": "(2,2)",
        ...         "up_color": 1,
        ...         "down_color": 2,
        ...         "left_color": 3,
        ...         "right_color": 4,
        ...         "max_up": 2,
        ...         "max_down": 2,
        ...         "max_left": 2,
        ...         "max_right": 2
        ...     }]
        ... }
        >>> build_S9_constraints(ctx, params, builder)
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

    # 2. Get grid dimensions
    # S9 draws cross arms into OUTPUT grid
    # Use output dimensions for indexing into y variables
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid to constrain
    C = task_context.C

    # 3. Process each seed
    seeds = schema_params.get("seeds", [])
    for seed in seeds:
        # Parse seed center
        center_str = seed.get("center", "(0,0)")
        try:
            r_center, c_center = literal_eval(center_str)
        except (ValueError, SyntaxError):
            continue  # Skip malformed seed

        # Get directional colors (None means skip that direction)
        up_color = seed.get("up_color")
        down_color = seed.get("down_color")
        left_color = seed.get("left_color")
        right_color = seed.get("right_color")

        # Get max propagation distances
        max_up = int(seed.get("max_up", 0))
        max_down = int(seed.get("max_down", 0))
        max_left = int(seed.get("max_left", 0))
        max_right = int(seed.get("max_right", 0))

        # 4. Propagate upward (decreasing row)
        if up_color is not None:
            color = int(up_color)
            if 0 <= color < C:
                for step in range(1, max_up + 1):
                    rr = r_center - step
                    cc = c_center
                    if rr < 0:
                        break  # Hit top boundary
                    p_idx = rr * W + cc
                    builder.fix_pixel_color(p_idx, color, C)

        # 5. Propagate downward (increasing row)
        if down_color is not None:
            color = int(down_color)
            if 0 <= color < C:
                for step in range(1, max_down + 1):
                    rr = r_center + step
                    cc = c_center
                    if rr >= H:
                        break  # Hit bottom boundary
                    p_idx = rr * W + cc
                    builder.fix_pixel_color(p_idx, color, C)

        # 6. Propagate leftward (decreasing col)
        if left_color is not None:
            color = int(left_color)
            if 0 <= color < C:
                for step in range(1, max_left + 1):
                    rr = r_center
                    cc = c_center - step
                    if cc < 0:
                        break  # Hit left boundary
                    p_idx = rr * W + cc
                    builder.fix_pixel_color(p_idx, color, C)

        # 7. Propagate rightward (increasing col)
        if right_color is not None:
            color = int(right_color)
            if 0 <= color < C:
                for step in range(1, max_right + 1):
                    rr = r_center
                    cc = c_center + step
                    if cc >= W:
                        break  # Hit right boundary
                    p_idx = rr * W + cc
                    builder.fix_pixel_color(p_idx, color, C)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S9 builder with toy example...")
    print("=" * 70)

    # Create a 5x5 input grid (content doesn't matter for cross propagation)
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Full cross from center (2,2)")
    print("-" * 70)
    print("Propagate: up=1, down=2, left=3, right=4")
    print("Max steps: 2 in each direction")

    params1 = {
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

    builder1 = ConstraintBuilder()
    build_S9_constraints(ctx, params1, builder1)

    # Up: (1,2), (0,2) = 2 pixels
    # Down: (3,2), (4,2) = 2 pixels
    # Left: (2,1), (2,0) = 2 pixels
    # Right: (2,3), (2,4) = 2 pixels
    # Total: 8 pixels (center not included)
    expected1 = 8
    print(f"  Expected: {expected1} constraints (cross with 2 steps each direction)")
    print(f"  Actual: {len(builder1.constraints)}")
    assert len(builder1.constraints) == expected1, \
        f"Expected {expected1} constraints, got {len(builder1.constraints)}"

    print("\nTest 2: Partial cross (only up and right)")
    print("-" * 70)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(2,2)",
            "up_color": 5,
            "down_color": None,  # Skip down
            "left_color": None,  # Skip left
            "right_color": 6,
            "max_up": 2,
            "max_down": 0,
            "max_left": 0,
            "max_right": 2
        }]
    }

    builder2 = ConstraintBuilder()
    build_S9_constraints(ctx, params2, builder2)

    # Up: 2 pixels, Right: 2 pixels
    # Total: 4 pixels
    expected2 = 4
    print(f"  Expected: {expected2} constraints (only up and right)")
    print(f"  Actual: {len(builder2.constraints)}")
    assert len(builder2.constraints) == expected2, \
        f"Expected {expected2} constraints, got {len(builder2.constraints)}"

    print("\nTest 3: Cross hitting boundaries")
    print("-" * 70)
    print("Seed at corner (0,0) with large max_steps")

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(0,0)",
            "up_color": 7,       # Can't go up from row 0
            "down_color": 8,
            "left_color": 9,     # Can't go left from col 0
            "right_color": 1,
            "max_up": 10,
            "max_down": 10,
            "max_left": 10,
            "max_right": 10
        }]
    }

    builder3 = ConstraintBuilder()
    build_S9_constraints(ctx, params3, builder3)

    # Up: 0 pixels (already at top)
    # Down: min(10, 4) = 4 pixels (rows 1-4)
    # Left: 0 pixels (already at left)
    # Right: min(10, 4) = 4 pixels (cols 1-4)
    # Total: 8 pixels
    expected3 = 8
    print(f"  Expected: {expected3} constraints (boundary clipping)")
    print(f"  Actual: {len(builder3.constraints)}")
    assert len(builder3.constraints) == expected3, \
        f"Expected {expected3} constraints, got {len(builder3.constraints)}"

    print("\nTest 4: Multiple seeds")
    print("-" * 70)

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [
            {
                "center": "(1,1)",
                "up_color": 1,
                "down_color": None,
                "left_color": None,
                "right_color": None,
                "max_up": 1,
                "max_down": 0,
                "max_left": 0,
                "max_right": 0
            },
            {
                "center": "(3,3)",
                "up_color": None,
                "down_color": 2,
                "left_color": None,
                "right_color": None,
                "max_up": 0,
                "max_down": 1,
                "max_left": 0,
                "max_right": 0
            }
        ]
    }

    builder4 = ConstraintBuilder()
    build_S9_constraints(ctx, params4, builder4)

    # Seed 1: up 1 step = 1 pixel
    # Seed 2: down 1 step = 1 pixel
    # Total: 2 pixels
    expected4 = 2
    print(f"  Expected: {expected4} constraints (2 seeds, 1 pixel each)")
    print(f"  Actual: {len(builder4.constraints)}")
    assert len(builder4.constraints) == expected4, \
        f"Expected {expected4} constraints, got {len(builder4.constraints)}"

    print("\n" + "=" * 70)
    print("✓ S9 builder self-test passed.")
