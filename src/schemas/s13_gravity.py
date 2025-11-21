"""
S13 schema builder: Gravity / Object Movement.

This implements the S13 schema for physics-based object movement:
    "Objects (mobile colors) move in a gravity direction until hitting
     a boundary or obstacle (stationary colors)."

S13 simulates conservation of matter:
  - Mobile pixels move to new positions (old position becomes empty)
  - Stationary pixels act as walls/obstacles
  - Movement continues until stable (no more changes)

S13 is geometry-preserving: output has same shape as input.
"""

from typing import Dict, Any
from ast import literal_eval

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder

# Import gravity simulation from miner
from src.law_mining.mine_s13 import simulate_gravity


def build_S13_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S13 preferences: simulate gravity and constrain final positions.

    S13 simulates object movement in a gravity direction. Mobile colors
    "fall" or move until hitting obstacles or boundaries.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "gravity_vector": "(dr,dc)",         # Direction tuple
          "mobile_colors": [int, ...]          # Colors that move
        }

    Where:
        - gravity_vector: Direction of gravity as (dr, dc) tuple
          * (1, 0): Down
          * (-1, 0): Up
          * (0, 1): Right
          * (0, -1): Left
        - mobile_colors: List of color values that can move
          * All other colors are stationary (act as walls)

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying gravity configuration
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> # Downward gravity, color 2 falls
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "gravity_vector": "(1,0)",
        ...     "mobile_colors": [2]
        ... }
        >>> build_S13_constraints(ctx, params, builder)
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
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid

    # 3. Parse gravity parameters
    gravity_vector_str = schema_params.get("gravity_vector", "(0,0)")
    mobile_colors = set(schema_params.get("mobile_colors", []))

    # Parse vector
    try:
        gravity_vector = literal_eval(gravity_vector_str)
    except (ValueError, SyntaxError):
        return  # Skip malformed vector

    if not mobile_colors:
        return  # No mobile colors, nothing to do

    # 4. Simulate gravity on input grid
    final_grid = simulate_gravity(ex.input_grid, gravity_vector, mobile_colors)

    # 5. Constrain all output pixels to their final positions
    # Weight = 50.0 (Tier 2 - Object Tier, same as S12)
    for r in range(H):
        for c in range(W):
            p_idx = r * W + c
            color = int(final_grid[r, c])
            builder.prefer_pixel_color(p_idx, color, weight=50.0)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S13 builder with toy example...")
    print("=" * 70)

    # Create a falling object scenario
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],  # Floor
    ], dtype=int)

    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=int)

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=3)

    print("Test 1: Downward gravity")
    print("-" * 70)

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "gravity_vector": "(1,0)",  # Down
        "mobile_colors": [2]
    }

    builder1 = ConstraintBuilder()
    build_S13_constraints(ctx, params1, builder1)

    # Should have preferences for all 30 pixels (6×5)
    expected1 = 30
    print(f"  Expected: {expected1} preferences (all pixels)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1, \
        f"Expected {expected1} preferences, got {len(builder1.preferences)}"

    # Check that falling object pixels are constrained correctly
    # Object should be at rows 3-4, cols 1-2
    # Preferences are tuples: (pixel_index, color, weight)
    for r in [3, 4]:
        for c in [1, 2]:
            p_idx = r * 5 + c
            prefs_for_pixel = [p for p in builder1.preferences if p[0] == p_idx]
            assert len(prefs_for_pixel) == 1, f"Expected 1 pref for pixel ({r},{c})"
            assert prefs_for_pixel[0][1] == 2, f"Expected color 2 at ({r},{c})"
            assert prefs_for_pixel[0][2] == 50.0, f"Expected weight 50.0"

    print("  ✓ Object correctly constrained at fallen position")

    # Check floor stays in place
    for c in range(5):
        p_idx = 5 * 5 + c
        prefs_for_pixel = [p for p in builder1.preferences if p[0] == p_idx]
        assert prefs_for_pixel[0][1] == 1, f"Expected floor (color 1) at row 5"

    print("  ✓ Floor correctly constrained in place")

    print("\n" + "=" * 70)
    print("✓ S13 builder self-test passed.")
