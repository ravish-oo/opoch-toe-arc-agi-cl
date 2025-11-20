"""
S10 schema builder: Frame / border vs interior.

This implements the S10 schema from the math kernel spec (section 2):
    "Different preferences for border band vs interior band of an object or whole grid.
     Border pixels get color b, interior pixels get color i."

S10 is geometry-preserving: output has same shape as input.
This builder applies pre-determined border/interior colors (from Pi-agent) using
existing border_info features from M1.
"""

from typing import Dict, Any

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S10_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S10 preferences: fix border and interior pixels to specific colors.

    S10 assigns different colors to border vs interior pixels of components,
    using the border_info feature computed in M1 (component_border_interior).

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "border_color": int,     # color for border pixels
          "interior_color": int    # color for interior pixels
        }

    Where:
        - border_info[(r,c)]["is_border"] identifies border pixels
        - border_info[(r,c)]["is_interior"] identifies interior pixels
        - Pixels that are neither are left unconstrained

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying border and interior colors
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> # Frame with border color 5 and interior color 7
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "border_color": 5,
        ...     "interior_color": 7
        ... }
        >>> build_S10_preferences(ctx, params, builder)
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

    # 2. Get grid dimensions and features
    # S10 applies border/interior preferences to OUTPUT grid
    # Use output dimensions for indexing into y variables
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid to constrain
    C = task_context.C
    border_info = ex.border_info  # Dict[(r,c)] -> {"is_border": bool, "is_interior": bool}

    # 3. Parse colors
    border_color = int(schema_params.get("border_color", 0))
    interior_color = int(schema_params.get("interior_color", 0))

    # Validate colors are in palette
    if not (0 <= border_color < C):
        border_color = 0
    if not (0 <= interior_color < C):
        interior_color = 0

    # 4. For each pixel, fix color based on border/interior status
    for r in range(H):
        for c in range(W):
            p_idx = r * W + c

            # Get border info for this pixel
            info = border_info.get((r, c), {})

            # Check if pixel is border or interior
            is_border = info.get("is_border", False)
            is_interior = info.get("is_interior", False)

            if is_border:
                # Prefer border color (Tier 2: Object, weight 50.0)
                builder.prefer_pixel_color(p_idx, border_color, weight=50.0)
            elif is_interior:
                # Prefer interior color (Tier 2: Object, weight 50.0)
                builder.prefer_pixel_color(p_idx, interior_color, weight=50.0)
            # else: pixel is neither border nor interior (e.g., background) → unconstrained


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S10 builder with toy example...")
    print("=" * 70)

    # Create a 5x5 input grid with a component in the center
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Check what border_info looks like
    print("Border info for input grid:")
    border_pixels = [(r, c) for (r, c), info in ex.border_info.items() if info.get("is_border")]
    interior_pixels = [(r, c) for (r, c), info in ex.border_info.items() if info.get("is_interior")]
    print(f"  Border pixels: {len(border_pixels)}")
    print(f"  Interior pixels: {len(interior_pixels)}")

    print("\nTest 1: Frame with border=5, interior=7")
    print("-" * 70)

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 5,
        "interior_color": 7
    }

    builder1 = ConstraintBuilder()
    build_S10_preferences(ctx, params1, builder1)

    # Expected: preferences for all border + interior pixels
    expected1 = len(border_pixels) + len(interior_pixels)
    print(f"  Expected: {expected1} preferences (border + interior)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1, \
        f"Expected {expected1} preferences, got {len(builder1.preferences)}"

    print("\nTest 2: Same color for border and interior")
    print("-" * 70)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 3,
        "interior_color": 3  # Same as border
    }

    builder2 = ConstraintBuilder()
    build_S10_preferences(ctx, params2, builder2)

    # Same expected count
    expected2 = len(border_pixels) + len(interior_pixels)
    print(f"  Expected: {expected2} preferences (all same color)")
    print(f"  Actual: {len(builder2.preferences)}")
    assert len(builder2.preferences) == expected2, \
        f"Expected {expected2} preferences, got {len(builder2.preferences)}"

    print("\nTest 3: Grid with all same color (forms one component)")
    print("-" * 70)

    # Create grid with all zeros (forms one big component)
    input_grid_bg = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=int)

    output_grid_bg = input_grid_bg.copy()

    ex_bg = build_example_context(input_grid_bg, output_grid_bg)
    ctx_bg = TaskContext(train_examples=[ex_bg], test_examples=[], C=10)

    # Count border/interior pixels in all-zero grid
    border_bg = [(r, c) for (r, c), info in ex_bg.border_info.items() if info.get("is_border")]
    interior_bg = [(r, c) for (r, c), info in ex_bg.border_info.items() if info.get("is_interior")]

    print(f"  Border pixels in 3x3 all-zero grid: {len(border_bg)}")
    print(f"  Interior pixels in 3x3 all-zero grid: {len(interior_bg)}")

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 1,
        "interior_color": 2
    }

    builder3 = ConstraintBuilder()
    build_S10_preferences(ctx_bg, params3, builder3)

    # All-zero grid forms one component with border/interior
    expected3 = len(border_bg) + len(interior_bg)
    print(f"  Expected: {expected3} preferences")
    print(f"  Actual: {len(builder3.preferences)}")
    assert len(builder3.preferences) == expected3, \
        f"Expected {expected3} preferences, got {len(builder3.preferences)}"

    print("\nTest 4: Multiple components")
    print("-" * 70)

    # Create grid with two separate components
    input_grid_multi = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid_multi = input_grid_multi.copy()

    ex_multi = build_example_context(input_grid_multi, output_grid_multi)
    ctx_multi = TaskContext(train_examples=[ex_multi], test_examples=[], C=10)

    # Count border/interior pixels
    border_multi = [(r, c) for (r, c), info in ex_multi.border_info.items() if info.get("is_border")]
    interior_multi = [(r, c) for (r, c), info in ex_multi.border_info.items() if info.get("is_interior")]

    print(f"  Border pixels in multi-component grid: {len(border_multi)}")
    print(f"  Interior pixels in multi-component grid: {len(interior_multi)}")

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 8,
        "interior_color": 9
    }

    builder4 = ConstraintBuilder()
    build_S10_preferences(ctx_multi, params4, builder4)

    expected4 = len(border_multi) + len(interior_multi)
    print(f"  Expected: {expected4} preferences")
    print(f"  Actual: {len(builder4.preferences)}")
    assert len(builder4.preferences) == expected4, \
        f"Expected {expected4} preferences, got {len(builder4.preferences)}"

    print("\n" + "=" * 70)
    print("✓ S10 builder self-test passed.")
