"""
S2 schema builder: Component-wise recolor map.

This implements the S2 schema from the math kernel spec (section 2):
    "For each connected component type (shape up to translation):
     1. Group pixels by object_id(p).
     2. For each object class k, learn a color map: input_color → output_color.
     Constraints: For all pixels p with object_id(p)=k:
         ∀c: y_{(p,c)} = 1 iff c = f_k(X(p))"

S2 is geometry-preserving: output has same shape as input.
This builder applies pre-mined recolor maps (from Pi-agent) as constraints.
"""

from typing import Dict, Any

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S2_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S2 constraints: recolor components based on size/attributes.

    S2 enforces that components with specific attributes get specific output colors.
    For M3.1, we use component size as the attribute.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "input_color": int,              # filter components by this input color
          "size_to_color": {               # map component size to output color
            "1": 3,                        # size 1 → color 3
            "2": 2,                        # size 2 → color 2
            "else": 1                      # default → color 1
          }
        }

    For each component matching input_color:
        - Determine output color from size_to_color mapping
        - For each pixel in component: y_{p,out_color} = 1

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying recolor mapping
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "input_color": 1,
        ...     "size_to_color": {"1": 3, "else": 2}
        ... }
        >>> build_S2_constraints(ctx, params, builder)
        # For each pixel in size-1 components of color 1: fix to color 3
        # For each pixel in other-size components of color 1: fix to color 2
    """
    # 1. Select the example
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

    # 2. Extract params
    input_color = int(schema_params.get("input_color", 0))
    size_to_color = schema_params.get("size_to_color", {})

    if not size_to_color:
        return  # No mapping specified

    # 3. Get grid dimensions
    # For S2 (geometry-preserving), output shape = input shape
    H = ex.input_H
    W = ex.input_W

    # 4. Process each component
    for comp in ex.components:
        # Filter by input color
        if comp.color != input_color:
            continue

        # Determine output color from size mapping
        size_str = str(comp.size)
        if size_str in size_to_color:
            out_color = int(size_to_color[size_str])
        elif "else" in size_to_color:
            out_color = int(size_to_color["else"])
        else:
            # No mapping for this size, skip
            continue

        # Fix color for all pixels in this component
        for (r, c) in comp.pixels:
            # Validate coordinates
            if not (0 <= r < H and 0 <= c < W):
                continue

            # Compute flat pixel index (row-major)
            p_idx = r * W + c

            # Add constraint: y_{p,out_color} = 1
            builder.fix_pixel_color(p_idx, out_color, task_context.C)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S2 builder with toy example...")
    print("=" * 70)

    # Create a simple 3x3 input grid with different component sizes
    # Color 1: three separate components (size 1, 1, 2)
    # Color 2: one component (size 3)
    input_grid = np.array([
        [0, 1, 0],
        [1, 2, 1],
        [2, 2, 2]
    ], dtype=int)

    # Output should recolor based on size
    output_grid = np.array([
        [0, 3, 0],  # size-1 component of color 1 → color 3
        [3, 5, 4],  # size-1 → 3, size-2 → 4
        [5, 5, 5]   # size-3 component of color 2 → color 5
    ], dtype=int)

    # Build example context
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(
        train_examples=[ex],
        test_examples=[],
        C=6  # colors 0-5
    )

    print(f"Number of components: {len(ex.components)}")
    for comp in ex.components:
        print(f"  Component {comp.id}: color={comp.color}, size={comp.size}, pixels={comp.pixels}")

    # Create params: recolor components of color 1 based on size
    params = {
        "example_type": "train",
        "example_index": 0,
        "input_color": 1,
        "size_to_color": {
            "1": 3,  # size 1 → color 3
            "2": 4,  # size 2 → color 4
            "else": 0
        }
    }

    # Build constraints
    builder = ConstraintBuilder()
    build_S2_constraints(ctx, params, builder)

    # Verify constraints were added
    print(f"\nConstraints added: {len(builder.constraints)}")

    # Count components of color 1
    color_1_comps = [c for c in ex.components if c.color == 1]
    total_pixels = sum(c.size for c in color_1_comps)
    print(f"Total pixels in color-1 components: {total_pixels}")
    print(f"Expected constraints: {total_pixels} (one per pixel)")

    assert len(builder.constraints) == total_pixels, \
        f"Expected {total_pixels} constraints, got {len(builder.constraints)}"

    # Inspect first constraint
    if builder.constraints:
        c0 = builder.constraints[0]
        print(f"\nSample constraint (first):")
        print(f"  indices: {c0.indices}")
        print(f"  coeffs: {c0.coeffs}")
        print(f"  rhs: {c0.rhs}")
        assert len(c0.indices) == 1, "Fix constraint should have 1 index"
        assert c0.coeffs == [1.0], "Fix constraint should have coeff [1.0]"
        assert c0.rhs == 1.0, "Fix constraint should have rhs=1.0"

    print("\n" + "=" * 70)
    print("✓ S2 builder self-test passed.")
