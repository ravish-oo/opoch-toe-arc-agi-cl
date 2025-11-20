"""
S4 schema builder: Periodicity / residue-class coloring.

This implements the S4 schema from the math kernel spec (section 2):
    "Colors determined purely by residue of coordinates mod K.
     For each pixel p: ∀c ≠ h(c(p) mod K): y_{(p,c)} = 0"

S4 is geometry-preserving: output has same shape as input.
This builder applies pre-mined residue→color mappings (from Pi-agent) as preferences.
"""

from typing import Dict, Any

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S4_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S4 preferences: fix pixel colors based on coordinate residues.

    S4 enforces that pixel colors are determined purely by the residue
    of their row or column index modulo K.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "axis": "row" | "col",
          "K": int,
          "residue_to_color": {
            "0": 1,   # residue 0 → color 1
            "1": 3    # residue 1 → color 3
          }
        }

    For each pixel, compute residue along specified axis mod K,
    then fix the pixel's color to the mapped value.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying axis, modulus, and color mapping
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> # Even columns get color 1, odd columns get color 3
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "axis": "col",
        ...     "K": 2,
        ...     "residue_to_color": {"0": 1, "1": 3}
        ... }
        >>> build_S4_preferences(ctx, params, builder)
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

    # 2. Extract params
    axis = schema_params.get("axis", "col")
    K = int(schema_params.get("K", 2))
    residue_to_color = schema_params.get("residue_to_color", {})

    if not residue_to_color:
        return  # No mapping specified

    # 3. Get grid dimensions
    # S4 applies residue-based coloring to OUTPUT grid
    # Use output dimensions for indexing into y variables
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid to constrain
    C = task_context.C

    # 4. For each pixel, fix color based on residue
    for r in range(H):
        for c in range(W):
            # Compute residue based on axis
            if axis == "row":
                # Use row residue
                if r not in ex.row_residues:
                    continue
                if K not in ex.row_residues[r]:
                    continue
                residue = ex.row_residues[r][K]
            elif axis == "col":
                # Use column residue
                if c not in ex.col_residues:
                    continue
                if K not in ex.col_residues[c]:
                    continue
                residue = ex.col_residues[c][K]
            else:
                continue  # Invalid axis

            # Look up color for this residue
            residue_str = str(residue)
            if residue_str not in residue_to_color:
                continue  # No mapping for this residue

            color = int(residue_to_color[residue_str])

            # Validate color is in palette
            if not (0 <= color < C):
                continue

            # Prefer this pixel's color (Tier 3: Local, weight 10.0)
            p_idx = r * W + c
            builder.prefer_pixel_color(p_idx, color, weight=10.0)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S4 builder with toy example...")
    print("=" * 70)

    # Create a 4x4 grid
    input_grid = np.array([
        [0, 1, 0, 1],
        [2, 3, 2, 3],
        [0, 1, 0, 1],
        [2, 3, 2, 3]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    print("Test 1: Column residue mod 2 (even/odd columns)")
    print("-" * 70)
    print("Grid shape: 4x4")
    print("Columns: 0(even), 1(odd), 2(even), 3(odd)")
    print("Mapping: even→1, odd→3")

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {
            "0": 1,  # even columns → color 1
            "1": 3   # odd columns → color 3
        }
    }

    builder1 = ConstraintBuilder()
    build_S4_preferences(ctx, params1, builder1)

    # Should have 16 preferences (4x4 grid = 16 pixels, one fix per pixel)
    expected1 = 16
    print(f"  Expected: {expected1} preferences (one per pixel)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1

    # Inspect a few preferences
    print("\n  Sample preferences:")
    for i in range(min(4, len(builder1.preferences))):
        c = builder1.preferences[i]
        print(f"    Constraint {i}: indices={c.indices}, coeffs={c.coeffs}, rhs={c.rhs}")

    print("\nTest 2: Row residue mod 2 (even/odd rows)")
    print("-" * 70)
    params2 = {
        "example_type": "train",
        "example_index": 0,
        "axis": "row",
        "K": 2,
        "residue_to_color": {
            "0": 2,  # even rows → color 2
            "1": 4   # odd rows → color 4
        }
    }

    builder2 = ConstraintBuilder()
    build_S4_preferences(ctx, params2, builder2)

    # Should have 16 preferences (4x4 grid = 16 pixels)
    expected2 = 16
    print(f"  Expected: {expected2} preferences (one per pixel)")
    print(f"  Actual: {len(builder2.preferences)}")
    assert len(builder2.preferences) == expected2

    print("\nTest 3: Column residue mod 4 (partial mapping)")
    print("-" * 70)
    params3 = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 4,
        "residue_to_color": {
            "0": 1,  # columns 0 → color 1
            "2": 3   # columns 2 → color 3
            # residues 1 and 3 not mapped (cols 1,3 unconstrained)
        }
    }

    builder3 = ConstraintBuilder()
    build_S4_preferences(ctx, params3, builder3)

    # Should have 8 preferences (only cols 0 and 2 mapped, 4 rows × 2 cols = 8)
    expected3 = 8
    print(f"  Expected: {expected3} preferences (only cols 0,2 mapped × 4 rows)")
    print(f"  Actual: {len(builder3.preferences)}")
    assert len(builder3.preferences) == expected3

    print("\n" + "=" * 70)
    print("✓ S4 builder self-test passed.")
