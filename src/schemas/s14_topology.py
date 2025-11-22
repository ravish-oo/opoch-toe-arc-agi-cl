"""
S14 schema builder: Topology / Flood Fill.

This implements the S14 schema for topological region-filling operations:
    "Fill holes inside boundaries" or "Fill background regions"

S14 uses scipy.ndimage for robust topological operations:
  - binary_fill_holes: Find enclosed regions
  - label: Connected component analysis

S14 is geometry-preserving: output has same shape as input.
"""

from typing import Dict, Any
import numpy as np

try:
    from scipy.ndimage import label, binary_fill_holes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S14_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S14 preferences: topology-based region filling.

    S14 supports two operations:
    - fill_enclosed: Fill holes inside a boundary with a specific color
    - fill_background: Fill the background region with a specific color

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "operation": "fill_enclosed" | "fill_background",
          "boundary_color": int,  # Only for fill_enclosed
          "fill_color": int
        }

    Args:
        task_context: TaskContext with all grids and features
        schema_params: Parameters specifying topology operation
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> # Fill holes inside red (1) boundary with blue (2)
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "operation": "fill_enclosed",
        ...     "boundary_color": 1,
        ...     "fill_color": 2
        ... }
        >>> build_S14_constraints(ctx, params, builder)
    """
    if not SCIPY_AVAILABLE:
        return  # Cannot build without scipy

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
    # FIX: Handle Test Examples (Future Geometry)
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        # Assume geometry preserving for Topological tasks
        H, W = ex.input_grid.shape

    # 3. Parse operation parameters
    operation = schema_params.get("operation", "")
    fill_color = schema_params.get("fill_color", 0)

    input_grid = ex.input_grid

    # 4. Execute operation based on type
    if operation == "fill_enclosed":
        boundary_color = schema_params.get("boundary_color", 0)

        # Create boundary mask
        boundary_mask = (input_grid == boundary_color)

        # Fill holes in boundary
        filled_mask = binary_fill_holes(boundary_mask)

        # Holes are filled but not boundary
        holes_mask = filled_mask & ~boundary_mask

        # Constrain hole pixels to fill_color
        # Weight = 20.0 (Tier 3 - Topological)
        for r in range(H):
            for c in range(W):
                if holes_mask[r, c]:
                    p_idx = r * W + c
                    builder.prefer_pixel_color(p_idx, fill_color, weight=20.0)

    elif operation == "fill_background":
        # Get background color (color at 0,0)
        background_color = int(input_grid[0, 0])

        # Create mask of same-color pixels
        same_color_mask = (input_grid == background_color)

        # Label connected components
        labeled, _ = label(same_color_mask)

        # Background component is the one containing (0,0)
        background_label = labeled[0, 0]

        if background_label == 0:
            return  # No background component

        background_mask = (labeled == background_label)

        # Constrain background pixels to fill_color
        # Weight = 20.0 (Tier 3 - Topological)
        for r in range(H):
            for c in range(W):
                if background_mask[r, c]:
                    p_idx = r * W + c
                    builder.prefer_pixel_color(p_idx, fill_color, weight=20.0)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S14 builder with toy example...")
    print("=" * 70)

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy not available, skipping test")
        exit(0)

    # Test 1: Fill enclosed hole
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],  # Hole at (2,2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],  # Hole filled with 2
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=3)

    print("Test 1: Fill enclosed hole")
    print("-" * 70)

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "operation": "fill_enclosed",
        "boundary_color": 1,
        "fill_color": 2
    }

    builder1 = ConstraintBuilder()
    build_S14_constraints(ctx, params1, builder1)

    # Should have 1 preference (for the hole at 2,2)
    print(f"  Preferences: {len(builder1.preferences)}")
    assert len(builder1.preferences) == 1, f"Expected 1 preference, got {len(builder1.preferences)}"

    # Check the preference is for pixel (2,2) = index 12 with color 2
    pref = builder1.preferences[0]
    expected_idx = 2 * 5 + 2  # row 2, col 2, width 5
    assert pref[0] == expected_idx, f"Expected pixel index {expected_idx}, got {pref[0]}"
    assert pref[1] == 2, f"Expected color 2, got {pref[1]}"
    assert pref[2] == 20.0, f"Expected weight 20.0, got {pref[2]}"

    print(f"  Preference: pixel={pref[0]}, color={pref[1]}, weight={pref[2]}")
    print("  Correctly constrained hole pixel to fill_color")

    # Test 2: Fill background
    print("\nTest 2: Fill background")
    print("-" * 70)

    input_grid2 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=int)

    output_grid2 = np.array([
        [3, 3, 3],
        [3, 1, 3],
        [3, 3, 3],
    ], dtype=int)

    ex2 = build_example_context(input_grid2, output_grid2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=4)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "operation": "fill_background",
        "fill_color": 3
    }

    builder2 = ConstraintBuilder()
    build_S14_constraints(ctx2, params2, builder2)

    # Should have 8 preferences (all background pixels except center)
    print(f"  Preferences: {len(builder2.preferences)}")
    assert len(builder2.preferences) == 8, f"Expected 8 preferences, got {len(builder2.preferences)}"

    print("  Correctly constrained 8 background pixels to fill_color")

    print("\n" + "=" * 70)
    print("S14 builder self-test passed.")
