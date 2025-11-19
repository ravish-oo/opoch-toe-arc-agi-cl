"""
S1 schema builder: Direct pixel color tie (copy/equality).

This implements the S1 schema from the math kernel spec (section 2):
    "If φ(p)=φ(q) in all training pairs and Y(p)=Y(q) always, then add:
     ∀c: y_{(p,c)} - y_{(q,c)} = 0"

S1 is geometry-preserving: output has same shape as input.
This builder applies pre-mined pixel ties (from Pi-agent) as constraints.
"""

from typing import Dict, Any

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S1_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S1 constraints: tie colors of specified pixel pairs.

    S1 enforces that pixels with equivalent features have the same color.
    This is the backbone for copy/equality operations.

    Param schema:
        {
          "ties": [
            {
              "example_type": "train" | "test",
              "example_index": int,
              "pairs": [((r1, c1), (r2, c2)), ...]
            },
            ...
          ]
        }

    For each pixel pair (p, q), adds constraints:
        y_{p,c} - y_{q,c} = 0  for all colors c

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying which pixel pairs to tie
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> params = {
        ...     "ties": [{
        ...         "example_type": "train",
        ...         "example_index": 0,
        ...         "pairs": [((0, 0), (0, 1)), ((1, 0), (1, 1))]
        ...     }]
        ... }
        >>> build_S1_constraints(ctx, params, builder)
        # Adds 2 * C tie constraints (one per color per pair)
    """
    # Extract ties from params
    ties = schema_params.get("ties", [])

    if not ties:
        # No ties specified, nothing to do
        return

    # Extract current example being solved (injected by dispatch layer)
    # Used to filter ties to only the current example
    current_example_type = schema_params.get("example_type")
    current_example_index = schema_params.get("example_index")

    for tie_group in ties:
        # 1. Select the example
        example_type = tie_group.get("example_type", "train")
        example_index = tie_group.get("example_index", 0)

        # Filter ties to only the current example being solved
        # This prevents IndexError when tie_groups contain ties from examples
        # with different grid dimensions (e.g., ex0: 10x10, ex3: 20x20)
        if current_example_type is not None and example_type != current_example_type:
            continue  # Skip ties from other example types
        if current_example_index is not None and example_index != current_example_index:
            continue  # Skip ties from other example indices

        if example_type == "train":
            if example_index >= len(task_context.train_examples):
                continue  # Skip invalid index
            ex = task_context.train_examples[example_index]
        else:  # "test"
            if example_index >= len(task_context.test_examples):
                continue  # Skip invalid index
            ex = task_context.test_examples[example_index]

        # 2. Get grid dimensions
        # S1 ties are output equalities; coordinates are in output space
        # Use output dimensions for indexing into y variables
        H = ex.output_H
        W = ex.output_W
        if H is None or W is None:
            return  # No output grid to constrain

        # 3. Process pixel pairs
        pairs = tie_group.get("pairs", [])

        for pair in pairs:
            if len(pair) != 2:
                continue  # Skip malformed pairs

            (r1, c1), (r2, c2) = pair

            # Validate coordinates are in bounds
            if not (0 <= r1 < H and 0 <= c1 < W):
                continue
            if not (0 <= r2 < H and 0 <= c2 < W):
                continue

            # Compute flat pixel indices (row-major)
            p_idx1 = r1 * W + c1
            p_idx2 = r2 * W + c2

            # Add tie constraints: y_{p1,c} - y_{p2,c} = 0 for all c
            builder.tie_pixel_colors(p_idx1, p_idx2, task_context.C)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S1 builder with toy example...")
    print("=" * 70)

    # Create a simple 2x2 input grid
    input_grid = np.array([
        [1, 2],
        [3, 4]
    ], dtype=int)

    # For geometry-preserving S1, output has same shape
    output_grid = np.array([
        [1, 1],
        [3, 3]
    ], dtype=int)

    # Build example context
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(
        train_examples=[ex],
        test_examples=[],
        C=5  # colors 0-4
    )

    # Create params: tie (0,0) to (0,1) and (1,0) to (1,1)
    params = {
        "ties": [{
            "example_type": "train",
            "example_index": 0,
            "pairs": [
                ((0, 0), (0, 1)),  # tie top-left to top-right
                ((1, 0), (1, 1))   # tie bottom-left to bottom-right
            ]
        }]
    }

    # Build constraints
    builder = ConstraintBuilder()
    build_S1_constraints(ctx, params, builder)

    # Verify constraints were added
    # Each tie adds C constraints (one per color)
    # 2 pairs × 5 colors = 10 constraints
    expected_constraints = 2 * ctx.C
    actual_constraints = len(builder.constraints)

    print(f"Expected constraints: {expected_constraints}")
    print(f"Actual constraints: {actual_constraints}")

    assert actual_constraints == expected_constraints, \
        f"Expected {expected_constraints} constraints, got {actual_constraints}"

    # Inspect first constraint
    if builder.constraints:
        c0 = builder.constraints[0]
        print(f"\nSample constraint (first):")
        print(f"  indices: {c0.indices}")
        print(f"  coeffs: {c0.coeffs}")
        print(f"  rhs: {c0.rhs}")
        assert c0.coeffs == [1.0, -1.0], "Tie constraint should have coeffs [1, -1]"
        assert c0.rhs == 0.0, "Tie constraint should have rhs=0"

    print("\n" + "=" * 70)
    print("✓ S1 builder self-test passed.")
