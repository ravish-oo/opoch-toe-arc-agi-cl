"""
S3 schema builder: Band / stripe laws (rows and columns).

This implements the S3 schema from the math kernel spec (section 2):
    "Rows and columns with same band features share a color pattern.
     For all rows in class R: enforce same pattern of colors across columns."

S3 is geometry-preserving: output has same shape as input.
This builder applies pre-mined band classes and periodicity (from Pi-agent) as constraints.
"""

from typing import Dict, Any, List

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S3_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S3 constraints: tie rows/columns in same band classes.

    S3 enforces that rows (or columns) with equivalent features share
    the same color pattern across their perpendicular axis.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "row_classes": [[0, 2], [1, 3, 4], ...],  # optional
          "col_classes": [[0, 2, 4], [1, 3, 5]],    # optional
          "col_period_K": int | None,               # optional
          "row_period_K": int | None                # optional
        }

    Row classes: tie all rows in same class pairwise across all columns.
    Col classes: tie all cols in same class pairwise across all rows.
    col_period_K: tie (r,c) with (r,c+K) for periodic column patterns.
    row_period_K: tie (r,c) with (r+K,c) for periodic row patterns.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying band classes and periodicity
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # Tie rows 0 and 2 (share same pattern across columns)
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "row_classes": [[0, 2]],
        ...     "col_classes": [],
        ...     "col_period_K": None,
        ...     "row_period_K": None
        ... }
        >>> build_S3_constraints(ctx, params, builder)
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
    # S3 applies band/stripe constraints to OUTPUT grid
    # Use output dimensions for indexing into y variables
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid to constrain
    C = task_context.C

    # 3. Row band ties (tie rows in same class)
    row_classes = schema_params.get("row_classes", [])
    for row_class in row_classes:
        if len(row_class) < 2:
            continue  # Need at least 2 rows to tie

        # Tie all pairs of rows in this class
        for i in range(len(row_class)):
            r1 = row_class[i]
            if not (0 <= r1 < H):
                continue  # Skip out-of-bounds row

            for j in range(i + 1, len(row_class)):
                r2 = row_class[j]
                if not (0 <= r2 < H):
                    continue  # Skip out-of-bounds row

                # Tie (r1, c) with (r2, c) for all columns
                for c in range(W):
                    p_idx1 = r1 * W + c
                    p_idx2 = r2 * W + c
                    builder.tie_pixel_colors_soft(p_idx1, p_idx2, C, weight=10.0)

    # 4. Column band ties (tie columns in same class)
    col_classes = schema_params.get("col_classes", [])
    for col_class in col_classes:
        if len(col_class) < 2:
            continue  # Need at least 2 columns to tie

        # Tie all pairs of columns in this class
        for i in range(len(col_class)):
            c1 = col_class[i]
            if not (0 <= c1 < W):
                continue  # Skip out-of-bounds column

            for j in range(i + 1, len(col_class)):
                c2 = col_class[j]
                if not (0 <= c2 < W):
                    continue  # Skip out-of-bounds column

                # Tie (r, c1) with (r, c2) for all rows
                for r in range(H):
                    p_idx1 = r * W + c1
                    p_idx2 = r * W + c2
                    builder.tie_pixel_colors_soft(p_idx1, p_idx2, C, weight=10.0)

    # 5. Column periodicity (tie columns with period K)
    col_period_K = schema_params.get("col_period_K")
    if col_period_K is not None:
        K = int(col_period_K)
        # For each row, tie (r,c) with (r,c+K) if c+K in bounds
        for r in range(H):
            for c in range(W):
                c2 = c + K
                if c2 < W:
                    p_idx1 = r * W + c
                    p_idx2 = r * W + c2
                    builder.tie_pixel_colors_soft(p_idx1, p_idx2, C, weight=10.0)

    # 6. Row periodicity (tie rows with period K)
    row_period_K = schema_params.get("row_period_K")
    if row_period_K is not None:
        K = int(row_period_K)
        # For each column, tie (r,c) with (r+K,c) if r+K in bounds
        for c in range(W):
            for r in range(H):
                r2 = r + K
                if r2 < H:
                    p_idx1 = r * W + c
                    p_idx2 = r2 * W + c
                    builder.tie_pixel_colors_soft(p_idx1, p_idx2, C, weight=10.0)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S3 builder with toy example...")
    print("=" * 70)

    # Create a 4x4 grid
    input_grid = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 2, 3, 4],  # Same as row 0 (should be tied)
        [9, 8, 7, 6]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Row classes (tie rows 0 and 2)")
    print("-" * 70)
    params1 = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [[0, 2]],  # Rows 0 and 2 should share pattern
        "col_classes": [],
        "col_period_K": None,
        "row_period_K": None
    }

    builder1 = ConstraintBuilder()
    build_S3_constraints(ctx, params1, builder1)

    # Should have 4 soft ties (one per row pair)
    expected1 = 4
    print(f"  Expected: {expected1} soft ties (4 row pairs)")
    print(f"  Actual: {len(builder1.soft_ties)}")
    assert len(builder1.soft_ties) == expected1

    print("\nTest 2: Column classes (tie cols 0 and 2)")
    print("-" * 70)
    params2 = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [],
        "col_classes": [[0, 2]],  # Cols 0 and 2 should share pattern
        "col_period_K": None,
        "row_period_K": None
    }

    builder2 = ConstraintBuilder()
    build_S3_constraints(ctx, params2, builder2)

    # Should have 4 soft ties (one per column pair)
    expected2 = 4
    print(f"  Expected: {expected2} soft ties (4 column pairs)")
    print(f"  Actual: {len(builder2.soft_ties)}")
    assert len(builder2.soft_ties) == expected2

    print("\nTest 3: Column periodicity K=2 (even/odd columns)")
    print("-" * 70)
    params3 = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [],
        "col_classes": [],
        "col_period_K": 2,  # Tie col 0↔2, 1↔3
        "row_period_K": None
    }

    builder3 = ConstraintBuilder()
    build_S3_constraints(ctx, params3, builder3)

    # For 4x4 grid with K=2:
    # Row 0: tie (0,0)↔(0,2), (0,1)↔(0,3)  = 2 ties
    # Row 1: tie (1,0)↔(1,2), (1,1)↔(1,3)  = 2 ties
    # Row 2: tie (2,0)↔(2,2), (2,1)↔(2,3)  = 2 ties
    # Row 3: tie (3,0)↔(3,2), (3,1)↔(3,3)  = 2 ties
    # Total: 8 soft ties
    expected3 = 8
    print(f"  Expected: {expected3} soft ties (8 pixel pairs)")
    print(f"  Actual: {len(builder3.soft_ties)}")
    assert len(builder3.soft_ties) == expected3

    print("\n" + "=" * 70)
    print("✓ S3 builder self-test passed.")
