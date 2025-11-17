"""
Core kernel runner for ARC-AGI constraint solver.

This module provides the main entrypoint for solving ARC tasks using
the constraint-based approach:
  1. Load task data
  2. Build TaskContext with all φ features (M1)
  3. Apply schema instances to generate constraints (M3)
  4. Solve LP/ILP per example (M4.1)
  5. Decode y → grids (M4.2)

This is the complete math kernel pipeline from the spec.
"""

from pathlib import Path
from typing import Dict, List

from src.core.grid_types import Grid
from src.schemas.context import load_arc_task, build_task_context_from_raw, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import TaskLawConfig
from src.solver.lp_solver import solve_constraints_for_grid, InfeasibleModelError, TaskSolveError
from src.solver.decoding import y_to_grid


def solve_arc_task(
    task_id: str,
    law_config: TaskLawConfig,
    challenges_path: Path = None
) -> Dict[str, List[Grid]]:
    """
    Solve an ARC-AGI task using a given law configuration.

    This implements the full math kernel pipeline:
      1. Load task from JSON
      2. Build TaskContext with all φ features
      3. For each example (train and test):
         a. Create ConstraintBuilder
         b. Apply all schema instances
         c. Solve ILP to get y
         d. Decode y → grid
      4. Return predicted grids

    Args:
        task_id: ARC task identifier
        law_config: TaskLawConfig listing which schema instances to apply
        challenges_path: Optional path to challenges JSON (defaults to standard location)

    Returns:
        Dictionary with:
          {
            "train_outputs_pred": [Grid, ...],  # predicted train outputs
            "test_outputs_pred":  [Grid, ...]   # predicted test outputs
          }

    Raises:
        TaskSolveError: If solving fails for any example (wraps InfeasibleModelError)

    Example:
        >>> from src.catalog.types import SchemaInstance, TaskLawConfig
        >>> config = TaskLawConfig(schema_instances=[
        ...     SchemaInstance("S1", {"ties": [...]})
        ... ])
        >>> result = solve_arc_task("00576224", config)
        >>> result["train_outputs_pred"]
        [<Grid>, <Grid>]
    """
    # 1. Load task
    if challenges_path is None:
        challenges_path = Path("data/arc-agi_training_challenges.json")

    task_data = load_arc_task(task_id, challenges_path)

    # 2. Build TaskContext
    ctx: TaskContext = build_task_context_from_raw(task_data)

    # 3. Prepare result containers
    train_outputs_pred: List[Grid] = []
    test_outputs_pred: List[Grid] = []

    # 4. Solve for each TRAIN example
    for i, ex in enumerate(ctx.train_examples):
        # Output dimensions from ground truth
        H_out = ex.output_H if ex.output_H is not None else ex.input_H
        W_out = ex.output_W if ex.output_W is not None else ex.input_W
        num_pixels = H_out * W_out
        num_colors = ctx.C

        # Create builder for this example
        builder = ConstraintBuilder()

        # Apply all schema instances
        for schema_instance in law_config.schema_instances:
            apply_schema_instance(
                family_id=schema_instance.family_id,
                schema_params=schema_instance.params,
                task_context=ctx,
                builder=builder,
                example_type="train",
                example_index=i
            )

        # Solve ILP for this example
        try:
            y = solve_constraints_for_grid(
                builder,
                num_pixels=num_pixels,
                num_colors=num_colors,
                objective="min_sum"
            )
        except InfeasibleModelError as e:
            raise TaskSolveError(task_id, "train", i, e)

        # Decode to grid
        grid_pred = y_to_grid(y, H_out, W_out, num_colors)
        train_outputs_pred.append(grid_pred)

    # 5. Solve for each TEST example
    for i, ex in enumerate(ctx.test_examples):
        # Output dimensions: from schema params (for S6/S7) or fallback to input dims
        # For geometry-preserving schemas (S1-S5, S8-S11), input dims = output dims
        # For S6/S7, law_config must specify output dimensions in params
        if ex.output_H is not None:
            H_out, W_out = ex.output_H, ex.output_W
        else:
            # Fallback to input dimensions (works for geometry-preserving)
            H_out, W_out = ex.input_H, ex.input_W

        num_pixels = H_out * W_out
        num_colors = ctx.C

        # Create builder for this example
        builder = ConstraintBuilder()

        # Apply all schema instances
        for schema_instance in law_config.schema_instances:
            apply_schema_instance(
                family_id=schema_instance.family_id,
                schema_params=schema_instance.params,
                task_context=ctx,
                builder=builder,
                example_type="test",
                example_index=i
            )

        # Solve ILP for this example
        try:
            y = solve_constraints_for_grid(
                builder,
                num_pixels=num_pixels,
                num_colors=num_colors,
                objective="min_sum"
            )
        except InfeasibleModelError as e:
            raise TaskSolveError(task_id, "test", i, e)

        # Decode to grid
        grid_pred = y_to_grid(y, H_out, W_out, num_colors)
        test_outputs_pred.append(grid_pred)

    # 6. Return results
    return {
        "train_outputs_pred": train_outputs_pred,
        "test_outputs_pred": test_outputs_pred,
    }


if __name__ == "__main__":
    # Self-test with toy law config
    from src.catalog.types import SchemaInstance, TaskLawConfig

    print("Testing kernel runner with full solver integration...")
    print("=" * 70)

    # Use a simple task and S1 (copy tie)
    task_id = "00576224"

    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={
                "ties": [{
                    "pairs": [((0, 0), (0, 1))]  # Tie top-left to top-right
                }]
            }
        )
    ])

    print(f"Task ID: {task_id}")
    print(f"Law config: {len(law_config.schema_instances)} schema instances")

    try:
        result = solve_arc_task(task_id, law_config)

        print(f"\n✓ Kernel solved successfully!")
        print(f"  Train outputs predicted: {len(result['train_outputs_pred'])}")
        print(f"  Test outputs predicted: {len(result['test_outputs_pred'])}")

        # Show first train output shape
        if result["train_outputs_pred"]:
            grid0 = result["train_outputs_pred"][0]
            print(f"\n  First train output shape: {grid0.shape}")
            print(f"  First train output:\n{grid0}")

    except TaskSolveError as e:
        print(f"\n✗ Task solve failed: {e}")
        print(f"  Task: {e.task_id}")
        print(f"  Example: {e.example_type}[{e.example_index}]")
        print(f"  Reason: {e.original_error}")

    print("\n" + "=" * 70)
    print("✓ Kernel runner self-test complete.")
