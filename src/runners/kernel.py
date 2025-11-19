"""
Core kernel runner for ARC-AGI constraint solver.

This module provides the main entrypoint for solving ARC tasks using
the constraint-based approach:
  1. Load task data
  2. Build TaskContext with all φ features (M1)
  3. Apply schema instances to generate constraints (M3)
  4. Solve LP/ILP per example (M4.1)
  5. Decode y → grids (M4.2)
  6. Return diagnostics for Pi-agent debugging (M5)

This is the complete math kernel pipeline from the spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.core.grid_types import Grid
from src.schemas.context import load_arc_task, build_task_context_from_raw, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import TaskLawConfig
from src.solver.lp_solver import solve_constraints_for_grid, InfeasibleModelError, TaskSolveError
from src.solver.decoding import y_to_grid
from src.runners.results import SolveDiagnostics, compute_train_mismatches, compute_grid_mismatches, ExampleSummary
from src.features.components import connected_components_by_color


def solve_arc_task_with_diagnostics(
    task_id: str,
    law_config: TaskLawConfig,
    use_training_labels: bool = False,
    use_test_labels: bool = False,
    challenges_path: Path | None = None,
    solutions_path: Path | None = None,
) -> Tuple[Dict[str, List[Grid]], SolveDiagnostics]:
    """
    Solve an ARC task and return both outputs and comprehensive diagnostics.

    This is the Pi-agent-friendly entrypoint that returns rich diagnostics about
    the solve attempt, including solver status, constraint counts, and detailed
    mismatch information for training and test examples.

    Args:
        task_id: ARC task identifier from challenges JSON
        law_config: TaskLawConfig with schema instances to apply
        use_training_labels: If True, compare predicted train outputs with
                            ground truth and populate mismatch information
        use_test_labels: If True, compare predicted test outputs with ground truth
                        from solutions file (requires solutions_path)
        challenges_path: Optional path to challenges JSON (defaults to training set)
        solutions_path: Optional path to solutions JSON (required if use_test_labels=True)

    Returns:
        Tuple of (outputs, diagnostics):
          - outputs: {
              "train": [Grid, ...],  # predicted train outputs
              "test": [Grid, ...]    # predicted test outputs
            }
          - diagnostics: SolveDiagnostics with:
              - status: "ok" | "infeasible" | "mismatch_train" | "mismatch_test" | "error"
              - solver_status: PuLP status string
              - num_constraints, num_variables
              - schema_ids_used
              - train_mismatches (if use_training_labels=True)
              - test_mismatches (if use_test_labels=True)
              - error_message (if status="error")

    Example:
        >>> config = TaskLawConfig(schema_instances=[...])
        >>> outputs, diag = solve_arc_task_with_diagnostics(
        ...     "00576224", config, use_training_labels=True, use_test_labels=True,
        ...     solutions_path=Path("data/arc-agi_training_solutions.json")
        ... )
        >>> if diag.status == "mismatch_test":
        ...     print(f"Test mismatches: {diag.test_mismatches}")
    """
    # Default to training challenges if not specified
    if challenges_path is None:
        challenges_path = Path("data/arc-agi_training_challenges.json")

    # Initialize diagnostics tracking
    train_outputs_pred: List[Grid] = []
    test_outputs_pred: List[Grid] = []
    total_constraints = 0
    total_variables = 0
    schema_ids_used = [inst.family_id for inst in law_config.schema_instances]
    status = "ok"
    solver_status_str = "Unknown"
    train_mismatches = []
    test_mismatches = []
    error_message: str | None = None

    # M5.X: Per-schema constraint counts and example summaries
    schema_constraint_counts: Dict[str, int] = {}
    example_summaries: List[ExampleSummary] = []

    try:
        # 1. Load task and build context
        task_data = load_arc_task(task_id, challenges_path)
        ctx: TaskContext = build_task_context_from_raw(task_data)

        # M5.X: Build example summaries for all train and test examples
        def components_per_color(grid: Grid) -> Dict[int, int]:
            """Count components per color in a grid."""
            comps = connected_components_by_color(grid)
            counts: Dict[int, int] = {}
            for comp in comps:
                counts[comp.color] = counts.get(comp.color, 0) + 1
            return counts

        # Summarize train examples
        for ex in ctx.train_examples:
            example_summaries.append(ExampleSummary(
                input_shape=tuple(ex.input_grid.shape),
                output_shape=tuple(ex.output_grid.shape) if ex.output_grid is not None else None,
                components_per_color=components_per_color(ex.input_grid)
            ))

        # Summarize test examples
        for ex in ctx.test_examples:
            example_summaries.append(ExampleSummary(
                input_shape=tuple(ex.input_grid.shape),
                output_shape=None,  # Test examples have no ground truth output
                components_per_color=components_per_color(ex.input_grid)
            ))

        # 2. Solve for each TRAIN example
        for i, ex in enumerate(ctx.train_examples):
            # Determine output dimensions
            H_out = ex.output_H if ex.output_H is not None else ex.input_H
            W_out = ex.output_W if ex.output_W is not None else ex.input_W
            num_pixels = H_out * W_out
            num_colors = ctx.C

            # Build constraints for this example
            builder = ConstraintBuilder()
            for schema_inst in law_config.schema_instances:
                apply_schema_instance(
                    family_id=schema_inst.family_id,
                    schema_params=schema_inst.params,
                    task_context=ctx,
                    builder=builder,
                    example_type="train",
                    example_index=i,
                    schema_constraint_counts=schema_constraint_counts,
                )

            # Track constraint/variable counts
            total_constraints += len(builder.constraints)
            total_variables += num_pixels * num_colors

            # Solve ILP
            y, solver_status_single = solve_constraints_for_grid(
                builder=builder,
                num_pixels=num_pixels,
                num_colors=num_colors,
                objective="min_sum"
            )
            solver_status_str = solver_status_single  # Keep last status

            # Decode to grid
            grid_pred = y_to_grid(y, H_out, W_out, num_colors)
            train_outputs_pred.append(grid_pred)

        # 3. Solve for each TEST example (with per-example exception handling)
        for i, ex in enumerate(ctx.test_examples):
            try:
                # Determine output dimensions
                if ex.output_H is not None:
                    H_out, W_out = ex.output_H, ex.output_W
                else:
                    # Fallback to input dimensions (works for geometry-preserving schemas)
                    H_out, W_out = ex.input_H, ex.input_W

                num_pixels = H_out * W_out
                num_colors = ctx.C

                # Build constraints for this example
                builder = ConstraintBuilder()
                for schema_inst in law_config.schema_instances:
                    apply_schema_instance(
                        family_id=schema_inst.family_id,
                        schema_params=schema_inst.params,
                        task_context=ctx,
                        builder=builder,
                        example_type="test",
                        example_index=i,
                        schema_constraint_counts=schema_constraint_counts,
                    )

                # Track constraint/variable counts
                total_constraints += len(builder.constraints)
                total_variables += num_pixels * num_colors

                # Solve ILP
                y, solver_status_single = solve_constraints_for_grid(
                    builder=builder,
                    num_pixels=num_pixels,
                    num_colors=num_colors,
                    objective="min_sum"
                )
                solver_status_str = solver_status_single

                # Decode to grid
                grid_pred = y_to_grid(y, H_out, W_out, num_colors)
                test_outputs_pred.append(grid_pred)

            except Exception as e:
                # Test example solve failed - cannot produce complete test outputs
                status = "error"
                error_message = f"Test solve failed for example {i}: {type(e).__name__}: {e}"
                # Break test loop - cannot reliably compare test outputs
                break

        # 4. Compare with training labels if requested
        if use_training_labels:
            # Extract ground truth from context
            true_train_outputs = [
                ex.output_grid for ex in ctx.train_examples
                if ex.output_grid is not None
            ]

            # Compute mismatches
            train_mismatches = compute_train_mismatches(
                true_train_outputs,
                train_outputs_pred
            )

        # Compute test mismatches if requested
        # Only proceed if we're not already in error/infeasible state
        if use_test_labels and status not in ("error", "infeasible"):
            if solutions_path is None:
                raise ValueError("solutions_path is required when use_test_labels=True")

            # Load test solutions from solutions JSON
            with solutions_path.open("r", encoding="utf-8") as f:
                solutions_data = json.load(f)

            if task_id not in solutions_data:
                raise ValueError(f"Task {task_id} not found in solutions file")

            true_test_outputs_raw = solutions_data[task_id]  # List of 2D grid arrays

            # Convert to numpy arrays
            true_test_outputs = [np.array(grid, dtype=int) for grid in true_test_outputs_raw]

            # Validate we have complete test predictions before comparing
            if len(test_outputs_pred) != len(true_test_outputs):
                # Cannot compare - incomplete test predictions (likely due to solve failure)
                status = "error"
                error_message = (
                    f"Number of test predictions ({len(test_outputs_pred)}) "
                    f"does not match number of true test outputs ({len(true_test_outputs)}). "
                    f"Test solving may have failed."
                )
            else:
                # Safe to compare - we have complete test predictions
                for ex_idx in range(len(true_test_outputs)):
                    true_grid = true_test_outputs[ex_idx]
                    pred_grid = test_outputs_pred[ex_idx]

                    mismatches = compute_grid_mismatches(true_grid, pred_grid)

                    if mismatches:
                        test_mismatches.append({
                            "example_idx": ex_idx,
                            "diff_cells": mismatches
                        })

        # Set status based on mismatches (priority order)
        # 1. Error/infeasible status already set above or in exception handlers
        # 2. Train mismatch takes precedence over test mismatch
        # 3. Test mismatch only if train is clean
        # 4. "ok" only if no errors and all comparisons pass
        if status not in ("error", "infeasible"):
            # Only update status if not already in error/infeasible state
            if use_training_labels and len(train_mismatches) > 0:
                status = "mismatch_train"
            elif use_test_labels and len(test_mismatches) > 0:
                status = "mismatch_test"
            else:
                status = "ok"

    except InfeasibleModelError as e:
        # Solver couldn't find a solution
        status = "infeasible"
        solver_status_str = str(e)
        error_message = f"ILP infeasible: {e}"

    except TaskSolveError as e:
        # Task-level solve error (wraps InfeasibleModelError with context)
        status = "infeasible"
        error_message = f"Task solve error: {e.example_type}[{e.example_index}]: {e.original_error}"

    except Exception as e:
        # Unexpected error
        status = "error"
        error_message = f"Unexpected error: {type(e).__name__}: {e}"

    # 5. Construct diagnostics
    diagnostics = SolveDiagnostics(
        task_id=task_id,
        law_config=law_config,
        status=status,
        solver_status=solver_status_str,
        num_constraints=total_constraints,
        num_variables=total_variables,
        schema_ids_used=schema_ids_used,
        train_mismatches=train_mismatches,
        test_mismatches=test_mismatches,
        schema_constraint_counts=schema_constraint_counts,
        example_summaries=example_summaries,
        error_message=error_message,
    )

    # 6. Return outputs and diagnostics
    outputs = {
        "train": train_outputs_pred,
        "test": test_outputs_pred,
    }

    return outputs, diagnostics


def solve_arc_task(
    task_id: str,
    law_config: TaskLawConfig,
    challenges_path: Path = None
) -> Dict[str, List[Grid]]:
    """
    Solve an ARC-AGI task using a given law configuration.

    This is a thin wrapper around solve_arc_task_with_diagnostics that
    returns only the predicted grids (for backward compatibility).
    For Pi-agent integration, use solve_arc_task_with_diagnostics directly.

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
    # Call new diagnostics version and extract just the outputs
    outputs, _ = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=False,
        challenges_path=challenges_path,
    )

    # Convert to old output format for backward compatibility
    return {
        "train_outputs_pred": outputs["train"],
        "test_outputs_pred": outputs["test"],
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
