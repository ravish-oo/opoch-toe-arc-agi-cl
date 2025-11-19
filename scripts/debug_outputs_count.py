"""
Debug how many outputs are predicted vs expected.
"""

import sys
import json
from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.solver.lp_solver import solve_constraints_for_grid
from src.solver.decoding import y_to_grid


def debug_outputs(task_id: str):
    """Check how many outputs are generated."""
    print(f"Debugging outputs for task: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    # Load
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(task_context)

    print(f"Test examples in context: {len(task_context.test_examples)}")

    # Load solutions
    with solutions_path.open("r") as f:
        solutions_data = json.load(f)

    true_test_outputs = solutions_data[task_id]
    print(f"True test outputs in solutions: {len(true_test_outputs)}")

    # Try to solve test examples
    test_outputs_pred = []

    for i, ex in enumerate(task_context.test_examples):
        print(f"\nSolving test example {i}...")

        # Determine output dimensions
        if ex.output_H is not None:
            H_out, W_out = ex.output_H, ex.output_W
        else:
            H_out, W_out = ex.input_H, ex.input_W

        print(f"  Output dimensions: {H_out}x{W_out}")

        num_pixels = H_out * W_out
        num_colors = task_context.C

        # Build constraints
        builder = ConstraintBuilder()
        for schema_inst in law_config.schema_instances:
            apply_schema_instance(
                family_id=schema_inst.family_id,
                schema_params=schema_inst.params,
                task_context=task_context,
                builder=builder,
                example_type="test",
                example_index=i,
                schema_constraint_counts={},
            )

        print(f"  Constraints: {len(builder.constraints)}")

        # Solve
        try:
            y, solver_status = solve_constraints_for_grid(
                builder=builder,
                num_pixels=num_pixels,
                num_colors=num_colors,
                objective="min_sum"
            )
            print(f"  Solver status: {solver_status}")

            grid_pred = y_to_grid(y, H_out, W_out, num_colors)
            test_outputs_pred.append(grid_pred)
            print(f"  ✓ Predicted output: {grid_pred.shape}")

        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")
            # Don't append anything - this is the bug!

    print(f"\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Expected test outputs: {len(true_test_outputs)}")
    print(f"  Predicted test outputs: {len(test_outputs_pred)}")
    print(f"  Match: {len(true_test_outputs) == len(test_outputs_pred)}")

    if len(true_test_outputs) != len(test_outputs_pred):
        print(f"\n  ✗ BUG: Kernel will try to access test_outputs_pred[{len(true_test_outputs)-1}]")
        print(f"         but list only has {len(test_outputs_pred)} elements!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.debug_outputs_count <task_id>")
        sys.exit(1)

    task_id = sys.argv[1]
    debug_outputs(task_id)
