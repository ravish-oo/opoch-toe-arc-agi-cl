"""
Example script showing how to deeply inspect a mismatch task.

This demonstrates accessing all available diagnostic information
for a task with status mismatch_train or mismatch_test.

Usage:
    PYTHONPATH=/Users/ravishq/code/opoch-toe-arc-agi-cl python scripts/inspect_mismatch_detailed.py <task_id>
"""

import sys
import json
from pathlib import Path
import numpy as np

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics


def inspect_mismatch(task_id: str):
    """Deep inspection of mismatch diagnostics for a task."""
    print("=" * 70)
    print(f"DETAILED MISMATCH INSPECTION: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    # 1. Solve with full diagnostics
    print("\n[1] Solving task with diagnostics...")
    raw_task = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(ctx)

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        use_test_labels=True,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
    )

    # 2. Display high-level diagnostics
    print(f"    Status: {diagnostics.status}")
    print(f"    Solver: {diagnostics.solver_status}")
    print(f"    Schemas applied: {diagnostics.schema_ids_used}")
    print(f"    Total constraints: {diagnostics.num_constraints}")
    print(f"    Total variables: {diagnostics.num_variables}")

    # 3. Schema-level breakdown
    print(f"\n[2] Schema constraint counts:")
    for schema, count in sorted(diagnostics.schema_constraint_counts.items()):
        pct = 100 * count / diagnostics.num_constraints if diagnostics.num_constraints > 0 else 0
        print(f"    {schema}: {count:4d} constraints ({pct:5.1f}%)")

    # 4. Example summaries
    print(f"\n[3] Example summaries:")
    for i, summary in enumerate(diagnostics.example_summaries):
        ex_type = "train" if i < len(ctx.train_examples) else "test"
        ex_idx = i if i < len(ctx.train_examples) else i - len(ctx.train_examples)
        print(f"    {ex_type}[{ex_idx}]: input={summary.input_shape}, output={summary.output_shape}")
        print(f"             components per color: {summary.components_per_color}")

    # 5. Train mismatches (if any)
    if diagnostics.train_mismatches:
        print(f"\n[4] TRAIN MISMATCHES: {len(diagnostics.train_mismatches)} example(s)")
        print("-" * 70)

        for mm in diagnostics.train_mismatches:
            ex_idx = mm["example_idx"]
            diff_cells = mm["diff_cells"]

            print(f"\n  Train example {ex_idx}:")

            # Check for shape mismatch
            if diff_cells and "shape_mismatch" in diff_cells[0]:
                print(f"    ✗ SHAPE MISMATCH")
                print(f"      True shape: {diff_cells[0]['true_shape']}")
                print(f"      Pred shape: {diff_cells[0]['pred_shape']}")
            else:
                print(f"    Mismatched cells: {len(diff_cells)}")

                # Show grids
                true_grid = ctx.train_examples[ex_idx].output_grid
                pred_grid = outputs["train"][ex_idx]

                print(f"\n    Ground truth grid:")
                print(f"{true_grid}")

                print(f"\n    Predicted grid:")
                print(f"{pred_grid}")

                # Show first 20 cell-level diffs
                print(f"\n    Cell-level differences (showing first 20):")
                for cell in diff_cells[:20]:
                    print(f"      ({cell['r']:2d}, {cell['c']:2d}): true={cell['true']} → pred={cell['pred']}")

                if len(diff_cells) > 20:
                    print(f"      ... and {len(diff_cells) - 20} more")

    # 6. Test mismatches (if any)
    if diagnostics.test_mismatches:
        print(f"\n[5] TEST MISMATCHES: {len(diagnostics.test_mismatches)} example(s)")
        print("-" * 70)

        # Load ground truth test outputs
        with solutions_path.open("r", encoding="utf-8") as f:
            solutions = json.load(f)
        true_test_grids = [np.array(g, dtype=int) for g in solutions[task_id]]

        for mm in diagnostics.test_mismatches:
            ex_idx = mm["example_idx"]
            diff_cells = mm["diff_cells"]

            print(f"\n  Test example {ex_idx}:")

            # Check for shape mismatch
            if diff_cells and "shape_mismatch" in diff_cells[0]:
                print(f"    ✗ SHAPE MISMATCH")
                print(f"      True shape: {diff_cells[0]['true_shape']}")
                print(f"      Pred shape: {diff_cells[0]['pred_shape']}")
            else:
                print(f"    Mismatched cells: {len(diff_cells)}")

                # Show grids
                true_grid = true_test_grids[ex_idx]
                pred_grid = outputs["test"][ex_idx]

                print(f"\n    Ground truth grid:")
                print(f"{true_grid}")

                print(f"\n    Predicted grid:")
                print(f"{pred_grid}")

                # Show first 20 cell-level diffs
                print(f"\n    Cell-level differences (showing first 20):")
                for cell in diff_cells[:20]:
                    print(f"      ({cell['r']:2d}, {cell['c']:2d}): true={cell['true']} → pred={cell['pred']}")

                if len(diff_cells) > 20:
                    print(f"      ... and {len(diff_cells) - 20} more")

    # 7. Summary
    print("\n" + "=" * 70)
    if diagnostics.status == "ok":
        print("✓ TASK STATUS: OK (all outputs match)")
    elif diagnostics.status == "mismatch_train":
        print("✗ TASK STATUS: MISMATCH_TRAIN (train outputs don't match)")
    elif diagnostics.status == "mismatch_test":
        print("✓ Train outputs match, ✗ test outputs don't match")
    elif diagnostics.status == "infeasible":
        print("✗ TASK STATUS: INFEASIBLE (solver couldn't find solution)")
        print(f"  {diagnostics.error_message}")
    else:  # error
        print("✗ TASK STATUS: ERROR")
        print(f"  {diagnostics.error_message}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_mismatch_detailed.py <task_id>")
        print("Example: python scripts/inspect_mismatch_detailed.py e872b94a")
        sys.exit(1)

    task_id = sys.argv[1]
    inspect_mismatch(task_id)
