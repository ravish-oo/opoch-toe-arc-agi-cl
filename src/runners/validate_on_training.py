"""
Training-set validation runner for ARC-AGI tasks.

This script runs the full kernel (schemas + constraints + solver + decoding)
on a training task and validates predictions against known solutions:
  - Train outputs: compared to ground truth from challenges file
  - Test outputs: compared to known solutions from solutions file

Usage:
    python -m src.runners.validate_on_training <task_id>
    python -m src.runners.validate_on_training 00576224 --challenges_path data/arc-agi_training_challenges.json

This is the core evaluation loop that Pi-agents will use to validate
law configs and identify mismatches for refinement.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.runners.kernel import solve_arc_task
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.schemas.context import load_arc_task
from src.core.arc_io import load_arc_training_solutions
from src.core.grid_types import Grid
from src.solver.lp_solver import InfeasibleModelError, TaskSolveError


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for validation runner.

    Returns:
        Parsed arguments with task_id, challenges_path, and solutions_path
    """
    parser = argparse.ArgumentParser(
        description="Validate law config on ARC-AGI training task."
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID from arc-agi_training_challenges.json"
    )
    parser.add_argument(
        "--challenges_path",
        type=Path,
        default=Path("data/arc-agi_training_challenges.json"),
        help="Path to arc-agi_training_challenges.json"
    )
    parser.add_argument(
        "--solutions_path",
        type=Path,
        default=Path("data/arc-agi_training_solutions.json"),
        help="Path to arc-agi_training_solutions.json"
    )
    return parser.parse_args()


def make_law_config_for_task(task_id: str) -> TaskLawConfig:
    """
    Construct a TaskLawConfig for this task.

    For now, returns a minimal working S1 configuration (same as test_kernel_smoke.py).
    Later, this will be replaced by:
      - Catalog lookup for hand-crafted configs
      - Pi-agent-generated configs based on law mining

    Args:
        task_id: ARC task identifier (unused for now, but will be used for lookup)

    Returns:
        TaskLawConfig with schema instances to apply
    """
    # Minimal working S1 config for immediate runnability
    return TaskLawConfig(
        schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={
                    "ties": [{
                        "pairs": [((0, 0), (0, 1))]  # Tie top-left to top-right
                    }]
                }
            )
        ]
    )


def get_true_train_grids(raw_task: Dict[str, Any]) -> List[Grid]:
    """
    Extract true training output grids from normalized raw_task.

    The raw_task comes from load_arc_task() which returns:
      {"train": [{"input": Grid, "output": Grid}], "test": [...]}

    Args:
        raw_task: Normalized task data from load_arc_task()

    Returns:
        List of true output grids for training examples
    """
    return [pair["output"] for pair in raw_task.get("train", [])]


def get_true_test_grids(task_id: str, solutions: Dict[str, List[Grid]]) -> List[Grid]:
    """
    Get true test output grids from solutions mapping.

    Args:
        task_id: ARC task identifier
        solutions: Dict mapping task_id -> list of test output grids

    Returns:
        List of true test output grids, or [] if not present
    """
    return solutions.get(task_id, [])


def compare_grids(pred: Grid, true: Grid) -> Dict[str, Any]:
    """
    Compare two grids and return detailed mismatch summary.

    Args:
        pred: Predicted grid
        true: True/expected grid

    Returns:
        Dictionary with:
          - match: bool (True if grids are identical)
          - reason: str (explanation of match/mismatch)
          - diff_coords: list of (r, c) tuples where grids differ
    """
    # Check shape first
    if pred.shape != true.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: pred {pred.shape}, true {true.shape}",
            "diff_coords": []
        }

    # Check pixel values
    equal_mask = (pred == true)
    if equal_mask.all():
        return {
            "match": True,
            "reason": "exact_match",
            "diff_coords": []
        }

    # Collect differing coordinates
    diff_coords = [(int(r), int(c)) for r, c in zip(*np.where(~equal_mask))]
    return {
        "match": False,
        "reason": "value_mismatch",
        "diff_coords": diff_coords
    }


def validate_on_training(
    task_id: str,
    challenges_path: Path,
    solutions_path: Path
) -> None:
    """
    Run kernel on a training task and validate predictions against ground truth.

    This is the core evaluation function that:
      1. Loads task data and solutions
      2. Builds law config for this task
      3. Runs solver to get predictions
      4. Compares train and test outputs to ground truth
      5. Prints detailed mismatch report

    No silent failures: all solver errors are printed explicitly.

    Args:
        task_id: ARC task identifier
        challenges_path: Path to arc-agi_training_challenges.json
        solutions_path: Path to arc-agi_training_solutions.json
    """
    print(f"=" * 70)
    print(f"VALIDATING TASK: {task_id}")
    print(f"=" * 70)

    # 1. Load task data and solutions
    try:
        raw_task = load_arc_task(task_id, challenges_path)
    except KeyError as e:
        print(f"[ERROR] Task not found: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load task: {e}")
        return

    try:
        solutions = load_arc_training_solutions(solutions_path)
    except Exception as e:
        print(f"[ERROR] Failed to load solutions: {e}")
        return

    # 2. Extract true grids
    true_train_grids = get_true_train_grids(raw_task)
    true_test_grids = get_true_test_grids(task_id, solutions)

    print(f"\nTask structure:")
    print(f"  Train examples: {len(true_train_grids)}")
    print(f"  Test examples (with solutions): {len(true_test_grids)}")

    # 3. Build law config
    law_config = make_law_config_for_task(task_id)
    print(f"\nLaw config:")
    print(f"  Schema instances: {len(law_config.schema_instances)}")
    for si in law_config.schema_instances:
        print(f"    - {si.family_id}")

    # 4. Run solver
    print(f"\nRunning solver...")
    try:
        result = solve_arc_task(task_id, law_config, challenges_path)
    except InfeasibleModelError as e:
        print(f"[ERROR] Infeasible ILP for task {task_id}:")
        print(f"  {e}")
        return
    except TaskSolveError as e:
        print(f"[ERROR] Task solve failed:")
        print(f"  Task: {e.task_id}")
        print(f"  Example: {e.example_type}[{e.example_index}]")
        print(f"  Reason: {e.original_error}")
        return
    except Exception as e:
        print(f"[ERROR] solve_arc_task failed for task {task_id}:")
        print(f"  {e}")
        return

    pred_train = result.get("train_outputs_pred", [])
    pred_test = result.get("test_outputs_pred", [])

    print(f"✓ Solver completed successfully")
    print(f"  Train predictions: {len(pred_train)}")
    print(f"  Test predictions: {len(pred_test)}")

    # 5. Compare train grids
    print(f"\n" + "-" * 70)
    print(f"TRAIN VALIDATION")
    print("-" * 70)

    train_matches = 0
    for i, true_grid in enumerate(true_train_grids):
        if i >= len(pred_train):
            print(f"  [TRAIN {i}] ✗ No prediction produced.")
            continue

        summary = compare_grids(pred_train[i], true_grid)
        if summary["match"]:
            print(f"  [TRAIN {i}] ✓ OK (exact match)")
            train_matches += 1
        else:
            print(f"  [TRAIN {i}] ✗ MISMATCH: {summary['reason']}")
            if summary["diff_coords"]:
                # Show first 10 differing cells
                diff_preview = summary["diff_coords"][:10]
                print(f"             Differing cells (first 10): {diff_preview}")
                if len(summary["diff_coords"]) > 10:
                    print(f"             ... and {len(summary['diff_coords']) - 10} more")

    print(f"\nTrain accuracy: {train_matches}/{len(true_train_grids)}")

    # 6. Compare test grids if solutions available
    if true_test_grids:
        print(f"\n" + "-" * 70)
        print(f"TEST VALIDATION")
        print("-" * 70)

        test_matches = 0
        for j, true_grid in enumerate(true_test_grids):
            if j >= len(pred_test):
                print(f"  [TEST {j}] ✗ No prediction produced.")
                continue

            summary = compare_grids(pred_test[j], true_grid)
            if summary["match"]:
                print(f"  [TEST {j}] ✓ OK (exact match)")
                test_matches += 1
            else:
                print(f"  [TEST {j}] ✗ MISMATCH: {summary['reason']}")
                if summary["diff_coords"]:
                    diff_preview = summary["diff_coords"][:10]
                    print(f"            Differing cells (first 10): {diff_preview}")
                    if len(summary["diff_coords"]) > 10:
                        print(f"            ... and {len(summary['diff_coords']) - 10} more")

        print(f"\nTest accuracy: {test_matches}/{len(true_test_grids)}")
    else:
        print(f"\n" + "-" * 70)
        print(f"No test solutions found in {solutions_path} for task {task_id}")
        print("-" * 70)

    print(f"\n" + "=" * 70)
    print(f"VALIDATION COMPLETE")
    print("=" * 70)


def main():
    """CLI entry point."""
    args = parse_args()
    validate_on_training(args.task_id, args.challenges_path, args.solutions_path)


if __name__ == "__main__":
    main()
