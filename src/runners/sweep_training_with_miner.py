"""
Training sweep with law miner: end-to-end validation.

This module sweeps through all ARC training tasks, applies the law miner,
validates results with the kernel, and stores successful configurations in
the catalog. Failed tasks are logged with diagnostics for later analysis.

This is pure orchestration - no law logic, no fallbacks, no patching.
It simply records ground truth about which tasks are solved by the current
miner implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

from src.schemas.context import load_arc_task, build_task_context_from_raw, TaskContext
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.catalog.types import TaskLawConfig
from src.catalog.store import save_task_law_config


def load_training_task_ids(challenges_path: Path) -> List[str]:
    """
    Load all task_ids from the ARC-AGI training challenges JSON.

    Args:
        challenges_path: Path to arc-agi_training_challenges.json

    Returns:
        Sorted list of task IDs (strings)

    Example:
        >>> path = Path("data/arc-agi_training_challenges.json")
        >>> task_ids = load_training_task_ids(path)
        >>> len(task_ids)  # typically 400 training tasks
        400
    """
    with challenges_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # keys are the task ids; keep as strings, return sorted
    return sorted(list(data.keys()))


def sweep_training_with_miner(
    challenges_path: Path,
    failures_log_path: Path,
    validate_test_labels: bool = False,
    solutions_path: Path | None = None,
    task_list_path: Path | None = None,
) -> None:
    """
    Run law mining + kernel validation over all training tasks.

    For each task_id:
      - build TaskContext,
      - mine TaskLawConfig,
      - run kernel with training labels (and optionally test labels),
      - store law_config in catalog if status == "ok",
      - otherwise log diagnostics for later inspection.

    This function does NOT adjust or patch law_config; it only records ground truth
    about which tasks are solved under current miners.

    Args:
        challenges_path: Path to arc-agi_training_challenges.json
        failures_log_path: Path to write failures log (JSONL format)
        validate_test_labels: If True, also compare test outputs with ground truth
        solutions_path: Path to arc-agi_training_solutions.json (required if validate_test_labels=True)

    Output:
        - Successful configs saved to: catalog/tasks/{task_id}.json
        - Failed tasks logged to: failures_log_path (one JSON object per line)
        - Status values: "ok", "mismatch_train", "mismatch_test", "infeasible", "error"

    Example:
        >>> challenges = Path("data/arc-agi_training_challenges.json")
        >>> solutions = Path("data/arc-agi_training_solutions.json")
        >>> failures = Path("logs/miner_training_failures.jsonl")
        >>> sweep_training_with_miner(challenges, failures, validate_test_labels=True, solutions_path=solutions)
        # Processes all 1000 training tasks with test validation
    """
    # Load task IDs (from custom list or all training tasks)
    if task_list_path is not None:
        # Load custom task list from JSON
        with task_list_path.open("r", encoding="utf-8") as f:
            task_list_data = json.load(f)
        task_ids = task_list_data.get("task_ids", [])
        sweep_name = f"Mini Sweep ({task_list_path.name})"
    else:
        # Load all task IDs from challenges
        task_ids = load_training_task_ids(challenges_path)
        sweep_name = "Training Sweep with Law Miner"

    print("=" * 70)
    print(sweep_name)
    print("=" * 70)
    print(f"Total tasks: {len(task_ids)}")
    print(f"Challenges: {challenges_path}")
    print(f"Failures log: {failures_log_path}")
    if task_list_path:
        print(f"Task list: {task_list_path}")
    print("=" * 70)

    # Create failures log directory if needed
    failures_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Counters for summary
    num_success = 0
    num_failures = 0
    num_errors = 0

    # Open failures log for writing (JSONL format)
    with failures_log_path.open("w", encoding="utf-8") as log_f:
        for i, task_id in enumerate(task_ids, 1):
            print(f"\n[{i}/{len(task_ids)}] Processing task: {task_id}")

            try:
                # 1) Load raw task and build TaskContext
                raw_task = load_arc_task(task_id, challenges_path)
                task_context: TaskContext = build_task_context_from_raw(raw_task)

                # 2) Mine laws
                law_config: TaskLawConfig = mine_law_config(task_context)

                print(f"  Mined {len(law_config.schema_instances)} schema instances")

                # 3) Validate with kernel (using training labels and optionally test labels)
                outputs, diagnostics = solve_arc_task_with_diagnostics(
                    task_id=task_id,
                    law_config=law_config,
                    use_training_labels=True,
                    use_test_labels=validate_test_labels,
                    challenges_path=challenges_path,
                    solutions_path=solutions_path,
                )

                print(f"  Status: {diagnostics.status}")
                print(f"  Solver: {diagnostics.solver_status}")
                print(f"  Constraints: {diagnostics.num_constraints}")
                print(f"  Variables: {diagnostics.num_variables}")

                # 4) Branch on diagnostics.status
                if diagnostics.status == "ok":
                    # Store successful law_config in catalog
                    save_task_law_config(task_id, law_config)
                    print(f"  ✓ Saved to catalog")
                    num_success += 1
                else:
                    # Log failure diagnostics
                    failure_record = {
                        "task_id": task_id,
                        "status": diagnostics.status,
                        "solver_status": diagnostics.solver_status,
                        "num_constraints": diagnostics.num_constraints,
                        "num_variables": diagnostics.num_variables,
                        "schema_ids_used": diagnostics.schema_ids_used,
                        "train_mismatches": diagnostics.train_mismatches,
                        "test_mismatches": diagnostics.test_mismatches,
                        "error_message": diagnostics.error_message,
                    }
                    log_f.write(json.dumps(failure_record) + "\n")
                    log_f.flush()  # Ensure written immediately
                    print(f"  ✗ Logged to failures")
                    num_failures += 1

            except Exception as e:
                # Hard failure (unexpected exception) – log and continue
                print(f"  ✗ Exception: {str(e)}")

                failure_record = {
                    "task_id": task_id,
                    "status": "error",
                    "solver_status": "EXCEPTION",
                    "num_constraints": 0,
                    "num_variables": 0,
                    "schema_ids_used": [],
                    "train_mismatches": [],
                    "error_message": str(e),
                }
                log_f.write(json.dumps(failure_record) + "\n")
                log_f.flush()
                num_errors += 1

    # Print summary
    print("\n" + "=" * 70)
    print("Training Sweep Complete")
    print("=" * 70)
    print(f"Total tasks: {len(task_ids)}")
    print(f"  Success: {num_success}")
    print(f"  Failures: {num_failures}")
    print(f"  Errors: {num_errors}")
    print(f"\nSuccessful configs saved to: catalog/tasks/")
    print(f"Failures logged to: {failures_log_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run law miner sweep over ARC tasks")
    parser.add_argument(
        "--task-list",
        type=str,
        default=None,
        help="Path to JSON file with custom task list (e.g., data/mini_sweep.json)"
    )
    args = parser.parse_args()

    # Default paths for CLI run
    challenges_path = Path("data/arc-agi_training_challenges.json")
    failures_log_path = Path("logs/miner_training_failures.jsonl")

    # Parse task list path
    task_list_path = Path(args.task_list) if args.task_list else None

    # Toggle for train-only vs train+test validation
    validate_test = True  # Set to False for train-only runs

    # Run the sweep
    sweep_training_with_miner(
        challenges_path=challenges_path,
        failures_log_path=failures_log_path,
        validate_test_labels=validate_test,
        solutions_path=Path("data/arc-agi_training_solutions.json") if validate_test else None,
        task_list_path=task_list_path,
    )
