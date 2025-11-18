"""
Diagnostic script for inspecting enriched solver diagnostics.

This script runs the kernel on a task and prints detailed diagnostics including:
  - Per-schema constraint counts (M5.X Part A)
  - Example summaries with shapes and component counts (M5.X Part B)

Usage:
    python -m src.runners.diagnose_task --task-id 00576224

    # With custom paths
    python -m src.runners.diagnose_task \
        --task-id 00576224 \
        --challenges-path data/arc-agi_training_challenges.json

Output:
    JSON-formatted diagnostics to stdout with:
      - task_id
      - status
      - num_constraints, num_variables
      - schema_ids_used
      - schema_constraint_counts: {schema_id: count, ...}
      - example_summaries: [{input_shape, output_shape, components_per_color}, ...]
      - train_mismatches (if training labels available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from src.catalog.store import load_task_law_config
from src.catalog.types import TaskLawConfig
from src.runners.kernel import solve_arc_task_with_diagnostics


def diagnose_task(
    task_id: str,
    challenges_path: Path,
    use_training_labels: bool = True,
) -> None:
    """
    Run solver diagnostics on a task and print enriched results as JSON.

    Args:
        task_id: ARC task identifier
        challenges_path: Path to challenges JSON
        use_training_labels: If True, compare with ground truth and report mismatches

    Output:
        Prints JSON diagnostics to stdout. Format:
        {
            "task_id": str,
            "status": "ok" | "infeasible" | "mismatch" | "error",
            "solver_status": str,
            "num_constraints": int,
            "num_variables": int,
            "schema_ids_used": [str, ...],
            "schema_constraint_counts": {schema_id: count},
            "example_summaries": [
                {
                    "input_shape": [H, W],
                    "output_shape": [H, W] or null,
                    "components_per_color": {color: count}
                },
                ...
            ],
            "train_mismatches": [...],
            "error_message": str or null
        }

    Example:
        >>> diagnose_task("00576224", Path("data/arc-agi_training_challenges.json"))
        {
          "task_id": "00576224",
          "status": "ok",
          "num_constraints": 42,
          "schema_constraint_counts": {"S1": 30, "S2": 12},
          "example_summaries": [
            {
              "input_shape": [5, 5],
              "output_shape": [5, 5],
              "components_per_color": {"0": 3, "1": 2}
            },
            ...
          ]
        }
    """
    # 1. Load existing law config for this task
    law_config: Optional[TaskLawConfig] = load_task_law_config(task_id)

    if law_config is None:
        print(json.dumps({
            "task_id": task_id,
            "status": "error",
            "error_message": f"No law config found for task {task_id}"
        }, indent=2))
        sys.exit(1)

    # 2. Run kernel with diagnostics
    try:
        outputs, diagnostics = solve_arc_task_with_diagnostics(
            task_id=task_id,
            law_config=law_config,
            use_training_labels=use_training_labels,
            challenges_path=challenges_path,
        )
    except Exception as e:
        print(json.dumps({
            "task_id": task_id,
            "status": "error",
            "error_message": f"Unexpected error: {type(e).__name__}: {e}"
        }, indent=2))
        sys.exit(1)

    # 3. Serialize diagnostics to JSON-friendly format
    output_data = {
        "task_id": diagnostics.task_id,
        "status": diagnostics.status,
        "solver_status": diagnostics.solver_status,
        "num_constraints": diagnostics.num_constraints,
        "num_variables": diagnostics.num_variables,
        "schema_ids_used": diagnostics.schema_ids_used,

        # M5.X Part A: Per-schema constraint counts
        "schema_constraint_counts": diagnostics.schema_constraint_counts,

        # M5.X Part B: Example summaries
        "example_summaries": [
            {
                "input_shape": list(summary.input_shape),
                "output_shape": list(summary.output_shape) if summary.output_shape else None,
                "components_per_color": summary.components_per_color
            }
            for summary in diagnostics.example_summaries
        ],

        "train_mismatches": diagnostics.train_mismatches,
        "error_message": diagnostics.error_message,
    }

    # 4. Print JSON to stdout
    print(json.dumps(output_data, indent=2))


def main():
    """CLI entrypoint for diagnostic script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run solver diagnostics on an ARC task and print enriched results."
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="ARC task identifier (e.g. 00576224)",
    )
    parser.add_argument(
        "--challenges-path",
        type=Path,
        default=Path("data/arc-agi_training_challenges.json"),
        help="Path to ARC challenges JSON (default: data/arc-agi_training_challenges.json)",
    )
    parser.add_argument(
        "--no-training-labels",
        action="store_true",
        help="Don't compare with ground truth (useful for test set)",
    )

    args = parser.parse_args()

    diagnose_task(
        task_id=args.task_id,
        challenges_path=args.challenges_path,
        use_training_labels=not args.no_training_labels,
    )


if __name__ == "__main__":
    main()
