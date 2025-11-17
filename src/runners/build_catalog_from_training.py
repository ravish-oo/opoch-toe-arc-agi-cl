"""
Training sweep and catalog builder script.

This script loops over all ARC training tasks, runs the kernel with existing
law configurations, and updates the catalog or logs failures. It does NO
law discovery - only validation and catalog maintenance.

Usage:
    # Process all tasks with existing configs
    python -m src.runners.build_catalog_from_training --only-with-configs

    # Process first 5 tasks for testing
    python -m src.runners.build_catalog_from_training --max-tasks 5

    # Custom paths
    python -m src.runners.build_catalog_from_training \
        --challenges-path data/arc-agi_training_challenges.json \
        --failure-log logs/failures.jsonl

Output:
    - Updates catalog/tasks/{task_id}.json for successful tasks
    - Appends failure diagnostics to logs/training_failures.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.core.arc_io import load_arc_task_ids
from src.catalog.types import TaskLawConfig
from src.catalog.store import (
    load_task_law_config,
    save_task_law_config,
)
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.runners.results import SolveDiagnostics


# Logger for this module
logger = logging.getLogger(__name__)


def sweep_training_tasks(
    challenges_path: Path,
    failure_log_path: Path,
    only_with_configs: bool = True,
    max_tasks: Optional[int] = None,
) -> None:
    """
    Sweep over training tasks, run the kernel with existing law configs,
    and update the catalog / log failures.

    This function does NO law discovery. It only:
      - Loads existing TaskLawConfig for each task
      - Runs solve_arc_task_with_diagnostics
      - Updates catalog if status == "ok"
      - Logs failures to JSONL if status != "ok"

    Args:
        challenges_path: Path to arc-agi_training_challenges.json
        failure_log_path: Path to JSONL file where failures will be appended
        only_with_configs: If True, skip tasks that have no stored TaskLawConfig
        max_tasks: If not None, limit to the first max_tasks task_ids (for testing)

    Example:
        >>> sweep_training_tasks(
        ...     challenges_path=Path("data/arc-agi_training_challenges.json"),
        ...     failure_log_path=Path("logs/training_failures.jsonl"),
        ...     only_with_configs=True,
        ...     max_tasks=10
        ... )
    """
    # 1. Get all task IDs
    logger.info("Loading task IDs from %s", challenges_path)
    task_ids = load_arc_task_ids(challenges_path)
    logger.info("Found %d tasks", len(task_ids))

    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]
        logger.info("Limiting to first %d tasks", max_tasks)

    # 2. Open failure log for appending (JSONL format)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_file = failure_log_path.open("a", encoding="utf-8")

    # 3. Loop over task IDs
    num_processed = 0
    num_ok = 0
    num_failures = 0
    num_skipped = 0

    for task_id in task_ids:
        logger.info("Processing task_id=%s", task_id)

        # Try to load existing law config
        law_config: Optional[TaskLawConfig] = load_task_law_config(task_id)

        if law_config is None:
            if only_with_configs:
                logger.info("  No law_config for task_id=%s, skipping.", task_id)
                num_skipped += 1
                continue
            else:
                logger.info("  No law_config for task_id=%s, using empty config.", task_id)
                law_config = TaskLawConfig(schema_instances=[])

        # 4. Call the kernel with diagnostics
        try:
            outputs, diagnostics = solve_arc_task_with_diagnostics(
                task_id=task_id,
                law_config=law_config,
                use_training_labels=True,
                challenges_path=challenges_path,
            )
        except Exception as e:
            logger.exception("Error while solving task_id=%s: %s", task_id, e)
            # Write a failure record with status="error"
            record = {
                "task_id": task_id,
                "status": "error",
                "error_message": str(e),
            }
            failure_log_file.write(json.dumps(record) + "\n")
            failure_log_file.flush()  # Ensure written immediately
            num_failures += 1
            continue

        # 5. Handle status
        num_processed += 1

        if diagnostics.status == "ok":
            logger.info("  ✓ OK for task_id=%s", task_id)
            # Mark this config as valid / up-to-date in the catalog
            save_task_law_config(task_id, law_config)
            num_ok += 1
            # No failure log entry for this task

        else:
            logger.warning(
                "  ✗ Failure for task_id=%s: status=%s, solver_status=%s",
                task_id,
                diagnostics.status,
                diagnostics.solver_status,
            )

            # Serialize diagnostics into a JSON-friendly dict
            failure_record = {
                "task_id": diagnostics.task_id,
                "status": diagnostics.status,
                "solver_status": diagnostics.solver_status,
                "num_constraints": diagnostics.num_constraints,
                "num_variables": diagnostics.num_variables,
                "schema_ids_used": diagnostics.schema_ids_used,
                "train_mismatches": diagnostics.train_mismatches,
                "error_message": diagnostics.error_message,
            }
            failure_log_file.write(json.dumps(failure_record) + "\n")
            failure_log_file.flush()
            num_failures += 1

    # 6. Close failure log
    failure_log_file.close()

    # 7. Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 70)
    logger.info("Total tasks: %d", len(task_ids))
    logger.info("Processed: %d", num_processed)
    logger.info("  OK: %d", num_ok)
    logger.info("  Failures: %d", num_failures)
    logger.info("Skipped (no config): %d", num_skipped)
    logger.info("=" * 70)


def main():
    """CLI entrypoint for catalog building sweep."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep ARC training tasks and build/update law catalog."
    )
    parser.add_argument(
        "--challenges-path",
        type=Path,
        default=Path("data/arc-agi_training_challenges.json"),
        help="Path to ARC training challenges JSON.",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("logs/training_failures.jsonl"),
        help="Path to JSONL file where failure diagnostics will be logged.",
    )
    parser.add_argument(
        "--only-with-configs",
        action="store_true",
        help="If set, skip tasks without existing law configs.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional limit on number of tasks to process (for quick tests).",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Run sweep
    sweep_training_tasks(
        challenges_path=args.challenges_path,
        failure_log_path=args.failure_log,
        only_with_configs=args.only_with_configs,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
