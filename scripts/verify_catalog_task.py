"""
Quick verification: check if a catalog task truly solves test examples.
"""

from pathlib import Path
import json

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics


def verify_task(task_id: str):
    """Verify if a task truly solves both train and test examples."""
    print(f"\nVerifying task: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    # Load and mine
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(task_context)

    print(f"Mined {len(law_config.schema_instances)} schema instances")

    # Test with train-only validation
    _, diag_train_only = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        use_test_labels=False,
        challenges_path=challenges_path,
    )

    print(f"\nTrain-only validation:")
    print(f"  Status: {diag_train_only.status}")
    print(f"  Train mismatches: {len(diag_train_only.train_mismatches)}")

    # Test with train+test validation
    _, diag_full = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        use_test_labels=True,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
    )

    print(f"\nTrain+test validation:")
    print(f"  Status: {diag_full.status}")
    print(f"  Train mismatches: {len(diag_full.train_mismatches)}")
    print(f"  Test mismatches: {len(diag_full.test_mismatches)}")

    if diag_full.status == "ok":
        print(f"\n✓ Task {task_id}: TRUE SUCCESS (train+test both match)")
    elif diag_full.status == "mismatch_test":
        print(f"\n✗ Task {task_id}: TRAIN OK, TEST FAILED")
        print(f"  Test mismatch details: {diag_full.test_mismatches}")
    else:
        print(f"\n✗ Task {task_id}: {diag_full.status}")


if __name__ == "__main__":
    # Get all catalog tasks
    catalog_dir = Path("catalog/tasks")
    catalog_files = sorted(catalog_dir.glob("*.json"))

    print("=" * 70)
    print(f"Verifying {len(catalog_files)} catalog tasks with test validation")
    print("=" * 70)

    # Verify first 5 catalog tasks
    for cf in catalog_files[:5]:
        task_id = cf.stem
        try:
            verify_task(task_id)
        except Exception as e:
            print(f"\n✗ Error verifying {task_id}: {e}")

    print("\n" + "=" * 70)
    print("Verification complete")
    print("=" * 70)
