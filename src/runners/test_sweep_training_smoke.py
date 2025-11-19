"""
Smoke test for training sweep with miner.

Tests the sweep on a small subset of tasks to verify:
  - Task loading works
  - Law mining works
  - Kernel validation works
  - Catalog saving works
  - Failure logging works
"""

from pathlib import Path
import json
import tempfile
import shutil

from src.runners.sweep_training_with_miner import load_training_task_ids, sweep_training_with_miner


def test_load_task_ids():
    """Test loading task IDs from challenges file."""
    print("=" * 70)
    print("Test: load_training_task_ids")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_ids = load_training_task_ids(challenges_path)

    print(f"Loaded {len(task_ids)} task IDs")
    print(f"First 5: {task_ids[:5]}")
    print(f"Last 5: {task_ids[-5:]}")

    assert len(task_ids) > 0, "Should load at least one task"
    assert all(isinstance(tid, str) for tid in task_ids), "Task IDs should be strings"
    assert task_ids == sorted(task_ids), "Task IDs should be sorted"

    print("✓ Test passed\n")


def test_sweep_subset():
    """Test sweep on a small subset of tasks."""
    print("=" * 70)
    print("Test: sweep_training_with_miner (subset)")
    print("=" * 70)

    # Create temporary directories for this test
    temp_catalog = Path(tempfile.mkdtemp())
    temp_logs = Path(tempfile.mkdtemp())

    try:
        # Prepare test paths
        challenges_path = Path("data/arc-agi_training_challenges.json")
        failures_log_path = temp_logs / "test_failures.jsonl"

        # Load all task IDs and select a subset
        all_task_ids = load_training_task_ids(challenges_path)
        test_task_ids = all_task_ids[:3]  # Just test first 3 tasks

        print(f"Testing on subset of {len(test_task_ids)} tasks: {test_task_ids}")

        # Create a temporary challenges file with just these tasks
        with challenges_path.open("r", encoding="utf-8") as f:
            all_tasks = json.load(f)

        subset_tasks = {tid: all_tasks[tid] for tid in test_task_ids}

        temp_challenges = temp_logs / "test_challenges.json"
        with temp_challenges.open("w", encoding="utf-8") as f:
            json.dump(subset_tasks, f)

        # Run sweep on subset
        # Temporarily override default catalog dir
        from src.catalog import store
        original_catalog_dir = store.DEFAULT_CATALOG_DIR
        store.DEFAULT_CATALOG_DIR = temp_catalog

        try:
            sweep_training_with_miner(temp_challenges, failures_log_path)
        finally:
            # Restore original catalog dir
            store.DEFAULT_CATALOG_DIR = original_catalog_dir

        # Verify outputs
        print("\nVerifying outputs:")

        # Check catalog directory
        catalog_files = list(temp_catalog.glob("*.json"))
        print(f"  Catalog files created: {len(catalog_files)}")
        for cf in catalog_files:
            print(f"    - {cf.name}")

        # Check failures log
        if failures_log_path.exists():
            with failures_log_path.open("r", encoding="utf-8") as f:
                failure_lines = f.readlines()
            print(f"  Failure log entries: {len(failure_lines)}")

            # Verify each line is valid JSON
            for i, line in enumerate(failure_lines, 1):
                try:
                    record = json.loads(line)
                    assert "task_id" in record, "Failure record should have task_id"
                    assert "status" in record, "Failure record should have status"
                    print(f"    - {record['task_id']}: {record['status']}")
                except json.JSONDecodeError:
                    raise AssertionError(f"Line {i} is not valid JSON: {line}")
        else:
            print(f"  No failures log created (all tasks succeeded)")

        # Verify total count
        total_processed = len(catalog_files) + (len(failure_lines) if failures_log_path.exists() else 0)
        assert total_processed == len(test_task_ids), \
            f"Should process all {len(test_task_ids)} tasks, got {total_processed}"

        print("\n✓ Test passed")

    finally:
        # Cleanup temporary directories
        shutil.rmtree(temp_catalog, ignore_errors=True)
        shutil.rmtree(temp_logs, ignore_errors=True)


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("Training Sweep Smoke Tests")
    print("=" * 70 + "\n")

    test_load_task_ids()
    test_sweep_subset()

    print("\n" + "=" * 70)
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
