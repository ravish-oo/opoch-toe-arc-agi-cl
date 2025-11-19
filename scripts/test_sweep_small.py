"""
Test sweep on a small number of tasks to check for any successes.
"""

from pathlib import Path
import tempfile
import shutil

from src.runners.sweep_training_with_miner import sweep_training_with_miner, load_training_task_ids
import json

# Create temp directories
temp_catalog = Path(tempfile.mkdtemp())
temp_logs = Path(tempfile.mkdtemp())

try:
    # Override catalog dir
    from src.catalog import store
    original_catalog_dir = store.DEFAULT_CATALOG_DIR
    store.DEFAULT_CATALOG_DIR = temp_catalog

    # Test on first 10 tasks
    challenges_path = Path("data/arc-agi_training_challenges.json")

    all_task_ids = load_training_task_ids(challenges_path)
    test_task_ids = all_task_ids[:10]

    # Create subset
    with challenges_path.open("r") as f:
        all_tasks = json.load(f)
    subset_tasks = {tid: all_tasks[tid] for tid in test_task_ids}

    temp_challenges = temp_logs / "test_challenges.json"
    with temp_challenges.open("w") as f:
        json.dump(subset_tasks, f)

    failures_log = temp_logs / "failures.jsonl"

    # Run sweep
    sweep_training_with_miner(temp_challenges, failures_log)

    # Check results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    catalog_files = list(temp_catalog.glob("*.json"))
    print(f"Successes: {len(catalog_files)}")
    for cf in catalog_files:
        print(f"  - {cf.stem}")

    if failures_log.exists():
        with failures_log.open("r") as f:
            failures = [json.loads(line) for line in f]
        print(f"\nFailures: {len(failures)}")
        for failure in failures:
            print(f"  - {failure['task_id']}: {failure['status']}")

    store.DEFAULT_CATALOG_DIR = original_catalog_dir

finally:
    shutil.rmtree(temp_catalog, ignore_errors=True)
    shutil.rmtree(temp_logs, ignore_errors=True)
