"""
Debug script to run sweep on a single task with full traceback.

Usage:
    python -m scripts.debug_single_task <task_id>
"""

import sys
from pathlib import Path
import traceback

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics


def debug_task(task_id: str):
    """Debug a single task with full error reporting."""
    print("=" * 70)
    print(f"Debugging Task: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    try:
        # Step 1: Load task
        print("\n[1] Loading task...")
        raw_task = load_arc_task(task_id, challenges_path)
        print(f"    ✓ Task loaded: {len(raw_task.get('train', []))} train, {len(raw_task.get('test', []))} test examples")

        # Step 2: Build context
        print("\n[2] Building TaskContext...")
        task_context = build_task_context_from_raw(raw_task)
        print(f"    ✓ Context built: {len(task_context.train_examples)} train, {len(task_context.test_examples)} test")
        print(f"    ✓ Color count: {task_context.C}")

        # Step 3: Mine laws
        print("\n[3] Mining laws...")
        law_config = mine_law_config(task_context)
        print(f"    ✓ Mined {len(law_config.schema_instances)} schema instances")

        # Show schema breakdown
        schema_counts = {}
        for inst in law_config.schema_instances:
            schema_counts[inst.family_id] = schema_counts.get(inst.family_id, 0) + 1
        print(f"    Schema breakdown: {schema_counts}")

        # Step 4: Run kernel
        print("\n[4] Running kernel with train+test validation...")
        outputs, diagnostics = solve_arc_task_with_diagnostics(
            task_id=task_id,
            law_config=law_config,
            use_training_labels=True,
            use_test_labels=True,
            challenges_path=challenges_path,
            solutions_path=solutions_path,
        )

        print(f"\n    Status: {diagnostics.status}")
        print(f"    Solver: {diagnostics.solver_status}")
        print(f"    Constraints: {diagnostics.num_constraints}")
        print(f"    Variables: {diagnostics.num_variables}")
        print(f"    Train mismatches: {len(diagnostics.train_mismatches)}")
        print(f"    Test mismatches: {len(diagnostics.test_mismatches)}")

        if diagnostics.error_message:
            print(f"\n    Error message: {diagnostics.error_message}")

        print("\n" + "=" * 70)
        print(f"✓ Task {task_id} completed with status: {diagnostics.status}")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ EXCEPTION at current step:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"\n   Full traceback:")
        traceback.print_exc()
        print("\n" + "=" * 70)
        print(f"✗ Task {task_id} FAILED with exception")
        print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.debug_single_task <task_id>")
        print("Example: python -m scripts.debug_single_task 0520fde7")
        sys.exit(1)

    task_id = sys.argv[1]
    debug_task(task_id)
