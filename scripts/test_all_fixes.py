"""
Test all geometry fixes on previously problematic tasks.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.solver.kernel import solve_task

# Tasks that previously had errors
error_tasks = [
    "ff28f65a",  # S9 invalid pixel indices
    "e872b94a",  # S4/S8 geometry bug
]

# Tasks that previously had exceptions
exception_tasks = [
    "0520fde7",  # S5/S11 reshape + S10 geometry
    "0a1d4ef5",  # S5 reshape
    "1be83260",  # S1 geometry
    "47c1f68c",  # S1 cross-example
    "5833af48",  # S1 cross-example
]

challenges_path = Path("data/arc-agi_training_challenges.json")

print("Testing all geometry fixes")
print("=" * 70)

all_tasks = error_tasks + exception_tasks
results = {}

for task_id in all_tasks:
    print(f"\nTask {task_id}:", end=" ")
    try:
        raw_task = load_arc_task(task_id, challenges_path)
        ctx = build_task_context_from_raw(raw_task)
        law_config = mine_law_config(ctx)
        status = solve_task(ctx, law_config)
        results[task_id] = status
        print(f"✓ {status}")
    except Exception as e:
        results[task_id] = f"error: {type(e).__name__}"
        print(f"✗ {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("Summary:")
print("-" * 70)

status_counts = {}
for task_id, status in results.items():
    if status.startswith("error:"):
        status_key = "error"
    else:
        status_key = status
    status_counts[status_key] = status_counts.get(status_key, 0) + 1

for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")

print()
if "error" not in status_counts:
    print("✓ All previously problematic tasks now run without errors")
else:
    print(f"✗ Still have {status_counts['error']} error(s)")
