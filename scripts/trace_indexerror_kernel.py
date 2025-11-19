"""
Trace IndexError in kernel execution for S5/S11-only tasks.
"""

import traceback
from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics

task_id = "0520fde7"
challenges_path = Path("data/arc-agi_training_challenges.json")
solutions_path = Path("data/arc-agi_training_solutions.json")

print(f"Tracing IndexError for task: {task_id}")
print("=" * 70)

# Load task
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)

print(f"Train examples: {len(task_context.train_examples)}")
print(f"Test examples: {len(task_context.test_examples)}")
print()

# Mine laws
law_config = mine_law_config(task_context)
print(f"Mined {len(law_config.schema_instances)} schema instances")

from collections import defaultdict
schema_breakdown = defaultdict(int)
for inst in law_config.schema_instances:
    schema_breakdown[inst.family_id] += 1
print(f"Schema breakdown: {dict(schema_breakdown)}")
print()

# Try to solve
print("Running kernel...")
print()

outputs, diagnostics = solve_arc_task_with_diagnostics(
    task_id=task_id,
    law_config=law_config,
    use_training_labels=True,
    use_test_labels=True,
    challenges_path=challenges_path,
    solutions_path=solutions_path,
)

print(f"Status: {diagnostics.status}")
print(f"Error message: {diagnostics.error_message}")
print(f"Test outputs pred: {len(outputs.get('test', []))}")
print(f"Train outputs pred: {len(outputs.get('train', []))}")

if outputs.get('test'):
    for idx, grid in enumerate(outputs['test']):
        print(f"Test output {idx} shape: {grid.shape}")

# If there was an error, let's try to reproduce it
if diagnostics.status == "error" and "IndexError" in str(diagnostics.error_message):
    print()
    print("IndexError detected in diagnostics!")
    print("This error was caught inside solve_arc_task_with_diagnostics.")
