"""
Check solutions file format for task 0520fde7.
"""

import json
from pathlib import Path

task_id = "0520fde7"
solutions_path = Path("data/arc-agi_training_solutions.json")

print(f"Checking solutions format for task: {task_id}")
print("=" * 70)

# Load solutions
with solutions_path.open("r", encoding="utf-8") as f:
    solutions_data = json.load(f)

print(f"Task {task_id} in solutions: {task_id in solutions_data}")
print()

if task_id not in solutions_data:
    print("Task not found in solutions!")
    import sys
    sys.exit(1)

# Get the solutions
true_test_outputs_raw = solutions_data[task_id]

print(f"Type of solutions_data[task_id]: {type(true_test_outputs_raw)}")
print(f"Length: {len(true_test_outputs_raw)}")
print()

for idx, grid_raw in enumerate(true_test_outputs_raw):
    print(f"Test output {idx}:")
    print(f"  Type: {type(grid_raw)}")
    if isinstance(grid_raw, list):
        print(f"  Outer length (rows): {len(grid_raw)}")
        if grid_raw:
            print(f"  Inner length (cols): {len(grid_raw[0])}")
            print(f"  Shape: {len(grid_raw)} × {len(grid_raw[0])}")
    print()
