"""
Simple runner to print a sample ARC task.

This verifies that our Grid loading and printing utilities work correctly.
It loads the first task from the training challenges and displays its grids.
"""

from pathlib import Path

from src.core.arc_io import load_arc_training_challenges
from src.core.grid_types import print_grid


def main() -> None:
    """Load and print the first ARC training task."""
    data_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(data_path)

    # Pick the first task_id (sorted for determinism)
    task_ids = sorted(tasks.keys())
    if not task_ids:
        print("No tasks loaded.")
        return

    task_id = task_ids[0]
    t = tasks[task_id]

    print(f"Task: {task_id}")
    print(f"  #train examples: {len(t['train'])}")
    print(f"  #test inputs:    {len(t['test'])}")

    if t["train"]:
        print("\nFirst train input grid:")
        print_grid(t["train"][0])

        if t["train_outputs"]:
            print("\nFirst train output grid:")
            print_grid(t["train_outputs"][0])

    if t["test"]:
        print("\nFirst test input grid:")
        print_grid(t["test"][0])


if __name__ == "__main__":
    main()
