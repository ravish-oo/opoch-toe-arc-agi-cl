"""
ARC-AGI JSON IO utilities.

This module loads ARC-AGI challenge and solution files from JSON format
and converts them into our Grid representation.

Expected JSON structure:

arc-agi_training_challenges.json:
{
  "task_id": {
    "train": [
      {"input": [[int, ...], ...], "output": [[int, ...], ...]},
      ...
    ],
    "test": [
      {"input": [[int, ...], ...]}
    ]
  }
}

arc-agi_training_solutions.json:
{
  "task_id": [
    [[int, ...], ...],  # test output 0
    ...
  ]
}
"""

from pathlib import Path
from typing import Dict, List
import json
import numpy as np

from src.core.grid_types import Grid


def load_arc_training_challenges(path: Path) -> Dict[str, Dict[str, List[Grid]]]:
    """
    Load ARC-AGI training challenges from the given JSON file.

    Output contract:
      {
        task_id: {
          "train": [Grid, ...],           # train input grids
          "train_outputs": [Grid, ...],   # train output grids (same length as train)
          "test": [Grid, ...]             # test input grids
        },
        ...
      }

    Args:
        path: Path to arc-agi_training_challenges.json

    Returns:
        Dictionary mapping task_id to train/test grids

    Notes:
        - All grids are converted to numpy arrays with dtype=int
        - Train outputs come from the "output" field in each train example
        - Test examples only have inputs (outputs in separate solutions file)
    """
    with open(path, 'r') as f:
        raw_data = json.load(f)

    tasks = {}

    for task_id, task_data in raw_data.items():
        train_inputs = []
        train_outputs = []
        test_inputs = []

        # Process training examples (have both input and output)
        for train_example in task_data.get("train", []):
            # Convert input grid
            input_grid = np.array(train_example["input"], dtype=int)
            train_inputs.append(input_grid)

            # Convert output grid
            output_grid = np.array(train_example["output"], dtype=int)
            train_outputs.append(output_grid)

        # Process test examples (only have input)
        for test_example in task_data.get("test", []):
            test_grid = np.array(test_example["input"], dtype=int)
            test_inputs.append(test_grid)

        tasks[task_id] = {
            "train": train_inputs,
            "train_outputs": train_outputs,
            "test": test_inputs
        }

    return tasks


def load_arc_training_solutions(path: Path) -> Dict[str, List[Grid]]:
    """
    Load ARC-AGI training solutions from JSON.

    Output contract:
      {
        task_id: [Grid, ...],  # test output grids
        ...
      }

    Args:
        path: Path to arc-agi_training_solutions.json

    Returns:
        Dictionary mapping task_id to list of test output grids

    Notes:
        - Solutions file contains only test outputs (not train outputs)
        - Each task_id maps to a list of grids (one per test input)
    """
    with open(path, 'r') as f:
        raw_data = json.load(f)

    solutions = {}

    for task_id, test_outputs in raw_data.items():
        # test_outputs is a list of grids (list of list of ints)
        grids = [np.array(grid, dtype=int) for grid in test_outputs]
        solutions[task_id] = grids

    return solutions


def load_arc_task_ids(challenges_path: Path) -> List[str]:
    """
    Load all task IDs from an ARC challenges JSON file.

    Args:
        path: Path to arc-agi_training_challenges.json or similar

    Returns:
        List of task_id strings (sorted for deterministic order)

    Example:
        >>> path = Path("data/arc-agi_training_challenges.json")
        >>> task_ids = load_arc_task_ids(path)
        >>> len(task_ids)
        400
    """
    with open(challenges_path, 'r') as f:
        data = json.load(f)

    # ARC challenges JSON is a dict mapping task_id -> task_data
    return sorted(data.keys())


if __name__ == "__main__":
    # Self-test: load and inspect first task
    from src.core.grid_types import print_grid

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    print("Loading challenges...")
    tasks = load_arc_training_challenges(challenges_path)
    print(f"Loaded {len(tasks)} tasks")

    # Pick first task
    task_id = sorted(tasks.keys())[0]
    task = tasks[task_id]

    print(f"\nTask: {task_id}")
    print(f"  Train examples: {len(task['train'])}")
    print(f"  Test examples: {len(task['test'])}")

    if task["train"]:
        print("\nFirst train input:")
        print_grid(task["train"][0])
        print("\nFirst train output:")
        print_grid(task["train_outputs"][0])

    print("\nLoading solutions...")
    solutions = load_arc_training_solutions(solutions_path)
    print(f"Loaded solutions for {len(solutions)} tasks")

    if task_id in solutions:
        print(f"\nTest output for {task_id}:")
        print_grid(solutions[task_id][0])
