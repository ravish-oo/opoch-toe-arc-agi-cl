#!/usr/bin/env python3
"""
Comprehensive test script for WO1b - arc_io.py

Validates JSON loading, Grid type conversion, and math kernel compliance.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path (so src.core imports work)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.arc_io import load_arc_training_challenges, load_arc_training_solutions
from src.core.grid_types import Grid


def test_file_existence():
    """Verify required data files exist"""
    print("Testing data file existence...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    assert challenges_path.exists(), f"Challenges file not found: {challenges_path}"
    assert solutions_path.exists(), f"Solutions file not found: {solutions_path}"

    print("  ✓ Data files exist")


def test_load_challenges_structure():
    """Test that challenges load with correct structure"""
    print("Testing challenges loading structure...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Check it's a dict
    assert isinstance(tasks, dict), "Tasks should be a dictionary"
    assert len(tasks) > 0, "Should load at least one task"

    # Check first task structure
    task_id = list(tasks.keys())[0]
    task = tasks[task_id]

    assert isinstance(task, dict), "Each task should be a dict"
    assert "train" in task, "Task should have 'train' key"
    assert "train_outputs" in task, "Task should have 'train_outputs' key"
    assert "test" in task, "Task should have 'test' key"

    assert isinstance(task["train"], list), "'train' should be a list"
    assert isinstance(task["train_outputs"], list), "'train_outputs' should be a list"
    assert isinstance(task["test"], list), "'test' should be a list"

    print(f"  ✓ Loaded {len(tasks)} tasks with correct structure")


def test_all_grids_are_numpy_arrays():
    """Verify all grids are numpy arrays with dtype=int"""
    print("Testing all grids are numpy arrays with dtype=int...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    total_grids = 0
    for task_id, task in tasks.items():
        # Check train inputs
        for i, grid in enumerate(task["train"]):
            assert isinstance(grid, np.ndarray), \
                f"Task {task_id} train[{i}] is not ndarray: {type(grid)}"
            assert grid.dtype == np.int64 or grid.dtype == np.int32, \
                f"Task {task_id} train[{i}] dtype is {grid.dtype}, expected int"
            total_grids += 1

        # Check train outputs
        for i, grid in enumerate(task["train_outputs"]):
            assert isinstance(grid, np.ndarray), \
                f"Task {task_id} train_outputs[{i}] is not ndarray: {type(grid)}"
            assert grid.dtype == np.int64 or grid.dtype == np.int32, \
                f"Task {task_id} train_outputs[{i}] dtype is {grid.dtype}, expected int"
            total_grids += 1

        # Check test inputs
        for i, grid in enumerate(task["test"]):
            assert isinstance(grid, np.ndarray), \
                f"Task {task_id} test[{i}] is not ndarray: {type(grid)}"
            assert grid.dtype == np.int64 or grid.dtype == np.int32, \
                f"Task {task_id} test[{i}] dtype is {grid.dtype}, expected int"
            total_grids += 1

    print(f"  ✓ All {total_grids} grids are numpy arrays with dtype=int")


def test_all_grids_are_2d():
    """Verify all grids are 2D (math kernel requirement)"""
    print("Testing all grids are 2D (math kernel compliance)...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    total_grids = 0
    for task_id, task in tasks.items():
        for i, grid in enumerate(task["train"]):
            assert grid.ndim == 2, \
                f"Task {task_id} train[{i}] is {grid.ndim}D, expected 2D. Shape: {grid.shape}"
            total_grids += 1

        for i, grid in enumerate(task["train_outputs"]):
            assert grid.ndim == 2, \
                f"Task {task_id} train_outputs[{i}] is {grid.ndim}D, expected 2D. Shape: {grid.shape}"
            total_grids += 1

        for i, grid in enumerate(task["test"]):
            assert grid.ndim == 2, \
                f"Task {task_id} test[{i}] is {grid.ndim}D, expected 2D. Shape: {grid.shape}"
            total_grids += 1

    print(f"  ✓ All {total_grids} grids are 2D")


def test_train_outputs_match_train_length():
    """Verify train and train_outputs have same length"""
    print("Testing train/train_outputs length matching...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    for task_id, task in tasks.items():
        train_len = len(task["train"])
        outputs_len = len(task["train_outputs"])

        assert train_len == outputs_len, \
            f"Task {task_id}: train has {train_len} examples but train_outputs has {outputs_len}"

    print(f"  ✓ All {len(tasks)} tasks have matching train/train_outputs lengths")


def test_grid_value_ranges():
    """Verify grid values are in valid ARC palette range [0-9]"""
    print("Testing grid values are in ARC palette [0-9]...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Sample first 10 tasks for speed
    sample_tasks = list(tasks.items())[:10]

    for task_id, task in sample_tasks:
        for grid in task["train"] + task["train_outputs"] + task["test"]:
            min_val = grid.min()
            max_val = grid.max()

            assert min_val >= 0, \
                f"Task {task_id}: grid has value {min_val} < 0"
            assert max_val <= 9, \
                f"Task {task_id}: grid has value {max_val} > 9 (ARC uses 0-9)"

    print(f"  ✓ Sampled {len(sample_tasks)} tasks, all values in [0-9]")


def test_grid_size_bounds():
    """Verify grids are within ARC size limits (max 30x30)"""
    print("Testing grid sizes are within ARC limits...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    max_h = 0
    max_w = 0

    # Sample first 50 tasks
    sample_tasks = list(tasks.items())[:50]

    for task_id, task in sample_tasks:
        for grid in task["train"] + task["train_outputs"] + task["test"]:
            h, w = grid.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)

            assert h >= 1, f"Task {task_id}: grid height {h} < 1"
            assert w >= 1, f"Task {task_id}: grid width {w} < 1"
            assert h <= 30, f"Task {task_id}: grid height {h} > 30"
            assert w <= 30, f"Task {task_id}: grid width {w} > 30"

    print(f"  ✓ All grids within bounds. Max observed: {max_h}x{max_w}")


def test_load_solutions():
    """Test solutions loading"""
    print("Testing solutions loading...")

    solutions_path = Path("data/arc-agi_training_solutions.json")
    solutions = load_arc_training_solutions(solutions_path)

    assert isinstance(solutions, dict), "Solutions should be a dict"
    assert len(solutions) > 0, "Should load at least one solution"

    # Check structure
    task_id = list(solutions.keys())[0]
    test_outputs = solutions[task_id]

    assert isinstance(test_outputs, list), "Test outputs should be a list"
    assert len(test_outputs) > 0, "Should have at least one test output"

    # Check all are grids
    total_grids = 0
    for task_id, test_outputs in solutions.items():
        for grid in test_outputs:
            assert isinstance(grid, np.ndarray), f"Solution grid should be ndarray"
            assert grid.dtype == np.int64 or grid.dtype == np.int32, \
                f"Solution grid dtype is {grid.dtype}, expected int"
            assert grid.ndim == 2, f"Solution grid should be 2D, got {grid.ndim}D"
            total_grids += 1

    print(f"  ✓ Loaded {len(solutions)} task solutions with {total_grids} total grids")


def test_solutions_match_test_count():
    """Verify number of test outputs matches number of test inputs"""
    print("Testing solutions count matches test inputs count...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    tasks = load_arc_training_challenges(challenges_path)
    solutions = load_arc_training_solutions(solutions_path)

    # Check for first 20 tasks
    sample_task_ids = sorted(tasks.keys())[:20]

    for task_id in sample_task_ids:
        if task_id not in solutions:
            # Not all tasks may have solutions in training set
            continue

        num_test_inputs = len(tasks[task_id]["test"])
        num_test_outputs = len(solutions[task_id])

        assert num_test_inputs == num_test_outputs, \
            f"Task {task_id}: {num_test_inputs} test inputs but {num_test_outputs} test outputs"

    print(f"  ✓ Sampled {len(sample_task_ids)} tasks, all have matching test input/output counts")


def test_specific_task_data_integrity():
    """Deep dive into specific task to verify data integrity"""
    print("Testing specific task data integrity...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Use the first task from the runner output: 00576224
    task_id = "00576224"
    assert task_id in tasks, f"Expected task {task_id} not found"

    task = tasks[task_id]

    # Verify counts from runner output
    assert len(task["train"]) == 2, f"Expected 2 train examples, got {len(task['train'])}"
    assert len(task["test"]) == 1, f"Expected 1 test input, got {len(task['test'])}"

    # Verify first train input is 2x2
    train_input_0 = task["train"][0]
    assert train_input_0.shape == (2, 2), f"Expected (2,2), got {train_input_0.shape}"

    # Verify values match what we saw in runner output: [[7,9], [4,3]]
    expected = np.array([[7, 9], [4, 3]], dtype=int)
    assert np.array_equal(train_input_0, expected), \
        f"Train input mismatch:\nExpected:\n{expected}\nGot:\n{train_input_0}"

    # Verify first train output is 6x6
    train_output_0 = task["train_outputs"][0]
    assert train_output_0.shape == (6, 6), f"Expected (6,6), got {train_output_0.shape}"

    print(f"  ✓ Task {task_id} data integrity verified")


def main():
    print("=" * 60)
    print("WO1b Comprehensive Test Suite - arc_io.py")
    print("=" * 60)
    print()

    tests = [
        test_file_existence,
        test_load_challenges_structure,
        test_all_grids_are_numpy_arrays,
        test_all_grids_are_2d,
        test_train_outputs_match_train_length,
        test_grid_value_ranges,
        test_grid_size_bounds,
        test_load_solutions,
        test_solutions_match_test_count,
        test_specific_task_data_integrity,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print()
    print("=" * 60)
    print(f"✅ ALL {len(tests)} TEST SUITES PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
