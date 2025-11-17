"""
Smoke test runner for TaskContext construction.

This script demonstrates building a TaskContext for a real ARC-AGI training task.
It loads one task from the training challenges file, builds the context using
all M1 feature operators, and prints diagnostics to verify everything works.

Usage:
    python -m src.runners.build_context_for_task [task_id]

If no task_id is provided, uses the first task in the challenges file.
"""

import sys
from pathlib import Path

from src.core.grid_types import print_grid
from src.schemas.context import load_arc_task, build_task_context_from_raw


def main():
    """Load a task, build context, and print diagnostics."""
    # Path to ARC-AGI training challenges
    challenges_path = Path("data/arc-agi_training_challenges.json")

    if not challenges_path.exists():
        print(f"ERROR: Challenges file not found at {challenges_path}")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    # Get task_id from command line or use first task
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
    else:
        # Load all tasks and pick first one
        from src.core.arc_io import load_arc_training_challenges
        all_tasks = load_arc_training_challenges(challenges_path)
        task_id = sorted(all_tasks.keys())[0]
        print(f"No task_id provided, using first task: {task_id}\n")

    print(f"Building TaskContext for task: {task_id}")
    print("=" * 70)

    # Load task and build context
    task_data = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(task_data)

    # Print task-level diagnostics
    print(f"\nTask-level info:")
    print(f"  Num train examples: {len(ctx.train_examples)}")
    print(f"  Num test examples: {len(ctx.test_examples)}")
    print(f"  Palette size C: {ctx.C}")

    # Print diagnostics for first training example
    if ctx.train_examples:
        print("\n" + "=" * 70)
        print("First training example:")
        print("=" * 70)

        ex = ctx.train_examples[0]

        print(f"\nInput grid shape: {ex.input_H}x{ex.input_W}")
        print(f"Output grid shape: {ex.output_H}x{ex.output_W}")

        print("\nInput grid:")
        print_grid(ex.input_grid)

        print("\nOutput grid:")
        print_grid(ex.output_grid)

        # Component info
        print(f"\nComponent info:")
        print(f"  Num components: {len(ex.components)}")
        for comp in ex.components[:3]:  # Show first 3 components
            print(f"    Component {comp.id}: color={comp.color}, size={comp.size}, bbox={comp.bbox}")
        if len(ex.components) > 3:
            print(f"    ... and {len(ex.components) - 3} more components")

        # Feature coverage
        print(f"\nFeature coverage:")
        print(f"  Num pixels with object_ids: {len(ex.object_ids)}")
        print(f"  Num pixels with sectors: {len(ex.sectors)}")
        print(f"  Num pixels with border_info: {len(ex.border_info)}")
        print(f"  Num neighborhood hashes: {len(ex.neighborhood_hashes)}")

        # Band labels
        print(f"\nBand labels:")
        print(f"  row_bands: {ex.row_bands}")
        print(f"  col_bands: {ex.col_bands}")

        # Row/col nonzero flags
        print(f"\nNonzero flags:")
        print(f"  row_nonzero: {ex.row_nonzero}")
        print(f"  col_nonzero: {ex.col_nonzero}")

        # Sample coordinate features for pixel (0,0)
        if (0, 0) in ex.coords:
            print(f"\nSample coordinate features for pixel (0,0):")
            print(f"  coords: {ex.coords[(0, 0)]}")
            if 0 in ex.row_residues:
                print(f"  row_residues[0]: {ex.row_residues[0]}")
            if 0 in ex.col_residues:
                print(f"  col_residues[0]: {ex.col_residues[0]}")

    # Print diagnostics for first test example
    if ctx.test_examples:
        print("\n" + "=" * 70)
        print("First test example:")
        print("=" * 70)

        ex_test = ctx.test_examples[0]

        print(f"\nInput grid shape: {ex_test.input_H}x{ex_test.input_W}")
        print(f"Output grid: {ex_test.output_grid} (None for test)")

        print("\nInput grid:")
        print_grid(ex_test.input_grid)

        print(f"\nNum components: {len(ex_test.components)}")
        print(f"Num pixels with object_ids: {len(ex_test.object_ids)}")

    print("\n" + "=" * 70)
    print("✓ TaskContext built successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
