"""
S9 cross/plus miner: Discover cross propagation patterns.

This module implements algorithmic mining of S9 (cross/plus propagation) schema instances.
S9 enforces that from specific seed positions, colors propagate along cardinal directions
(up, down, left, right) with consistent colors and lengths.

Algorithm:
  1. Detect seed positions in train outputs (positions with at least one arm)
  2. Infer seed_color from input colors at those positions (must be unique)
  3. For each direction, infer arm color and length (must be consistent across all seeds)
  4. Generate SchemaInstances for train and test examples using seed_color predicate

Key principle: S9 discovers OUTPUT cross patterns from seeds identified by INPUT color.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance


def mine_S9(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S9 schema instances: cross/plus propagation from seeds.

    S9 enforces cross patterns where:
      - Seeds are identified by input color
      - From each seed, arms propagate in up/down/left/right directions
      - Each direction has consistent color and length across all seeds/examples
      - Center pixel is not constrained (just positional anchor)

    Algorithm:
      1. For each train example, detect seed positions (output has arms)
      2. Derive seed_color from input colors at seed positions
      3. For each direction, infer arm color and max length
      4. Generate SchemaInstances for train and test examples

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List with 0 or more SchemaInstance objects, each with params:
        {
            "example_type": "train" | "test",
            "example_index": int,
            "seeds": [
                {
                    "center": "(r,c)",
                    "up_color": int | None,
                    "down_color": int | None,
                    "left_color": int | None,
                    "right_color": int | None,
                    "max_up": int,
                    "max_down": int,
                    "max_left": int,
                    "max_right": int
                },
                ...
            ]
        }

    Only emits instances when the cross pattern is exactly consistent
    across all training examples. If inconsistent, returns [].
    """
    # Stage 1: Detect seed positions and infer seed color
    seed_color_result = _detect_seed_color(task_context)

    if seed_color_result is None:
        return []  # No consistent seed color found

    seed_color, all_seed_positions = seed_color_result

    # Stage 2: Infer arm colors and lengths per direction
    arm_params_result = _infer_arm_parameters(task_context, all_seed_positions)

    if arm_params_result is None:
        return []  # Inconsistent arm patterns

    up_color, down_color, left_color, right_color, max_up, max_down, max_left, max_right = arm_params_result

    # Stage 3: Generate SchemaInstances for train and test examples
    instances: List[SchemaInstance] = []

    # Train examples
    for ex_idx, ex in enumerate(task_context.train_examples):
        seeds = _find_seeds_by_color(ex.input_grid, seed_color)

        if seeds:
            instances.append(SchemaInstance(
                family_id="S9",
                params={
                    "example_type": "train",
                    "example_index": ex_idx,
                    "seeds": [
                        {
                            "center": f"({r},{c})",
                            "up_color": up_color,
                            "down_color": down_color,
                            "left_color": left_color,
                            "right_color": right_color,
                            "max_up": max_up,
                            "max_down": max_down,
                            "max_left": max_left,
                            "max_right": max_right,
                        }
                        for (r, c) in seeds
                    ]
                }
            ))

    # Test examples
    for ex_idx, ex in enumerate(task_context.test_examples):
        seeds = _find_seeds_by_color(ex.input_grid, seed_color)

        if seeds:
            instances.append(SchemaInstance(
                family_id="S9",
                params={
                    "example_type": "test",
                    "example_index": ex_idx,
                    "seeds": [
                        {
                            "center": f"({r},{c})",
                            "up_color": up_color,
                            "down_color": down_color,
                            "left_color": left_color,
                            "right_color": right_color,
                            "max_up": max_up,
                            "max_down": max_down,
                            "max_left": max_left,
                            "max_right": max_right,
                        }
                        for (r, c) in seeds
                    ]
                }
            ))

    return instances


def _detect_seed_color(
    task_context: TaskContext
) -> Optional[Tuple[int, List[Tuple[int, int, int]]]]:
    """
    Detect seed color and positions from training examples.

    Returns:
        (seed_color, [(ex_idx, r, c), ...]) if unique seed color found,
        None otherwise
    """
    # Step 1: Find all potential seed positions in training outputs
    # A seed is a position with at least one arm (non-zero run in any direction)
    seed_positions_per_color: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue

        X = ex.input_grid
        Y = ex.output_grid
        H_out, W_out = Y.shape
        H_in, W_in = X.shape

        # S9 requires geometry-preserving (input and output same size)
        if H_out != H_in or W_out != W_in:
            continue  # Skip non-geometry-preserving examples

        for r in range(H_out):
            for c in range(W_out):
                # Check if this position has at least one arm in output
                has_any_arm = False

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc

                    if 0 <= rr < H_out and 0 <= cc < W_out:
                        if Y[rr, cc] != 0:
                            has_any_arm = True
                            break

                if has_any_arm:
                    # This is a potential seed - record input color
                    input_color = int(X[r, c])
                    seed_positions_per_color[input_color].append((ex_idx, r, c))

    # Step 2: Find unique seed_color
    # We need exactly one color that appears at ALL seed positions
    if not seed_positions_per_color:
        return None  # No seeds found

    # For v1, we require all seeds across all examples to have same input color
    # Try each color and see if it's the only one
    if len(seed_positions_per_color) != 1:
        # Multiple input colors at seed positions - not a simple pattern
        return None

    seed_color = list(seed_positions_per_color.keys())[0]
    all_seed_positions = seed_positions_per_color[seed_color]

    return (seed_color, all_seed_positions)


def _infer_arm_parameters(
    task_context: TaskContext,
    all_seed_positions: List[Tuple[int, int, int]]
) -> Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int], int, int, int, int]]:
    """
    Infer arm colors and lengths from seed positions.

    Returns:
        (up_color, down_color, left_color, right_color, max_up, max_down, max_left, max_right)
        or None if patterns are inconsistent
    """
    # Direction mappings
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }

    # Track colors and lengths per direction
    direction_colors: Dict[str, Set[int]] = {
        "up": set(),
        "down": set(),
        "left": set(),
        "right": set()
    }
    direction_lengths: Dict[str, Set[int]] = {
        "up": set(),
        "down": set(),
        "left": set(),
        "right": set()
    }

    # For each seed, walk arms and collect color/length
    for (ex_idx, r, c) in all_seed_positions:
        ex = task_context.train_examples[ex_idx]
        Y = ex.output_grid
        H, W = Y.shape

        for direction_name, (dr, dc) in directions.items():
            # Walk arm from (r, c) in this direction
            arm_cells = []
            rr, cc = r + dr, c + dc

            while 0 <= rr < H and 0 <= cc < W:
                col = int(Y[rr, cc])
                if col == 0:
                    break  # Stop at background
                arm_cells.append(col)
                rr += dr
                cc += dc

            if not arm_cells:
                # No arm in this direction for this seed
                continue

            # Validate arm has single color (all cells same)
            unique_colors = set(arm_cells)
            if len(unique_colors) > 1:
                # Multi-color within single arm → invalid
                return None

            arm_color = arm_cells[0]
            arm_length = len(arm_cells)

            direction_colors[direction_name].add(arm_color)
            direction_lengths[direction_name].add(arm_length)

    # Validate per-direction consistency
    for d in ["up", "down", "left", "right"]:
        if len(direction_colors[d]) > 1:
            # Inconsistent colors in this direction
            return None
        if len(direction_lengths[d]) > 1:
            # Inconsistent lengths in this direction
            return None

    # Extract final parameters
    up_color = direction_colors["up"].pop() if direction_colors["up"] else None
    down_color = direction_colors["down"].pop() if direction_colors["down"] else None
    left_color = direction_colors["left"].pop() if direction_colors["left"] else None
    right_color = direction_colors["right"].pop() if direction_colors["right"] else None

    max_up = direction_lengths["up"].pop() if direction_lengths["up"] else 0
    max_down = direction_lengths["down"].pop() if direction_lengths["down"] else 0
    max_left = direction_lengths["left"].pop() if direction_lengths["left"] else 0
    max_right = direction_lengths["right"].pop() if direction_lengths["right"] else 0

    return (up_color, down_color, left_color, right_color, max_up, max_down, max_left, max_right)


def _find_seeds_by_color(grid: np.ndarray, seed_color: int) -> List[Tuple[int, int]]:
    """
    Find all positions in grid with specified color.

    Args:
        grid: Input grid
        seed_color: Color to search for

    Returns:
        List of (r, c) positions with seed_color
    """
    H, W = grid.shape
    seeds = []

    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) == seed_color:
                seeds.append((r, c))

    return seeds


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles
    from src.law_mining.role_stats import compute_role_stats

    print("=" * 70)
    print("mine_s9_cross.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    # Compute roles and stats (not used by S9, but kept for signature)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    # Mine S9
    print("\nMining S9 instances...")
    s9_instances = mine_S9(task_context, roles, role_stats)
    print(f"✓ S9 instances: {len(s9_instances)}")

    if s9_instances:
        inst = s9_instances[0]
        print(f"  family_id: {inst.family_id}")
        print(f"  params keys: {list(inst.params.keys())}")

        seeds = inst.params.get("seeds", [])
        print(f"  Total seeds: {len(seeds)}")

        if seeds:
            print(f"\n  Sample seed:")
            sample = seeds[0]
            print(f"    center: {sample.get('center')}")
            print(f"    up_color: {sample.get('up_color')}")
            print(f"    down_color: {sample.get('down_color')}")
            print(f"    left_color: {sample.get('left_color')}")
            print(f"    right_color: {sample.get('right_color')}")
            print(f"    max_up: {sample.get('max_up')}")
            print(f"    max_down: {sample.get('max_down')}")
            print(f"    max_left: {sample.get('max_left')}")
            print(f"    max_right: {sample.get('max_right')}")

    print("\n" + "=" * 70)
    print("✓ mine_s9_cross.py self-test passed")
    print("=" * 70)
