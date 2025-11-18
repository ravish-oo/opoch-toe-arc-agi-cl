"""
Role statistics aggregator.

This module aggregates all appearances of each role_id across train_in,
train_out, and test_in grids, including their colors.

This is pure factual compression - no heuristics, no defaults, no inference.
Miners (M6.3) will use this data to find always-true invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from collections import defaultdict

from src.schemas.context import TaskContext, ExampleContext
from src.core.grid_types import Grid
from src.law_mining.roles import RolesMapping, NodeKind


@dataclass
class RoleStats:
    """
    Aggregated appearances of a single role_id across all grids of a task.

    Each entry is a tuple: (example_idx, r, c, color)

    Attributes:
        train_in: List of appearances in training input grids
        train_out: List of appearances in training output grids
        test_in: List of appearances in test input grids
    """
    train_in: List[Tuple[int, int, int, int]] = field(default_factory=list)
    train_out: List[Tuple[int, int, int, int]] = field(default_factory=list)
    test_in: List[Tuple[int, int, int, int]] = field(default_factory=list)


def compute_role_stats(
    task_context: TaskContext,
    roles: RolesMapping,
) -> Dict[int, RoleStats]:
    """
    For each role_id, collect all its appearances in train_in, train_out,
    test_in grids with associated colors.

    This function performs pure aggregation with no inference. It simply
    collects all (example_idx, r, c, color) tuples for each role_id based
    on the roles mapping and the actual grid contents.

    Args:
        task_context: TaskContext containing train_examples and test_examples
        roles: RolesMapping from compute_roles() mapping (kind, ex_idx, r, c) -> role_id

    Returns:
        Dictionary mapping role_id -> RoleStats with aggregated appearances

    Example:
        >>> from src.law_mining.roles import compute_roles
        >>> roles = compute_roles(task_context)
        >>> stats = compute_role_stats(task_context, roles)
        >>> # stats[5].train_in contains all train_in appearances of role 5
        >>> # stats[5].train_out contains all train_out appearances of role 5
    """
    # Initialize storage
    role_stats: Dict[int, RoleStats] = defaultdict(RoleStats)

    # Iterate over all role assignments
    for (kind, ex_idx, r, c), role_id in roles.items():
        if kind == "train_in":
            ex: ExampleContext = task_context.train_examples[ex_idx]
            grid: Grid = ex.input_grid
            color = int(grid[r, c])
            role_stats[role_id].train_in.append((ex_idx, r, c, color))

        elif kind == "train_out":
            ex: ExampleContext = task_context.train_examples[ex_idx]
            grid: Grid | None = ex.output_grid
            if grid is None:
                # For training tasks this should not happen, but we fail-safe by skipping.
                continue
            color = int(grid[r, c])
            role_stats[role_id].train_out.append((ex_idx, r, c, color))

        elif kind == "test_in":
            ex: ExampleContext = task_context.test_examples[ex_idx]
            grid: Grid = ex.input_grid
            color = int(grid[r, c])
            role_stats[role_id].test_in.append((ex_idx, r, c, color))

        else:
            # Unknown kind -> this should never happen if RolesMapping is well-formed.
            raise ValueError(f"Unknown node kind in roles mapping: {kind!r}")

    # Return a normal dict (not defaultdict)
    return dict(role_stats)


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles

    print("=" * 70)
    print("role_stats.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    print("\nComputing roles...")
    roles = compute_roles(task_context)
    num_roles = len(set(roles.values()))
    print(f"✓ Computed {num_roles} distinct roles")

    print("\nComputing role stats...")
    role_stats = compute_role_stats(task_context, roles)
    print(f"✓ Computed stats for {len(role_stats)} roles")

    # Verify counts match
    total_train_in = sum(len(stats.train_in) for stats in role_stats.values())
    total_train_out = sum(len(stats.train_out) for stats in role_stats.values())
    total_test_in = sum(len(stats.test_in) for stats in role_stats.values())
    print(f"\n  Total train_in entries: {total_train_in}")
    print(f"  Total train_out entries: {total_train_out}")
    print(f"  Total test_in entries: {total_test_in}")

    # Show sample stats
    print("\nSample role stats:")
    print("-" * 70)
    for i, (role_id, stats) in enumerate(list(role_stats.items())[:5]):
        print(f"  Role {role_id}:")
        print(f"    train_in count:  {len(stats.train_in)}")
        print(f"    train_out count: {len(stats.train_out)}")
        print(f"    test_in count:   {len(stats.test_in)}")

        # Show one entry from train_in if available
        if stats.train_in:
            ex_idx, r, c, color = stats.train_in[0]
            print(f"    Sample train_in: ex={ex_idx} ({r},{c}) color={color}")

    # Verify all role_ids from roles appear in role_stats
    role_ids_from_roles = set(roles.values())
    role_ids_from_stats = set(role_stats.keys())
    assert role_ids_from_roles == role_ids_from_stats, \
        "All role_ids from roles should appear in role_stats"
    print(f"\n✓ Verified: all {len(role_ids_from_roles)} role_ids accounted for")

    print("\n" + "=" * 70)
    print("✓ role_stats.py self-test passed")
    print("=" * 70)
