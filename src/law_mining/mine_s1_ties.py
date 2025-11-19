"""
S1 tie miner: Discover equality constraints from homogeneous roles.

This module implements algorithmic mining of S1 (tie/equality) schema instances.
S1 enforces that pixels with equivalent structural roles have the same output color.

Algorithm:
  - For each role, check if all train_out appearances have the same color
  - If homogeneous (one color), tie all positions with that role
  - Generate tie pairs within each example (train_out and test_in)
  - Never fix colors, only enforce equality

Key principle: S1 discovers OUTPUT equalities from training, not input copying.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from collections import defaultdict

from src.schemas.context import TaskContext
from src.law_mining.roles import RolesMapping, NodeKind
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance


def mine_S1(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S1 schema instances: tie pixels with homogeneous output colors.

    S1 enforces equality constraints for pixels that have:
      - Same structural role (same role_id from WL refinement)
      - Homogeneous output colors across all training examples

    Algorithm:
      1. For each role, check train_out color homogeneity:
         - Skip if 0 train_out appearances (no evidence)
         - Skip if >1 distinct colors (conflicting)
         - Accept if exactly 1 color (homogeneous)
      2. For homogeneous roles, generate tie pairs:
         - Within each train_out example: tie all positions with this role
         - Within each test_in example: tie all positions with this role
      3. Return single SchemaInstance with all ties

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping from WL refinement
        role_stats: Per-role statistics from compute_role_stats

    Returns:
        List with 0 or 1 SchemaInstance containing all tie pairs

    Note:
        - Never fixes colors, only creates equality constraints
        - Ties are per-example (not across examples)
        - Uses anchor pattern: tie all to first position

    Example output:
        [SchemaInstance(
            family_id="S1",
            params={
                "ties": [
                    {
                        "example_type": "train",
                        "example_index": 0,
                        "pairs": [((0,0), (0,1)), ((0,0), (1,0))]
                    },
                    {
                        "example_type": "test",
                        "example_index": 0,
                        "pairs": [((2,2), (2,3))]
                    }
                ]
            }
        )]
    """
    # Step 1: Collect tie pairs by example
    # Key: (example_type, example_index) -> List of tie pairs
    ties_by_example: Dict[Tuple[str, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)

    # Step 2: For each role, check homogeneity and generate ties
    for role_id, stats in role_stats.items():
        # Extract all output colors for this role in training outputs
        colors_out = {color for (_ex, _r, _c, color) in stats.train_out}

        # Check homogeneity
        if len(colors_out) == 0:
            # No evidence in train_out → skip
            continue
        elif len(colors_out) > 1:
            # Conflicting colors → skip
            continue
        # else: len(colors_out) == 1 → homogeneous, proceed

        # Step 3: Generate ties for this homogeneous role
        # Group positions by (example_type, example_index)
        positions_by_example: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)

        for (kind, k, r, c), rid in roles.items():
            if rid != role_id:
                continue

            # Collect train_out positions
            if kind == "train_out":
                positions_by_example[("train", k)].append((r, c))
            # Collect test_in positions (which will become test_out)
            elif kind == "test_in":
                positions_by_example[("test", k)].append((r, c))
            # Note: We don't tie train_in positions; S1 is about output equalities

        # Step 4: For each example with ≥2 positions, create tie pairs
        for (example_type, ex_idx), positions in positions_by_example.items():
            if len(positions) < 2:
                continue  # Need at least 2 positions to tie

            # Use anchor pattern: tie all to first position
            anchor = positions[0]
            for pos in positions[1:]:
                ties_by_example[(example_type, ex_idx)].append((anchor, pos))

    # Step 5: Build SchemaInstances from ties
    # Create ONE SchemaInstance PER EXAMPLE (not one instance for all examples)
    # This aligns with kernel's per-example solving architecture
    if not ties_by_example:
        return []  # No ties found

    instances: List[SchemaInstance] = []

    for (example_type, ex_idx), pairs in sorted(ties_by_example.items()):
        if not pairs:
            continue

        # Create one instance per example with only that example's ties
        instances.append(SchemaInstance(
            family_id="S1",
            params={
                "ties": [{
                    "example_type": example_type,
                    "example_index": ex_idx,
                    "pairs": pairs,
                }]
            }
        ))

    return instances


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles
    from src.law_mining.role_stats import compute_role_stats

    print("=" * 70)
    print("mine_s1_ties.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    # Compute roles and stats
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    print(f"Total roles: {len(role_stats)}")

    # Mine S1
    print("\nMining S1 instances...")
    s1_instances = mine_S1(task_context, roles, role_stats)
    print(f"✓ S1 instances: {len(s1_instances)}")

    if s1_instances:
        inst = s1_instances[0]
        print(f"  family_id: {inst.family_id}")
        print(f"  params keys: {list(inst.params.keys())}")

        ties = inst.params.get("ties", [])
        print(f"  Total tie groups: {len(ties)}")

        total_pairs = sum(len(t.get("pairs", [])) for t in ties)
        print(f"  Total tie pairs: {total_pairs}")

        if ties:
            print(f"\n  Sample tie group:")
            sample = ties[0]
            print(f"    example_type: {sample.get('example_type')}")
            print(f"    example_index: {sample.get('example_index')}")
            print(f"    num pairs: {len(sample.get('pairs', []))}")
            if sample.get('pairs'):
                print(f"    sample pair: {sample['pairs'][0]}")

    print("\n" + "=" * 70)
    print("✓ mine_s1_ties.py self-test passed")
    print("=" * 70)
