"""
Law miner orchestrator: mine_law_config.

This module provides the high-level law mining function that:
  1. Computes structural roles (WL/q refinement)
  2. Computes role-level statistics
  3. Invokes all schema miners (S1-S12 + S_Default)
  4. Assembles the results into a TaskLawConfig

The orchestrator does not filter, rank, or validate coverage - it simply
aggregates whatever always-true laws the miners discover. The kernel and
diagnostics will later determine if the laws are sufficient to solve the task.
"""

from __future__ import annotations

from typing import List

from src.schemas.context import TaskContext
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.catalog.types import TaskLawConfig, SchemaInstance

from src.law_mining.mine_s1_ties import mine_S1
from src.law_mining.mine_s1_s2_s10 import mine_S2, mine_S10
from src.law_mining.mine_s3_s4_s8_s9 import mine_S3, mine_S4, mine_S8, mine_S9
from src.law_mining.mine_s5_s6_s7_s11 import mine_S5, mine_S6, mine_S7, mine_S11
from src.law_mining.mine_s12 import mine_S12
from src.law_mining.mine_s_default import mine_S_Default


def mine_law_config(task_context: TaskContext) -> TaskLawConfig:
    """
    High-level law miner for a single ARC task.

    Steps:
      - compute structural roles for all pixels across all grids,
      - aggregate role-level statistics,
      - invoke miners for all schema families S1..S12 and S_Default,
      - concatenate all mined SchemaInstance objects into a TaskLawConfig.

    This function does NOT try to judge coverage or correctness; it only
    encodes what the miners infer as always-true laws from training data.
    Any mismatch/infeasibility will be exposed later by the kernel +
    diagnostics when this TaskLawConfig is used.

    Args:
        task_context: TaskContext with train/test examples and φ features

    Returns:
        TaskLawConfig containing all mined SchemaInstance objects

    Example:
        >>> from src.schemas.context import load_arc_task, build_task_context_from_raw
        >>> from pathlib import Path
        >>>
        >>> raw_task = load_arc_task("00576224", Path("data/arc-agi_training_challenges.json"))
        >>> task_context = build_task_context_from_raw(raw_task)
        >>> law_config = mine_law_config(task_context)
        >>> print(f"Mined {len(law_config.schema_instances)} schema instances")
    """
    # 1) Structural role labels (WL/q) for all pixels
    roles = compute_roles(task_context)

    # 2) Aggregate role-level statistics across train_in, train_out, test_in
    role_stats = compute_role_stats(task_context, roles)

    # 3) Initialize schema instance list
    schema_instances: List[SchemaInstance] = []

    # 4) Call each miner in fixed, explicit order
    # S1: tie/equality constraints
    schema_instances.extend(mine_S1(task_context, roles, role_stats))

    # S2, S10: component recolor / frame
    schema_instances.extend(mine_S2(task_context, roles, role_stats))
    schema_instances.extend(mine_S10(task_context, roles, role_stats))

    # S3, S4, S8, S9: bands, residues, tiling, plus-propagation
    schema_instances.extend(mine_S3(task_context, roles, role_stats))
    schema_instances.extend(mine_S4(task_context, roles, role_stats))
    schema_instances.extend(mine_S8(task_context, roles, role_stats))
    schema_instances.extend(mine_S9(task_context, roles, role_stats))

    # S5, S6, S7, S11: template stamping, crop, summary, local codebook
    schema_instances.extend(mine_S5(task_context, roles, role_stats))
    schema_instances.extend(mine_S6(task_context, roles, role_stats))
    schema_instances.extend(mine_S7(task_context, roles, role_stats))
    schema_instances.extend(mine_S11(task_context, roles, role_stats))

    # S12: generalized raycasting (8-directional projection)
    schema_instances.extend(mine_S12(task_context, roles, role_stats))

    # S_Default: law of inertia for unconstrained pixels
    schema_instances.extend(mine_S_Default(task_context, roles, role_stats))

    # 5) Return the TaskLawConfig
    return TaskLawConfig(schema_instances=schema_instances)


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw

    print("=" * 70)
    print("mine_law_config.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    # Mine law config
    print("\nMining law config...")
    law_config = mine_law_config(task_context)
    print(f"✓ Total schema instances mined: {len(law_config.schema_instances)}")

    # Show breakdown by family
    family_counts = {}
    for inst in law_config.schema_instances:
        family_counts[inst.family_id] = family_counts.get(inst.family_id, 0) + 1

    print("\nBreakdown by schema family:")
    for family_id in sorted(family_counts.keys()):
        count = family_counts[family_id]
        print(f"  {family_id}: {count} instances")

    # Show first few instances
    if law_config.schema_instances:
        print(f"\nFirst 3 schema instances:")
        for i, inst in enumerate(law_config.schema_instances[:3]):
            print(f"\n  Instance {i+1}:")
            print(f"    family_id: {inst.family_id}")
            print(f"    params keys: {list(inst.params.keys())}")

    print("\n" + "=" * 70)
    print("✓ mine_law_config.py self-test passed")
    print("=" * 70)
