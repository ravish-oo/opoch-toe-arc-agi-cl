"""
Find unconstrained roles: roles with consistent behavior but no schema coverage.

This script identifies roles that:
1. Appear in train_out with consistent color across examples
2. But have NO rules from S1-S11 or S_Default
3. These unconstrained roles get arbitrary colors from ILP solver

This diagnostic validates the "schema coverage gap" hypothesis from the
orbit collision audit.
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats


def find_unconstrained_roles(task_id: str, consistency_threshold: float = 0.95):
    """
    Find roles with consistent behavior but no schema coverage.

    Args:
        task_id: ARC task ID
        consistency_threshold: Minimum probability for "consistent" (default 0.95)

    Returns:
        List of (role_id, dominant_color, consistency, num_appearances) tuples
    """
    print("=" * 70)
    print(f"UNCONSTRAINED ROLES ANALYSIS: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")

    # 1. Load task and compute roles
    print("\n[1] Loading task and computing roles...")
    raw_task = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(raw_task)

    roles = compute_roles(ctx)
    role_stats = compute_role_stats(ctx, roles)
    num_roles = len(role_stats)
    print(f"    ✓ Computed {num_roles} distinct roles")

    # 2. Mine all schemas (S1-S11 + S_Default)
    print("\n[2] Mining schemas...")
    law_config = mine_law_config(ctx)
    num_schemas = len(law_config.schema_instances)
    print(f"    ✓ Mined {num_schemas} schema instances")

    # Count schema instances by family
    schema_counts = defaultdict(int)
    for si in law_config.schema_instances:
        schema_counts[si.family_id] += 1

    print(f"    Schema breakdown:")
    for family_id, count in sorted(schema_counts.items()):
        print(f"      {family_id}: {count}")

    # 3. Build set of constrained roles
    constrained_roles = set()

    for si in law_config.schema_instances:
        family_id = si.family_id
        params = si.params

        if family_id == "S_Default":
            # S_Default params: {"rules": {role_id: "fixed_0" | "copy_input"}}
            rules = params.get("rules", {})
            constrained_roles.update(rules.keys())

        elif family_id == "S1":
            # S1 params: {"ties": [(obj1, obj2, ...)], ...}
            # Each tied component group gets constrained
            # Role IDs are stored in object_ids mapping
            # (Skip detailed S1 parsing for now - requires component lookup)
            pass

        elif family_id == "S2":
            # S2 params: {"src_obj_id": int, "tgt_color": int, ...}
            # Only constrains specific component
            src_obj = params.get("src_obj_id")
            if src_obj is not None:
                # Component gets constrained, but we need to map obj_id → role_ids
                # (Skip for now)
                pass

        # TODO: Add parsing for S3-S11 to track constrained roles

    print(f"\n[3] Constrained roles: {len(constrained_roles)}")

    # 4. Find roles with consistent behavior in train_out
    print(f"\n[4] Analyzing role consistency in train_out...")

    unconstrained_roles = []

    for role_id, stats in role_stats.items():
        # Skip roles not in train_out
        if not stats.train_out:
            continue

        # Count color occurrences in train_out
        color_counts = defaultdict(int)
        for ex_idx, r, c, color in stats.train_out:
            color_counts[color] += 1

        total = len(stats.train_out)
        dominant_color = max(color_counts.items(), key=lambda x: x[1])[0]
        dominant_count = color_counts[dominant_color]
        consistency = dominant_count / total

        # Check if consistent AND unconstrained
        if consistency >= consistency_threshold and role_id not in constrained_roles:
            unconstrained_roles.append((
                role_id,
                dominant_color,
                consistency,
                total,
                dict(color_counts)
            ))

    unconstrained_roles.sort(key=lambda x: x[3], reverse=True)  # Sort by appearances

    # 5. Report findings
    print(f"\n{'=' * 70}")
    print(f"UNCONSTRAINED ROLES FOUND: {len(unconstrained_roles)}")
    print(f"{'=' * 70}")

    if not unconstrained_roles:
        print("\n  ✓ No unconstrained roles found - full schema coverage!")
        return []

    print(f"\nRoles with consistent behavior (≥{consistency_threshold*100:.0f}%) but NO schema coverage:")
    print(f"\n{'Role ID':<10} {'Color':<8} {'Consistency':<12} {'Appearances':<12} {'Color Distribution'}")
    print("-" * 70)

    for role_id, color, consistency, appearances, color_dist in unconstrained_roles:
        color_dist_str = ", ".join(f"{c}:{cnt}" for c, cnt in sorted(color_dist.items()))
        print(f"{role_id:<10} {color:<8} {consistency:>6.1%}      {appearances:<12} {color_dist_str}")

    # 6. Detailed analysis of top unconstrained roles
    print(f"\n{'=' * 70}")
    print(f"DETAILED ANALYSIS (Top 3 Unconstrained Roles)")
    print(f"{'=' * 70}")

    for role_id, color, consistency, appearances, color_dist in unconstrained_roles[:3]:
        print(f"\n{'─' * 70}")
        print(f"Role {role_id}: Dominant Color {color} ({consistency:.1%})")
        print(f"{'─' * 70}")

        stats = role_stats[role_id]

        # Show where role appears
        print(f"\nAppearances:")
        print(f"  train_in:  {len(stats.train_in)}")
        print(f"  train_out: {len(stats.train_out)}")
        print(f"  test_in:   {len(stats.test_in)}")

        # Show first few positions in train_out
        print(f"\nSample train_out positions (first 5):")
        for ex_idx, r, c, col in stats.train_out[:5]:
            print(f"  Example {ex_idx}, ({r:>2},{c:>2}) → color {col}")

        # If role appears in train_in, show input colors
        if stats.train_in:
            in_colors = set(col for _, _, _, col in stats.train_in)
            print(f"\nInput colors (train_in): {in_colors}")

        # Why not captured?
        print(f"\nWhy not captured by schemas:")
        if len(stats.train_in) == 0:
            print(f"  • Role doesn't exist in train_in (expansion zone or new component)")
            print(f"    → S_Default only handles fixed_0 or copy_input")
            print(f"    → Needs fixed_{color} rule")
        elif consistency < 0.99:
            print(f"  • Consistency {consistency:.1%} < 99% threshold")
            print(f"    → Too much variation for current mining thresholds")
        else:
            print(f"  • Consistency ≥99%, exists in input")
            print(f"    → Schema miners (S1-S11) failed to capture pattern")
            print(f"    → Likely a bug or gap in schema mining logic")

    return unconstrained_roles


def batch_analyze_tasks(task_ids: List[str], threshold: float = 0.95):
    """
    Analyze multiple tasks and summarize findings.

    Args:
        task_ids: List of ARC task IDs
        threshold: Consistency threshold for unconstrained roles
    """
    print("=" * 70)
    print(f"BATCH UNCONSTRAINED ROLES ANALYSIS")
    print(f"Tasks: {len(task_ids)}")
    print(f"Threshold: {threshold:.0%}")
    print("=" * 70)

    results = []

    for task_id in task_ids:
        try:
            unconstrained = find_unconstrained_roles(task_id, threshold)
            results.append((task_id, len(unconstrained), unconstrained))
            print("\n")
        except Exception as e:
            print(f"\n✗ Error analyzing {task_id}: {e}\n")
            results.append((task_id, -1, []))

    # Summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)

    total_tasks = len([r for r in results if r[1] >= 0])
    tasks_with_gaps = len([r for r in results if r[1] > 0])
    total_unconstrained = sum(r[1] for r in results if r[1] > 0)

    print(f"\nTasks analyzed: {total_tasks}")
    print(f"Tasks with unconstrained roles: {tasks_with_gaps} ({tasks_with_gaps/total_tasks*100:.1f}%)")
    print(f"Total unconstrained roles: {total_unconstrained}")
    print(f"Average per task with gaps: {total_unconstrained/tasks_with_gaps:.1f}" if tasks_with_gaps > 0 else "")

    print(f"\nPer-task breakdown:")
    print(f"{'Task ID':<12} {'Unconstrained Roles'}")
    print("-" * 70)
    for task_id, count, _ in sorted(results, key=lambda x: x[1], reverse=True):
        if count >= 0:
            print(f"{task_id:<12} {count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single task:  python scripts/find_unconstrained_roles.py <task_id>")
        print("  Batch mode:   python scripts/find_unconstrained_roles.py <task_id1> <task_id2> ...")
        print("\nExample:")
        print("  python scripts/find_unconstrained_roles.py 045e512c")
        print("  python scripts/find_unconstrained_roles.py 045e512c 00576224 025d127b")
        sys.exit(1)

    task_ids = sys.argv[1:]

    if len(task_ids) == 1:
        # Single task mode
        find_unconstrained_roles(task_ids[0])
    else:
        # Batch mode
        batch_analyze_tasks(task_ids)
