"""
S_Default miner: Law of Inertia & Colored Vacuum.

This miner discovers the default behavior for pixels not covered by S1-S11.
It analyzes role statistics to determine if roles exhibit:
  - "fixed_{C}": Always becomes specific color C (e.g., fixed_4, fixed_0)
  - "copy_input": Always preserves input color (inertia)
  - "vacuum_{C}": Global rule for expansion zones with consistent color C

This prevents the ILP solver from assigning arbitrary colors to unconstrained pixels.
Handles both geometry-preserving and geometry-changing tasks.
Supports ANY consistent color, not just 0.
"""

from typing import Dict, List, Any

from src.schemas.context import TaskContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance


def mine_S_Default(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
    threshold: float = 0.99
) -> List[SchemaInstance]:
    """
    Mine default behavior rules from role statistics.

    For each role_id, analyze train_in -> train_out transitions:
      - P_zero: Probability of becoming color 0
      - P_copy: Probability of preserving input color

    If P_zero >= threshold: emit "fixed_0" rule
    Elif P_copy >= threshold: emit "copy_input" rule
    Else: ignore (active role, let S1-S11 handle)

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping from compute_roles()
        role_stats: Dict[role_id -> RoleStats] from compute_role_stats()
        threshold: Probability threshold for rule emission (default 0.99)

    Returns:
        List containing single SchemaInstance("S_Default") with rules,
        or empty list if no default rules found

    Example:
        >>> rules = mine_S_Default(ctx, roles, role_stats)
        >>> if rules:
        ...     assert rules[0].family_id == "S_Default"
        ...     assert "rules" in rules[0].params
    """
    rules: Dict[int, str] = {}

    for role_id, stats in role_stats.items():
        # Only analyze roles that appear in train_out
        if not stats.train_out:
            continue

        # Build lookup for train_in colors by (ex_idx, r, c)
        train_in_lookup: Dict[tuple, int] = {}
        for ex_idx, r, c, color in stats.train_in:
            train_in_lookup[(ex_idx, r, c)] = color

        # Analyze train_out appearances
        num_out = len(stats.train_out)
        count_copy = 0
        counts_by_color: Dict[int, int] = {}

        for ex_idx, r, c, c_out in stats.train_out:
            # Count occurrences of each specific color
            counts_by_color[c_out] = counts_by_color.get(c_out, 0) + 1

            # Count copies (where train_in color matches train_out color)
            c_in = train_in_lookup.get((ex_idx, r, c))
            if c_in is not None and c_in == c_out:
                count_copy += 1

        # Calculate probabilities
        P_copy = count_copy / num_out if num_out > 0 else 0.0

        # Emit rules based on thresholds
        # Prioritize copy_input (strongest invariant)
        if P_copy >= threshold:
            rules[role_id] = "copy_input"
        elif counts_by_color:
            # Find most consistent color
            best_color = max(counts_by_color, key=counts_by_color.get)
            P_best_color = counts_by_color[best_color] / num_out if num_out > 0 else 0.0
            if P_best_color >= threshold:
                # Generalized: support ANY fixed color (not just 0)
                rules[role_id] = f"fixed_{best_color}"
        # else: active role, skip

    # Mine Global Vacuum Rule (for expansion zones in geometry-changing tasks)
    # Look at ALL train_out pixels that are OUTSIDE train_in bounds
    vacuum_pixels = 0
    vacuum_counts: Dict[int, int] = {}

    for i, ex in enumerate(task_context.train_examples):
        # Check if output is larger than input (geometry-changing)
        if ex.output_H <= ex.input_H and ex.output_W <= ex.input_W:
            continue  # No expansion

        # Iterate only pixels outside input bounds
        for r in range(ex.output_H):
            for c in range(ex.output_W):
                if r >= ex.input_H or c >= ex.input_W:
                    # This is a vacuum pixel (expansion zone)
                    vacuum_pixels += 1
                    color_out = int(ex.output_grid[r, c])
                    vacuum_counts[color_out] = vacuum_counts.get(color_out, 0) + 1

    # If we have vacuum pixels and they are consistently one color
    if vacuum_pixels > 0 and vacuum_counts:
        # Find most consistent color in expansion zone
        best_vac_color = max(vacuum_counts, key=vacuum_counts.get)
        P_vacuum = vacuum_counts[best_vac_color] / vacuum_pixels
        if P_vacuum >= threshold:
            # Special Rule ID -1 for Global Vacuum (any color)
            rules[-1] = f"vacuum_{best_vac_color}"

    # If no rules found, return empty list
    if not rules:
        return []

    # Return single SchemaInstance with all rules
    # Include roles_mapping so builder can lookup role_ids for pixels
    # FIX: Convert tuple keys to string representation for JSON serialization
    roles_mapping_str = {str(k): v for k, v in roles.items()}

    schema_instance = SchemaInstance(
        family_id="S_Default",
        params={
            "rules": rules,
            "roles_mapping": roles_mapping_str  # String keys for JSON safety
        }
    )

    return [schema_instance]


if __name__ == "__main__":
    # Self-test with toy data
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles
    from src.law_mining.role_stats import compute_role_stats

    print("Testing S_Default miner with real task...")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"Loading task: {task_id}")
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

    print("\nMining S_Default rules...")
    schema_instances = mine_S_Default(task_context, roles, role_stats)

    if schema_instances:
        print(f"✓ Mined {len(schema_instances)} schema instance(s)")
        s_default = schema_instances[0]
        print(f"  Schema family_id: {s_default.family_id}")
        rules = s_default.params["rules"]
        print(f"  Number of rules: {len(rules)}")

        # Count rule types
        num_fixed = sum(1 for r in rules.values() if r.startswith("fixed_"))
        num_copy = sum(1 for r in rules.values() if r == "copy_input")
        num_vacuum = sum(1 for r in rules.values() if r.startswith("vacuum_"))
        print(f"  Fixed-color rules: {num_fixed}")
        print(f"  Copy-input rules: {num_copy}")
        print(f"  Vacuum rules: {num_vacuum}")
        if -1 in rules:
            print(f"  ✓ Global vacuum rule present: {rules[-1]}")

        # Show sample rules
        print("\nSample rules (first 5):")
        print("-" * 70)
        for i, (role_id, rule) in enumerate(list(rules.items())[:5]):
            stats = role_stats[role_id]
            print(f"  Role {role_id}: {rule}")
            print(f"    train_in:  {len(stats.train_in)} appearances")
            print(f"    train_out: {len(stats.train_out)} appearances")
    else:
        print("  No default rules found (all roles are active)")

    print("\n" + "=" * 70)
    print("✓ S_Default miner self-test passed.")
