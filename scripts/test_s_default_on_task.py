"""
Test S_Default miner and builder on a specific task.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s_default import mine_S_Default
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance

task_id = "045e512c"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Testing S_Default on task: {task_id}")
print("=" * 70)

# Load task
raw_task = load_arc_task(task_id, challenges_path)
ctx = build_task_context_from_raw(raw_task)

print(f"Train examples: {len(ctx.train_examples)}")
print(f"Test examples: {len(ctx.test_examples)}")
print()

# Check geometry
for i, ex in enumerate(ctx.train_examples):
    print(f"Train {i}: input {ex.input_H}×{ex.input_W} → output {ex.output_H}×{ex.output_W}")
print()

# Compute roles
print("Computing roles...")
roles = compute_roles(ctx)
num_roles = len(set(roles.values()))
print(f"✓ {num_roles} distinct roles")
print()

# Compute role stats
print("Computing role stats...")
role_stats = compute_role_stats(ctx, roles)
print(f"✓ Stats for {len(role_stats)} roles")
print()

# Mine S_Default
print("Mining S_Default...")
schema_instances = mine_S_Default(ctx, roles, role_stats)

if not schema_instances:
    print("✗ No S_Default instance mined!")
else:
    s_default = schema_instances[0]
    rules = s_default.params["rules"]
    print(f"✓ Mined S_Default with {len(rules)} rules")

    # Count rule types
    num_fixed_0 = sum(1 for r in rules.values() if r == "fixed_0")
    num_copy = sum(1 for r in rules.values() if r == "copy_input")
    print(f"  Fixed-0 rules: {num_fixed_0}")
    print(f"  Copy-input rules: {num_copy}")
    print()

    # Show sample rules
    print("Sample rules (first 10):")
    print("-" * 70)
    for i, (role_id, rule) in enumerate(list(rules.items())[:10]):
        stats = role_stats[role_id]
        print(f"  Role {role_id}: {rule}")
        print(f"    train_in:  {len(stats.train_in)} appearances")
        print(f"    train_out: {len(stats.train_out)} appearances")

        # Show sample colors
        if stats.train_out:
            colors_out = [c for _, _, _, c in stats.train_out]
            print(f"    train_out colors: {set(colors_out)}")
    print()

    # Now test building constraints for train example 0
    print("Building constraints for train example 0...")
    ex = ctx.train_examples[0]
    print(f"  Output: {ex.output_H}×{ex.output_W} = {ex.output_H * ex.output_W} pixels")
    print()

    builder = ConstraintBuilder()
    apply_schema_instance(
        family_id="S_Default",
        schema_params=s_default.params,
        task_context=ctx,
        builder=builder,
        example_type="train",
        example_index=0,
    )

    print(f"  Constraints added: {len(builder.constraints)}")

    if len(builder.constraints) == 0:
        print("  ✗ NO CONSTRAINTS ADDED!")
        print()
        print("  Debugging: Check role lookup")
        print("-" * 70)

        # Check if roles_mapping has train_out entries
        roles_mapping = s_default.params.get("roles_mapping", {})
        train_out_keys = [k for k in roles_mapping.keys() if "train_out" in str(k)]
        print(f"  roles_mapping has {len(train_out_keys)} train_out entries")
        if train_out_keys:
            print(f"  Sample train_out keys: {train_out_keys[:5]}")
        else:
            print("  ✗ NO train_out entries in roles_mapping!")

            # Check what keys exist
            sample_keys = list(roles_mapping.keys())[:10]
            print(f"  roles_mapping has these keys: {sample_keys}")
    else:
        print(f"  ✓ Successfully added {len(builder.constraints)} constraints")

print()
print("=" * 70)
