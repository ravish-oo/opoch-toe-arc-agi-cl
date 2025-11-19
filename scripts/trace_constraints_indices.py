"""
Trace constraints and check for invalid pixel indices.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.constraints.indexing import y_index_to_pc

task_id = "ff28f65a"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Tracing constraints for task: {task_id}")
print("=" * 70)

# Load task
raw_task = load_arc_task(task_id, challenges_path)
ctx = build_task_context_from_raw(raw_task)

# Mine laws
law_config = mine_law_config(ctx)

# Build constraints for train example 0
ex = ctx.train_examples[0]
H_out = ex.output_H if ex.output_H is not None else ex.input_H
W_out = ex.output_W if ex.output_W is not None else ex.input_W
num_pixels = H_out * W_out
num_colors = ctx.C

print(f"Train example 0:")
print(f"  Output: {H_out} × {W_out} = {num_pixels} pixels")
print(f"  Colors: {num_colors}")
print(f"  Valid pixel indices: 0 to {num_pixels - 1}")
print(f"  Valid y indices: 0 to {num_pixels * num_colors - 1}")
print()

# Build constraints
builder = ConstraintBuilder()
schema_constraint_counts = {}

for schema_inst in law_config.schema_instances:
    apply_schema_instance(
        family_id=schema_inst.family_id,
        schema_params=schema_inst.params,
        task_context=ctx,
        builder=builder,
        example_type="train",
        example_index=0,
        schema_constraint_counts=schema_constraint_counts,
    )

print(f"Total constraints built: {len(builder.constraints)}")
print()

# Check each constraint for invalid indices
invalid_constraints = []

for i, lc in enumerate(builder.constraints):
    for idx in lc.indices:
        p_idx, color = y_index_to_pc(idx, num_colors, 0)

        if p_idx >= num_pixels:
            invalid_constraints.append({
                'constraint_num': i,
                'y_index': idx,
                'p_idx': p_idx,
                'color': color,
            })

if invalid_constraints:
    print(f"✗ Found {len(invalid_constraints)} invalid indices in constraints")
    print()

    # Show first few
    for inv in invalid_constraints[:10]:
        print(f"Constraint {inv['constraint_num']}:")
        print(f"  y_index: {inv['y_index']}")
        print(f"  Decoded to: p_idx={inv['p_idx']}, color={inv['color']}")
        print(f"  But num_pixels={num_pixels} (max p_idx={num_pixels - 1})")
        print()

    # Group by pixel index
    p_indices = sorted(set(inv['p_idx'] for inv in invalid_constraints))
    print(f"Invalid pixel indices: {p_indices[:20]}")
    if p_indices:
        print(f"Range: {min(p_indices)} to {max(p_indices)}")
        print(f"Should be: 0 to {num_pixels - 1}")
else:
    print("✓ All constraint indices are valid")
