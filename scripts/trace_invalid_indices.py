"""
Debug script to find invalid pixel indices in constraints.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.constraints.indexing import y_index_to_pc

task_id = "00d62c1b"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Finding invalid indices for task: {task_id}")
print("=" * 70)

# Load and mine
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)
law_config = mine_law_config(task_context)

# Build constraints for train example 0
ex = task_context.train_examples[0]
H_out = ex.output_H if ex.output_H is not None else ex.input_H
W_out = ex.output_W if ex.output_W is not None else ex.input_W
num_pixels = H_out * W_out
num_colors = task_context.C

print(f"Train example 0:")
print(f"  Output dimensions: {H_out}x{W_out}")
print(f"  num_pixels: {num_pixels} (valid indices: 0-{num_pixels-1})")
print(f"  num_colors: {num_colors}")

# Build constraints
builder = ConstraintBuilder()

for schema_inst in law_config.schema_instances:
    apply_schema_instance(
        family_id=schema_inst.family_id,
        schema_params=schema_inst.params,
        task_context=task_context,
        builder=builder,
        example_type="train",
        example_index=0,
        schema_constraint_counts={},
    )

print(f"\nTotal constraints: {len(builder.constraints)}")

# Check each constraint for invalid indices
invalid_found = []

for i, lc in enumerate(builder.constraints):
    for idx in lc.indices:
        p_idx, color = y_index_to_pc(idx, num_colors, 0)

        if p_idx >= num_pixels:
            invalid_found.append({
                'constraint_num': i,
                'y_index': idx,
                'p_idx': p_idx,
                'color': color,
                'constraint': lc
            })
            if len(invalid_found) <= 5:  # Show first 5
                print(f"\n✗ Invalid index in constraint {i}:")
                print(f"    y_index: {idx}")
                print(f"    Decoded to: p_idx={p_idx}, color={color}")
                print(f"    But num_pixels={num_pixels} (max p_idx={num_pixels-1})")
                print(f"    Constraint: {lc}")

if invalid_found:
    print(f"\n{'=' * 70}")
    print(f"Total invalid indices: {len(invalid_found)}")

    # Group by pixel index to see pattern
    p_indices = sorted(set(inv['p_idx'] for inv in invalid_found))
    print(f"Invalid pixel indices: {p_indices[:20]}")  # Show first 20

    if max(p_indices) >= 0:
        print(f"Range: {min(p_indices)} to {max(p_indices)}")
        print(f"Should be: 0 to {num_pixels-1}")
else:
    print("\n✓ No invalid indices found")
