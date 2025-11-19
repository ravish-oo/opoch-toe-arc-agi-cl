"""
Find which schema created invalid pixel indices.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.constraints.indexing import y_index_to_pc

task_id = "00d62c1b"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Finding which schema created invalid indices for task: {task_id}")
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

print(f"Train example 0: {H_out}x{W_out}, num_pixels={num_pixels}")

# Apply schemas one by one and check for invalid indices
for schema_idx, schema_inst in enumerate(law_config.schema_instances):
    builder = ConstraintBuilder()

    apply_schema_instance(
        family_id=schema_inst.family_id,
        schema_params=schema_inst.params,
        task_context=task_context,
        builder=builder,
        example_type="train",
        example_index=0,
        schema_constraint_counts={},
    )

    # Check if this schema created invalid indices
    has_invalid = False
    for lc in builder.constraints:
        for idx in lc.indices:
            p_idx, color = y_index_to_pc(idx, num_colors, 0)
            if p_idx >= num_pixels:
                if not has_invalid:
                    print(f"\n✗ Schema {schema_idx+1} ({schema_inst.family_id}) created invalid indices:")
                    print(f"   Params: {schema_inst.params}")
                    print(f"   First invalid: p_idx={p_idx} (max={num_pixels-1})")
                has_invalid = True
                break
        if has_invalid:
            break

if not has_invalid:
    print("\n✓ No invalid indices found")
