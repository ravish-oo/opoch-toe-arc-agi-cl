"""
Debug script to trace IndexError during test example processing.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.solver.lp_solver import solve_constraints_for_grid
from src.solver.decoding import y_to_grid

task_id = "00d62c1b"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Tracing test example processing for task: {task_id}")
print("=" * 70)

# Load and mine
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)
law_config = mine_law_config(task_context)

print(f"Test examples: {len(task_context.test_examples)}")

# Process TRAIN examples first
print("\n[TRAIN] Solving train examples...")
for i, ex in enumerate(task_context.train_examples):
    print(f"  Train {i}: input={ex.input_grid.shape}, output={ex.output_grid.shape if ex.output_grid is not None else None}")

# Try to process TEST example 0
print("\n[TEST] Processing test example 0...")
ex = task_context.test_examples[0]
print(f"  Test 0: input={ex.input_grid.shape}")
print(f"  output_H={ex.output_H}, output_W={ex.output_W}")

# Determine output dimensions
if ex.output_H is not None:
    H_out, W_out = ex.output_H, ex.output_W
else:
    H_out, W_out = ex.input_H, ex.input_W

print(f"  Determined output dimensions: {H_out}x{W_out}")

num_pixels = H_out * W_out
num_colors = task_context.C

print(f"  num_pixels={num_pixels}, num_colors={num_colors}")

# Build constraints for test example
print("\n  Applying schemas to test example 0...")
builder = ConstraintBuilder()

for i, schema_inst in enumerate(law_config.schema_instances):
    try:
        apply_schema_instance(
            family_id=schema_inst.family_id,
            schema_params=schema_inst.params,
            task_context=task_context,
            builder=builder,
            example_type="test",
            example_index=0,
            schema_constraint_counts={},
        )
        if i % 5 == 0 or i == len(law_config.schema_instances) - 1:
            print(f"    [{i+1}/{len(law_config.schema_instances)}] constraints: {len(builder.constraints)}")
    except Exception as e:
        print(f"\n  ✗ EXCEPTION at schema {i+1} ({schema_inst.family_id}):")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

if len(builder.constraints) > 0:
    print(f"\n  Attempting to solve with {len(builder.constraints)} constraints...")
    try:
        y, solver_status = solve_constraints_for_grid(
            builder=builder,
            num_pixels=num_pixels,
            num_colors=num_colors,
            objective="min_sum"
        )
        print(f"  Solver status: {solver_status}")

        grid_pred = y_to_grid(y, H_out, W_out, num_colors)
        print(f"  ✓ Predicted grid: {grid_pred.shape}")
    except Exception as e:
        print(f"  ✗ EXCEPTION during solving:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
