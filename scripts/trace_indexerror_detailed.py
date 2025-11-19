"""
Detailed trace of IndexError in test solving.
"""

import traceback
from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.solver.lp_solver import solve_constraints_for_grid
from src.solver.decoding import y_to_grid

task_id = "ff28f65a"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Detailed trace of IndexError for task: {task_id}")
print("=" * 70)

# Load task
raw_task = load_arc_task(task_id, challenges_path)
ctx = build_task_context_from_raw(raw_task)

print(f"Train examples: {len(ctx.train_examples)}")
print(f"Test examples: {len(ctx.test_examples)}")
print()

# Mine laws
law_config = mine_law_config(ctx)
print(f"Mined {len(law_config.schema_instances)} schema instances")
print()

# Train solving for train example 0
print("Solving TRAIN example 0...")
print("-" * 70)

ex = ctx.train_examples[0]
print(f"Train example 0:")
print(f"  input shape: {ex.input_grid.shape}")
print(f"  output shape: {ex.output_grid.shape if ex.output_grid is not None else 'None'}")
print(f"  ex.output_H: {ex.output_H}")
print(f"  ex.output_W: {ex.output_W}")
print()

# Determine output dimensions
H_out = ex.output_H if ex.output_H is not None else ex.input_H
W_out = ex.output_W if ex.output_W is not None else ex.input_W

print(f"Determined output dimensions: {H_out} × {W_out}")
num_pixels = H_out * W_out
num_colors = ctx.C
print(f"num_pixels: {num_pixels}, num_colors: {num_colors}")
print()

# Build constraints
print("Building constraints...")
builder = ConstraintBuilder()
schema_constraint_counts = {}

for idx, schema_inst in enumerate(law_config.schema_instances):
    print(f"  Applying schema {idx+1}/{len(law_config.schema_instances)}: {schema_inst.family_id}")
    try:
        apply_schema_instance(
            family_id=schema_inst.family_id,
            schema_params=schema_inst.params,
            task_context=ctx,
            builder=builder,
            example_type="train",
            example_index=0,
            schema_constraint_counts=schema_constraint_counts,
        )
        print(f"    ✓ Success, total constraints now: {len(builder.constraints)}")
    except Exception as e:
        print(f"    ✗ FAILED: {type(e).__name__}: {e}")
        print()
        print("Full traceback:")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)
        break

print()
print(f"Total constraints built: {len(builder.constraints)}")
print()

# Solve ILP
print("Solving ILP...")
try:
    y, solver_status = solve_constraints_for_grid(
        builder=builder,
        num_pixels=num_pixels,
        num_colors=num_colors,
        objective="min_sum"
    )
    print(f"✓ Solver status: {solver_status}")
    print(f"✓ Solution shape: {len(y)} pixels")
except Exception as e:
    print(f"✗ Solver FAILED: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    print("=" * 70)
    traceback.print_exc()
    print("=" * 70)
    import sys
    sys.exit(1)

# Decode to grid
print()
print("Decoding to grid...")
try:
    grid_pred = y_to_grid(y, H_out, W_out, num_colors)
    print(f"✓ Grid decoded, shape: {grid_pred.shape}")
except Exception as e:
    print(f"✗ Decode FAILED: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    print("=" * 70)
    traceback.print_exc()
    print("=" * 70)
    import sys
    sys.exit(1)

print()
print("=" * 70)
print("✓ Train example 0 solved successfully")
