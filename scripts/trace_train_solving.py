"""
Debug script to trace where train solving fails.
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

print(f"Tracing TRAIN solving for task: {task_id}")
print("=" * 70)

# Load and mine
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)
law_config = mine_law_config(task_context)

print(f"Train examples: {len(task_context.train_examples)}")

# Try to solve FIRST train example
print("\nSolving train example 0...")
ex = task_context.train_examples[0]
print(f"  Input: {ex.input_grid.shape}")
print(f"  Output: {ex.output_grid.shape if ex.output_grid is not None else None}")

H_out = ex.output_H if ex.output_H is not None else ex.input_H
W_out = ex.output_W if ex.output_W is not None else ex.input_W
num_pixels = H_out * W_out
num_colors = task_context.C

print(f"  Output dimensions: {H_out}x{W_out}")
print(f"  num_pixels={num_pixels}, num_colors={num_colors}")

# Build constraints
print("\n  Building constraints...")
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

print(f"  Total constraints: {len(builder.constraints)}")

# Solve
print("\n  Solving ILP...")
try:
    y, solver_status = solve_constraints_for_grid(
        builder=builder,
        num_pixels=num_pixels,
        num_colors=num_colors,
        objective="min_sum"
    )
    print(f"  Solver status: {solver_status}")
    print(f"  y shape: {y.shape}")

    # Decode
    print(f"\n  Decoding to grid...")
    grid_pred = y_to_grid(y, H_out, W_out, num_colors)
    print(f"  ✓ Predicted grid: {grid_pred.shape}")

except Exception as e:
    print(f"  ✗ EXCEPTION:")
    print(f"    {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
