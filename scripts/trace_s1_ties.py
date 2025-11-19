"""
Trace S1 ties for task 1be83260.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config

task_id = "1be83260"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Tracing S1 ties for task: {task_id}")
print("=" * 70)

# Load task
raw_task = load_arc_task(task_id, challenges_path)
ctx = build_task_context_from_raw(raw_task)

print(f"Train examples: {len(ctx.train_examples)}")
for idx, ex in enumerate(ctx.train_examples):
    print(f"  Train {idx}: input {ex.input_grid.shape}, output {ex.output_grid.shape if ex.output_grid is not None else None}")
print()

print(f"Test examples: {len(ctx.test_examples)}")
for idx, ex in enumerate(ctx.test_examples):
    print(f"  Test {idx}: input {ex.input_grid.shape}")
print()

# Mine laws
law_config = mine_law_config(ctx)
print(f"Mined {len(law_config.schema_instances)} schema instances")
print()

# Find S1 instances
for inst in law_config.schema_instances:
    if inst.family_id == "S1":
        print(f"S1 instance:")
        ties = inst.params.get("ties", [])
        print(f"  Tie groups: {len(ties)}")

        for tie_group in ties:
            example_type = tie_group.get("example_type")
            example_index = tie_group.get("example_index")
            pairs = tie_group.get("pairs", [])

            print(f"\n  Tie group: {example_type}[{example_index}]")
            print(f"    Pairs: {len(pairs)}")

            # Get grid dimensions
            if example_type == "train":
                ex = ctx.train_examples[example_index]
            else:
                ex = ctx.test_examples[example_index]

            input_H, input_W = ex.input_grid.shape
            output_H = ex.output_H if ex.output_H is not None else ex.input_H
            output_W = ex.output_W if ex.output_W is not None else ex.input_W

            print(f"    Input dims: {input_H} × {input_W} = {input_H * input_W} pixels")
            print(f"    Output dims: {output_H} × {output_W} = {output_H * output_W} pixels")

            # Check for invalid indices
            max_input_idx = -1
            max_output_idx_if_using_input_w = -1

            for (r1, c1), (r2, c2) in pairs[:5]:  # Show first 5
                # What S1 builder currently does (uses input W)
                p_idx1_wrong = r1 * input_W + c1
                p_idx2_wrong = r2 * input_W + c2

                # What it should do (use output W)
                p_idx1_correct = r1 * output_W + c1
                p_idx2_correct = r2 * output_W + c2

                max_input_idx = max(max_input_idx, p_idx1_wrong, p_idx2_wrong)

                if p_idx1_wrong >= output_H * output_W or p_idx2_wrong >= output_H * output_W:
                    print(f"    ✗ INVALID: ({r1},{c1}), ({r2},{c2})")
                    print(f"      Using input W={input_W}: p_idx1={p_idx1_wrong}, p_idx2={p_idx2_wrong}")
                    print(f"      Max valid idx for output: {output_H * output_W - 1}")

            if max_input_idx >= output_H * output_W:
                print(f"    ✗ Max index {max_input_idx} exceeds output size {output_H * output_W}")

print()
print("=" * 70)
