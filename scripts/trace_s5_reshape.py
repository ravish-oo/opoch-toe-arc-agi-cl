"""
Debug S5 reshape error.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw

task_id = "0520fde7"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Tracing S5 reshape error for task: {task_id}")
print("=" * 70)

# Load task
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)

print(f"Train examples: {len(task_context.train_examples)}\n")

# Check input vs output shapes for each example
for ex_idx, ex in enumerate(task_context.train_examples):
    input_shape = ex.input_grid.shape
    output_shape = ex.output_grid.shape if ex.output_grid is not None else None

    print(f"Train example {ex_idx}:")
    print(f"  Input shape:  {input_shape}")
    print(f"  Output shape: {output_shape}")
    print(f"  Same shape: {input_shape == output_shape}")
    print()

# Now trace through S5 mining logic
print("=" * 70)
print("Simulating S5 miner logic:")
print()

PATCH_RADIUS = 1
patch_size = 2 * PATCH_RADIUS + 1
print(f"PATCH_RADIUS = {PATCH_RADIUS}")
print(f"Expected patch size = {patch_size} × {patch_size} = {patch_size * patch_size} elements")
print()

for ex_idx, ex in enumerate(task_context.train_examples):
    if ex.output_grid is None:
        continue

    grid_in = ex.input_grid
    grid_out = ex.output_grid

    # This is what the buggy code does:
    H, W = grid_in.shape  # <-- Uses INPUT shape

    print(f"Train example {ex_idx}:")
    print(f"  H, W from grid_in: {H} × {W}")
    print(f"  grid_out actual shape: {grid_out.shape}")

    nbh = ex.neighborhood_hashes

    print(f"  neighborhood_hashes count: {len(nbh)}")

    # Check a few pixels
    checked = 0
    for (r, c), h_val in nbh.items():
        r_min = r - PATCH_RADIUS
        r_max = r + PATCH_RADIUS + 1
        c_min = c - PATCH_RADIUS
        c_max = c + PATCH_RADIUS + 1

        bounds_check = not (r_min < 0 or r_max > H or c_min < 0 or c_max > W)

        if bounds_check:
            # Extract patch from output grid
            patch = grid_out[r_min:r_max, c_min:c_max]
            patch_bytes = patch.tobytes()

            if checked < 3:  # Show first 3
                print(f"  Pixel ({r},{c}): hash={h_val}")
                print(f"    Bounds: r[{r_min}:{r_max}], c[{c_min}:{c_max}]")
                print(f"    Bounds check (vs input H={H}, W={W}): PASS")
                print(f"    Extracted patch shape from output: {patch.shape}")
                print(f"    Patch bytes size: {len(patch_bytes) // 8}")  # int64 = 8 bytes
                print(f"    Expected size: {patch_size * patch_size}")

                if len(patch_bytes) // 8 != patch_size * patch_size:
                    print(f"    ✗ MISMATCH! This will cause reshape error.")
                print()

            checked += 1

    print(f"  Total pixels that passed bounds check: {checked}")
    print()

print("=" * 70)
print("Root cause: Bounds check uses INPUT shape, but extraction uses OUTPUT grid.")
print("When input.shape != output.shape, patches can be wrong size.")
