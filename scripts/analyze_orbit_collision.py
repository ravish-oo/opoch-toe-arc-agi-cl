"""
Analyze orbit collisions: find pixels that got same predicted color but should differ.

This script identifies "missing features" by finding pixel pairs where:
- Ground truth colors are different
- Predicted colors are the same
- We analyze what features distinguish them to find what's missing from φ(p)
"""

import sys
import json
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics


def get_pixel_features(ex_ctx, r: int, c: int) -> Dict[str, Any]:
    """
    Extract all φ(p) features for a pixel at position (r,c) in INPUT grid.

    Returns dict with all available features for this pixel.
    """
    pixel = (r, c)
    features = {
        "coord": pixel,
        "input_color": int(ex_ctx.input_grid[r, c]),
    }

    # Component features
    if pixel in ex_ctx.object_ids:
        obj_id = ex_ctx.object_ids[pixel]
        features["object_id"] = obj_id

        # Find component
        comp = None
        for c_obj in ex_ctx.components:
            if c_obj.id == obj_id:
                comp = c_obj
                break

        if comp:
            features["component_color"] = comp.color
            features["component_size"] = len(comp.pixels)
            features["component_bbox"] = comp.bbox  # (r_min, r_max, c_min, c_max)

            # Role bits
            if obj_id in ex_ctx.role_bits:
                features["role_bits"] = ex_ctx.role_bits[obj_id]

    # Sector info
    if pixel in ex_ctx.sectors:
        features["sectors"] = ex_ctx.sectors[pixel]

    # Border info
    if pixel in ex_ctx.border_info:
        features["border_info"] = ex_ctx.border_info[pixel]

    # Bands
    if r in ex_ctx.row_bands:
        features["row_band"] = ex_ctx.row_bands[r]
    if c in ex_ctx.col_bands:
        features["col_band"] = ex_ctx.col_bands[c]

    # Nonzero flags
    if r in ex_ctx.row_nonzero:
        features["row_has_nonzero"] = ex_ctx.row_nonzero[r]
    if c in ex_ctx.col_nonzero:
        features["col_has_nonzero"] = ex_ctx.col_nonzero[c]

    # Neighborhood hash
    if pixel in ex_ctx.neighborhood_hashes:
        features["neighborhood_hash"] = ex_ctx.neighborhood_hashes[pixel]

    # Residues
    if r in ex_ctx.row_residues:
        features["row_residues"] = ex_ctx.row_residues[r]
    if c in ex_ctx.col_residues:
        features["col_residues"] = ex_ctx.col_residues[c]

    return features


def find_collision_example(
    task_id: str,
    example_idx: int,
    ex_ctx,
    pred_grid: np.ndarray,
    true_grid: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int], int, int, int]:
    """
    Find a collision: two pixels with different ground truth but same prediction.

    Returns:
        ((r1, c1), (r2, c2), true_color_1, true_color_2, pred_color)
        or None if no collision found
    """
    H, W = true_grid.shape

    # Build map: pred_color -> list of pixels
    pred_to_pixels: Dict[int, list] = {}
    for r in range(H):
        for c in range(W):
            pred_color = int(pred_grid[r, c])
            if pred_color not in pred_to_pixels:
                pred_to_pixels[pred_color] = []
            pred_to_pixels[pred_color].append((r, c))

    # For each predicted color, find pixels with different ground truth
    for pred_color, pixels in pred_to_pixels.items():
        if len(pixels) < 2:
            continue

        # Find two pixels with different ground truth colors
        for i, p1 in enumerate(pixels):
            r1, c1 = p1
            true1 = int(true_grid[r1, c1])

            for p2 in pixels[i+1:]:
                r2, c2 = p2
                true2 = int(true_grid[r2, c2])

                if true1 != true2:
                    # Found collision!
                    return (p1, p2, true1, true2, pred_color)

    return None


def analyze_collision(task_id: str):
    """
    Deep analysis of orbit collisions for a mismatch_train task.
    """
    print("=" * 70)
    print(f"ORBIT COLLISION ANALYSIS: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    # 1. Solve task
    print("\n[1] Solving task...")
    raw_task = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(ctx)

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        use_test_labels=False,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
    )

    print(f"    Status: {diagnostics.status}")
    print(f"    Schemas: {set(diagnostics.schema_ids_used)}")

    if diagnostics.status != "mismatch_train":
        print(f"\n✗ Task status is {diagnostics.status}, not mismatch_train")
        return

    # 2. Analyze each mismatched training example
    print(f"\n[2] Analyzing train mismatches...")

    for mm in diagnostics.train_mismatches:
        ex_idx = mm["example_idx"]
        diff_cells = mm["diff_cells"]

        if not diff_cells or "shape_mismatch" in diff_cells[0]:
            continue

        print(f"\n{'=' * 70}")
        print(f"TRAIN EXAMPLE {ex_idx}")
        print(f"{'=' * 70}")

        ex_ctx = ctx.train_examples[ex_idx]
        true_grid = ex_ctx.output_grid
        pred_grid = outputs["train"][ex_idx]

        print(f"\nGrid shape: {true_grid.shape}")
        print(f"Mismatched cells: {len(diff_cells)}")

        # 3. Find a collision
        collision = find_collision_example(
            task_id, ex_idx, ex_ctx, pred_grid, true_grid
        )

        if not collision:
            print("  No collision found (all predicted colors unique)")
            continue

        p1, p2, true1, true2, pred_color = collision
        r1, c1 = p1
        r2, c2 = p2

        print(f"\n{'─' * 70}")
        print(f"🔴 COLLISION DETECTED")
        print(f"{'─' * 70}")
        print(f"\nPixel 1: ({r1}, {c1})")
        print(f"  Ground truth color: {true1}")
        print(f"  Predicted color: {pred_color}")

        print(f"\nPixel 2: ({r2}, {c2})")
        print(f"  Ground truth color: {true2}")
        print(f"  Predicted color: {pred_color}")

        print(f"\n⚠️  Kernel assigned SAME color ({pred_color}) to both pixels")
        print(f"    but ground truth requires DIFFERENT colors ({true1} vs {true2})")

        # 4. Extract features for both pixels
        print(f"\n{'─' * 70}")
        print(f"FEATURE COMPARISON")
        print(f"{'─' * 70}")

        # NOTE: Features are on INPUT grid, but collision is on OUTPUT grid
        # We need to map output positions back to input positions if geometry changed

        # For geometry-preserving tasks, positions are same
        if ex_ctx.input_H == ex_ctx.output_H and ex_ctx.input_W == ex_ctx.output_W:
            features1 = get_pixel_features(ex_ctx, r1, c1)
            features2 = get_pixel_features(ex_ctx, r2, c2)

            print(f"\nPixel 1 ({r1}, {c1}) features:")
            for key, val in features1.items():
                print(f"  {key:25s}: {val}")

            print(f"\nPixel 2 ({r2}, {c2}) features:")
            for key, val in features2.items():
                print(f"  {key:25s}: {val}")

            # Compare features
            print(f"\n{'─' * 70}")
            print(f"FEATURE DIFFERENCES")
            print(f"{'─' * 70}")

            all_keys = set(features1.keys()) | set(features2.keys())
            same_features = []
            diff_features = []

            for key in sorted(all_keys):
                val1 = features1.get(key, "N/A")
                val2 = features2.get(key, "N/A")

                if val1 == val2:
                    same_features.append(key)
                else:
                    diff_features.append((key, val1, val2))

            print(f"\nFeatures that are SAME (these created the collision):")
            for key in same_features:
                print(f"  ✓ {key}: {features1[key]}")

            if diff_features:
                print(f"\nFeatures that DIFFER (but were not captured by schemas):")
                for key, val1, val2 in diff_features:
                    print(f"  ✗ {key:25s}: {val1} vs {val2}")
            else:
                print(f"\nNO DIFFERING FEATURES FOUND")
                print(f"  → Pixels have IDENTICAL φ(p) features in INPUT")
                print(f"  → But DIFFERENT colors in OUTPUT")
                print(f"  → Missing feature must be related to OUTPUT structure!")

            # 5. Visual context
            print(f"\n{'─' * 70}")
            print(f"VISUAL CONTEXT")
            print(f"{'─' * 70}")

            print(f"\nInput grid (features extracted from here):")
            print(ex_ctx.input_grid)

            print(f"\nGround truth output:")
            print(true_grid)

            print(f"\nPredicted output:")
            print(pred_grid)

            # Mark collision positions
            print(f"\nCollision positions marked:")
            display_grid = np.array(true_grid, dtype=object)
            display_grid[r1, c1] = f"[{true1}]"
            display_grid[r2, c2] = f"({true2})"
            print(display_grid)
            print(f"  [{true1}] = Pixel 1 at ({r1},{c1})")
            print(f"  ({true2}) = Pixel 2 at ({r2},{c2})")

        else:
            print(f"\n⚠️  Geometry-changing task (input {ex_ctx.input_H}×{ex_ctx.input_W} → output {ex_ctx.output_H}×{ex_ctx.output_W})")
            print(f"    Cannot directly map output positions to input features")
            print(f"    Collision analysis requires understanding transformation")

        # Only analyze first collision per example
        break

    print(f"\n{'=' * 70}")
    print(f"END OF ANALYSIS")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_orbit_collision.py <task_id>")
        print("Example: python scripts/analyze_orbit_collision.py 00576224")
        sys.exit(1)

    task_id = sys.argv[1]
    analyze_collision(task_id)
