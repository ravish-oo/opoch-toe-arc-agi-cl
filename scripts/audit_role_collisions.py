"""
Audit role collisions: Find distinct objects with same role_id.

This script identifies missing topological invariants by finding objects/pixels
that should behave differently but were assigned the same role_id.

The "Distinguishability Audit" (Law 1 Check):
- Find pixels with different ground truth outputs
- Check if they have the SAME role_id
- If yes: Identify the missing topological invariant
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.runners.kernel import solve_arc_task_with_diagnostics


def get_pixel_features_detailed(ex_ctx, r: int, c: int) -> Dict[str, Any]:
    """Extract all φ(p) features for a pixel."""
    pixel = (r, c)
    features = {
        "coord": pixel,
        "input_color": int(ex_ctx.input_grid[r, c]) if r < ex_ctx.input_H and c < ex_ctx.input_W else None,
    }

    # Component features
    if pixel in ex_ctx.object_ids:
        obj_id = ex_ctx.object_ids[pixel]
        features["object_id"] = obj_id

        comp = next((c for c in ex_ctx.components if c.id == obj_id), None)
        if comp:
            features["component_color"] = comp.color
            features["component_size"] = len(comp.pixels)
            features["component_bbox"] = comp.bbox

            if obj_id in ex_ctx.role_bits:
                features["role_bits"] = ex_ctx.role_bits[obj_id]

    # Spatial features
    if pixel in ex_ctx.sectors:
        features["sectors"] = ex_ctx.sectors[pixel]
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

    # Neighborhood
    if pixel in ex_ctx.neighborhood_hashes:
        features["neighborhood_hash"] = ex_ctx.neighborhood_hashes[pixel]

    # Residues
    if r in ex_ctx.row_residues:
        features["row_residues"] = ex_ctx.row_residues[r]
    if c in ex_ctx.col_residues:
        features["col_residues"] = ex_ctx.col_residues[c]

    return features


def audit_role_collisions(task_id: str):
    """
    Perform distinguishability audit on a task.

    Find pixels that:
    1. Have different ground truth outputs
    2. But have the SAME role_id
    3. Identify what topological property distinguishes them
    """
    print("=" * 70)
    print(f"ROLE COLLISION AUDIT: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")
    solutions_path = Path("data/arc-agi_training_solutions.json")

    # 1. Load and solve task
    print("\n[1] Loading task and computing roles...")
    raw_task = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(raw_task)

    # Compute roles
    roles = compute_roles(ctx)
    role_stats = compute_role_stats(ctx, roles)
    num_roles = len(set(roles.values()))
    print(f"    ✓ Computed {num_roles} distinct roles across all grids")

    # Mine laws and solve
    law_config = mine_law_config(ctx)
    print(f"    ✓ Mined {len(law_config.schema_instances)} schema instances")

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        use_test_labels=False,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
    )

    print(f"\n[2] Task Status: {diagnostics.status}")
    print(f"    Schemas: {set(diagnostics.schema_ids_used)}")
    print(f"    Constraints: {diagnostics.num_constraints}")
    print(f"    Variables: {diagnostics.num_variables}")

    if diagnostics.status not in ("mismatch_train", "infeasible"):
        print(f"\n✗ Task status is {diagnostics.status}, not mismatch_train or infeasible")
        return

    # 2. Analyze role collisions
    print(f"\n[3] Analyzing role collisions...")

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

        print(f"\nGrid: {true_grid.shape[0]}×{true_grid.shape[1]}")
        print(f"Mismatched cells: {len(diff_cells)}")

        # Build role mapping for this example's output
        role_to_pixels: Dict[int, List[Tuple[int, int]]] = {}
        for r in range(ex_ctx.output_H):
            for c in range(ex_ctx.output_W):
                role_key = ("train_out", ex_idx, r, c)
                role_id = roles.get(role_key)
                if role_id is not None:
                    if role_id not in role_to_pixels:
                        role_to_pixels[role_id] = []
                    role_to_pixels[role_id].append((r, c))

        # Find role collisions: same role_id, different ground truth
        collisions_found = 0

        for role_id, pixels in role_to_pixels.items():
            if len(pixels) < 2:
                continue

            # Check if this role has pixels with different ground truth colors
            colors = set(int(true_grid[r, c]) for r, c in pixels)

            if len(colors) > 1:
                # COLLISION FOUND!
                collisions_found += 1

                print(f"\n{'─' * 70}")
                print(f"🔴 COLLISION #{collisions_found}: Role {role_id}")
                print(f"{'─' * 70}")
                print(f"This role has {len(pixels)} pixels with {len(colors)} different ground truth colors: {colors}")

                # Show first few pixels from each color
                color_examples = {}
                for r, c in pixels[:20]:  # Limit to first 20
                    color = int(true_grid[r, c])
                    if color not in color_examples:
                        color_examples[color] = []
                    if len(color_examples[color]) < 3:
                        color_examples[color].append((r, c))

                print(f"\nExample pixels:")
                for color, examples in sorted(color_examples.items()):
                    print(f"  Color {color}: {examples}")

                # Deep dive: Compare two pixels with different colors
                if len(color_examples) >= 2:
                    colors_list = sorted(color_examples.keys())
                    c1, c2 = colors_list[0], colors_list[1]
                    p1 = color_examples[c1][0]
                    p2 = color_examples[c2][0]

                    r1, c1_coord = p1
                    r2, c2_coord = p2

                    print(f"\n{'─' * 70}")
                    print(f"FEATURE COMPARISON")
                    print(f"{'─' * 70}")
                    print(f"\nPixel 1: ({r1}, {c1_coord}) - Ground Truth Color: {c1}")
                    print(f"Pixel 2: ({r2}, {c2_coord}) - Ground Truth Color: {c2}")

                    # Get input-space features (if geometry-preserving)
                    if ex_ctx.input_H == ex_ctx.output_H and ex_ctx.input_W == ex_ctx.output_W:
                        features1 = get_pixel_features_detailed(ex_ctx, r1, c1_coord)
                        features2 = get_pixel_features_detailed(ex_ctx, r2, c2_coord)

                        print(f"\nPixel 1 features:")
                        for k, v in features1.items():
                            print(f"  {k:25s}: {v}")

                        print(f"\nPixel 2 features:")
                        for k, v in features2.items():
                            print(f"  {k:25s}: {v}")

                        # Compare
                        all_keys = set(features1.keys()) | set(features2.keys())
                        same = []
                        diff = []

                        for k in sorted(all_keys):
                            v1 = features1.get(k, "N/A")
                            v2 = features2.get(k, "N/A")
                            if v1 == v2:
                                same.append(k)
                            else:
                                diff.append((k, v1, v2))

                        print(f"\n{'─' * 70}")
                        print(f"SAME FEATURES (caused collision):")
                        print(f"{'─' * 70}")
                        for k in same:
                            print(f"  ✓ {k}: {features1[k]}")

                        if diff:
                            print(f"\n{'─' * 70}")
                            print(f"DIFFERENT FEATURES (not captured by roles):")
                            print(f"{'─' * 70}")
                            for k, v1, v2 in diff:
                                print(f"  ✗ {k:25s}: {v1} vs {v2}")

                        # Visual context
                        print(f"\n{'─' * 70}")
                        print(f"VISUAL CONTEXT")
                        print(f"{'─' * 70}")

                        # Show local neighborhoods
                        print(f"\nInput grid (5×5 neighborhood around pixels):")
                        for label, (rr, cc) in [("Pixel 1", p1), ("Pixel 2", p2)]:
                            print(f"\n{label} at ({rr},{cc}):")
                            for dr in range(-2, 3):
                                row_str = "  "
                                for dc in range(-2, 3):
                                    r_n, c_n = rr + dr, cc + dc
                                    if 0 <= r_n < ex_ctx.input_H and 0 <= c_n < ex_ctx.input_W:
                                        val = ex_ctx.input_grid[r_n, c_n]
                                        if dr == 0 and dc == 0:
                                            row_str += f"[{val}]"
                                        else:
                                            row_str += f" {val} "
                                    else:
                                        row_str += " . "
                                print(row_str)

                        # Show output context
                        print(f"\nGround truth output (5×5 neighborhood):")
                        for label, (rr, cc), color in [("Pixel 1", p1, c1), ("Pixel 2", p2, c2)]:
                            print(f"\n{label} at ({rr},{cc}) - should be {color}:")
                            for dr in range(-2, 3):
                                row_str = "  "
                                for dc in range(-2, 3):
                                    r_n, c_n = rr + dr, cc + dc
                                    if 0 <= r_n < true_grid.shape[0] and 0 <= c_n < true_grid.shape[1]:
                                        val = true_grid[r_n, c_n]
                                        if dr == 0 and dc == 0:
                                            row_str += f"[{val}]"
                                        else:
                                            row_str += f" {val} "
                                    else:
                                        row_str += " . "
                                print(row_str)
                    else:
                        print("\n⚠️  Geometry-changing task - cannot map output to input features")

                # Only show first 2 collisions per example
                if collisions_found >= 2:
                    break

        if collisions_found == 0:
            print("\n  No role collisions found (all pixels with same role have same ground truth)")

    # 3. Analyze infeasible status
    if diagnostics.status == "infeasible":
        print(f"\n{'=' * 70}")
        print(f"INFEASIBLE ANALYSIS")
        print(f"{'=' * 70}")
        print(f"\nTask is INFEASIBLE - constraints are contradictory.")
        print(f"This likely means:")
        print(f"  1. Two schemas fixed the same pixel to different colors")
        print(f"  2. OR: Over-constrained roles (same role_id forced to multiple colors)")
        print(f"\nChecking role statistics for contradictions...")

        # Look for roles that appear in train_out with different colors
        for role_id, stats in role_stats.items():
            if not stats.train_out:
                continue

            colors = set(color for _, _, _, color in stats.train_out)
            if len(colors) > 1:
                print(f"\n  ⚠️  Role {role_id} appears with {len(colors)} different colors: {colors}")
                print(f"      Appearances: {len(stats.train_out)}")
                # Show sample positions
                for color in colors:
                    examples = [(ex, r, c) for ex, r, c, col in stats.train_out if col == color][:3]
                    print(f"      Color {color}: {examples}")

    print(f"\n{'=' * 70}")
    print(f"END OF AUDIT")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/audit_role_collisions.py <task_id>")
        print("Example: python scripts/audit_role_collisions.py 045e512c")
        sys.exit(1)

    task_id = sys.argv[1]
    audit_role_collisions(task_id)
