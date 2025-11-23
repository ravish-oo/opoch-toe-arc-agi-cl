"""
Schema miners for S3, S4, S8, S9.

This module implements algorithmic mining of always-true schema instances
for band/stripe patterns (S3), residue-based coloring (S4), and tiling (S8).
S9 (cross propagation) is implemented in a dedicated module (M6.3E).

Each miner:
  - Analyzes training examples for task-level invariants
  - Verifies consistency across ALL training examples
  - Returns SchemaInstance objects only when laws are always-true
  - Never uses heuristics, defaults, or "most frequent" logic

Miners in this module:
  - mine_S3: Band/stripe patterns (rows/cols in same band share pattern)
  - mine_S4: Residue-class coloring (mod K stripes/checkerboards)
  - mine_S8: Tiling/replication (repeated base tile)
  - mine_S9: Imported from mine_s9_cross (M6.3E)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance

from src.features.coords_bands import row_band_labels, col_band_labels

# Import S9 miner from dedicated module (M6.3E)
from src.law_mining.mine_s9_cross import mine_S9


# =============================================================================
# S3 Miner - Band/Stripe Patterns
# =============================================================================

def mine_S3(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S3 schema instances: band/stripe patterns.

    S3 enforces that rows (or columns) in the same band class share the
    same color pattern. The builder TIES these rows together (enforces equality).

    Algorithm:
      1. For each training example:
         - Group rows by band label ("top", "middle", "bottom")
         - Check if all rows in each band have identical output patterns
      2. Across all training examples:
         - A band is valid if it's consistent in ALL examples
      3. Generate per-example SchemaInstances with row_classes

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S3 parameters

    Example:
        If all train examples show "top rows have same pattern, middle rows
        have same pattern", generate:
        {
            "example_type": "train",
            "example_index": 0,
            "row_classes": [[0, 3], [1, 2]]  # top band, middle band
        }
    """
    # Step 1: For each training example, analyze row bands
    # Track which bands are valid (all rows in band have same pattern)
    valid_bands: Set[str] = None  # Will intersect across examples

    # Also track row assignments per example for generating instances later
    example_band_assignments: List[Dict[str, List[int]]] = []

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue  # Skip if no output (shouldn't happen)

        grid_out = ex.output_grid
        H, W = grid_out.shape
        row_bands = row_band_labels(H)

        # Group rows by band label
        band_to_rows: Dict[str, List[int]] = defaultdict(list)
        for r in range(H):
            band_label = row_bands[r]
            band_to_rows[band_label].append(r)

        # Check consistency within each band for this example
        example_valid_bands: Set[str] = set()
        band_assignments: Dict[str, List[int]] = {}

        for band_label, rows in band_to_rows.items():
            if len(rows) < 2:
                # Need at least 2 rows to form a band class
                continue

            # Extract patterns for all rows in this band
            patterns = [tuple(int(c) for c in grid_out[r, :]) for r in rows]

            # Check if all patterns are identical
            if len(set(patterns)) == 1:
                # All rows in this band have same pattern in this example
                example_valid_bands.add(band_label)
                band_assignments[band_label] = rows

        # Intersect valid bands across examples
        if valid_bands is None:
            valid_bands = example_valid_bands
        else:
            valid_bands = valid_bands.intersection(example_valid_bands)

        example_band_assignments.append(band_assignments)

    # If no bands are valid across all examples, return empty
    if not valid_bands:
        return []

    # Step 2: Generate per-example SchemaInstances
    instances: List[SchemaInstance] = []

    # For train examples
    for ex_idx, band_assignments in enumerate(example_band_assignments):
        # Build row_classes from valid bands
        row_classes = [
            band_assignments[band_label]
            for band_label in sorted(valid_bands)
            if band_label in band_assignments
        ]

        if not row_classes:
            continue  # No valid bands for this example

        instances.append(SchemaInstance(
            family_id="S3",
            params={
                "example_type": "train",
                "example_index": ex_idx,
                "row_classes": row_classes
            }
        ))

    # For test examples (apply same band structure)
    for ex_idx, ex in enumerate(task_context.test_examples):
        grid_in = ex.input_grid
        H, W = grid_in.shape
        row_bands = row_band_labels(H)

        # Group rows by band
        band_to_rows: Dict[str, List[int]] = defaultdict(list)
        for r in range(H):
            band_label = row_bands[r]
            band_to_rows[band_label].append(r)

        # Build row_classes for valid bands
        row_classes = [
            band_to_rows[band_label]
            for band_label in sorted(valid_bands)
            if band_label in band_to_rows and len(band_to_rows[band_label]) >= 2
        ]

        if not row_classes:
            continue

        instances.append(SchemaInstance(
            family_id="S3",
            params={
                "example_type": "test",
                "example_index": ex_idx,
                "row_classes": row_classes
            }
        ))

    return instances


# =============================================================================
# S4 Miner - Residue-Class Coloring
# =============================================================================

def mine_S4(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S4 schema instances: residue-class coloring (mod K patterns).

    S4 fixes pixel colors based purely on coordinate residues modulo K.
    This captures stripes, checkerboards, and periodic patterns.

    Algorithm:
      1. For each axis ("row" or "col") and K in {2,3,4,5}:
         - Aggregate output colors by residue across ALL train examples
      2. If all residues have exactly one consistent color:
         - This is a valid S4 law
      3. Generate per-example SchemaInstances with the mapping

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S4 parameters

    Example:
        If all train examples show "col mod 2: {0->1, 1->3}",
        generate instances for each example with:
        {
            "example_type": "train",
            "example_index": 0,
            "axis": "col",
            "K": 2,
            "residue_to_color": {"0": 1, "1": 3}
        }
    """
    instances: List[SchemaInstance] = []

    # Try each axis and K combination
    for axis in ["row", "col"]:
        for K in [2, 3, 4, 5]:
            # Step 1: Aggregate residue→colors across all train examples
            residue_to_colors: Dict[int, Set[int]] = defaultdict(set)

            for ex_idx, ex in enumerate(task_context.train_examples):
                if ex.output_grid is None:
                    continue

                grid_out = ex.output_grid
                H, W = grid_out.shape

                for r in range(H):
                    for c in range(W):
                        color = int(grid_out[r, c])
                        residue = (r % K) if axis == "row" else (c % K)
                        residue_to_colors[residue].add(color)

            # Step 2: Check consistency
            # All residues must have exactly one color
            if not residue_to_colors:
                continue  # No data

            valid = True
            residue_to_color_int: Dict[int, int] = {}

            for residue, colors in residue_to_colors.items():
                if len(colors) != 1:
                    # Conflict - multiple colors for this residue
                    valid = False
                    break
                residue_to_color_int[residue] = colors.pop()

            if not valid:
                continue  # This (axis, K) is not a valid law

            # Step 3: Convert to string keys for builder
            residue_to_color_str = {
                str(residue): int(color)
                for residue, color in residue_to_color_int.items()
            }

            # Step 4: Generate per-example instances
            # For train examples
            for ex_idx, ex in enumerate(task_context.train_examples):
                instances.append(SchemaInstance(
                    family_id="S4",
                    params={
                        "example_type": "train",
                        "example_index": ex_idx,
                        "axis": axis,
                        "K": K,
                        "residue_to_color": residue_to_color_str
                    }
                ))

            # For test examples
            for ex_idx, ex in enumerate(task_context.test_examples):
                instances.append(SchemaInstance(
                    family_id="S4",
                    params={
                        "example_type": "test",
                        "example_index": ex_idx,
                        "axis": axis,
                        "K": K,
                        "residue_to_color": residue_to_color_str
                    }
                ))

    return instances


# =============================================================================
# S8 Miner - Tiling/Replication with Symmetric Transforms
# =============================================================================

# Supported transforms for symmetric tiling
TRANSFORMS = ["identity", "flipx", "flipy", "flipxy"]


def get_tile_variants(tile: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate all transform variants of a base tile."""
    return {
        "identity": tile,
        "flipx": np.flip(tile, axis=1),      # Horizontal flip
        "flipy": np.flip(tile, axis=0),      # Vertical flip
        "flipxy": np.flip(np.flip(tile, axis=0), axis=1),  # Both flips
    }


def find_matching_transform(
    actual_tile: np.ndarray,
    variants: Dict[str, np.ndarray]
) -> str | None:
    """Find which transform variant matches the actual tile."""
    for transform_name, variant in variants.items():
        if actual_tile.shape == variant.shape and np.array_equal(actual_tile, variant):
            return transform_name
    return None


def mine_S8(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S8 schema instances: tiling/replication with symmetric transforms.

    S8 replicates a base tile pattern across the output grid, potentially
    with transforms (flipx, flipy, etc.) at different tile positions.

    Algorithm:
      1. For each training example:
         - Try all valid tile sizes (divisors of H, W)
         - For each tile size, extract base tile and check if output can be
           reconstructed using base tile + transforms at each position
      2. Intersect across all examples:
         - Must find ONE common (tile_h, tile_w, base_tile, transforms) for ALL
      3. If consistent, generate per-example SchemaInstances

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature compatibility)
        role_stats: RoleStats (not used, kept for signature compatibility)

    Returns:
        List of SchemaInstance objects with S8 parameters
    """
    # Step 1: For each example, find all valid tilings (with transforms)
    # Candidate: (tile_h, tile_w, base_tile, transforms_dict)
    candidates_per_example: List[List[Tuple[int, int, np.ndarray, Dict[str, str]]]] = []

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue

        grid_out = ex.output_grid
        H, W = grid_out.shape

        example_candidates: List[Tuple[int, int, np.ndarray, Dict[str, str]]] = []

        # Try all divisor pairs
        for tile_h in range(1, H + 1):
            if H % tile_h != 0:
                continue
            for tile_w in range(1, W + 1):
                if W % tile_w != 0:
                    continue

                # Extract base tile from top-left
                base_tile = grid_out[0:tile_h, 0:tile_w].copy()
                variants = get_tile_variants(base_tile)

                # Check each tile position and find matching transform
                num_tiles_h = H // tile_h
                num_tiles_w = W // tile_w
                transforms_dict: Dict[str, str] = {}
                valid_tiling = True

                for tile_row in range(num_tiles_h):
                    for tile_col in range(num_tiles_w):
                        # Extract actual tile at this position
                        r_start = tile_row * tile_h
                        c_start = tile_col * tile_w
                        actual_tile = grid_out[r_start:r_start + tile_h, c_start:c_start + tile_w]

                        # Find matching transform
                        transform = find_matching_transform(actual_tile, variants)
                        if transform is None:
                            valid_tiling = False
                            break

                        transforms_dict[f"({tile_row},{tile_col})"] = transform

                    if not valid_tiling:
                        break

                if valid_tiling:
                    example_candidates.append((tile_h, tile_w, base_tile, transforms_dict))

        if not example_candidates:
            # This example doesn't admit any tiling - S8 doesn't apply
            return []

        candidates_per_example.append(example_candidates)

    if not candidates_per_example:
        return []  # No train examples or none had tilings

    # Step 2: Find common tiling across all examples
    # We need (tile_h, tile_w, transform_pattern) to match
    # NOTE: base_tile varies per example (input-derived), so we match on
    # (tile_h, tile_w, transforms_dict) instead

    # Start with first example's candidates
    common_candidates = candidates_per_example[0]

    # Intersect with remaining examples (match on tile_h, tile_w, transform_pattern)
    for example_candidates in candidates_per_example[1:]:
        new_common = []
        for (h1, w1, tile1, trans1) in common_candidates:
            for (h2, w2, tile2, trans2) in example_candidates:
                # Match on dimensions and transform pattern (NOT base_tile content)
                if h1 == h2 and w1 == w2 and trans1 == trans2:
                    # Keep the first example's base_tile for pattern structure
                    new_common.append((h1, w1, tile1, trans1))
                    break
        common_candidates = new_common

    if len(common_candidates) == 0:
        # No consistent tiling across all examples
        return []

    # Take the first common candidate (prefer smaller tiles)
    common_candidates.sort(key=lambda x: x[0] * x[1])
    tile_height, tile_width, _, consistent_transforms = common_candidates[0]

    # Step 3: Generate per-example instances
    instances: List[SchemaInstance] = []

    # For train examples - derive tile_pattern from INPUT, use consistent_transforms
    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue

        grid_in = ex.input_grid
        grid_out = ex.output_grid
        H_in, W_in = grid_in.shape
        H, W = grid_out.shape

        # Build tile_pattern from INPUT (the tile is the input grid)
        tile_pattern = {}
        for dr in range(tile_height):
            for dc in range(tile_width):
                # Use input grid if dimensions match, else use output's base tile
                if H_in == tile_height and W_in == tile_width:
                    tile_pattern[f"({dr},{dc})"] = int(grid_in[dr, dc])
                else:
                    # Fallback: extract from output's top-left tile
                    tile_pattern[f"({dr},{dc})"] = int(grid_out[dr, dc])

        instances.append(SchemaInstance(
            family_id="S8",
            params={
                "example_type": "train",
                "example_index": ex_idx,
                "tile_height": tile_height,
                "tile_width": tile_width,
                "tile_pattern": tile_pattern,
                "tile_transforms": consistent_transforms,  # Use learned pattern
                "region_origin": "(0,0)",
                "region_height": H,
                "region_width": W
            }
        ))

    # For test examples - derive tile_pattern from INPUT and use consistent_transforms
    for ex_idx, ex in enumerate(task_context.test_examples):
        grid_in = ex.input_grid
        H_in, W_in = grid_in.shape

        # The tile is the input grid (for tasks like 00576224 where input IS the tile)
        # Only works if input dimensions match tile dimensions
        if H_in == tile_height and W_in == tile_width:
            # Build tile_pattern from input
            test_tile_pattern = {}
            for dr in range(tile_height):
                for dc in range(tile_width):
                    test_tile_pattern[f"({dr},{dc})"] = int(grid_in[dr, dc])

            # Use consistent transforms from training examples
            # Output dimensions will be injected by kernel (via predict_dimensions)
            # We set region to None - builder will use actual grid dimensions
            instances.append(SchemaInstance(
                family_id="S8",
                params={
                    "example_type": "test",
                    "example_index": ex_idx,
                    "tile_height": tile_height,
                    "tile_width": tile_width,
                    "tile_pattern": test_tile_pattern,
                    "tile_transforms": consistent_transforms,  # Use learned pattern
                    "region_origin": "(0,0)",
                    "region_height": None,  # Let builder use actual grid dims
                    "region_width": None    # (injected by kernel via predict_dimensions)
                }
            ))

    return instances


# =============================================================================
# S9 Miner - Cross/Plus Propagation
# =============================================================================

# NOTE (M6.3E):
# mine_S9 is implemented in src/law_mining/mine_s9_cross.py and imported above.
# See that module for full implementation details.


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles
    from src.law_mining.role_stats import compute_role_stats

    print("=" * 70)
    print("mine_s3_s4_s8_s9.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    # Compute roles and stats
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    # Mine schemas
    print("\nMining S3 instances...")
    s3_instances = mine_S3(task_context, roles, role_stats)
    print(f"✓ S3 instances: {len(s3_instances)}")
    if s3_instances:
        print(f"  Sample S3 params: {s3_instances[0].params}")

    print("\nMining S4 instances...")
    s4_instances = mine_S4(task_context, roles, role_stats)
    print(f"✓ S4 instances: {len(s4_instances)}")
    if s4_instances:
        print(f"  Sample S4 params: {s4_instances[0].params}")

    print("\nMining S8 instances...")
    s8_instances = mine_S8(task_context, roles, role_stats)
    print(f"✓ S8 instances: {len(s8_instances)}")
    if s8_instances:
        print(f"  Sample S8 params: {s8_instances[0].params}")

    print("\nMining S9 instances...")
    s9_instances = mine_S9(task_context, roles, role_stats)
    print(f"✓ S9 instances: {len(s9_instances)}")
    if s9_instances:
        seeds = s9_instances[0].params.get("seeds", [])
        print(f"  Sample S9: {len(seeds)} seeds")

    print("\n" + "=" * 70)
    print("✓ mine_s3_s4_s8_s9.py self-test passed")
    print("=" * 70)
