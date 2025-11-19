"""
Schema miners for S5, S6, S7, S11.

This module implements algorithmic mining of always-true schema instances
for template stamping, cropping, aggregation, and local codebook schemas.

Miners in this module:
  - mine_S5: Template stamping (seed → patch)
  - mine_S6: Crop to ROI
  - mine_S7: Block/summary grids
  - mine_S11: Local 3×3 codebook
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance

from src.features.components import connected_components_by_color


# =============================================================================
# S5 Miner - Template Stamping (Seed → Template)
# =============================================================================

def mine_S5(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S5 schema instances: template stamping at seed pixels.

    S5 detects seed pixels by their neighborhood hash and stamps a
    pre-defined template patch around each seed occurrence.

    Algorithm:
      1. For each train example:
         - Compute neighborhood hashes on input
         - For each pixel, extract output patch
         - Record hash -> patch mappings
      2. Aggregate across all train examples:
         - Track all patches seen for each hash
      3. Keep only hashes with exactly one consistent patch
      4. Generate SchemaInstance for each (train + test) example

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S5 parameters

    Example mined rule:
        If all training examples show "hash 12345 -> 3×3 patch P",
        then generate SchemaInstances with:
        {
            "example_type": "train",
            "example_index": 0,
            "seed_templates": {
                "12345": { "(0,0)": 5, "(0,1)": 5, ... }
            }
        }
    """
    # Use radius=1 for 3×3 patches (standard S5)
    PATCH_RADIUS = 1

    # Step 1: Mine task-wide invariants
    # Key: hash_value -> Set[patch_as_bytes]
    hash_to_patches: Dict[int, Set[bytes]] = defaultdict(set)

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue  # Skip if no output

        grid_in = ex.input_grid
        grid_out = ex.output_grid
        H, W = grid_in.shape

        # Get neighborhood hashes (already computed in ExampleContext)
        nbh = ex.neighborhood_hashes  # Dict[(r,c), int]

        # For each pixel, check if full patch fits
        for (r, c), h_val in nbh.items():
            # Check if patch fits in output grid
            r_min = r - PATCH_RADIUS
            r_max = r + PATCH_RADIUS + 1
            c_min = c - PATCH_RADIUS
            c_max = c + PATCH_RADIUS + 1

            if r_min < 0 or r_max > H or c_min < 0 or c_max > W:
                continue  # Patch doesn't fit

            # Extract output patch
            patch = grid_out[r_min:r_max, c_min:c_max]

            # Convert to bytes for set comparison
            patch_bytes = patch.tobytes()

            # Record this hash -> patch mapping
            hash_to_patches[h_val].add(patch_bytes)

    # Step 2: Filter to only consistent mappings
    # Build dict of hash -> canonical_patch for always-true rules
    consistent_templates: Dict[int, np.ndarray] = {}

    for h_val, patch_set in hash_to_patches.items():
        if len(patch_set) == 1:
            # All train examples agree on this hash -> patch mapping
            patch_bytes = patch_set.pop()
            # Convert back to numpy array
            patch_size = 2 * PATCH_RADIUS + 1
            patch = np.frombuffer(patch_bytes, dtype=np.int64).reshape(patch_size, patch_size)
            consistent_templates[h_val] = patch

    if not consistent_templates:
        return []  # No S5 rules found

    # Step 3: Convert to builder parameter format
    # seed_templates: { hash_str: { "(dr,dc)": color } }
    seed_templates: Dict[str, Dict[str, int]] = {}

    for h_val, patch in consistent_templates.items():
        offset_dict = {}
        for dr in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            for dc in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                offset_key = f"({dr},{dc})"
                color = int(patch[dr + PATCH_RADIUS, dc + PATCH_RADIUS])
                offset_dict[offset_key] = color

        seed_templates[str(h_val)] = offset_dict

    # Step 4: Generate SchemaInstances for each example
    instances: List[SchemaInstance] = []

    # Generate for train examples
    for ex_idx, ex in enumerate(task_context.train_examples):
        instances.append(SchemaInstance(
            family_id="S5",
            params={
                "example_type": "train",
                "example_index": ex_idx,
                "seed_templates": seed_templates
            }
        ))

    # Generate for test examples
    for ex_idx, ex in enumerate(task_context.test_examples):
        instances.append(SchemaInstance(
            family_id="S5",
            params={
                "example_type": "test",
                "example_index": ex_idx,
                "seed_templates": seed_templates
            }
        ))

    return instances


# =============================================================================
# S6 Miner - Crop to ROI
# =============================================================================

def mine_S6(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S6 schema instances: crop to region of interest.

    S6 creates a smaller output grid that is a crop of the input.
    We mine by detecting that outputs are exact subgrids of inputs
    and finding a consistent cropping rule from a finite set of explicit rules.

    Rule families tested (in priority order):
      - Rule A: Fixed offset (same r0, c0 across all examples)
      - Rule B: Largest component of color c (for each color that appears)
      - Rule C: Largest component (any color)

    Algorithm:
      1. For each train example:
         - Check if output is smaller than input
         - Find ALL possible crop positions (candidates)
      2. Test rules in order:
         - Rule A: Fixed offset
         - Rule B: For each color, test "largest component of that color"
         - Rule C: "Largest component (any color)"
      3. If a rule matches, generate SchemaInstance with out_to_in mapping
         computed per-example by applying the rule

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S6 parameters

    Note:
        The out_to_in mapping is computed PER EXAMPLE by applying the
        discovered rule to that example's input. For Rule A, all examples
        use the same offset. For Rules B/C, each example computes its crop
        from its own largest component.
    """
    # Step S6.0: Detect that each train output is a crop of its input
    # For each example, find ALL valid crop positions
    candidates_per_example: List[List[Tuple[int, int]]] = []
    output_dims: List[Tuple[int, int]] = []

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            return []  # No output

        grid_in = ex.input_grid
        grid_out = ex.output_grid
        H_in, W_in = grid_in.shape
        H_out, W_out = grid_out.shape

        # Check if output is smaller or equal
        if H_out > H_in or W_out > W_in:
            return []  # Not a crop

        output_dims.append((H_out, W_out))

        # Find ALL valid crop positions
        candidates = []
        for r0 in range(H_in - H_out + 1):
            for c0 in range(W_in - W_out + 1):
                sub = grid_in[r0:r0+H_out, c0:c0+W_out]
                if np.array_equal(sub, grid_out):
                    candidates.append((r0, c0))

        if not candidates:
            return []  # No crop position found for this example

        candidates_per_example.append(candidates)

    # Check that all examples have same output dimensions
    if not output_dims:
        return []

    first_H_out, first_W_out = output_dims[0]
    for (H_out, W_out) in output_dims:
        if H_out != first_H_out or W_out != first_W_out:
            return []  # Inconsistent output dimensions

    # Step S6.1: Test explicit selection rules

    # Rule A: Fixed offset
    # Check if there exists (r0*, c0*) such that it's in ALL candidates
    if candidates_per_example:
        # Find intersection of all candidate sets
        common_offsets = set(candidates_per_example[0])
        for candidates in candidates_per_example[1:]:
            common_offsets &= set(candidates)

        if common_offsets:
            # Rule A matches! Pick one (deterministic: smallest r0, then c0)
            r0_star, c0_star = min(common_offsets)

            # Build out_to_in mapping (same for all examples)
            out_to_in = {}
            for r_out in range(first_H_out):
                for c_out in range(first_W_out):
                    r_in = r0_star + r_out
                    c_in = c0_star + c_out
                    out_to_in[f"({r_out},{c_out})"] = f"({r_in},{c_in})"

            # Generate instances
            instances = []
            for ex_idx, ex in enumerate(task_context.train_examples):
                instances.append(SchemaInstance(
                    family_id="S6",
                    params={
                        "example_type": "train",
                        "example_index": ex_idx,
                        "output_height": first_H_out,
                        "output_width": first_W_out,
                        "background_color": 0,
                        "out_to_in": out_to_in
                    }
                ))

            for ex_idx, ex in enumerate(task_context.test_examples):
                instances.append(SchemaInstance(
                    family_id="S6",
                    params={
                        "example_type": "test",
                        "example_index": ex_idx,
                        "output_height": first_H_out,
                        "output_width": first_W_out,
                        "background_color": 0,
                        "out_to_in": out_to_in
                    }
                ))

            return instances

    # Rule B: Largest component of color c
    # Collect all colors that appear in any training input
    all_colors = set()
    for ex in task_context.train_examples:
        all_colors.update(int(c) for c in ex.input_grid.flatten())

    # Test each color
    for color_c in sorted(all_colors):
        rule_b_valid = True
        crop_positions_b = []

        for ex_idx, ex in enumerate(task_context.train_examples):
            grid_in = ex.input_grid

            # Get all components of color c
            components = connected_components_by_color(grid_in)
            components_c = [comp for comp in components if comp.color == color_c]

            if not components_c:
                # This example has no components of color c
                rule_b_valid = False
                break

            # Find largest component by size, break ties by (r_min, c_min)
            largest = max(components_c, key=lambda c: (c.size, -min(p[0] for p in c.pixels), -min(p[1] for p in c.pixels)))

            # Get bounding box
            r_coords = [p[0] for p in largest.pixels]
            c_coords = [p[1] for p in largest.pixels]
            r_min = min(r_coords)
            c_min = min(c_coords)
            r_max = max(r_coords)
            c_max = max(c_coords)

            # Check if this bbox matches one of the valid crop candidates
            # Crop position is (r_min, c_min)
            if (r_min, c_min) not in candidates_per_example[ex_idx]:
                rule_b_valid = False
                break

            crop_positions_b.append((r_min, c_min))

        if rule_b_valid:
            # Rule B matches for training! Now validate test inputs
            # Before accepting, ensure ALL test inputs have components of color c
            test_applicable = True
            test_crop_positions_b = []

            for ex_idx, ex in enumerate(task_context.test_examples):
                grid_in = ex.input_grid

                # Get components of color c
                components = connected_components_by_color(grid_in)
                components_c = [comp for comp in components if comp.color == color_c]

                if not components_c:
                    # Test input doesn't have this color → rule not applicable
                    test_applicable = False
                    break

                # Find largest component
                largest = max(components_c, key=lambda c: (c.size, -min(p[0] for p in c.pixels), -min(p[1] for p in c.pixels)))
                r_coords = [p[0] for p in largest.pixels]
                c_coords = [p[1] for p in largest.pixels]
                r0 = min(r_coords)
                c0 = min(c_coords)
                test_crop_positions_b.append((r0, c0))

            if not test_applicable:
                # Rule B doesn't apply to all test inputs → try next rule
                continue

            # Rule B is valid for both train AND test!
            # Generate instances with per-example out_to_in
            instances = []

            # For train examples, use the computed crop positions
            for ex_idx, (r0, c0) in enumerate(crop_positions_b):
                out_to_in = {}
                for r_out in range(first_H_out):
                    for c_out in range(first_W_out):
                        r_in = r0 + r_out
                        c_in = c0 + c_out
                        out_to_in[f"({r_out},{c_out})"] = f"({r_in},{c_in})"

                instances.append(SchemaInstance(
                    family_id="S6",
                    params={
                        "example_type": "train",
                        "example_index": ex_idx,
                        "output_height": first_H_out,
                        "output_width": first_W_out,
                        "background_color": 0,
                        "out_to_in": out_to_in
                    }
                ))

            # For test examples, use the validated crop positions
            for ex_idx, (r0, c0) in enumerate(test_crop_positions_b):
                out_to_in = {}
                for r_out in range(first_H_out):
                    for c_out in range(first_W_out):
                        r_in = r0 + r_out
                        c_in = c0 + c_out
                        out_to_in[f"({r_out},{c_out})"] = f"({r_in},{c_in})"

                instances.append(SchemaInstance(
                    family_id="S6",
                    params={
                        "example_type": "test",
                        "example_index": ex_idx,
                        "output_height": first_H_out,
                        "output_width": first_W_out,
                        "background_color": 0,
                        "out_to_in": out_to_in
                    }
                ))

            return instances

    # Rule C: Largest component (any color)
    rule_c_valid = True
    crop_positions_c = []

    for ex_idx, ex in enumerate(task_context.train_examples):
        grid_in = ex.input_grid

        # Get all components (any color)
        components = connected_components_by_color(grid_in)

        if not components:
            rule_c_valid = False
            break

        # Find largest component by size, break ties by (r_min, c_min)
        largest = max(components, key=lambda c: (c.size, -min(p[0] for p in c.pixels), -min(p[1] for p in c.pixels)))

        # Get bounding box
        r_coords = [p[0] for p in largest.pixels]
        c_coords = [p[1] for p in largest.pixels]
        r_min = min(r_coords)
        c_min = min(c_coords)

        # Check if this bbox matches one of the valid crop candidates
        if (r_min, c_min) not in candidates_per_example[ex_idx]:
            rule_c_valid = False
            break

        crop_positions_c.append((r_min, c_min))

    if rule_c_valid:
        # Rule C matches for training! Now validate test inputs
        # Before accepting, ensure ALL test inputs have at least one component
        test_applicable = True
        test_crop_positions_c = []

        for ex_idx, ex in enumerate(task_context.test_examples):
            grid_in = ex.input_grid

            # Get all components
            components = connected_components_by_color(grid_in)

            if not components:
                # Test input has no components → rule not applicable
                test_applicable = False
                break

            # Find largest component
            largest = max(components, key=lambda c: (c.size, -min(p[0] for p in c.pixels), -min(p[1] for p in c.pixels)))
            r_coords = [p[0] for p in largest.pixels]
            c_coords = [p[1] for p in largest.pixels]
            r0 = min(r_coords)
            c0 = min(c_coords)
            test_crop_positions_c.append((r0, c0))

        if not test_applicable:
            # Rule C doesn't apply to all test inputs → reject
            return []

        # Rule C is valid for both train AND test!
        instances = []

        # For train examples
        for ex_idx, (r0, c0) in enumerate(crop_positions_c):
            out_to_in = {}
            for r_out in range(first_H_out):
                for c_out in range(first_W_out):
                    r_in = r0 + r_out
                    c_in = c0 + c_out
                    out_to_in[f"({r_out},{c_out})"] = f"({r_in},{c_in})"

            instances.append(SchemaInstance(
                family_id="S6",
                params={
                    "example_type": "train",
                    "example_index": ex_idx,
                    "output_height": first_H_out,
                    "output_width": first_W_out,
                    "background_color": 0,
                    "out_to_in": out_to_in
                }
            ))

        # For test examples, use the validated crop positions
        for ex_idx, (r0, c0) in enumerate(test_crop_positions_c):
            out_to_in = {}
            for r_out in range(first_H_out):
                for c_out in range(first_W_out):
                    r_in = r0 + r_out
                    c_in = c0 + c_out
                    out_to_in[f"({r_out},{c_out})"] = f"({r_in},{c_in})"

            instances.append(SchemaInstance(
                family_id="S6",
                params={
                    "example_type": "test",
                    "example_index": ex_idx,
                    "output_height": first_H_out,
                    "output_width": first_W_out,
                    "background_color": 0,
                    "out_to_in": out_to_in
                }
            ))

        return instances

    # Step S6.2: No rule matched
    return []


# =============================================================================
# S7 Miner - Summary/Aggregation Grids
# =============================================================================

def mine_S7(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S7 schema instances: block/summary grids.

    S7 creates a smaller summary grid where each output cell represents
    an aggregated region of the input (e.g., unique non-zero color per block).

    Current implementation (M6.3C scope):
      - Only implements "unique non-zero or zero" summary rule
      - Each block must have either:
        * Exactly one non-zero color → output that color
        * All zeros → output 0
      - If any block has multiple non-zero colors → S7 returns []

    Future extensions:
      - Dominant color (mode)
      - Color counts
      - Other deterministic summary functions

    Algorithm:
      1. For each train example:
         - Check if output is smaller and divides input cleanly
         - Partition input into blocks
         - Test "unique non-zero or zero" rule
      2. Verify rule is consistent across all train examples
      3. Generate SchemaInstance with summary_colors

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S7 parameters

    Note:
        This is a conscious scope limitation, not a bug. Tasks that use
        other summary rules (e.g., majority color, counts) will not get
        an S7 law and will need to be solved by other schemas.

    Example mined rule:
        If all train examples follow block aggregation,
        generate SchemaInstances with:
        {
            "example_type": "train",
            "example_index": 0,
            "output_height": 2,
            "output_width": 2,
            "summary_colors": { "(0,0)": 1, "(0,1)": 2, ... }
        }
    """
    # Step 1: Check if all training examples follow block aggregation pattern
    block_info_per_example: List[Tuple[int, int, int, int]] = []

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            return []  # No output

        grid_in = ex.input_grid
        grid_out = ex.output_grid
        H_in, W_in = grid_in.shape
        H_out, W_out = grid_out.shape

        # Check if output is smaller
        if H_out >= H_in or W_out >= W_in:
            return []  # Not aggregation

        # Check if dimensions divide cleanly
        if H_in % H_out != 0 or W_in % W_out != 0:
            return []  # Not clean block division

        block_h = H_in // H_out
        block_w = W_in // W_out

        block_info_per_example.append((H_out, W_out, block_h, block_w))

    if not block_info_per_example:
        return []

    # Step 2: Check consistency across examples
    first_H_out, first_W_out, first_block_h, first_block_w = block_info_per_example[0]

    for (H_out, W_out, block_h, block_w) in block_info_per_example:
        if H_out != first_H_out or W_out != first_W_out:
            return []  # Inconsistent output dimensions
        if block_h != first_block_h or block_w != first_block_w:
            return []  # Inconsistent block sizes

    # Step 3: Test "unique non-zero color" summary rule
    # For each training example, verify this rule holds
    summary_rule_valid = True

    for ex_idx, ex in enumerate(task_context.train_examples):
        grid_in = ex.input_grid
        grid_out = ex.output_grid

        for i in range(first_H_out):
            for j in range(first_W_out):
                # Get block
                r0 = i * first_block_h
                c0 = j * first_block_w
                block = grid_in[r0:r0+first_block_h, c0:c0+first_block_w]

                # Get summary color from output
                summary_out_color = int(grid_out[i, j])

                # Apply rule: unique non-zero color
                nonzero_colors = {int(c) for c in block.flatten() if c != 0}

                if len(nonzero_colors) == 0:
                    # All zeros -> expect output 0
                    if summary_out_color != 0:
                        summary_rule_valid = False
                        break
                elif len(nonzero_colors) == 1:
                    # One unique non-zero -> expect that color
                    if summary_out_color != nonzero_colors.pop():
                        summary_rule_valid = False
                        break
                else:
                    # Multiple non-zero colors -> rule doesn't apply
                    summary_rule_valid = False
                    break

            if not summary_rule_valid:
                break

        if not summary_rule_valid:
            break

    if not summary_rule_valid:
        return []  # Rule doesn't hold

    # Step 3.5: Validate that rule also applies to ALL test inputs
    # Before accepting, check that test blocks follow "unique non-zero or zero"
    test_rule_valid = True

    for ex_idx, ex in enumerate(task_context.test_examples):
        grid_in = ex.input_grid

        for i in range(first_H_out):
            for j in range(first_W_out):
                r0 = i * first_block_h
                c0 = j * first_block_w
                block = grid_in[r0:r0+first_block_h, c0:c0+first_block_w]

                # Check if block follows the rule
                nonzero_colors = {int(c) for c in block.flatten() if c != 0}

                if len(nonzero_colors) > 1:
                    # Ambiguous block in test → rule not applicable
                    test_rule_valid = False
                    break

            if not test_rule_valid:
                break

        if not test_rule_valid:
            break

    if not test_rule_valid:
        return []  # Rule doesn't apply to test inputs

    # Step 4: Generate SchemaInstances for each example
    instances: List[SchemaInstance] = []

    # For each example, compute summary_colors
    for ex_idx, ex in enumerate(task_context.train_examples):
        grid_in = ex.input_grid
        summary_colors = {}

        for i in range(first_H_out):
            for j in range(first_W_out):
                r0 = i * first_block_h
                c0 = j * first_block_w
                block = grid_in[r0:r0+first_block_h, c0:c0+first_block_w]

                # Apply summary rule
                nonzero_colors = {int(c) for c in block.flatten() if c != 0}

                if len(nonzero_colors) == 0:
                    color = 0
                elif len(nonzero_colors) == 1:
                    color = nonzero_colors.pop()
                else:
                    # Should not happen (we validated above)
                    color = 0

                summary_colors[f"({i},{j})"] = color

        instances.append(SchemaInstance(
            family_id="S7",
            params={
                "example_type": "train",
                "example_index": ex_idx,
                "output_height": first_H_out,
                "output_width": first_W_out,
                "summary_colors": summary_colors
            }
        ))

    # For test examples, compute from test input (now validated)
    for ex_idx, ex in enumerate(task_context.test_examples):
        grid_in = ex.input_grid
        summary_colors = {}

        for i in range(first_H_out):
            for j in range(first_W_out):
                r0 = i * first_block_h
                c0 = j * first_block_w
                block = grid_in[r0:r0+first_block_h, c0:c0+first_block_w]

                # Apply summary rule (validated to work)
                nonzero_colors = {int(c) for c in block.flatten() if c != 0}

                if len(nonzero_colors) == 0:
                    color = 0
                else:
                    # len == 1 (validated above)
                    color = nonzero_colors.pop()

                summary_colors[f"({i},{j})"] = color

        instances.append(SchemaInstance(
            family_id="S7",
            params={
                "example_type": "test",
                "example_index": ex_idx,
                "output_height": first_H_out,
                "output_width": first_W_out,
                "summary_colors": summary_colors
            }
        ))

    return instances


# =============================================================================
# S11 Miner - Local Neighborhood Codebook
# =============================================================================

def mine_S11(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S11 schema instances: local neighborhood codebook.

    S11 is the most general local schema: for each input neighborhood hash H,
    the local output pattern around that pixel is always the same P(H).

    This is similar to S5 but applies to ALL pixels with matching hashes,
    not just selective seeds.

    Algorithm:
      1. For each train example:
         - Hash local neighborhoods in input
         - Extract corresponding patches in output
         - Build codebook H → P
      2. Aggregate across all train examples:
         - Track all patches seen for each hash
      3. Keep only hashes with exactly one consistent patch
      4. Generate SchemaInstance for each (train + test) example

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S11 parameters

    Example mined rule:
        If all training examples show "hash 12345 -> 3×3 patch P",
        generate SchemaInstances with:
        {
            "example_type": "train",
            "example_index": 0,
            "hash_templates": {
                "12345": { "(0,0)": 5, "(0,1)": 5, ... }
            }
        }
    """
    # Use radius=1 for 3×3 patches (standard S11)
    PATCH_RADIUS = 1

    # Step 1: Mine task-wide invariants
    # Key: hash_value -> Set[patch_as_bytes]
    hash_to_patches: Dict[int, Set[bytes]] = defaultdict(set)

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue  # Skip if no output

        grid_in = ex.input_grid
        grid_out = ex.output_grid
        H, W = grid_in.shape

        # Get neighborhood hashes (already computed in ExampleContext)
        nbh = ex.neighborhood_hashes  # Dict[(r,c), int]

        # For each pixel, check if full patch fits
        for (r, c), h_val in nbh.items():
            # Check if patch fits in output grid
            r_min = r - PATCH_RADIUS
            r_max = r + PATCH_RADIUS + 1
            c_min = c - PATCH_RADIUS
            c_max = c + PATCH_RADIUS + 1

            if r_min < 0 or r_max > H or c_min < 0 or c_max > W:
                continue  # Patch doesn't fit

            # Extract output patch
            patch = grid_out[r_min:r_max, c_min:c_max]

            # Convert to bytes for set comparison
            patch_bytes = patch.tobytes()

            # Record this hash -> patch mapping
            hash_to_patches[h_val].add(patch_bytes)

    # Step 2: Filter to only consistent mappings
    # Build dict of hash -> canonical_patch for always-true rules
    consistent_templates: Dict[int, np.ndarray] = {}

    for h_val, patch_set in hash_to_patches.items():
        if len(patch_set) == 1:
            # All train examples agree on this hash -> patch mapping
            patch_bytes = patch_set.pop()
            # Convert back to numpy array
            patch_size = 2 * PATCH_RADIUS + 1
            patch = np.frombuffer(patch_bytes, dtype=np.int64).reshape(patch_size, patch_size)
            consistent_templates[h_val] = patch

    if not consistent_templates:
        return []  # No S11 rules found

    # Step 3: Convert to builder parameter format
    # hash_templates: { hash_str: { "(dr,dc)": color } }
    hash_templates: Dict[str, Dict[str, int]] = {}

    for h_val, patch in consistent_templates.items():
        offset_dict = {}
        for dr in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            for dc in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                offset_key = f"({dr},{dc})"
                color = int(patch[dr + PATCH_RADIUS, dc + PATCH_RADIUS])
                offset_dict[offset_key] = color

        hash_templates[str(h_val)] = offset_dict

    # Step 4: Generate SchemaInstances for each example
    instances: List[SchemaInstance] = []

    # Generate for train examples
    for ex_idx, ex in enumerate(task_context.train_examples):
        instances.append(SchemaInstance(
            family_id="S11",
            params={
                "example_type": "train",
                "example_index": ex_idx,
                "hash_templates": hash_templates
            }
        ))

    # Generate for test examples
    for ex_idx, ex in enumerate(task_context.test_examples):
        instances.append(SchemaInstance(
            family_id="S11",
            params={
                "example_type": "test",
                "example_index": ex_idx,
                "hash_templates": hash_templates
            }
        ))

    return instances


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles
    from src.law_mining.role_stats import compute_role_stats

    print("=" * 70)
    print("mine_s5_s6_s7_s11.py self-test")
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
    print("\nMining S5 instances...")
    s5_instances = mine_S5(task_context, roles, role_stats)
    print(f"✓ S5 instances: {len(s5_instances)}")
    if s5_instances:
        print(f"  Sample S5 params keys: {list(s5_instances[0].params.keys())}")

    print("\nMining S6 instances...")
    s6_instances = mine_S6(task_context, roles, role_stats)
    print(f"✓ S6 instances: {len(s6_instances)}")
    if s6_instances:
        print(f"  Sample S6 params keys: {list(s6_instances[0].params.keys())}")

    print("\nMining S7 instances...")
    s7_instances = mine_S7(task_context, roles, role_stats)
    print(f"✓ S7 instances: {len(s7_instances)}")
    if s7_instances:
        print(f"  Sample S7 params keys: {list(s7_instances[0].params.keys())}")

    print("\nMining S11 instances...")
    s11_instances = mine_S11(task_context, roles, role_stats)
    print(f"✓ S11 instances: {len(s11_instances)}")
    if s11_instances:
        print(f"  Sample S11 params keys: {list(s11_instances[0].params.keys())}")

    print("\n" + "=" * 70)
    print("✓ mine_s5_s6_s7_s11.py self-test passed")
    print("=" * 70)
