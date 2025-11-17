"""
S11 schema builder: Local neighborhood codebook (hash → template).

This implements the S11 schema from the math kernel spec (section 2):
    "For each 3×3 neighborhood pattern type, assign a corresponding output pattern
     via a learned codebook H → P. Acts as safety net for any local weirdness."

S11 is geometry-preserving: output has same shape as input.
This builder applies pre-mined hash-to-template mappings (from Pi-agent) as constraints.
"""

from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S11_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S11 constraints: apply template based on neighborhood hash.

    S11 treats each 3×3 neighborhood hash as a symbol and maps it to
    an output template. This is more general than S5 (which only stamps
    at selective "seed" locations).

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "hash_templates": {
            "123456": { "(0,0)": 5, "(0,1)": 5, "(1,0)": 5, "(1,1)": 5 },
            "222222": { "(0,0)": 0 },  # maybe overwrite center only
            ...
          }
        }

    Where:
        - Keys of hash_templates are stringified neighborhood hashes
        - Each template is a dict of offset strings "(dr,dc)" → color
        - Template is applied at (r+dr, c+dc) for each pixel (r,c) with matching hash

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying hash types and templates
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # For all pixels with hash 123456, rewrite their 2x2 neighborhood
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "hash_templates": {
        ...         "123456": { "(0,0)": 5, "(0,1)": 5, "(1,0)": 5, "(1,1)": 5 }
        ...     }
        ... }
        >>> build_S11_constraints(ctx, params, builder)
    """
    # 1. Select example
    example_type = schema_params.get("example_type", "train")
    example_index = schema_params.get("example_index", 0)

    if example_type == "train":
        if example_index >= len(task_context.train_examples):
            return  # Invalid index
        ex = task_context.train_examples[example_index]
    else:  # "test"
        if example_index >= len(task_context.test_examples):
            return  # Invalid index
        ex = task_context.test_examples[example_index]

    # 2. Get grid dimensions and features
    # S11 is geometry-preserving, so output shape = input shape
    H = ex.input_H
    W = ex.input_W
    C = task_context.C
    nbh = ex.neighborhood_hashes  # Dict[(r,c)] -> hash_value (int)

    # 3. Parse hash templates from params
    raw_templates = schema_params.get("hash_templates", {})
    if not raw_templates:
        return  # No templates specified

    # Convert stringified keys to actual types:
    #   hash_str -> int
    #   offset_str "(dr,dc)" -> (dr, dc) tuple
    parsed_templates: Dict[int, Dict[Tuple[int, int], int]] = {}

    for hash_str, offset_map in raw_templates.items():
        h_val = int(hash_str)
        tmpl: Dict[Tuple[int, int], int] = {}

        for offset_str, color in offset_map.items():
            # offset_str is like "(0,1)" or "(-1,0)"
            # Strip parentheses and split by comma
            offset_str = offset_str.strip()
            dr_dc = offset_str.strip("()").split(",")
            dr = int(dr_dc[0].strip())
            dc = int(dr_dc[1].strip())
            tmpl[(dr, dc)] = int(color)

        parsed_templates[h_val] = tmpl

    # 4. Apply templates for each pixel with matching hash
    for (r, c), h_val in nbh.items():
        if h_val not in parsed_templates:
            continue  # No template for this hash

        # Get template for this hash
        tmpl = parsed_templates[h_val]

        # Apply template: for each offset, fix the color
        for (dr, dc), color in tmpl.items():
            rr = r + dr
            cc = c + dc

            # Check bounds
            if 0 <= rr < H and 0 <= cc < W:
                # Validate color is in palette
                if not (0 <= color < C):
                    continue  # Skip invalid colors

                # Fix pixel color
                p_idx = rr * W + cc
                builder.fix_pixel_color(p_idx, color, C)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S11 builder with toy example...")
    print("=" * 70)

    # Create a 5x5 grid with different patterns
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],  # pattern at (1,1) and (1,2)
        [0, 1, 1, 0, 0],  # pattern at (2,1) and (2,2)
        [0, 0, 0, 2, 0],  # different pattern at (3,3)
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # S11 applies to ALL pixels with matching hashes
    # Let's count how many unique hashes we have
    unique_hashes = set(ex.neighborhood_hashes.values())
    print(f"Total pixels with hashes: {len(ex.neighborhood_hashes)}")
    print(f"Unique hash values: {len(unique_hashes)}")

    # Get a representative hash (pick any from the grid)
    if ex.neighborhood_hashes:
        sample_hash = list(ex.neighborhood_hashes.values())[0]
    else:
        sample_hash = 999999  # dummy

    print(f"Using sample hash for testing: {sample_hash}")

    print("\nTest 1: Apply template to all pixels with specific hash")
    print("-" * 70)
    print("Template: overwrite center pixel only (0,0) → color 5")

    # Count how many pixels have this hash
    pixels_with_hash = sum(1 for h in ex.neighborhood_hashes.values() if h == sample_hash)
    print(f"Pixels with hash {sample_hash}: {pixels_with_hash}")

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(sample_hash): {
                "(0,0)": 5  # Overwrite center only
            }
        }
    }

    builder1 = ConstraintBuilder()
    build_S11_constraints(ctx, params1, builder1)

    # Should have 1 constraint per pixel with this hash
    expected1 = pixels_with_hash
    print(f"  Constraints added: {len(builder1.constraints)}")
    print(f"  Expected: {expected1} (1 constraint per matching pixel)")
    assert len(builder1.constraints) == expected1, \
        f"Expected {expected1} constraints, got {len(builder1.constraints)}"

    print("\nTest 2: Multiple templates for different hashes")
    print("-" * 70)

    # Get two different hashes if available
    hash_list = list(ex.neighborhood_hashes.values())
    if len(hash_list) >= 2:
        hash1 = hash_list[0]
        hash2 = hash_list[1]
        # Find a different hash
        for h in hash_list:
            if h != hash1:
                hash2 = h
                break
    else:
        hash1 = sample_hash
        hash2 = sample_hash + 1  # dummy different hash

    count1 = sum(1 for h in ex.neighborhood_hashes.values() if h == hash1)
    count2 = sum(1 for h in ex.neighborhood_hashes.values() if h == hash2)

    print(f"Hash {hash1}: {count1} pixels")
    print(f"Hash {hash2}: {count2} pixels")

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(hash1): {
                "(0,0)": 3
            },
            str(hash2): {
                "(0,0)": 4
            }
        }
    }

    builder2 = ConstraintBuilder()
    build_S11_constraints(ctx, params2, builder2)

    expected2 = count1 + count2
    print(f"  Constraints added: {len(builder2.constraints)}")
    print(f"  Expected: {expected2} ({count1} + {count2})")
    assert len(builder2.constraints) == expected2, \
        f"Expected {expected2} constraints, got {len(builder2.constraints)}"

    print("\nTest 3: Template with multiple offsets (3x3 pattern)")
    print("-" * 70)

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(sample_hash): {
                "(-1,-1)": 7,
                "(-1,0)": 7,
                "(-1,1)": 7,
                "(0,-1)": 7,
                "(0,0)": 7,
                "(0,1)": 7,
                "(1,-1)": 7,
                "(1,0)": 7,
                "(1,1)": 7
            }
        }
    }

    builder3 = ConstraintBuilder()
    build_S11_constraints(ctx, params3, builder3)

    # Each pixel with matching hash stamps 9 offsets, but some may go out of bounds
    # For a 5x5 grid, interior pixels can stamp all 9, edge pixels fewer
    # Let's just check we got constraints
    print(f"  Constraints added: {len(builder3.constraints)}")
    print(f"  (varies based on edge effects, but should be > 0)")
    assert len(builder3.constraints) > 0, "Should have generated some constraints"

    print("\n" + "=" * 70)
    print("✓ S11 builder self-test passed.")
