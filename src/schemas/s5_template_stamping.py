"""
S5 schema builder: Template stamping (seed → template).

This implements the S5 schema from the math kernel spec (section 2):
    "Small stencil templates (e.g. 3×3) centered on special seeds.
     For each seed type t, stamp a fixed output patch P_t around it."

S5 is geometry-preserving: output has same shape as input.
This builder applies pre-mined seed templates (from Pi-agent) as constraints.
"""

from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S5_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S5 constraints: stamp templates around seed pixels.

    S5 detects seed pixels by their neighborhood hash and stamps
    a pre-defined template patch around each seed occurrence.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "seed_templates": {
            "123456": { "(0,0)": 5, "(0,1)": 5, "(1,0)": 5, "(1,1)": 5 },
            "987654": { "(0,0)": 2, "(-1,0)": 2, "(1,0)": 2, "(0,-1)": 2, "(0,1)": 2 }
          }
        }

    Where:
        - Keys of seed_templates are stringified neighborhood hashes
        - Each template is a dict of offset strings "(dr,dc)" → color
        - Template is stamped at (r+dr, c+dc) for each seed pixel (r,c)

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying seed types and templates
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # Stamp a 2x2 blue square around pixels with hash 123456
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "seed_templates": {
        ...         "123456": { "(0,0)": 5, "(0,1)": 5, "(1,0)": 5, "(1,1)": 5 }
        ...     }
        ... }
        >>> build_S5_constraints(ctx, params, builder)
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
    # S5 stamps templates into OUTPUT grid at positions detected in INPUT grid
    # Use output dimensions for indexing into y variables
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid to constrain
    C = task_context.C
    nbh = ex.neighborhood_hashes  # Dict[(r,c)] -> hash_value (int) from input grid

    # 3. Parse seed templates from params
    raw_templates = schema_params.get("seed_templates", {})
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

    # 4. Stamp templates for each matching seed pixel
    for (r, c), h_val in nbh.items():
        if h_val not in parsed_templates:
            continue  # This pixel's hash is not a seed

        # Get template for this seed type
        tmpl = parsed_templates[h_val]

        # Stamp template: for each offset, fix the color
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

    print("Testing S5 builder with toy example...")
    print("=" * 70)

    # Create a 5x5 grid
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],  # seed at (1,1)
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],  # seed at (3,3)
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Get hash at position (1,1) - the seed pixel
    if (1, 1) in ex.neighborhood_hashes:
        hash_at_1_1 = ex.neighborhood_hashes[(1, 1)]
        print(f"Neighborhood hash at (1,1): {hash_at_1_1}")
    else:
        hash_at_1_1 = 999999  # dummy hash for testing
        print(f"Using dummy hash for testing: {hash_at_1_1}")

    if (3, 3) in ex.neighborhood_hashes:
        hash_at_3_3 = ex.neighborhood_hashes[(3, 3)]
        print(f"Neighborhood hash at (3,3): {hash_at_3_3}")
    else:
        hash_at_3_3 = hash_at_1_1  # same hash for testing
        print(f"Using same hash for (3,3): {hash_at_3_3}")

    print("\nTest 1: Stamp 2x2 template at seeds")
    print("-" * 70)
    print("Template: 2x2 square of color 5")
    print("Seeds: pixels with specific hash (1,1) and (3,3)")

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(hash_at_1_1): {
                "(0,0)": 5,
                "(0,1)": 5,
                "(1,0)": 5,
                "(1,1)": 5
            }
        }
    }

    builder1 = ConstraintBuilder()
    build_S5_constraints(ctx, params1, builder1)

    # Should have 2 seeds × 4 pixels/template = 8 constraints
    # (assuming both (1,1) and (3,3) have the same hash)
    print(f"  Constraints added: {len(builder1.constraints)}")
    print(f"  Expected: 8 (2 seeds × 4 pixels each)")
    assert len(builder1.constraints) == 8, \
        f"Expected 8 constraints, got {len(builder1.constraints)}"

    print("\nTest 2: Cross pattern template")
    print("-" * 70)
    print("Template: 5-pixel cross (center + 4 cardinal directions)")

    # Get a different hash for variety (or use same one)
    test_hash = hash_at_1_1

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(test_hash): {
                "(0,0)": 2,     # center
                "(-1,0)": 2,    # up
                "(1,0)": 2,     # down
                "(0,-1)": 2,    # left
                "(0,1)": 2      # right
            }
        }
    }

    builder2 = ConstraintBuilder()
    build_S5_constraints(ctx, params2, builder2)

    # Seeds at (1,1) and (3,3), each stamps 5 pixels
    # But some may go out of bounds:
    #   (1,1): up (-1,0) → (0,1) ✓, down (1,0) → (2,1) ✓, left (0,-1) → (1,0) ✓, right (0,1) → (1,2) ✓, center (0,0) → (1,1) ✓ = 5
    #   (3,3): up → (2,3) ✓, down → (4,3) ✓, left → (3,2) ✓, right → (3,4) ✓, center → (3,3) ✓ = 5
    # Total: 2 seeds × 5 pixels = 10 constraints
    expected2 = 10
    print(f"  Constraints added: {len(builder2.constraints)}")
    print(f"  Expected: {expected2} (2 seeds × 5 pixels each)")
    assert len(builder2.constraints) == expected2, \
        f"Expected {expected2} constraints, got {len(builder2.constraints)}"

    print("\nTest 3: Template with out-of-bounds offsets")
    print("-" * 70)
    print("Template with large offsets that go outside grid")

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(test_hash): {
                "(0,0)": 3,
                "(10,10)": 3,   # Out of bounds
                "(-10,-10)": 3  # Out of bounds
            }
        }
    }

    builder3 = ConstraintBuilder()
    build_S5_constraints(ctx, params3, builder3)

    # Only center (0,0) is in bounds for both seeds
    # So 2 seeds × 1 valid pixel = 2 constraints
    expected3 = 2
    print(f"  Constraints added: {len(builder3.constraints)}")
    print(f"  Expected: {expected3} (only center pixels in bounds)")
    assert len(builder3.constraints) == expected3, \
        f"Expected {expected3} constraints, got {len(builder3.constraints)}"

    print("\n" + "=" * 70)
    print("✓ S5 builder self-test passed.")
