"""
S12 schema builder: Generalized Raycasting / Projection.

This implements the S12 schema for directional ray projection:
    "From seed pixels (identified by hash/color), cast rays in specified
     directions (including diagonals), painting pixels until stop condition met."

S12 generalizes S9 by supporting:
  - 8 directions (N, S, E, W, NE, NW, SE, SW) instead of 4
  - Configurable stop conditions (border, collision, specific colors)
  - Hash-based seed selection (context-aware triggers)

S12 is geometry-preserving: output has same shape as input.
"""

from typing import Dict, Any, List, Tuple
from ast import literal_eval

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S12_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S12 preferences: cast rays from seed pixels in specified directions.

    S12 creates directional projections by casting rays from seed pixels,
    continuing until a stop condition is met.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "rays": [
            {
              "seed_type": "color" | "hash",    # How to identify seed pixels
              "seed_color": int,                # (if seed_type="color") Color that shoots rays
              "seed_hashes": [int, ...],        # (if seed_type="hash") Neighborhood hashes
              "vector": "(dr,dc)",              # Direction tuple (e.g., "(1,1)" for SE)
              "draw_color": int,                # Color to paint along ray
              "stop_condition": str,            # "border" | "any_nonzero" | "color_X"
              "include_seed": bool              # Paint the seed pixel itself?
            },
            ...
          ]
        }

    Where:
        - seed_type: How to identify seed pixels (default: "hash")
          * "color": All pixels of seed_color shoot rays (Color Law)
          * "hash": Only pixels with matching neighborhood hashes shoot (Pattern Law)
        - seed_color: (if seed_type="color") Color value that identifies seed pixels
        - seed_hashes: (if seed_type="hash") List of neighborhood hash values
        - vector: Direction of ray as (dr, dc) tuple
        - draw_color: Color to paint pixels along ray path
        - stop_condition: When to stop the ray
          * "border": Stop at grid boundary
          * "any_nonzero": Stop when hitting any non-zero pixel
          * "color_X": Stop when hitting specific color X
        - include_seed: Whether to color the seed pixel itself

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying ray configurations
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> # Diagonal rays (SE) from specific seed patterns
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "rays": [{
        ...         "seed_hashes": [12345],
        ...         "vector": "(1,1)",
        ...         "draw_color": 6,
        ...         "stop_condition": "border",
        ...         "include_seed": False
        ...     }]
        ... }
        >>> build_S12_constraints(ctx, params, builder)
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

    # 2. Get grid dimensions
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid
    C = task_context.C

    # 3. Get neighborhood hashes
    nbh = ex.neighborhood_hashes  # Dict[(r, c), int]

    # 4. Process each ray configuration
    rays = schema_params.get("rays", [])

    for ray_config in rays:
        vector_str = ray_config.get("vector", "(0,0)")
        draw_color = ray_config.get("draw_color", 0)
        stop_condition = ray_config.get("stop_condition", "border")
        include_seed = ray_config.get("include_seed", False)
        seed_type = ray_config.get("seed_type", "hash")  # "color" or "hash"

        # Parse vector
        try:
            dr, dc = literal_eval(vector_str)
        except (ValueError, SyntaxError):
            continue  # Skip malformed vector

        # Validate draw_color
        if not (0 <= draw_color < C):
            continue

        # Find seed pixels based on seed_type
        seed_pixels: List[Tuple[int, int]] = []

        if seed_type == "color":
            # CASE A: Color Law (e.g., "Red shoots North")
            # Fast: iterate all pixels, select by color
            seed_color = ray_config.get("seed_color")
            if seed_color is None:
                continue  # Missing seed_color param

            H_in, W_in = ex.input_grid.shape
            for r in range(H_in):
                for c in range(W_in):
                    # Check if pixel has the seed color
                    if int(ex.input_grid[r, c]) == seed_color:
                        # Also check if position has a neighborhood hash (not on edge)
                        if (r, c) in nbh:
                            seed_pixels.append((r, c))

        elif seed_type == "hash":
            # CASE B: Pattern Law (e.g., "Line ends shoot")
            # Existing hash lookup logic
            seed_hashes = ray_config.get("seed_hashes", [])
            for (r, c), h_val in nbh.items():
                if h_val in seed_hashes:
                    seed_pixels.append((r, c))

        else:
            continue  # Unknown seed_type

        # Cast ray from each seed pixel
        for r_seed, c_seed in seed_pixels:
            # Skip if seed is outside output grid bounds (geometry-changing case)
            if not (0 <= r_seed < H and 0 <= c_seed < W):
                continue

            # Optionally color the seed itself
            if include_seed:
                p_idx = r_seed * W + c_seed
                builder.prefer_pixel_color(p_idx, draw_color, weight=50.0)

            # Cast ray in the specified direction
            curr_r = r_seed + dr
            curr_c = c_seed + dc

            while True:
                # Check bounds
                if not (0 <= curr_r < H and 0 <= curr_c < W):
                    break  # Hit boundary

                # Check stop condition
                if should_stop(ex, curr_r, curr_c, stop_condition):
                    break

                # Paint this pixel
                p_idx = curr_r * W + curr_c
                builder.prefer_pixel_color(p_idx, draw_color, weight=50.0)

                # Move to next position
                curr_r += dr
                curr_c += dc


def should_stop(ex, r: int, c: int, stop_condition: str) -> bool:
    """
    Determine if ray should stop at position (r, c).

    Args:
        ex: ExampleContext with input/output grids
        r: Row position
        c: Column position
        stop_condition: Stop rule ("border", "any_nonzero", "color_X")

    Returns:
        True if ray should stop, False to continue
    """
    if stop_condition == "border":
        # Never stop mid-grid (boundary checked in main loop)
        return False

    # Check if position is within input grid bounds (geometry-changing case)
    H_in, W_in = ex.input_grid.shape
    if not (0 <= r < H_in and 0 <= c < W_in):
        # Ray position is outside input grid, can't check collision
        # Continue ray (don't stop)
        return False

    # Get current pixel color from input grid
    input_color = int(ex.input_grid[r, c])

    if stop_condition == "any_nonzero":
        return input_color != 0

    if stop_condition.startswith("color_"):
        # Parse target color
        try:
            target_color = int(stop_condition.split("_")[1])
            return input_color == target_color
        except (ValueError, IndexError):
            return False

    return False


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S12 builder with toy example...")
    print("=" * 70)

    # Create a 5x5 grid
    # Seed at (2,2), should cast diagonal ray to SE
    input_grid = np.zeros((5, 5), dtype=int)
    input_grid[2, 2] = 1  # Seed pixel

    output_grid = np.zeros((5, 5), dtype=int)

    ex = build_example_context(input_grid, output_grid)
    # Manually set a neighborhood hash for the seed
    ex.neighborhood_hashes = {(2, 2): 12345}

    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Diagonal ray (SE) from seed (hash-based)")
    print("-" * 70)

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "rays": [{
            "seed_type": "hash",  # Pattern Law
            "seed_hashes": [12345],
            "vector": "(1,1)",  # SE diagonal
            "draw_color": 6,
            "stop_condition": "border",
            "include_seed": False
        }]
    }

    builder1 = ConstraintBuilder()
    build_S12_constraints(ctx, params1, builder1)

    # Should have preferences for (3,3) and (4,4) - diagonal from (2,2)
    # 2 pixels along diagonal
    expected1 = 2
    print(f"  Expected: {expected1} preferences (diagonal ray)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1, \
        f"Expected {expected1} preferences, got {len(builder1.preferences)}"

    print("\nTest 2: Vertical ray (S) with include_seed (hash-based)")
    print("-" * 70)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "rays": [{
            "seed_type": "hash",  # Pattern Law
            "seed_hashes": [12345],
            "vector": "(1,0)",  # South
            "draw_color": 3,
            "stop_condition": "border",
            "include_seed": True  # Include seed
        }]
    }

    builder2 = ConstraintBuilder()
    build_S12_constraints(ctx, params2, builder2)

    # Should have preferences for (2,2), (3,2), (4,2) - 3 pixels
    expected2 = 3
    print(f"  Expected: {expected2} preferences (seed + 2 ray pixels)")
    print(f"  Actual: {len(builder2.preferences)}")
    assert len(builder2.preferences) == expected2, \
        f"Expected {expected2} preferences, got {len(builder2.preferences)}"

    print("\nTest 3: Color Law (seed_type='color', all color 1 pixels shoot)")
    print("-" * 70)

    # Create grid with TWO color 1 pixels (both should shoot rays)
    input_grid3 = np.zeros((5, 5), dtype=int)
    input_grid3[1, 1] = 1  # Seed 1
    input_grid3[1, 3] = 1  # Seed 2

    output_grid3 = np.zeros((5, 5), dtype=int)

    ex3 = build_example_context(input_grid3, output_grid3)
    ex3.neighborhood_hashes = {(1, 1): 100, (1, 3): 200}  # Different hashes

    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=10)

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "rays": [{
            "seed_type": "color",  # Color Law (generalized)
            "seed_color": 1,       # All color 1 pixels shoot
            "vector": "(1,0)",     # South
            "draw_color": 7,
            "stop_condition": "border",
            "include_seed": False
        }]
    }

    builder3 = ConstraintBuilder()
    build_S12_constraints(ctx3, params3, builder3)

    # Should cast rays from BOTH (1,1) and (1,3)
    # From (1,1): (2,1), (3,1), (4,1) = 3 pixels
    # From (1,3): (2,3), (3,3), (4,3) = 3 pixels
    # Total: 6 pixels
    expected3 = 6
    print(f"  Expected: {expected3} preferences (2 seeds × 3 ray pixels each)")
    print(f"  Actual: {len(builder3.preferences)}")
    assert len(builder3.preferences) == expected3, \
        f"Expected {expected3} preferences, got {len(builder3.preferences)}"
    print("  ✓ Color Law correctly found ALL pixels with seed_color=1")

    print("\n" + "=" * 70)
    print("✓ S12 builder self-test passed.")
