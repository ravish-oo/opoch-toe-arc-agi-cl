"""
S8 schema builder: Tiling / replication with symmetric transforms.

This implements the S8 schema from the math kernel spec (section 2):
    "Copy a small patch periodically to fill an area, possibly with padding.
     For each tile position (i,j), stamp tile pattern T at that location."

Extended to support symmetric tiling (wallpaper groups):
    - Each tile position can have a transform: identity, flipx, flipy, rot90, etc.
    - This captures mirror tiling and other repeating patterns with symmetry.

S8 is geometry-preserving: output has same shape as input.
This builder applies pre-determined tiling patterns (from Pi-agent) as preferences.
"""

from typing import Dict, Any, Tuple
from ast import literal_eval

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


# =============================================================================
# Transform Operations
# =============================================================================

def apply_inverse_transform(
    dr: int, dc: int,
    tile_h: int, tile_w: int,
    transform: str
) -> Tuple[int, int]:
    """
    Apply inverse transform to offset (dr, dc) to get source offset in base tile.

    When a tile is transformed (e.g., FlipX), the pixel at offset (dr, dc)
    in the transformed tile corresponds to a different offset in the base tile.
    This function computes that source offset.

    Args:
        dr, dc: Offset within the transformed tile
        tile_h, tile_w: Tile dimensions
        transform: Transform name ("identity", "flipx", "flipy", "rot90", etc.)

    Returns:
        (src_dr, src_dc): Source offset in the base tile
    """
    if transform == "identity":
        return (dr, dc)
    elif transform == "flipx":
        # Horizontal flip: column is mirrored
        return (dr, tile_w - 1 - dc)
    elif transform == "flipy":
        # Vertical flip: row is mirrored
        return (tile_h - 1 - dr, dc)
    elif transform == "flipxy":
        # Both flips (180 rotation)
        return (tile_h - 1 - dr, tile_w - 1 - dc)
    elif transform == "rot90":
        # 90 degrees clockwise: (dr, dc) -> (dc, tile_h - 1 - dr)
        # Inverse is 270 degrees (or 90 counter-clockwise)
        return (tile_w - 1 - dc, dr)
    elif transform == "rot180":
        # 180 degrees: same as flipxy
        return (tile_h - 1 - dr, tile_w - 1 - dc)
    elif transform == "rot270":
        # 270 degrees clockwise (90 counter-clockwise)
        # Inverse is 90 degrees clockwise
        return (dc, tile_h - 1 - dr)
    else:
        # Unknown transform - fallback to identity
        return (dr, dc)


def build_S8_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S8 preferences: tile a pattern over a specified region.

    S8 replicates a base tile pattern across a rectangular region
    with specified stride (tile dimensions).

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "tile_height": int,
          "tile_width": int,
          "tile_pattern": {
            "(0,0)": 1,   # offset (0,0) in tile → color 1
            "(0,1)": 2,   # offset (0,1) in tile → color 2
            "(1,0)": 3,   # offset (1,0) in tile → color 3
            "(1,1)": 4    # offset (1,1) in tile → color 4
          },
          "tile_transforms": {           # OPTIONAL: per-tile-position transforms
            "(0,0)": "identity",         # tile at position (0,0) uses identity
            "(0,1)": "flipx",            # tile at position (0,1) is horizontally flipped
            "(1,0)": "flipx",            # etc.
            "(1,1)": "identity"
          },
          "region_origin": "(r0,c0)",  # top-left of tiling region
          "region_height": int,
          "region_width": int
        }

    Where:
        - tile_pattern defines colors at offsets relative to tile origin (base tile)
        - tile_transforms specifies transform for each tile position (default: identity)
        - Supported transforms: identity, flipx, flipy, flipxy, rot90, rot180, rot270
        - region defines where to tile within the output grid
        - tiles are stamped with stride (tile_height, tile_width)

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying tile pattern and tiling region
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> # Tile a 2x2 pattern across a 4x4 region
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "tile_height": 2,
        ...     "tile_width": 2,
        ...     "tile_pattern": {
        ...         "(0,0)": 1,
        ...         "(0,1)": 2,
        ...         "(1,0)": 3,
        ...         "(1,1)": 4
        ...     },
        ...     "region_origin": "(0,0)",
        ...     "region_height": 4,
        ...     "region_width": 4
        ... }
        >>> build_S8_constraints(ctx, params, builder)
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
    # S8 tiles pattern into OUTPUT grid
    # For test examples, output_H/W may be None - use region_height/width from params
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        # Fall back to region dimensions from params (required for test examples)
        H = schema_params.get("region_height")
        W = schema_params.get("region_width")
        if H is None or W is None:
            return  # No dimensions available
    C = task_context.C

    # 3. Parse tiling parameters
    tile_h = int(schema_params.get("tile_height", 1))
    tile_w = int(schema_params.get("tile_width", 1))

    if tile_h <= 0 or tile_w <= 0:
        return  # Invalid tile dimensions

    # Parse region origin
    region_origin_str = schema_params.get("region_origin", "(0,0)")
    try:
        r0, c0 = literal_eval(region_origin_str)
    except (ValueError, SyntaxError):
        return  # Invalid region origin

    region_h = int(schema_params.get("region_height", H))
    region_w = int(schema_params.get("region_width", W))

    # Parse tile pattern
    raw_pattern = schema_params.get("tile_pattern", {})
    tile_pattern: Dict[Tuple[int, int], int] = {}

    for k_str, color in raw_pattern.items():
        try:
            dr, dc = literal_eval(k_str)
            tile_pattern[(dr, dc)] = int(color)
        except (ValueError, SyntaxError):
            continue  # Skip malformed entries

    if not tile_pattern:
        return  # No pattern to tile

    # Parse tile_transforms (optional - defaults to identity for all positions)
    raw_transforms = schema_params.get("tile_transforms", {})
    tile_transforms: Dict[Tuple[int, int], str] = {}

    for k_str, transform in raw_transforms.items():
        try:
            tile_r, tile_c = literal_eval(k_str)
            tile_transforms[(tile_r, tile_c)] = str(transform)
        except (ValueError, SyntaxError):
            continue

    # 4. Tile the pattern across the region
    # Loop over tile origins with stride (tile_h, tile_w)
    tile_row = 0  # Tile position counter (row)
    tr = r0
    while tr < r0 + region_h:
        tile_col = 0  # Tile position counter (col)
        tc = c0
        while tc < c0 + region_w:
            # Get transform for this tile position (default: identity)
            transform = tile_transforms.get((tile_row, tile_col), "identity")

            # Stamp tile pattern at (tr, tc) with transform applied
            for dr in range(tile_h):
                for dc in range(tile_w):
                    rr = tr + dr
                    cc = tc + dc

                    # Check bounds
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue

                    # Apply inverse transform to get source offset in base tile
                    src_dr, src_dc = apply_inverse_transform(dr, dc, tile_h, tile_w, transform)

                    # Look up color from base tile pattern
                    color = tile_pattern.get((src_dr, src_dc))
                    if color is None:
                        continue  # No color defined for this offset

                    # Validate color is in palette
                    if 0 <= color < C:
                        # Prefer this pixel's color (Tier 3: Local, weight 10.0)
                        p_idx = rr * W + cc
                        builder.prefer_pixel_color(p_idx, color, weight=10.0)

            tc += tile_w
            tile_col += 1
        tr += tile_h
        tile_row += 1


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S8 builder with toy example...")
    print("=" * 70)

    # Create a 4x4 input grid (content doesn't matter for tiling)
    input_grid = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Tile 2x2 pattern across 4x4 grid")
    print("-" * 70)
    print("Tile pattern:")
    print("  (0,0)→1  (0,1)→2")
    print("  (1,0)→3  (1,1)→4")
    print("Region: entire 4x4 grid")

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4
    }

    builder1 = ConstraintBuilder()
    build_S8_constraints(ctx, params1, builder1)

    # Should have 16 preferences (4x4 grid fully tiled with 2x2 pattern)
    # 2x2 tiles fit perfectly: (0,0), (0,2), (2,0), (2,2) = 4 tiles
    # Each tile has 4 pixels → 16 total
    expected1 = 16
    print(f"  Expected: {expected1} preferences (4x4 grid, 2x2 tiles)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1, \
        f"Expected {expected1} preferences, got {len(builder1.preferences)}"

    print("\nTest 2: Tile 3x3 pattern in partial region")
    print("-" * 70)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 3,
        "tile_width": 3,
        "tile_pattern": {
            "(0,0)": 5,
            "(0,1)": 5,
            "(0,2)": 5,
            "(1,0)": 5,
            "(1,1)": 6,
            "(1,2)": 5,
            "(2,0)": 5,
            "(2,1)": 5,
            "(2,2)": 5
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4
    }

    builder2 = ConstraintBuilder()
    build_S8_constraints(ctx, params2, builder2)

    # 3x3 tile in 4x4 region: only (0,0) origin fits fully
    # But (0,3) column is partially outside region_width=4, so we get:
    # - Tile at (0,0): 3x3 = 9 pixels (but col 3 doesn't exist in 4x4 region_width)
    # Actually region is 4x4, tile is 3x3, so first tile at (0,0) gives us 3x3 = 9 pixels
    # Next tile would be at (0,3) but that's at column 3, and with width 3 goes to col 5 (out of region_width=4)
    # So we only get partial tiles or just the first one
    # Let me recalculate: region starts at (0,0), height=4, width=4
    # Tile stride is 3x3
    # Tile origins: (0,0), (0,3) for columns (but 0+3=3, and region is width 4, so col 3 is last col)
    # Actually, tr goes 0, 3, 6,... and we check tr < 0+4, so tr=0,3 are valid
    # tc goes 0, 3, 6,... and we check tc < 0+4, so tc=0,3 are valid
    # So we get tiles at (0,0), (0,3), (3,0), (3,3)
    # Each tile has 9 offsets, but many will be out of bounds for a 4x4 grid
    # Tile at (0,0): offsets (0,0) to (2,2) → all in bounds → 9 pixels
    # Tile at (0,3): offsets (0,3) to (2,5) → only col 3 in bounds → 3 pixels (rows 0-2, col 3)
    # Tile at (3,0): offsets (3,0) to (5,2) → only row 3 in bounds → 3 pixels (row 3, cols 0-2)
    # Tile at (3,3): offset (3,3) to (5,5) → only (3,3) in bounds → 1 pixel
    # Total: 9 + 3 + 3 + 1 = 16 pixels
    expected2 = 16
    print(f"  Expected: {expected2} preferences (partial tiling with clipping)")
    print(f"  Actual: {len(builder2.preferences)}")
    assert len(builder2.preferences) == expected2, \
        f"Expected {expected2} preferences, got {len(builder2.preferences)}"

    print("\nTest 3: Small region (2x2) with 1x1 tile")
    print("-" * 70)

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 1,
        "tile_width": 1,
        "tile_pattern": {
            "(0,0)": 7
        },
        "region_origin": "(1,1)",
        "region_height": 2,
        "region_width": 2
    }

    builder3 = ConstraintBuilder()
    build_S8_constraints(ctx, params3, builder3)

    # 1x1 tile in 2x2 region starting at (1,1)
    # Tile origins: (1,1), (1,2), (2,1), (2,2)
    # Each has 1 pixel → 4 preferences
    expected3 = 4
    print(f"  Expected: {expected3} preferences (2x2 region, 1x1 tiles)")
    print(f"  Actual: {len(builder3.preferences)}")
    assert len(builder3.preferences) == expected3, \
        f"Expected {expected3} preferences, got {len(builder3.preferences)}"

    print("\nTest 4: Empty tile pattern")
    print("-" * 70)

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {},  # Empty
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4
    }

    builder4 = ConstraintBuilder()
    build_S8_constraints(ctx, params4, builder4)

    expected4 = 0
    print(f"  Expected: {expected4} preferences (empty pattern)")
    print(f"  Actual: {len(builder4.preferences)}")
    assert len(builder4.preferences) == expected4, \
        f"Expected {expected4} preferences, got {len(builder4.preferences)}"

    print("\n" + "=" * 70)
    print("✓ S8 builder self-test passed.")
