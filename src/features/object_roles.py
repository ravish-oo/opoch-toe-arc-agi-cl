"""
Component-relative roles and sectors for ARC grids.

This module implements higher-level features derived from components
(math kernel spec section 1.2.8 and related):
  - Per-pixel quadrant/sector within component's bounding box
  - Per-pixel interior vs border within component
  - Per-component role bits (is_small, is_big, is_unique_shape)

All features are deterministic and non-learned.
"""

from typing import Dict, Tuple, List
import numpy as np
from collections import Counter

from src.core.grid_types import Grid, Pixel
from src.features.components import Component, connected_components_by_color, compute_shape_signature


def component_sectors(
    components: List[Component]
) -> Dict[Pixel, Dict[str, str]]:
    """
    For each pixel (r,c) in each component, assign sector labels
    relative to that component's bounding box.

    This implements the quadrant/sector feature from the math kernel spec
    (section 1.2.8): relative position in bounding box of its component.

    Args:
        components: List of Component objects

    Returns:
        Dictionary mapping (r,c) -> {
          "vert_sector": "top" | "center" | "bottom",
          "horiz_sector": "left" | "center" | "right"
        }

    Notes:
        - Sectors are computed relative to each component's bbox, not the grid
        - Uses same edge/middle logic as row/col band labels
        - If bbox height=1, all pixels get "center" vertically
        - If bbox height=2, first row is "top", last is "bottom"
        - If bbox height≥3, first is "top", last is "bottom", middle is "center"
        - Same logic for horizontal sectors

    Example:
        >>> # Component with bbox (0, 2, 0, 2) - 3×3 component
        >>> # Pixels at row 0 get "top", row 1 get "center", row 2 get "bottom"
    """
    result = {}

    for comp in components:
        r_min, r_max, c_min, c_max = comp.bbox
        h = r_max - r_min + 1  # component height
        w = c_max - c_min + 1  # component width

        # Determine vertical sector mapping for this component
        if h == 1:
            vert_map = {r_min: "center"}
        elif h == 2:
            vert_map = {r_min: "top", r_max: "bottom"}
        else:  # h >= 3
            vert_map = {r_min: "top", r_max: "bottom"}
            for r in range(r_min + 1, r_max):
                vert_map[r] = "center"

        # Determine horizontal sector mapping for this component
        if w == 1:
            horiz_map = {c_min: "center"}
        elif w == 2:
            horiz_map = {c_min: "left", c_max: "right"}
        else:  # w >= 3
            horiz_map = {c_min: "left", c_max: "right"}
            for c in range(c_min + 1, c_max):
                horiz_map[c] = "center"

        # Assign sectors to all pixels in this component
        for (r, c) in comp.pixels:
            result[(r, c)] = {
                "vert_sector": vert_map[r],
                "horiz_sector": horiz_map[c],
            }

    return result


def component_border_interior(
    grid: Grid,
    components: List[Component]
) -> Dict[Pixel, Dict[str, bool]]:
    """
    For each pixel (r,c) in each component, mark whether it is
    'border' or 'interior' with respect to that component.

    A pixel is 'interior' if all 4-connected neighbors of same
    color are also in the component. Otherwise it's 'border'.

    This is the per-component version needed for schema S10 (frame/border laws).

    Args:
        grid: Input grid (H, W)
        components: List of Component objects

    Returns:
        Dictionary mapping (r,c) -> {
          "is_border": bool,
          "is_interior": bool
        }

    Notes:
        - Border and interior are mutually exclusive (exactly one is True)
        - A pixel is border if ANY of its 4 neighbors is:
          1. Out of grid bounds
          2. Different color
          3. Same color but different component ID
        - Uses 4-connectivity (up, down, left, right)

    Example:
        >>> # Single isolated pixel: is_border=True (all neighbors are different/absent)
        >>> # Pixel fully surrounded by same component: is_interior=True
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    H, W = grid.shape
    result = {}

    # Build lookup: pixel -> component ID
    pixel_to_comp_id: Dict[Pixel, int] = {}
    for comp in components:
        for (r, c) in comp.pixels:
            pixel_to_comp_id[(r, c)] = comp.id

    # 4-connected neighbors (up, down, left, right)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for comp in components:
        for (r, c) in comp.pixels:
            is_border = False

            # Check all 4 neighbors
            for dr, dc in neighbor_offsets:
                rr, cc = r + dr, c + dc

                # Check if neighbor makes this a border pixel
                if not (0 <= rr < H and 0 <= cc < W):
                    # Out of bounds
                    is_border = True
                    break
                elif grid[rr, cc] != comp.color:
                    # Different color
                    is_border = True
                    break
                elif pixel_to_comp_id.get((rr, cc)) != comp.id:
                    # Same color but different component
                    is_border = True
                    break

            # Interior is the opposite of border
            is_interior = not is_border

            result[(r, c)] = {
                "is_border": is_border,
                "is_interior": is_interior,
            }

    return result


def component_role_bits(
    components: List[Component]
) -> Dict[int, Dict[str, bool]]:
    """
    Assign simple 'role bits' to each component.id:

      - is_small: size in lowest third of sizes
      - is_big:   size in highest third of sizes
      - is_unique_shape: this (color, shape_signature) occurs only once

    This provides role flags that schemas can use to select components
    (e.g., "apply rule only to big components").

    Args:
        components: List of Component objects

    Returns:
        Dictionary mapping comp.id -> {
          "is_small": bool,
          "is_big": bool,
          "is_unique_shape": bool
        }

    Notes:
        - Size thresholds use rank-based percentiles (33rd and 66th)
        - Shape uniqueness is per (color, shape_signature) pair
        - A component can be both is_small and is_big if all sizes are equal
        - Ensures all components have shape_signature computed

    Example:
        >>> # Components with sizes [1, 2, 5, 10]
        >>> # small_cutoff = sizes[4//3] = sizes[1] = 2
        >>> # big_cutoff = sizes[2*4//3] = sizes[2] = 5
        >>> # size=1 → is_small=True, is_big=False
        >>> # size=10 → is_small=False, is_big=True
    """
    if not components:
        return {}

    # Ensure all components have shape_signature
    for comp in components:
        if comp.shape_signature is None:
            comp.shape_signature = compute_shape_signature(comp)

    # Collect sizes and compute thresholds
    sizes = sorted([comp.size for comp in components])
    n = len(sizes)
    small_idx = n // 3
    big_idx = (2 * n) // 3
    small_cutoff = sizes[small_idx]
    big_cutoff = sizes[big_idx]

    # Count (color, shape_signature) occurrences
    key_counts = Counter((comp.color, comp.shape_signature) for comp in components)

    # Assign role bits
    roles = {}
    for comp in components:
        key = (comp.color, comp.shape_signature)
        roles[comp.id] = {
            "is_small": comp.size <= small_cutoff,
            "is_big": comp.size >= big_cutoff,
            "is_unique_shape": key_counts[key] == 1,
        }

    return roles


if __name__ == "__main__":
    # Simple grid with two components of the same shape and one different
    grid = np.array([
        [0, 1, 1, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 2],
        [3, 3, 0, 0, 0],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    # Get components first
    comps = connected_components_by_color(grid)
    print("\nComponents:")
    for comp in comps:
        print(f"  id={comp.id}, color={comp.color}, size={comp.size}, bbox={comp.bbox}")

    # Sectors
    sectors = component_sectors(comps)
    print("\nSectors for a few pixels:")
    sample_pixels = list(sectors.keys())[:5]
    for p in sample_pixels:
        print(f"  {p}: {sectors[p]}")

    # Border vs interior
    border_info = component_border_interior(grid, comps)
    print("\nBorder/interior flags for a few pixels:")
    for p in sample_pixels:
        print(f"  {p}: {border_info[p]}")

    # Role bits
    roles = component_role_bits(comps)
    print("\nComponent role bits:")
    for comp in comps:
        print(f"  comp.id={comp.id}: {roles[comp.id]}")
