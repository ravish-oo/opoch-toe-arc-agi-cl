"""
Connected components per color for ARC grids.

This module implements per-color connected component extraction from the
math kernel spec (section 1.2.4):
  - component id: comp_id(p) for each color class
  - component size, bounding box (min/max row/col per component)

Uses 4-connectivity (horizontal/vertical neighbors only).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

from src.core.grid_types import Grid, Pixel

# Try to import scipy for efficient component labeling
try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None


@dataclass
class Component:
    """
    Represents a connected component in a grid.

    Attributes:
        id: Unique integer identifier across all components in the grid (0,1,2,...)
        color: Color value of this component (int)
        pixels: List of (row, col) tuples belonging to this component
        size: Number of pixels in this component (= len(pixels))
        bbox: Bounding box as (r_min, r_max, c_min, c_max), inclusive, 0-based
        shape_signature: Translation-invariant shape representation (tuple of relative coords)
    """
    id: int
    color: int
    pixels: List[Pixel]
    size: int
    bbox: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
    shape_signature: Any = field(default=None)


def connected_components_by_color(grid: Grid) -> List[Component]:
    """
    Compute 4-connected components for each distinct color in the grid.

    For each color value in the grid:
      - Create a binary mask (grid == color)
      - Run 4-connected component labeling on that mask
      - For each non-empty component:
          - Collect all pixels (r,c)
          - Compute size and bbox
          - Create a Component object

    Components across all colors are returned in a flat list with
    monotonically increasing ids (0,1,2,...).

    Args:
        grid: Input grid (H, W) with integer color values

    Returns:
        List of Component objects, ordered by color (ascending) then
        component label index

    Example:
        >>> grid = np.array([[0, 0], [1, 1]], dtype=int)
        >>> comps = connected_components_by_color(grid)
        >>> len(comps)
        2
        >>> comps[0].color
        0
        >>> comps[0].size
        2
    """
    assert grid.ndim == 2, f"Grid must be 2D, got {grid.ndim}D"

    # Get all distinct colors in ascending order
    colors = np.unique(grid)

    components = []
    component_id = 0

    for color in colors:
        # Create binary mask for this color
        mask = (grid == color)

        if ndi is not None:
            # Use scipy.ndimage.label with 4-connectivity
            components_for_color = _label_with_scipy(mask, color, component_id)
        else:  # pragma: no cover
            # Fallback to BFS implementation
            components_for_color = _label_with_bfs(mask, color, component_id)

        components.extend(components_for_color)
        component_id += len(components_for_color)

    return components


def _label_with_scipy(mask: np.ndarray, color: int, start_id: int) -> List[Component]:
    """
    Label connected components using scipy.ndimage.label with 4-connectivity.

    Args:
        mask: Boolean mask (H, W) where True indicates pixels of this color
        color: Color value for these components
        start_id: Starting component ID for this color

    Returns:
        List of Component objects for this color
    """
    # Define 4-connectivity structure (no diagonals)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    # Label connected components
    labels, num_components = ndi.label(mask, structure=structure)

    components = []

    for label_id in range(1, num_components + 1):
        # Get coordinates of all pixels with this label
        coords = np.argwhere(labels == label_id)

        if coords.size == 0:  # pragma: no cover
            continue  # Should not happen, but be safe

        # Extract pixels as list of tuples
        pixels = [(int(r), int(c)) for r, c in coords]

        # Compute bounding box
        rows = coords[:, 0]
        cols = coords[:, 1]
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())

        # Compute size
        size = len(pixels)

        # Create component
        comp = Component(
            id=start_id + label_id - 1,
            color=int(color),
            pixels=pixels,
            size=size,
            bbox=(r_min, r_max, c_min, c_max)
        )
        components.append(comp)

    return components


def _label_with_bfs(mask: np.ndarray, color: int, start_id: int) -> List[Component]:  # pragma: no cover
    """
    Label connected components using BFS with 4-connectivity (fallback without scipy).

    Args:
        mask: Boolean mask (H, W) where True indicates pixels of this color
        color: Color value for these components
        start_id: Starting component ID for this color

    Returns:
        List of Component objects for this color
    """
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)

    components = []
    component_id = start_id

    # 4-connectivity offsets (up, down, left, right)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                # Start BFS from this pixel
                pixels = []
                queue = [(r, c)]
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    pixels.append((curr_r, curr_c))

                    # Check 4 neighbors
                    for dr, dc in neighbors:
                        nr, nc = curr_r + dr, curr_c + dc

                        # Check bounds and conditions
                        if (0 <= nr < H and 0 <= nc < W and
                            mask[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                # Compute bounding box
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                r_min, r_max = min(rows), max(rows)
                c_min, c_max = min(cols), max(cols)

                # Create component
                comp = Component(
                    id=component_id,
                    color=int(color),
                    pixels=pixels,
                    size=len(pixels),
                    bbox=(r_min, r_max, c_min, c_max)
                )
                components.append(comp)
                component_id += 1

    return components


def compute_shape_signature(comp: Component) -> Tuple[Tuple[int, int], ...]:
    """
    Compute a translation-invariant shape signature for a component.

    Steps:
      - Use comp.bbox = (r_min, r_max, c_min, c_max)
      - For each pixel (r,c) in comp.pixels:
          dr = r - r_min
          dc = c - c_min
      - Collect all (dr, dc) pairs
      - Sort them in lexicographic order
      - Return them as a tuple of (dr, dc) pairs

    This makes shapes equal up to translation (same pattern, different location).

    Args:
        comp: Component to compute signature for

    Returns:
        Tuple of sorted (dr, dc) pairs representing the shape

    Example:
        >>> comp = Component(id=0, color=1, pixels=[(5,7), (5,8), (6,7)],
        ...                  size=3, bbox=(5,6,7,8))
        >>> compute_shape_signature(comp)
        ((0, 0), (0, 1), (1, 0))
    """
    (r_min, r_max, c_min, c_max) = comp.bbox
    rel_coords: List[Tuple[int, int]] = []
    for (r, c) in comp.pixels:
        dr = r - r_min
        dc = c - c_min
        rel_coords.append((dr, dc))

    rel_coords_sorted = sorted(rel_coords)
    signature: Tuple[Tuple[int, int], ...] = tuple(rel_coords_sorted)
    return signature


def assign_object_ids(components: List[Component]) -> Dict[Pixel, int]:
    """
    Given a list of Components, compute shape_signature for each,
    group components with the same (color, shape_signature),
    and assign an object_id (0,1,2,...) per group.

    Args:
        components: List of Component objects

    Returns:
        Dictionary mapping (r,c) -> object_id for all pixels in all components

    Notes:
        - Object classes are defined by (color, shape_signature) pairs
        - Two components with the same shape but different colors get different object_ids
        - All components in the same object class share one object_id
        - Background pixels (not in any component) are not in the returned dict

    Example:
        >>> # Two identical 2x2 squares of color 1 at different positions
        >>> # will get the same object_id
        >>> comps = connected_components_by_color(grid)
        >>> obj_ids = assign_object_ids(comps)
        >>> # obj_ids maps each pixel to its object class
    """
    # First, ensure all components have shape_signature computed
    for comp in components:
        if comp.shape_signature is None:
            comp.shape_signature = compute_shape_signature(comp)

    # Group by (color, shape_signature)
    groups: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], List[Component]] = {}
    for comp in components:
        key = (comp.color, comp.shape_signature)
        groups.setdefault(key, []).append(comp)

    # Assign object_ids
    pixel_to_object_id: Dict[Pixel, int] = {}
    current_object_id = 0
    for key, comps_in_group in groups.items():
        # all comps in this group share the same object_id
        for comp in comps_in_group:
            for (r, c) in comp.pixels:
                pixel_to_object_id[(r, c)] = current_object_id
        current_object_id += 1

    return pixel_to_object_id


if __name__ == "__main__":
    # Simple grid with two identical shapes in different places
    grid = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    comps = connected_components_by_color(grid)
    print("\nComponents:")
    for comp in comps:
        comp.shape_signature = compute_shape_signature(comp)
        print(f"  id={comp.id}, color={comp.color}, size={comp.size}, bbox={comp.bbox}, shape_sig={comp.shape_signature}")

    pixel_to_obj = assign_object_ids(comps)
    print("\nPixel to object_id:")
    # print object_id map in grid form for clarity
    H, W = grid.shape
    obj_grid = -np.ones_like(grid)
    for (r, c), oid in pixel_to_obj.items():
        obj_grid[r, c] = oid
    print(obj_grid)
