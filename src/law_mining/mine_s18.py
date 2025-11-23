"""
S18 Miner: Symmetry & Crystallography (D4 + Local Axis-Relative Symmetry).

Detects two types of symmetry transformations:

1. GLOBAL SYMMETRY: Entire input transforms to output via D4 operation
   - Identity, Rot90, Rot180, Rot270
   - FlipX, FlipY, FlipDiag, FlipAntiDiag

2. LOCAL/AXIS-RELATIVE SYMMETRY: Source pixels reflect across axis into target
   - Uses axis as "hinge" - positions defined by distance from axis
   - Handles asymmetric regions naturally (different sizes on each side)
   - Validates pixel-by-pixel, not rectangular region comparison

Algorithm:
  1. Detect background color (mode of grid)
  2. Try global D4 transforms first (fast)
  3. If no global match, try axis-relative local symmetry
  4. Validate consistency across ALL training examples
  5. Return SchemaInstance with detected parameters
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Callable, Tuple
import numpy as np
from scipy import stats

from src.schemas.context import TaskContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance


# =============================================================================
# D4 Symmetry Group Transforms (Global)
# =============================================================================

D4_TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda g: g.copy(),
    "rot90": lambda g: np.rot90(g, k=-1),  # Clockwise
    "rot180": lambda g: np.rot90(g, k=2),
    "rot270": lambda g: np.rot90(g, k=1),  # Counter-clockwise
    "flip_x": lambda g: np.flip(g, axis=1),
    "flip_y": lambda g: np.flip(g, axis=0),
    "flip_diag": lambda g: g.T,
    "flip_antidiag": lambda g: np.flip(np.flip(g, axis=0), axis=1).T,
}


# =============================================================================
# Background Detection
# =============================================================================

def detect_background(grid: np.ndarray) -> int:
    """
    Detect the background color of a grid.
    Uses mode (most frequent color) as the background.
    """
    if grid.size == 0:
        return 0
    flat = grid.flatten()
    mode_result = stats.mode(flat, keepdims=False)
    return int(mode_result.mode)


# =============================================================================
# Axis Detection (Semantic - by Color)
# =============================================================================

def find_axis_candidates(grid: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Find candidate axis positions for splitting the grid.
    Returns (row_axes, col_axes).
    """
    H, W = grid.shape
    row_axes = []
    col_axes = []

    # Find uniform rows (potential horizontal dividers)
    for r in range(H):
        if len(np.unique(grid[r, :])) == 1:
            if 0 < r < H - 1:
                row_axes.append(r)

    # Find uniform cols (potential vertical dividers)
    for c in range(W):
        if len(np.unique(grid[:, c])) == 1:
            if 0 < c < W - 1:
                col_axes.append(c)

    # Try center split if no uniform dividers found
    if not row_axes and H >= 3:
        row_axes.append(H // 2)
    if not col_axes and W >= 3:
        col_axes.append(W // 2)

    return row_axes, col_axes


def find_axis_candidates_with_color(grid: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Find candidate axis positions WITH their colors for splitting the grid.
    Returns (row_axes, col_axes) where each is a list of (index, color) tuples.

    Semantic axis detection: axis is identified by its COLOR, not coordinate.
    """
    H, W = grid.shape
    row_axes = []  # List of (row_index, axis_color)
    col_axes = []  # List of (col_index, axis_color)

    # Find uniform rows (potential horizontal dividers)
    for r in range(H):
        unique_colors = np.unique(grid[r, :])
        if len(unique_colors) == 1:
            if 0 < r < H - 1:
                row_axes.append((r, int(unique_colors[0])))

    # Find uniform cols (potential vertical dividers)
    for c in range(W):
        unique_colors = np.unique(grid[:, c])
        if len(unique_colors) == 1:
            if 0 < c < W - 1:
                col_axes.append((c, int(unique_colors[0])))

    return row_axes, col_axes


def resolve_axis_from_color(
    grid: np.ndarray,
    axis_type: str,
    axis_color: int
) -> Optional[int]:
    """
    Resolve axis position from color - find row/col with uniform axis_color.

    Args:
        grid: The grid to search
        axis_type: "row" or "col"
        axis_color: The color the axis should have

    Returns:
        The index of the axis, or None if not found
    """
    H, W = grid.shape

    if axis_type == "row":
        for r in range(H):
            unique_colors = np.unique(grid[r, :])
            if len(unique_colors) == 1 and int(unique_colors[0]) == axis_color:
                return r
    elif axis_type == "col":
        for c in range(W):
            unique_colors = np.unique(grid[:, c])
            if len(unique_colors) == 1 and int(unique_colors[0]) == axis_color:
                return c

    return None


# =============================================================================
# Dynamic Uniform Axis Detection (Abstract - not tied to specific color)
# =============================================================================

def is_uniform_line(
    grid: np.ndarray,
    axis_type: str,
    axis_idx: int,
    bg_color: int,
    uniformity_threshold: float = 0.9
) -> bool:
    """
    Check if a row/col is a uniform line distinct from background.

    A uniform line is:
    1. Mostly one color (>= threshold)
    2. That color is distinct from background

    Args:
        grid: The grid to check
        axis_type: "row" or "col"
        axis_idx: Index of the row/col
        bg_color: Background color to exclude
        uniformity_threshold: Minimum fraction of pixels that must be same color

    Returns:
        True if the line is a valid uniform divider
    """
    H, W = grid.shape

    if axis_type == "row":
        if not (0 < axis_idx < H - 1):  # Must not be edge
            return False
        line = grid[axis_idx, :]
    elif axis_type == "col":
        if not (0 < axis_idx < W - 1):  # Must not be edge
            return False
        line = grid[:, axis_idx]
    else:
        return False

    # Count color frequencies
    unique, counts = np.unique(line, return_counts=True)
    if len(unique) == 0:
        return False

    # Find dominant color
    dominant_idx = np.argmax(counts)
    dominant_color = int(unique[dominant_idx])
    dominant_count = counts[dominant_idx]

    # Check uniformity threshold
    uniformity = dominant_count / len(line)
    if uniformity < uniformity_threshold:
        return False

    # Check distinct from background
    if dominant_color == bg_color:
        return False

    return True


def find_uniform_axis(
    grid: np.ndarray,
    axis_type: str,
    bg_color: int,
    uniformity_threshold: float = 0.9
) -> Optional[int]:
    """
    Find a uniform divider line in the grid (dynamic detection at runtime).

    Searches for a row/col that is:
    1. Uniform (mostly one color)
    2. Distinct from background
    3. Spanning (full width/height)

    Tie-breaker: Pick line closest to center.

    Args:
        grid: The grid to search
        axis_type: "row" or "col"
        bg_color: Background color
        uniformity_threshold: Minimum uniformity fraction

    Returns:
        Index of the uniform axis, or None if not found
    """
    H, W = grid.shape
    candidates = []

    if axis_type == "row":
        center = H // 2
        for r in range(1, H - 1):  # Skip edges
            if is_uniform_line(grid, "row", r, bg_color, uniformity_threshold):
                candidates.append((r, abs(r - center)))
    elif axis_type == "col":
        center = W // 2
        for c in range(1, W - 1):  # Skip edges
            if is_uniform_line(grid, "col", c, bg_color, uniformity_threshold):
                candidates.append((c, abs(c - center)))

    if not candidates:
        return None

    # Sort by distance to center (ascending), pick closest
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# =============================================================================
# Axis-Relative Reflection Helpers
# =============================================================================

def get_source_pixels(
    grid: np.ndarray,
    bg_color: int,
    axis_row: Optional[int],
    axis_col: Optional[int],
    source_side: str
) -> List[Tuple[int, int, int]]:
    """
    Get list of active (non-background) pixels on the source side of the axis.

    Returns list of (row, col, color) tuples.
    """
    H, W = grid.shape
    pixels = []

    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == bg_color:
                continue

            # Skip axis row/col
            if axis_row is not None and r == axis_row:
                continue
            if axis_col is not None and c == axis_col:
                continue

            # Check if pixel is on source side
            in_source = False

            if source_side == "left" and axis_col is not None:
                in_source = c < axis_col
            elif source_side == "right" and axis_col is not None:
                in_source = c > axis_col
            elif source_side == "top" and axis_row is not None:
                in_source = r < axis_row
            elif source_side == "bottom" and axis_row is not None:
                in_source = r > axis_row
            elif source_side == "top_left" and axis_row is not None and axis_col is not None:
                in_source = r < axis_row and c < axis_col
            elif source_side == "top_right" and axis_row is not None and axis_col is not None:
                in_source = r < axis_row and c > axis_col
            elif source_side == "bottom_left" and axis_row is not None and axis_col is not None:
                in_source = r > axis_row and c < axis_col
            elif source_side == "bottom_right" and axis_row is not None and axis_col is not None:
                in_source = r > axis_row and c > axis_col

            if in_source:
                pixels.append((r, c, color))

    return pixels


def reflect_coordinate(
    r: int, c: int,
    axis_row: Optional[int],
    axis_col: Optional[int],
    transform: str
) -> Tuple[int, int]:
    """
    Reflect a coordinate across the axis using axis-relative math.

    The axis acts as a "hinge" - the reflected position is equidistant
    from the axis on the opposite side.

    FlipX: c_target = 2 * axis_col - c (reflect across vertical axis)
    FlipY: r_target = 2 * axis_row - r (reflect across horizontal axis)
    FlipBoth: apply both
    """
    r_target, c_target = r, c

    if transform in ["flip_x", "flip_both"]:
        if axis_col is not None:
            c_target = 2 * axis_col - c

    if transform in ["flip_y", "flip_both"]:
        if axis_row is not None:
            r_target = 2 * axis_row - r

    return r_target, c_target


def validate_axis_reflection(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    bg_color: int,
    axis_row: Optional[int],
    axis_col: Optional[int],
    source_side: str,
    transform: str
) -> bool:
    """
    Validate that source pixels reflect correctly into output.

    Uses axis-relative coordinates - handles asymmetric regions.
    """
    H, W = output_grid.shape

    # Get source pixels
    source_pixels = get_source_pixels(input_grid, bg_color, axis_row, axis_col, source_side)

    if not source_pixels:
        return False  # No source content

    # Check each source pixel maps correctly to output
    for r, c, color in source_pixels:
        r_target, c_target = reflect_coordinate(r, c, axis_row, axis_col, transform)

        # Bounds check - SKIP if out of bounds (asymmetric regions)
        # Don't fail - just skip pixels that can't fit in target region
        if not (0 <= r_target < H and 0 <= c_target < W):
            continue  # Skip this pixel, validate only what fits

        # Check output has correct color at target
        if int(output_grid[r_target, c_target]) != color:
            return False

    return True


def identify_source_and_transform(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    bg_color: int,
    axis_row: Optional[int],
    axis_col: Optional[int],
    split_type: str
) -> Optional[Tuple[str, str]]:
    """
    Identify which side is source and what transform maps it to output.

    Returns (source_side, transform) or None.
    """
    if split_type == "vertical":
        # Try left -> right (flip_x)
        if validate_axis_reflection(input_grid, output_grid, bg_color,
                                    axis_row, axis_col, "left", "flip_x"):
            return "left", "flip_x"
        # Try right -> left (flip_x)
        if validate_axis_reflection(input_grid, output_grid, bg_color,
                                    axis_row, axis_col, "right", "flip_x"):
            return "right", "flip_x"

    elif split_type == "horizontal":
        # Try top -> bottom (flip_y)
        if validate_axis_reflection(input_grid, output_grid, bg_color,
                                    axis_row, axis_col, "top", "flip_y"):
            return "top", "flip_y"
        # Try bottom -> top (flip_y)
        if validate_axis_reflection(input_grid, output_grid, bg_color,
                                    axis_row, axis_col, "bottom", "flip_y"):
            return "bottom", "flip_y"

    elif split_type == "quadrant":
        # Try each quadrant as source with appropriate transforms
        quadrants = ["top_left", "top_right", "bottom_left", "bottom_right"]
        transforms_map = {
            "top_left": [("top_right", "flip_x"), ("bottom_left", "flip_y"), ("bottom_right", "flip_both")],
            "top_right": [("top_left", "flip_x"), ("bottom_right", "flip_y"), ("bottom_left", "flip_both")],
            "bottom_left": [("top_left", "flip_y"), ("bottom_right", "flip_x"), ("top_right", "flip_both")],
            "bottom_right": [("top_right", "flip_x"), ("bottom_left", "flip_y"), ("top_left", "flip_both")],
        }

        for source in quadrants:
            # Check if this quadrant has content
            source_pixels = get_source_pixels(input_grid, bg_color, axis_row, axis_col, source)
            if not source_pixels:
                continue

            # Validate all target transforms work
            all_valid = True
            for target, transform in transforms_map[source]:
                if not validate_axis_reflection(input_grid, output_grid, bg_color,
                                                axis_row, axis_col, source, transform):
                    all_valid = False
                    break

            if all_valid:
                return source, "quadrant"

    return None


# =============================================================================
# Main Mining Functions
# =============================================================================

def mine_global_symmetry(task_context: TaskContext) -> Optional[str]:
    """Try to find a global D4 transform that works for all training examples."""
    if not task_context.train_examples:
        return None

    valid_transforms: Set[str] = set(D4_TRANSFORMS.keys())

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        example_valid: Set[str] = set()
        for transform_name, transform_fn in D4_TRANSFORMS.items():
            try:
                transformed = transform_fn(ex.input_grid)
                if transformed.shape == ex.output_grid.shape:
                    if np.array_equal(transformed, ex.output_grid):
                        example_valid.add(transform_name)
            except Exception:
                continue

        valid_transforms = valid_transforms.intersection(example_valid)
        if not valid_transforms:
            return None

    if not valid_transforms:
        return None

    # Prefer non-identity
    if len(valid_transforms) > 1 and "identity" in valid_transforms:
        valid_transforms.discard("identity")

    return sorted(valid_transforms)[0]


def mine_local_symmetry(task_context: TaskContext) -> Optional[Dict[str, any]]:
    """
    Try to find local axis-relative symmetry pattern.
    Uses pixel-by-pixel validation with axis as hinge.

    SEMANTIC AXIS DETECTION: Axis is identified by COLOR, not coordinate.
    This allows axis position to vary between examples while maintaining
    the same semantic meaning (same color = same axis).
    """
    if not task_context.train_examples:
        return None

    # Need geometry-preserving tasks
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return None

    # Detect background
    bg_color = detect_background(task_context.train_examples[0].input_grid)

    # =========================================================================
    # SEMANTIC AXIS DETECTION: Find common axis COLORS across all examples
    # =========================================================================

    # Collect axis colors for each example
    row_axis_colors_per_example = []
    col_axis_colors_per_example = []

    for ex in task_context.train_examples:
        row_axes_with_color, col_axes_with_color = find_axis_candidates_with_color(ex.input_grid)

        # Extract just the colors
        row_colors = set(color for _, color in row_axes_with_color)
        col_colors = set(color for _, color in col_axes_with_color)

        row_axis_colors_per_example.append(row_colors)
        col_axis_colors_per_example.append(col_colors)

    # Find common axis colors across ALL examples (intersection)
    common_row_colors: Set[int] = set()
    common_col_colors: Set[int] = set()

    if row_axis_colors_per_example:
        common_row_colors = row_axis_colors_per_example[0]
        for colors in row_axis_colors_per_example[1:]:
            common_row_colors = common_row_colors.intersection(colors)

    if col_axis_colors_per_example:
        common_col_colors = col_axis_colors_per_example[0]
        for colors in col_axis_colors_per_example[1:]:
            common_col_colors = common_col_colors.intersection(colors)

    # =========================================================================
    # Try vertical splits with semantic axis (by color)
    # =========================================================================
    for axis_color in sorted(common_col_colors):
        if axis_color == bg_color:
            continue  # Skip background color as axis

        params = try_axis_symmetry_by_color(
            task_context, None, axis_color, "vertical", bg_color
        )
        if params:
            return params

    # =========================================================================
    # Try horizontal splits with semantic axis (by color)
    # =========================================================================
    for axis_color in sorted(common_row_colors):
        if axis_color == bg_color:
            continue  # Skip background color as axis

        params = try_axis_symmetry_by_color(
            task_context, axis_color, None, "horizontal", bg_color
        )
        if params:
            return params

    # =========================================================================
    # Try quadrant splits with semantic axes (by color)
    # =========================================================================
    for row_axis_color in sorted(common_row_colors):
        if row_axis_color == bg_color:
            continue
        for col_axis_color in sorted(common_col_colors):
            if col_axis_color == bg_color:
                continue

            params = try_axis_symmetry_by_color(
                task_context, row_axis_color, col_axis_color, "quadrant", bg_color
            )
            if params:
                return params

    # =========================================================================
    # DYNAMIC UNIFORM: Axis color varies between examples, but each has a uniform line
    # This is the most abstract level - "reflect across THE divider" (whatever color)
    # =========================================================================

    # Try vertical splits with dynamic uniform axis
    params = try_axis_symmetry_dynamic_uniform(task_context, "col", "vertical", bg_color)
    if params:
        return params

    # Try horizontal splits with dynamic uniform axis
    params = try_axis_symmetry_dynamic_uniform(task_context, "row", "horizontal", bg_color)
    if params:
        return params

    # =========================================================================
    # FALLBACK: Try fixed coordinate axes (for center splits without uniform dividers)
    # =========================================================================
    # Collect axis candidates by coordinate
    all_row_axes: Set[int] = set()
    all_col_axes: Set[int] = set()

    for ex in task_context.train_examples:
        row_axes, col_axes = find_axis_candidates(ex.input_grid)
        all_row_axes.update(row_axes)
        all_col_axes.update(col_axes)

    # Try vertical splits (flip_x across vertical axis)
    for axis_col in sorted(all_col_axes):
        params = try_axis_symmetry(task_context, None, axis_col, "vertical", bg_color)
        if params:
            return params

    # Try horizontal splits (flip_y across horizontal axis)
    for axis_row in sorted(all_row_axes):
        params = try_axis_symmetry(task_context, axis_row, None, "horizontal", bg_color)
        if params:
            return params

    # Try quadrant splits (both axes)
    for axis_row in sorted(all_row_axes):
        for axis_col in sorted(all_col_axes):
            params = try_axis_symmetry(task_context, axis_row, axis_col, "quadrant", bg_color)
            if params:
                return params

    return None


def try_axis_symmetry(
    task_context: TaskContext,
    axis_row: Optional[int],
    axis_col: Optional[int],
    split_type: str,
    bg_color: int
) -> Optional[Dict[str, any]]:
    """Validate axis-relative symmetry for given axis positions."""
    consistent_source = None
    consistent_transform = None

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        result = identify_source_and_transform(
            ex.input_grid, ex.output_grid, bg_color,
            axis_row, axis_col, split_type
        )

        if result is None:
            return None

        source, transform = result

        # Check consistency
        if consistent_source is None:
            consistent_source = source
            consistent_transform = transform
        else:
            if source != consistent_source or transform != consistent_transform:
                return None

    if consistent_source and consistent_transform:
        return {
            "mode": "local",
            "split_type": split_type,
            "axis_row": axis_row,
            "axis_col": axis_col,
            "source": consistent_source,
            "transform": consistent_transform,
            "bg_color": bg_color
        }

    return None


def try_axis_symmetry_by_color(
    task_context: TaskContext,
    row_axis_color: Optional[int],
    col_axis_color: Optional[int],
    split_type: str,
    bg_color: int
) -> Optional[Dict[str, any]]:
    """
    Validate axis-relative symmetry using SEMANTIC axis detection (by color).

    Resolves axis position from color for each example independently,
    then validates that the symmetry pattern is consistent.

    Args:
        task_context: TaskContext with training examples
        row_axis_color: Color of the horizontal axis (None for vertical-only)
        col_axis_color: Color of the vertical axis (None for horizontal-only)
        split_type: "vertical", "horizontal", or "quadrant"
        bg_color: Background color to exclude

    Returns:
        Dict with params including axis_row_color/axis_col_color, or None
    """
    consistent_source = None
    consistent_transform = None

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        # Resolve axis position from color for THIS example
        axis_row = None
        axis_col = None

        if row_axis_color is not None:
            axis_row = resolve_axis_from_color(ex.input_grid, "row", row_axis_color)
            if axis_row is None:
                return None  # Axis color not found in this example

        if col_axis_color is not None:
            axis_col = resolve_axis_from_color(ex.input_grid, "col", col_axis_color)
            if axis_col is None:
                return None  # Axis color not found in this example

        # Validate symmetry with resolved axis
        result = identify_source_and_transform(
            ex.input_grid, ex.output_grid, bg_color,
            axis_row, axis_col, split_type
        )

        if result is None:
            return None

        source, transform = result

        # Check consistency across examples
        if consistent_source is None:
            consistent_source = source
            consistent_transform = transform
        else:
            if source != consistent_source or transform != consistent_transform:
                return None

    if consistent_source and consistent_transform:
        # Return params with axis COLORS (semantic), not coordinates
        return {
            "mode": "local",
            "split_type": split_type,
            "axis_row_color": row_axis_color,  # Semantic: axis identified by color
            "axis_col_color": col_axis_color,  # Semantic: axis identified by color
            "source": consistent_source,
            "transform": consistent_transform,
            "bg_color": bg_color
        }

    return None


def try_axis_symmetry_dynamic_uniform(
    task_context: TaskContext,
    axis_type: str,  # "row" or "col"
    split_type: str,
    bg_color: int
) -> Optional[Dict[str, any]]:
    """
    Validate axis-relative symmetry using DYNAMIC UNIFORM axis detection.

    This is the most abstract level: instead of requiring a specific axis color,
    we just require that EACH example has a uniform line that acts as the axis.
    The color can vary between examples.

    Algorithm:
    1. For each example, find a uniform line (distinct from bg)
    2. Validate symmetry works with that axis
    3. If ALL examples have consistent source/transform, emit law with axis_rule="dynamic_uniform"

    Args:
        task_context: TaskContext with training examples
        axis_type: "row" for horizontal split, "col" for vertical split
        split_type: "vertical" or "horizontal"
        bg_color: Background color

    Returns:
        Dict with params including axis_rule="dynamic_uniform", or None
    """
    consistent_source = None
    consistent_transform = None

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        # Find uniform axis dynamically for THIS example
        axis_idx = find_uniform_axis(ex.input_grid, axis_type, bg_color)

        if axis_idx is None:
            return None  # No uniform axis found in this example

        # Set up axis_row/axis_col based on type
        axis_row = axis_idx if axis_type == "row" else None
        axis_col = axis_idx if axis_type == "col" else None

        # Validate symmetry with detected axis
        result = identify_source_and_transform(
            ex.input_grid, ex.output_grid, bg_color,
            axis_row, axis_col, split_type
        )

        if result is None:
            return None

        source, transform = result

        # Check consistency across examples
        if consistent_source is None:
            consistent_source = source
            consistent_transform = transform
        else:
            if source != consistent_source or transform != consistent_transform:
                return None

    if consistent_source and consistent_transform:
        # Return params with DYNAMIC UNIFORM rule (most abstract)
        return {
            "mode": "local",
            "split_type": split_type,
            "axis_rule": "dynamic_uniform",  # Abstract: "find the uniform divider"
            "axis_type": axis_type,  # "row" or "col" - which direction to search
            "source": consistent_source,
            "transform": consistent_transform,
            "bg_color": bg_color
        }

    return None


def mine_S18(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S18 schema instances: global and local symmetry transforms.

    Two-stage algorithm:
      1. Try global D4 transforms (fast)
      2. If no global match, try axis-relative local symmetry
    """
    instances: List[SchemaInstance] = []

    # Stage 1: Try global symmetry
    global_transform = mine_global_symmetry(task_context)

    if global_transform:
        for ex_idx in range(len(task_context.train_examples)):
            instances.append(SchemaInstance(
                family_id="S18",
                params={
                    "mode": "global",
                    "example_type": "train",
                    "example_index": ex_idx,
                    "transform": global_transform
                }
            ))

        for ex_idx in range(len(task_context.test_examples)):
            instances.append(SchemaInstance(
                family_id="S18",
                params={
                    "mode": "global",
                    "example_type": "test",
                    "example_index": ex_idx,
                    "transform": global_transform
                }
            ))

        return instances

    # Stage 2: Try local symmetry
    local_params = mine_local_symmetry(task_context)

    if local_params:
        for ex_idx in range(len(task_context.train_examples)):
            params = dict(local_params)
            params["example_type"] = "train"
            params["example_index"] = ex_idx
            instances.append(SchemaInstance(
                family_id="S18",
                params=params
            ))

        for ex_idx in range(len(task_context.test_examples)):
            params = dict(local_params)
            params["example_type"] = "test"
            params["example_index"] = ex_idx
            instances.append(SchemaInstance(
                family_id="S18",
                params=params
            ))

    return instances


if __name__ == "__main__":
    print("=" * 70)
    print("S18 Miner self-test (Axis-Relative Reflection)")
    print("=" * 70)

    from src.schemas.context import build_example_context

    # Test 1: Global FlipX
    print("\nTest 1: Global FlipX")
    print("-" * 70)

    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    output1 = np.array([[3, 2, 1], [6, 5, 4]], dtype=int)

    ex1 = build_example_context(input1, output1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=10)

    instances1 = mine_S18(ctx1, {}, {})
    print(f"  Mined: {len(instances1)} instance(s)")
    if instances1:
        print(f"  Mode: {instances1[0].params.get('mode')}")
        print(f"  Transform: {instances1[0].params.get('transform')}")
        assert instances1[0].params["mode"] == "global"
    print("  OK")

    # Test 2: Asymmetric vertical reflection (axis off-center)
    print("\nTest 2: Asymmetric vertical reflection (axis at col 2)")
    print("-" * 70)

    # Axis at col 2 (uniform 5s)
    # Left side: 2 cols (0, 1)
    # Right side: 3 cols (3, 4, 5)
    # Source on left, reflects to right
    input2 = np.array([
        [1, 2, 5, 0, 0, 0],
        [3, 4, 5, 0, 0, 0],
        [5, 6, 5, 0, 0, 0],
    ], dtype=int)
    output2 = np.array([
        [1, 2, 5, 2, 1, 0],  # Note: col 5 stays 0 (asymmetric)
        [3, 4, 5, 4, 3, 0],
        [5, 6, 5, 6, 5, 0],
    ], dtype=int)

    ex2 = build_example_context(input2, output2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=10)

    instances2 = mine_S18(ctx2, {}, {})
    print(f"  Mined: {len(instances2)} instance(s)")
    if instances2:
        print(f"  Mode: {instances2[0].params.get('mode')}")
        print(f"  Split type: {instances2[0].params.get('split_type')}")
        print(f"  Source: {instances2[0].params.get('source')}")
        print(f"  Transform: {instances2[0].params.get('transform')}")
        print(f"  Axis col: {instances2[0].params.get('axis_col')}")
        assert instances2[0].params["mode"] == "local"
        assert instances2[0].params["source"] == "left"
    print("  OK: Detected asymmetric vertical reflection")

    # Test 3: Quadrant symmetry with axis
    print("\nTest 3: Quadrant symmetry")
    print("-" * 70)

    input3 = np.array([
        [1, 2, 5, 0, 0],
        [3, 4, 5, 0, 0],
        [5, 5, 5, 5, 5],
        [0, 0, 5, 0, 0],
        [0, 0, 5, 0, 0]
    ], dtype=int)
    output3 = np.array([
        [1, 2, 5, 2, 1],
        [3, 4, 5, 4, 3],
        [5, 5, 5, 5, 5],
        [3, 4, 5, 4, 3],
        [1, 2, 5, 2, 1]
    ], dtype=int)

    ex3 = build_example_context(input3, output3)
    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=10)

    instances3 = mine_S18(ctx3, {}, {})
    print(f"  Mined: {len(instances3)} instance(s)")
    if instances3:
        print(f"  Mode: {instances3[0].params.get('mode')}")
        print(f"  Source: {instances3[0].params.get('source')}")
    print("  OK")

    # Test 4: No symmetry
    print("\nTest 4: No symmetry")
    print("-" * 70)

    input4 = np.array([[1, 2], [3, 4]], dtype=int)
    output4 = np.array([[9, 9], [9, 9]], dtype=int)

    ex4 = build_example_context(input4, output4)
    ctx4 = TaskContext(train_examples=[ex4], test_examples=[], C=10)

    instances4 = mine_S18(ctx4, {}, {})
    print(f"  Mined: {len(instances4)} instance(s)")
    assert len(instances4) == 0
    print("  OK: Correctly rejected")

    # Test 5: SEMANTIC AXIS DETECTION (varying positions, same color)
    print("\nTest 5: Semantic axis detection (axis position varies, color constant)")
    print("-" * 70)

    # Example 1: Axis at col 2 (color 5)
    input5a = np.array([
        [1, 2, 5, 0, 0],
        [3, 4, 5, 0, 0],
    ], dtype=int)
    output5a = np.array([
        [1, 2, 5, 2, 1],
        [3, 4, 5, 4, 3],
    ], dtype=int)

    # Example 2: Axis at col 4 (SAME color 5, DIFFERENT position)
    input5b = np.array([
        [1, 2, 3, 4, 5, 0, 0, 0, 0],
        [6, 7, 8, 9, 5, 0, 0, 0, 0],
    ], dtype=int)
    output5b = np.array([
        [1, 2, 3, 4, 5, 4, 3, 2, 1],
        [6, 7, 8, 9, 5, 9, 8, 7, 6],
    ], dtype=int)

    ex5a = build_example_context(input5a, output5a)
    ex5b = build_example_context(input5b, output5b)
    ctx5 = TaskContext(train_examples=[ex5a, ex5b], test_examples=[], C=10)

    instances5 = mine_S18(ctx5, {}, {})
    print(f"  Mined: {len(instances5)} instance(s)")
    if instances5:
        params = instances5[0].params
        print(f"  Mode: {params.get('mode')}")
        print(f"  Split type: {params.get('split_type')}")
        print(f"  Source: {params.get('source')}")
        print(f"  axis_col (fixed): {params.get('axis_col')}")
        print(f"  axis_col_color (semantic): {params.get('axis_col_color')}")

        # KEY: Should have axis_col_color=5, not fixed axis_col
        assert params.get("axis_col_color") == 5, \
            f"Expected axis_col_color=5, got {params.get('axis_col_color')}"
        assert params.get("axis_col") is None, \
            f"Expected axis_col=None (semantic), got {params.get('axis_col')}"
        print("  OK: Correctly detected SEMANTIC axis by color (not coordinate)")
    else:
        print("  FAILED: No instances mined")
        assert False, "Expected to mine semantic axis symmetry"

    # Test 6: DYNAMIC UNIFORM (axis COLOR varies between examples)
    print("\nTest 6: Dynamic uniform axis (axis color VARIES between examples)")
    print("-" * 70)

    # Example 1: Blue axis (color 1) at col 2
    input6a = np.array([
        [2, 3, 1, 0, 0],
        [4, 5, 1, 0, 0],
    ], dtype=int)
    output6a = np.array([
        [2, 3, 1, 3, 2],
        [4, 5, 1, 5, 4],
    ], dtype=int)

    # Example 2: Pink axis (color 6) at col 2 - DIFFERENT COLOR, same structure
    input6b = np.array([
        [2, 3, 6, 0, 0],
        [4, 5, 6, 0, 0],
    ], dtype=int)
    output6b = np.array([
        [2, 3, 6, 3, 2],
        [4, 5, 6, 5, 4],
    ], dtype=int)

    ex6a = build_example_context(input6a, output6a)
    ex6b = build_example_context(input6b, output6b)
    ctx6 = TaskContext(train_examples=[ex6a, ex6b], test_examples=[], C=10)

    instances6 = mine_S18(ctx6, {}, {})
    print(f"  Mined: {len(instances6)} instance(s)")
    if instances6:
        params = instances6[0].params
        print(f"  Mode: {params.get('mode')}")
        print(f"  Split type: {params.get('split_type')}")
        print(f"  Source: {params.get('source')}")
        print(f"  axis_rule: {params.get('axis_rule')}")
        print(f"  axis_type: {params.get('axis_type')}")

        # KEY: Should have axis_rule="dynamic_uniform" (not specific color)
        assert params.get("axis_rule") == "dynamic_uniform", \
            f"Expected axis_rule='dynamic_uniform', got {params.get('axis_rule')}"
        assert params.get("axis_type") == "col", \
            f"Expected axis_type='col', got {params.get('axis_type')}"
        print("  OK: Correctly detected DYNAMIC UNIFORM axis (color-agnostic)")
    else:
        print("  FAILED: No instances mined")
        assert False, "Expected to mine dynamic uniform symmetry"

    print("\n" + "=" * 70)
    print("S18 miner self-test passed.")
    print("=" * 70)
