"""
S18 schema builder: Symmetry & Crystallography (D4 + Local Axis-Relative Symmetry).

Handles two modes of symmetry:

1. GLOBAL MODE: Apply a D4 transform to entire input grid
   - Params: {"mode": "global", "transform": "flip_x", ...}

2. LOCAL MODE: Reflect source pixels across axis using axis-relative math
   - Uses axis as "hinge" - target position = 2*axis - source position
   - Handles asymmetric regions naturally
   - Only applies preferences to TARGET pixels (axis handled by S1/Inertia)

Uses weight 100.0 as symmetry is a Hard Geometric Law.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


# =============================================================================
# D4 Transform Functions (Global)
# =============================================================================

def apply_global_transform(grid: np.ndarray, transform: str) -> np.ndarray:
    """Apply a D4 transform to a grid."""
    if transform == "identity":
        return grid.copy()
    elif transform == "rot90":
        return np.rot90(grid, k=-1)
    elif transform == "rot180":
        return np.rot90(grid, k=2)
    elif transform == "rot270":
        return np.rot90(grid, k=1)
    elif transform == "flip_x":
        return np.flip(grid, axis=1)
    elif transform == "flip_y":
        return np.flip(grid, axis=0)
    elif transform == "flip_diag":
        return grid.T
    elif transform == "flip_antidiag":
        return np.flip(np.flip(grid, axis=0), axis=1).T
    else:
        return grid.copy()


# =============================================================================
# Axis Resolution (Semantic - by Color)
# =============================================================================

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


def find_uniform_axis(
    grid: np.ndarray,
    axis_type: str,
    bg_color: int,
    uniformity_threshold: float = 0.9
) -> Optional[int]:
    """
    Find a uniform divider line in the grid (dynamic detection at runtime).

    This is the DYNAMIC UNIFORM scanner - finds any uniform line regardless of color.
    Used when axis_rule="dynamic_uniform".

    Searches for a row/col that is:
    1. Uniform (mostly one color, >= threshold)
    2. Distinct from background
    3. Spanning (full width/height)

    Tie-breaker: Pick line closest to center.

    Args:
        grid: The grid to search
        axis_type: "row" or "col"
        bg_color: Background color to exclude
        uniformity_threshold: Minimum fraction of pixels that must be same color

    Returns:
        Index of the uniform axis, or None if not found
    """
    H, W = grid.shape
    candidates = []

    if axis_type == "row":
        center = H // 2
        for r in range(1, H - 1):  # Skip edges
            line = grid[r, :]
            unique, counts = np.unique(line, return_counts=True)
            if len(unique) == 0:
                continue
            dominant_idx = np.argmax(counts)
            dominant_color = int(unique[dominant_idx])
            dominant_count = counts[dominant_idx]
            uniformity = dominant_count / len(line)

            if uniformity >= uniformity_threshold and dominant_color != bg_color:
                candidates.append((r, abs(r - center)))

    elif axis_type == "col":
        center = W // 2
        for c in range(1, W - 1):  # Skip edges
            line = grid[:, c]
            unique, counts = np.unique(line, return_counts=True)
            if len(unique) == 0:
                continue
            dominant_idx = np.argmax(counts)
            dominant_color = int(unique[dominant_idx])
            dominant_count = counts[dominant_idx]
            uniformity = dominant_count / len(line)

            if uniformity >= uniformity_threshold and dominant_color != bg_color:
                candidates.append((c, abs(c - center)))

    if not candidates:
        return None

    # Sort by distance to center (ascending), pick closest
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# =============================================================================
# Axis-Relative Helpers
# =============================================================================

def get_source_pixels(
    grid: np.ndarray,
    bg_color: int,
    axis_row: Optional[int],
    axis_col: Optional[int],
    source_side: str
) -> List[Tuple[int, int, int]]:
    """
    Get list of active (non-background) pixels on the source side.
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

    FlipX: c_target = 2 * axis_col - c
    FlipY: r_target = 2 * axis_row - r
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


# =============================================================================
# S18 Builder
# =============================================================================

def build_S18_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S18 preferences: apply symmetry transform (global or local).

    Global mode: Apply D4 transform to entire input.
    Local mode: Reflect source pixels across axis using axis-relative math.
                Only applies preferences to TARGET pixels.

    Uses weight 100.0 as Symmetry is a Hard Geometric Law.
    """
    mode = schema_params.get("mode", "global")

    if mode == "global":
        _build_global_symmetry(task_context, schema_params, builder)
    elif mode == "local":
        _build_local_symmetry(task_context, schema_params, builder)


def _build_global_symmetry(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """Handle global symmetry mode."""
    example_type = schema_params.get("example_type", "train")
    example_index = schema_params.get("example_index", 0)

    if example_type == "train":
        if example_index >= len(task_context.train_examples):
            return
        ex = task_context.train_examples[example_index]
    else:
        if example_index >= len(task_context.test_examples):
            return
        ex = task_context.test_examples[example_index]

    transform = schema_params.get("transform", "identity")
    input_grid = ex.input_grid

    try:
        transformed = apply_global_transform(input_grid, transform)
    except Exception:
        return

    H_out = ex.output_H if ex.output_H is not None else transformed.shape[0]
    W_out = ex.output_W if ex.output_W is not None else transformed.shape[1]

    if transformed.shape != (H_out, W_out):
        return

    C = task_context.C

    for r in range(H_out):
        for c in range(W_out):
            color = int(transformed[r, c])
            if 0 <= color < C:
                p_idx = r * W_out + c
                builder.prefer_pixel_color(p_idx, color, weight=100.0)


def _build_local_symmetry(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Handle local/axis-relative symmetry mode.

    Uses axis as hinge - for each source pixel, compute target coordinate
    and emit preference for that target pixel.

    Only emits preferences for TARGET pixels (reflections of source).
    Source pixels and axis lines are NOT constrained by S18.

    SEMANTIC AXIS DETECTION: If axis_row_color or axis_col_color params are
    present, resolve axis position from color at runtime (handles varying
    axis positions across examples with same semantic meaning).
    """
    example_type = schema_params.get("example_type", "train")
    example_index = schema_params.get("example_index", 0)

    if example_type == "train":
        if example_index >= len(task_context.train_examples):
            return
        ex = task_context.train_examples[example_index]
    else:
        if example_index >= len(task_context.test_examples):
            return
        ex = task_context.test_examples[example_index]

    # Get parameters
    source = schema_params.get("source", "left")
    transform = schema_params.get("transform", "flip_x")
    split_type = schema_params.get("split_type", "vertical")
    bg_color = schema_params.get("bg_color", 0)

    input_grid = ex.input_grid
    H, W = input_grid.shape

    # =========================================================================
    # AXIS RESOLUTION: Multiple strategies in order of abstraction
    # =========================================================================
    axis_row = schema_params.get("axis_row")
    axis_col = schema_params.get("axis_col")

    # Check for axis_rule (most abstract - dynamic uniform detection)
    axis_rule = schema_params.get("axis_rule")
    axis_type = schema_params.get("axis_type")  # "row" or "col" (for single-axis)

    # STRATEGY 1: Dynamic Uniform (find ANY uniform line at runtime)
    if axis_rule == "dynamic_uniform":
        # For QUADRANT: Find BOTH row and col axes dynamically
        if split_type == "quadrant":
            resolved_row = find_uniform_axis(input_grid, "row", bg_color)
            resolved_col = find_uniform_axis(input_grid, "col", bg_color)
            if resolved_row is not None and resolved_col is not None:
                axis_row = resolved_row
                axis_col = resolved_col
            else:
                # Need BOTH axes for quadrant - can't build constraints
                return
        # For SINGLE-AXIS: Use axis_type to determine which axis to find
        elif axis_type == "col":
            resolved_col = find_uniform_axis(input_grid, "col", bg_color)
            if resolved_col is not None:
                axis_col = resolved_col
            else:
                # No uniform axis found - can't build constraints
                return
        elif axis_type == "row":
            resolved_row = find_uniform_axis(input_grid, "row", bg_color)
            if resolved_row is not None:
                axis_row = resolved_row
            else:
                # No uniform axis found - can't build constraints
                return

    # STRATEGY 2: Semantic by color (axis identified by specific color)
    else:
        axis_row_color = schema_params.get("axis_row_color")
        axis_col_color = schema_params.get("axis_col_color")

        # Resolve axis from color if semantic params are present
        if axis_row_color is not None:
            resolved_row = resolve_axis_from_color(input_grid, "row", axis_row_color)
            if resolved_row is not None:
                axis_row = resolved_row
            else:
                # Axis color not found - can't build constraints
                return

        if axis_col_color is not None:
            resolved_col = resolve_axis_from_color(input_grid, "col", axis_col_color)
            if resolved_col is not None:
                axis_col = resolved_col
            else:
                # Axis color not found - can't build constraints
                return

    # STRATEGY 3: Use center as default if not specified
    if axis_row is None and split_type in ["horizontal", "quadrant"]:
        axis_row = H // 2
    if axis_col is None and split_type in ["vertical", "quadrant"]:
        axis_col = W // 2

    H_out = ex.output_H if ex.output_H is not None else H
    W_out = ex.output_W if ex.output_W is not None else W

    C = task_context.C

    # Handle dynamic source detection for quadrant symmetry
    # When source == "dynamic", auto-detect which quadrant has content
    resolved_source = source
    if source == "dynamic" and split_type == "quadrant":
        # Find quadrant with most non-bg content
        quadrants = ["top_left", "top_right", "bottom_left", "bottom_right"]
        max_content = -1
        for q in quadrants:
            q_pixels = get_source_pixels(input_grid, bg_color, axis_row, axis_col, q)
            if len(q_pixels) > max_content:
                max_content = len(q_pixels)
                resolved_source = q
        if max_content == 0:
            return  # No content in any quadrant

    # Get source pixels
    source_pixels = get_source_pixels(input_grid, bg_color, axis_row, axis_col, resolved_source)

    # For quadrant mode, we need to emit preferences for all three target quadrants
    if split_type == "quadrant" and transform == "quadrant":
        # Source quadrant reflects to 3 targets with different transforms
        transforms_map = {
            "top_left": [("flip_x", ), ("flip_y", ), ("flip_both", )],
            "top_right": [("flip_x", ), ("flip_y", ), ("flip_both", )],
            "bottom_left": [("flip_y", ), ("flip_x", ), ("flip_both", )],
            "bottom_right": [("flip_x", ), ("flip_y", ), ("flip_both", )],
        }

        for xform in ["flip_x", "flip_y", "flip_both"]:
            for r, c, color in source_pixels:
                r_target, c_target = reflect_coordinate(r, c, axis_row, axis_col, xform)

                # Bounds check
                if not (0 <= r_target < H_out and 0 <= c_target < W_out):
                    continue

                p_idx = r_target * W_out + c_target
                if 0 <= color < C:
                    builder.prefer_pixel_color(p_idx, color, weight=100.0)
    else:
        # Simple reflection (vertical or horizontal)
        for r, c, color in source_pixels:
            r_target, c_target = reflect_coordinate(r, c, axis_row, axis_col, transform)

            # Bounds check
            if not (0 <= r_target < H_out and 0 <= c_target < W_out):
                continue

            p_idx = r_target * W_out + c_target
            if 0 <= color < C:
                builder.prefer_pixel_color(p_idx, color, weight=100.0)


if __name__ == "__main__":
    import numpy as np
    from src.schemas.context import build_example_context

    print("=" * 70)
    print("S18 builder self-test (Axis-Relative Reflection)")
    print("=" * 70)

    # Test 1: Global FlipX
    print("\nTest 1: Global FlipX transform")
    print("-" * 70)

    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    output1 = np.array([[3, 2, 1], [6, 5, 4]], dtype=int)

    ex1 = build_example_context(input1, output1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=10)

    params1 = {
        "mode": "global",
        "example_type": "train",
        "example_index": 0,
        "transform": "flip_x"
    }

    builder1 = ConstraintBuilder()
    build_S18_constraints(ctx1, params1, builder1)

    print(f"  Preferences generated: {len(builder1.preferences)}")
    assert len(builder1.preferences) == 6
    print("  OK")

    # Test 2: Asymmetric vertical reflection
    print("\nTest 2: Asymmetric vertical reflection (axis at col 2)")
    print("-" * 70)

    # Axis at col 2, left side has content, right side gets reflection
    input2 = np.array([
        [1, 2, 5, 0, 0, 0],
        [3, 4, 5, 0, 0, 0],
        [5, 6, 5, 0, 0, 0],
    ], dtype=int)

    ex2 = build_example_context(input2, input2)  # Output same shape
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=10)

    params2 = {
        "mode": "local",
        "split_type": "vertical",
        "axis_row": None,
        "axis_col": 2,
        "source": "left",
        "transform": "flip_x",
        "bg_color": 0,
        "example_type": "train",
        "example_index": 0
    }

    builder2 = ConstraintBuilder()
    build_S18_constraints(ctx2, params2, builder2)

    print(f"  Preferences generated: {len(builder2.preferences)}")

    # Source pixels: (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4, (2,0)=5, (2,1)=6
    # Reflected to: (0,4)=1, (0,3)=2, (1,4)=3, (1,3)=4, (2,4)=5, (2,3)=6
    # axis_col=2, so:
    # (0,0) -> c_target = 2*2 - 0 = 4 -> (0,4)
    # (0,1) -> c_target = 2*2 - 1 = 3 -> (0,3)
    assert len(builder2.preferences) == 6, f"Expected 6, got {len(builder2.preferences)}"

    # Check specific targets
    pref_dict = {p_idx: color for p_idx, color, _ in builder2.preferences}
    # Row 0, col 4 -> idx = 0*6 + 4 = 4 should be color 1
    assert pref_dict.get(4) == 1, f"Expected pixel 4 to be 1, got {pref_dict.get(4)}"
    # Row 0, col 3 -> idx = 0*6 + 3 = 3 should be color 2
    assert pref_dict.get(3) == 2, f"Expected pixel 3 to be 2, got {pref_dict.get(3)}"

    print("  Verified axis-relative target coordinates")
    print("  OK")

    # Test 3: Quadrant symmetry
    print("\nTest 3: Quadrant symmetry (top-left -> all)")
    print("-" * 70)

    input3 = np.array([
        [1, 2, 5, 0, 0],
        [3, 4, 5, 0, 0],
        [5, 5, 5, 5, 5],
        [0, 0, 5, 0, 0],
        [0, 0, 5, 0, 0]
    ], dtype=int)

    ex3 = build_example_context(input3, input3)
    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=10)

    params3 = {
        "mode": "local",
        "split_type": "quadrant",
        "axis_row": 2,
        "axis_col": 2,
        "source": "top_left",
        "transform": "quadrant",
        "bg_color": 0,
        "example_type": "train",
        "example_index": 0
    }

    builder3 = ConstraintBuilder()
    build_S18_constraints(ctx3, params3, builder3)

    print(f"  Preferences generated: {len(builder3.preferences)}")
    # Source: 4 pixels (1,2,3,4 at positions (0,0),(0,1),(1,0),(1,1))
    # Each reflects to 3 targets = 12 preferences
    assert len(builder3.preferences) == 12, f"Expected 12, got {len(builder3.preferences)}"
    print("  OK")

    # Test 4: SEMANTIC AXIS (axis_col_color instead of axis_col)
    print("\nTest 4: Semantic axis (resolve axis from color)")
    print("-" * 70)

    # Grid with axis at col 2 (color 5)
    input4 = np.array([
        [1, 2, 5, 0, 0],
        [3, 4, 5, 0, 0],
    ], dtype=int)

    ex4 = build_example_context(input4, input4)
    ctx4 = TaskContext(train_examples=[ex4], test_examples=[], C=10)

    # Use semantic axis_col_color=5 instead of fixed axis_col=2
    params4 = {
        "mode": "local",
        "split_type": "vertical",
        "axis_col_color": 5,  # SEMANTIC: axis identified by color, not coordinate
        "source": "left",
        "transform": "flip_x",
        "bg_color": 0,
        "example_type": "train",
        "example_index": 0
    }

    builder4 = ConstraintBuilder()
    build_S18_constraints(ctx4, params4, builder4)

    print(f"  Preferences generated: {len(builder4.preferences)}")
    # Source pixels: (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4
    # Reflected via axis at col 2: (0,4)=1, (0,3)=2, (1,4)=3, (1,3)=4
    assert len(builder4.preferences) == 4, f"Expected 4, got {len(builder4.preferences)}"

    # Verify correct target coordinates (axis resolved at runtime to col 2)
    pref_dict = {p_idx: color for p_idx, color, _ in builder4.preferences}
    # Row 0, col 4 -> idx = 0*5 + 4 = 4 should be color 1
    assert pref_dict.get(4) == 1, f"Expected pixel 4 to be 1, got {pref_dict.get(4)}"
    # Row 0, col 3 -> idx = 0*5 + 3 = 3 should be color 2
    assert pref_dict.get(3) == 2, f"Expected pixel 3 to be 2, got {pref_dict.get(3)}"

    print("  Verified axis resolved from color 5 -> col 2 at runtime")
    print("  OK")

    # Test 5: DYNAMIC UNIFORM AXIS (axis_rule="dynamic_uniform")
    print("\nTest 5: Dynamic uniform axis (find ANY uniform line at runtime)")
    print("-" * 70)

    # Grid with axis at col 2 (color 7 - any uniform line works)
    input5 = np.array([
        [1, 2, 7, 0, 0],
        [3, 4, 7, 0, 0],
    ], dtype=int)

    ex5 = build_example_context(input5, input5)
    ctx5 = TaskContext(train_examples=[ex5], test_examples=[], C=10)

    # Use dynamic_uniform: finds ANY uniform line, regardless of color
    params5 = {
        "mode": "local",
        "split_type": "vertical",
        "axis_rule": "dynamic_uniform",  # ABSTRACT: find the uniform divider
        "axis_type": "col",
        "source": "left",
        "transform": "flip_x",
        "bg_color": 0,
        "example_type": "train",
        "example_index": 0
    }

    builder5 = ConstraintBuilder()
    build_S18_constraints(ctx5, params5, builder5)

    print(f"  Preferences generated: {len(builder5.preferences)}")
    # Source pixels: (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4
    # Should find uniform axis at col 2 (color 7) and reflect
    assert len(builder5.preferences) == 4, f"Expected 4, got {len(builder5.preferences)}"

    # Verify correct target coordinates (axis found dynamically at col 2)
    pref_dict = {p_idx: color for p_idx, color, _ in builder5.preferences}
    assert pref_dict.get(4) == 1, f"Expected pixel 4 to be 1, got {pref_dict.get(4)}"
    assert pref_dict.get(3) == 2, f"Expected pixel 3 to be 2, got {pref_dict.get(3)}"

    print("  Verified dynamic uniform axis found at col 2 (color 7)")
    print("  OK")

    print("\n" + "=" * 70)
    print("S18 builder self-test passed.")
    print("=" * 70)
