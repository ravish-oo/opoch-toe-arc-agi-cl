"""
S14 law miner: Discover topology/flood fill patterns.

This miner finds topological region-filling rules:
  - Fill holes inside enclosed boundaries
  - Fill background regions connected to edges
  - Uses scipy.ndimage for robust topological operations

Mining algorithm:
  1. Identify unexplained pixels: diff = (input != output)
  2. Use scipy.ndimage.label to find connected diff regions
  3. Hypothesize rules:
     - fill_enclosed: holes inside a boundary color get filled
     - fill_background: regions connected to (0,0) get filled
  4. Validate on ALL training examples
  5. Emit SchemaInstance for valid patterns
"""

from typing import List, Dict, Any, Set, Tuple
import numpy as np

try:
    from scipy.ndimage import label, binary_fill_holes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.schemas.context import TaskContext
from src.catalog.types import SchemaInstance


def find_holes_in_boundary(grid: np.ndarray, boundary_color: int) -> np.ndarray:
    """
    Find holes (enclosed regions) inside a boundary of a specific color.

    Uses scipy.ndimage.binary_fill_holes for robust topological detection.

    Args:
        grid: Input grid
        boundary_color: Color that forms the boundary

    Returns:
        Boolean mask where True = hole pixel (inside but not boundary)
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(grid.shape, dtype=bool)

    # Create binary mask of boundary
    boundary_mask = (grid == boundary_color)

    # Fill holes in the boundary mask
    filled_mask = binary_fill_holes(boundary_mask)

    # Holes are pixels that are filled but not part of boundary
    holes_mask = filled_mask & ~boundary_mask

    return holes_mask


def find_background_region(grid: np.ndarray) -> np.ndarray:
    """
    Find the background region connected to (0,0).

    Uses scipy.ndimage.label for connected component analysis.

    Args:
        grid: Input grid

    Returns:
        Boolean mask where True = background pixel (connected to corner)
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(grid.shape, dtype=bool)

    # Get the color at (0,0) - this is typically background
    background_color = int(grid[0, 0])

    # Create mask of pixels with same color as background
    same_color_mask = (grid == background_color)

    # Label connected components (4-connectivity)
    labeled, num_features = label(same_color_mask)

    # The background component is the one containing (0,0)
    background_label = labeled[0, 0]

    if background_label == 0:
        # (0,0) is not part of any component (shouldn't happen)
        return np.zeros(grid.shape, dtype=bool)

    background_mask = (labeled == background_label)

    return background_mask


def validate_fill_enclosed(
    task_context: TaskContext,
    boundary_color: int,
    fill_color: int
) -> bool:
    """
    Validate that fill_enclosed(boundary_color, fill_color) holds on all training examples.

    The rule: All holes inside the boundary become fill_color in output.

    Args:
        task_context: TaskContext with training examples
        boundary_color: Color that forms the boundary
        fill_color: Color that fills the holes

    Returns:
        True if rule is valid on all training examples
    """
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return False  # S14 requires geometry-preserving

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Find holes in input
        holes_mask = find_holes_in_boundary(input_grid, boundary_color)

        if not np.any(holes_mask):
            continue  # No holes in this example, vacuously valid

        # Check: all hole pixels in output must have fill_color
        hole_output_colors = output_grid[holes_mask]
        if not np.all(hole_output_colors == fill_color):
            return False

    return True


def validate_fill_background(
    task_context: TaskContext,
    fill_color: int
) -> bool:
    """
    Validate that fill_background(fill_color) holds on all training examples.

    The rule: All background pixels (connected to 0,0) become fill_color in output.

    Args:
        task_context: TaskContext with training examples
        fill_color: Color that fills the background

    Returns:
        True if rule is valid on all training examples
    """
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return False  # S14 requires geometry-preserving

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Find background in input
        background_mask = find_background_region(input_grid)

        if not np.any(background_mask):
            continue  # No background, vacuously valid

        # Check: all background pixels in output must have fill_color
        bg_output_colors = output_grid[background_mask]
        if not np.all(bg_output_colors == fill_color):
            return False

    return True


def count_affected_pixels_enclosed(
    task_context: TaskContext,
    boundary_color: int,
    fill_color: int
) -> int:
    """
    Count how many pixels are actually changed by fill_enclosed.

    Args:
        task_context: TaskContext with training examples
        boundary_color: Color that forms the boundary
        fill_color: Color that fills the holes

    Returns:
        Total count of pixels changed across all training examples
    """
    total = 0
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        holes_mask = find_holes_in_boundary(input_grid, boundary_color)

        # Count pixels where hole exists AND input color != fill_color
        input_colors = input_grid[holes_mask]
        total += np.sum(input_colors != fill_color)

    return total


def count_affected_pixels_background(
    task_context: TaskContext,
    fill_color: int
) -> int:
    """
    Count how many pixels are actually changed by fill_background.

    Args:
        task_context: TaskContext with training examples
        fill_color: Color that fills the background

    Returns:
        Total count of pixels changed across all training examples
    """
    total = 0
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        background_mask = find_background_region(input_grid)

        # Count pixels where background exists AND input color != fill_color
        input_colors = input_grid[background_mask]
        total += np.sum(input_colors != fill_color)

    return total


def mine_S14(
    task_context: TaskContext,
    roles: Dict[Any, int],
    role_stats: Dict[int, Dict[str, Any]]
) -> List[SchemaInstance]:
    """
    Mine S14 topology patterns from training examples.

    Algorithm:
      1. Check geometry-preserving constraint
      2. Collect all colors in palette
      3. For each boundary_color, for each fill_color:
         - Test fill_enclosed hypothesis
         - If valid and has kinetic utility >= 1, emit instance
      4. For each fill_color:
         - Test fill_background hypothesis
         - If valid and has kinetic utility >= 1, emit instance

    Args:
        task_context: TaskContext with training examples
        roles: Role mapping (unused but kept for API consistency)
        role_stats: Role statistics (unused but kept for API consistency)

    Returns:
        List of SchemaInstance objects for valid topology patterns
    """
    instances: List[SchemaInstance] = []

    if not SCIPY_AVAILABLE:
        return instances  # Cannot mine without scipy

    # Step 0: S14 only applies to geometry-preserving tasks
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return instances  # Not geometry-preserving

    # Collect all colors present in inputs and outputs
    all_colors: Set[int] = set()
    for ex in task_context.train_examples:
        all_colors.update(set(ex.input_grid.flatten()))
        if ex.output_grid is not None:
            all_colors.update(set(ex.output_grid.flatten()))

    # Remove background (0) from boundary candidates but keep as fill candidate
    boundary_colors = all_colors - {0}
    fill_colors = all_colors

    # Mine fill_enclosed patterns
    for boundary_color in boundary_colors:
        for fill_color in fill_colors:
            if boundary_color == fill_color:
                continue  # Boundary and fill must be different

            if validate_fill_enclosed(task_context, boundary_color, fill_color):
                affected = count_affected_pixels_enclosed(task_context, boundary_color, fill_color)
                if affected >= 1:
                    instances.append(SchemaInstance(
                        family_id="S14",
                        params={
                            "operation": "fill_enclosed",
                            "boundary_color": boundary_color,
                            "fill_color": fill_color
                        }
                    ))

    # Mine fill_background patterns
    for fill_color in fill_colors:
        if validate_fill_background(task_context, fill_color):
            affected = count_affected_pixels_background(task_context, fill_color)
            if affected >= 1:
                instances.append(SchemaInstance(
                    family_id="S14",
                    params={
                        "operation": "fill_background",
                        "fill_color": fill_color
                    }
                ))

    return instances


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S14 miner with toy example...")
    print("=" * 70)

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy not available, skipping test")
        exit(0)

    # Test 1: Fill enclosed hole
    # Input: Red (1) boundary with hole, Output: Hole filled with Blue (2)
    input_grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],  # Hole at (2,2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    output_grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],  # Hole filled with Blue (2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    ex1 = build_example_context(input_grid1, output_grid1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=3)

    print("Test 1: Fill enclosed hole")
    print("-" * 70)
    print("Input:")
    print(input_grid1)
    print("\nOutput:")
    print(output_grid1)

    instances1 = mine_S14(ctx1, {}, {})
    print(f"\nMined instances: {len(instances1)}")

    found_enclosed = False
    for inst in instances1:
        print(f"  {inst.params}")
        if inst.params.get("operation") == "fill_enclosed":
            if inst.params.get("boundary_color") == 1 and inst.params.get("fill_color") == 2:
                found_enclosed = True

    assert found_enclosed, "Expected to find fill_enclosed(boundary=1, fill=2)"
    print("\n  Found fill_enclosed(boundary=1, fill=2)")

    # Test 2: Fill background
    # Input: objects on background, Output: background filled
    input_grid2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    output_grid2 = np.array([
        [3, 3, 3, 3, 3],
        [3, 1, 1, 3, 3],
        [3, 1, 1, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
    ], dtype=int)

    ex2 = build_example_context(input_grid2, output_grid2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=4)

    print("\nTest 2: Fill background")
    print("-" * 70)
    print("Input:")
    print(input_grid2)
    print("\nOutput:")
    print(output_grid2)

    instances2 = mine_S14(ctx2, {}, {})
    print(f"\nMined instances: {len(instances2)}")

    found_background = False
    for inst in instances2:
        print(f"  {inst.params}")
        if inst.params.get("operation") == "fill_background":
            if inst.params.get("fill_color") == 3:
                found_background = True

    assert found_background, "Expected to find fill_background(fill=3)"
    print("\n  Found fill_background(fill=3)")

    print("\n" + "=" * 70)
    print("S14 miner self-test passed.")
