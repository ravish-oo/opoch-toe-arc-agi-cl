"""
S13 law miner: Discover gravity / object movement patterns.

This miner finds physics-based object movement rules:
  - Objects (connected components) move in a gravity direction
  - Movement continues until hitting boundary or obstacle
  - Empirically detect which colors are mobile vs stationary

Mining algorithm:
  1. Try 4 gravity vectors: Down, Up, Right, Left
  2. For each vector, empirically detect mobile colors
  3. Simulate gravity physics on all training examples
  4. Validate: simulated output must match ground truth exactly
  5. Emit SchemaInstance for patterns that pass on ALL training examples
"""

from typing import List, Dict, Any, Tuple, Set
import numpy as np

from src.schemas.context import TaskContext
from src.catalog.types import SchemaInstance
from src.features.components import connected_components_by_color, Component


# 4 gravity vectors: Down, Up, Right, Left
GRAVITY_VECTORS = [
    (1, 0),   # Down
    (-1, 0),  # Up
    (0, 1),   # Right
    (0, -1)   # Left
]


def compute_centroid(component: Component) -> Tuple[float, float]:
    """
    Compute the centroid (center of mass) of a component.

    Args:
        component: Component with pixels list

    Returns:
        Tuple of (row_avg, col_avg)
    """
    if not component.pixels:
        return (0.0, 0.0)

    r_sum = sum(r for r, c in component.pixels)
    c_sum = sum(c for r, c in component.pixels)
    count = len(component.pixels)

    return (r_sum / count, c_sum / count)


def detect_mobile_colors(
    task_context: TaskContext,
    vector: Tuple[int, int],
    threshold: float = 0.0  # Relaxed: ANY movement counts
) -> Set[int]:
    """
    Empirically detect which colors move in the given vector direction.

    CALIBRATED: A color is mobile if ANY of its components move in the
    vector direction. This handles "blocked" scenarios like Tetris stacks
    where only the top block moves while others are stationary.

    The simulator will handle collision detection - if a component can't
    move because it's blocked, the simulator figures that out.

    Args:
        task_context: TaskContext with training examples
        vector: (dr, dc) gravity direction
        threshold: Minimum consistency ratio (default 0.0 = any movement)

    Returns:
        Set of mobile color values
    """
    mobile_colors: Set[int] = set()
    dr, dc = vector

    # Get all colors present in inputs (pooled across all examples)
    all_colors: Set[int] = set()
    for ex in task_context.train_examples:
        all_colors.update(set(ex.input_grid.flatten()) - {0})

    # For each color, compute pooled consistency
    for color in all_colors:
        total_components = 0
        moved_components = 0

        # Pool statistics across ALL training examples
        for ex in task_context.train_examples:
            if ex.output_grid is None:
                continue

            input_grid = ex.input_grid
            output_grid = ex.output_grid

            # Get components of this color in input and output
            components_in = [comp for comp in connected_components_by_color(input_grid)
                           if comp.color == color]
            components_out = [comp for comp in connected_components_by_color(output_grid)
                            if comp.color == color]

            total_components += len(components_in)

            # Check which components moved in vector direction
            for comp_in in components_in:
                centroid_in = compute_centroid(comp_in)

                # Try to find matching component in output
                moved = False
                for comp_out in components_out:
                    centroid_out = compute_centroid(comp_out)

                    # Calculate shift
                    shift_r = centroid_out[0] - centroid_in[0]
                    shift_c = centroid_out[1] - centroid_in[1]

                    # Check if shift aligns with vector (shift == vector * k, k > 0)
                    if dr != 0:
                        # Primary direction is vertical
                        k = shift_r / dr
                        if k > 0.5:  # Moved at least half a pixel in gravity direction
                            # Check if horizontal shift is consistent
                            expected_c_shift = dc * k
                            if abs(shift_c - expected_c_shift) < 0.5:
                                moved = True
                                break
                    elif dc != 0:
                        # Primary direction is horizontal
                        k = shift_c / dc
                        if k > 0.5:  # Moved at least half a pixel in gravity direction
                            # Check if vertical shift is consistent
                            expected_r_shift = dr * k
                            if abs(shift_r - expected_r_shift) < 0.5:
                                moved = True
                                break

                if moved:
                    moved_components += 1

        # CALIBRATED: Accept if ANY component moves in the vector direction
        # The simulator handles collision detection for blocked components
        if moved_components > 0:
            mobile_colors.add(color)

    return mobile_colors


def simulate_gravity(
    grid: np.ndarray,
    vector: Tuple[int, int],
    mobile_colors: Set[int]
) -> np.ndarray:
    """
    Simulate gravity: mobile colors fall/move, stationary colors act as walls.

    Args:
        grid: Input grid
        vector: (dr, dc) gravity direction
        mobile_colors: Set of colors that can move

    Returns:
        Simulated output grid after gravity settles
    """
    result = grid.copy()
    dr, dc = vector
    H, W = grid.shape

    if not mobile_colors:
        return result  # No mobile colors, nothing moves

    changed = True
    iterations = 0
    max_iterations = max(H, W) * 2  # Safety limit

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Scan in OPPOSITE direction to gravity
        # (e.g., bottom-up for downward gravity to avoid moving same pixel twice)
        if dr > 0:
            rows = range(H - 1, -1, -1)  # Bottom to top
        elif dr < 0:
            rows = range(H)  # Top to bottom
        else:
            rows = range(H)  # Any order for horizontal

        if dc > 0:
            cols = range(W - 1, -1, -1)  # Right to left
        elif dc < 0:
            cols = range(W)  # Left to right
        else:
            cols = range(W)  # Any order for vertical

        for r in rows:
            for c in cols:
                color = int(result[r, c])

                # Only mobile colors can move
                if color not in mobile_colors:
                    continue

                # Try to move in gravity direction
                new_r = r + dr
                new_c = c + dc

                # Bounds check
                if not (0 <= new_r < H and 0 <= new_c < W):
                    continue  # Hit boundary, can't move

                # Collision check: can only move into empty space (0)
                dest_color = int(result[new_r, new_c])
                if dest_color == 0:
                    # Move the pixel
                    result[new_r, new_c] = color
                    result[r, c] = 0
                    changed = True
                # Else: blocked by wall (stationary color) or another pixel

    return result


def mine_S13(
    task_context: TaskContext,
    roles: Dict[Any, int],
    role_stats: Dict[int, Dict[str, Any]]
) -> List[SchemaInstance]:
    """
    Mine S13 gravity patterns from training examples.

    Algorithm:
      1. Check geometry-preserving constraint
      2. Try 4 gravity vectors (Down, Up, Right, Left)
      3. For each vector, empirically detect mobile colors
      4. Simulate gravity on all training examples
      5. Validate: simulated output == ground truth
      6. Emit SchemaInstance for valid gravity laws

    Args:
        task_context: TaskContext with training examples
        roles: Role mapping (unused but kept for API consistency)
        role_stats: Role statistics (unused but kept for API consistency)

    Returns:
        List of SchemaInstance objects for valid gravity patterns
    """
    instances: List[SchemaInstance] = []

    # Step 0: S13 only applies to geometry-preserving tasks
    # If any training example has input.shape != output.shape, S13 is not applicable
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            # S13 requires same pixel positions in input and output
            # For non-geometry-preserving tasks (crop, summary, etc.), return empty
            return instances

    # Try each gravity vector
    for vector in GRAVITY_VECTORS:
        # Empirically detect mobile colors for this vector
        mobile_colors = detect_mobile_colors(task_context, vector)

        if not mobile_colors:
            continue  # No mobile colors for this vector

        # Validate: simulate on all training examples
        valid = True
        for ex in task_context.train_examples:
            if ex.output_grid is None:
                continue

            simulated = simulate_gravity(ex.input_grid, vector, mobile_colors)

            if not np.array_equal(simulated, ex.output_grid):
                valid = False
                break

        # If valid on all training examples, emit SchemaInstance
        if valid:
            instances.append(SchemaInstance(
                family_id="S13",
                params={
                    "gravity_vector": str(vector),
                    "mobile_colors": sorted(list(mobile_colors))
                }
            ))

    return instances


if __name__ == "__main__":
    # Self-test with toy gravity example
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S13 miner with toy gravity example...")
    print("=" * 70)

    # Create a simple falling object scenario
    # Input: object at top
    # Output: object at bottom
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],  # Floor (stationary)
    ], dtype=int)

    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [1, 1, 1, 1, 1],  # Floor stays
    ], dtype=int)

    ex_train = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex_train], test_examples=[], C=3)

    print("Test 1: Object falling downward")
    print("-" * 70)
    print("Input:")
    print(input_grid)
    print("\nExpected output:")
    print(output_grid)

    instances = mine_S13(ctx, {}, {})

    print(f"\nMined instances: {len(instances)}")

    # Should find downward gravity
    found_down = False
    for inst in instances:
        if inst.family_id == "S13":
            vec = inst.params.get("gravity_vector")
            mobile = inst.params.get("mobile_colors")
            print(f"  Found: vector={vec}, mobile_colors={mobile}")
            if vec == "(1, 0)" and 2 in mobile and 1 not in mobile:
                found_down = True

    assert found_down, "Expected to find downward gravity with mobile=2, stationary=1"
    print("\n✓ Correctly detected: Down gravity, color 2 mobile, color 1 stationary")

    print("\n" + "=" * 70)
    print("✓ S13 miner self-test passed.")
