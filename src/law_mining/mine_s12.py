"""
S12 law miner: Discover generalized raycasting patterns.

This miner finds directional ray projection rules:
  - Seed pixels identified by neighborhood hash
  - Cast rays in 8 directions (N, S, E, W, NE, NW, SE, SW)
  - Paint with draw_color until stop condition met
  - Validate: all ray pixels must match ground truth

Mining algorithm:
  1. Enumerate candidates: seed_hash × vector × draw_color × stop_condition
  2. Simulate rays on all training examples
  3. Validate: every ray pixel must have draw_color in GT
  4. Emit SchemaInstance for patterns that pass on ALL training examples
"""

from typing import List, Dict, Any, Tuple, Set
import numpy as np

from src.schemas.context import TaskContext
from src.catalog.types import SchemaInstance


# 8-directional vectors (N, S, E, W, NE, NW, SE, SW)
VECTORS = [
    (0, 1),   # E (East)
    (0, -1),  # W (West)
    (1, 0),   # S (South)
    (-1, 0),  # N (North)
    (1, 1),   # SE (Southeast)
    (1, -1),  # SW (Southwest)
    (-1, 1),  # NE (Northeast)
    (-1, -1)  # NW (Northwest)
]

# Stop conditions to test
STOP_CONDITIONS = ["border", "any_nonzero"] + [f"color_{c}" for c in range(10)]


def should_stop_ray(grid: np.ndarray, r: int, c: int, stop_condition: str) -> bool:
    """
    Determine if ray should stop at position (r, c).

    Args:
        grid: Input grid to check collision
        r: Row position
        c: Column position
        stop_condition: Stop rule

    Returns:
        True if ray should stop, False to continue
    """
    if stop_condition == "border":
        return False  # Never stop mid-grid (boundary checked in caller)

    input_color = int(grid[r, c])

    if stop_condition == "any_nonzero":
        return input_color != 0

    if stop_condition.startswith("color_"):
        try:
            target_color = int(stop_condition.split("_")[1])
            return input_color == target_color
        except (ValueError, IndexError):
            return False

    return False


def simulate_ray(
    seed_r: int,
    seed_c: int,
    vector: Tuple[int, int],
    input_grid: np.ndarray,
    stop_condition: str,
    include_seed: bool
) -> List[Tuple[int, int]]:
    """
    Simulate a ray from seed pixel in specified direction.

    Args:
        seed_r: Seed row
        seed_c: Seed column
        vector: (dr, dc) direction
        input_grid: Input grid for collision detection
        stop_condition: When to stop ray
        include_seed: Whether to include seed pixel in ray

    Returns:
        List of (r, c) pixel coordinates along ray path
    """
    H, W = input_grid.shape
    dr, dc = vector
    ray_pixels = []

    if include_seed:
        ray_pixels.append((seed_r, seed_c))

    curr_r = seed_r + dr
    curr_c = seed_c + dc

    while 0 <= curr_r < H and 0 <= curr_c < W:
        if should_stop_ray(input_grid, curr_r, curr_c, stop_condition):
            break

        ray_pixels.append((curr_r, curr_c))

        curr_r += dr
        curr_c += dc

    return ray_pixels


def validate_ray_pattern(
    task_context: TaskContext,
    seed_hash: int,
    vector: Tuple[int, int],
    draw_color: int,
    stop_condition: str,
    include_seed: bool
) -> bool:
    """
    Validate that a ray pattern holds across ALL training examples.

    Args:
        task_context: TaskContext with training examples
        seed_hash: Neighborhood hash identifying seed pixels
        vector: (dr, dc) direction
        draw_color: Color to paint along ray
        stop_condition: When to stop ray
        include_seed: Whether to paint seed pixel

    Returns:
        True if pattern is valid on all training examples, False otherwise
    """
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue  # Skip examples without output

        input_grid = ex.input_grid
        output_grid = ex.output_grid
        nbh = ex.neighborhood_hashes
        H_out, W_out = output_grid.shape

        # Find seed pixels matching this hash
        seed_pixels = [(r, c) for (r, c), h_val in nbh.items() if h_val == seed_hash]

        if not seed_pixels:
            continue  # No seeds in this example (pattern is vacuously valid)

        # Simulate rays from each seed
        for seed_r, seed_c in seed_pixels:
            ray_pixels = simulate_ray(
                seed_r, seed_c, vector, input_grid, stop_condition, include_seed
            )

            # Validate: every ray pixel must have draw_color in ground truth
            for r, c in ray_pixels:
                if not (0 <= r < H_out and 0 <= c < W_out):
                    return False  # Ray extends outside output grid

                gt_color = int(output_grid[r, c])
                if gt_color != draw_color:
                    return False  # Ray pixel doesn't match ground truth

    return True  # Pattern is valid on all training examples


def mine_S12(
    task_context: TaskContext,
    roles: Dict[Any, int],
    role_stats: Dict[int, Dict[str, Any]]
) -> List[SchemaInstance]:
    """
    Mine S12 ray patterns from training examples.

    Algorithm (PHYSICS-FIRST):
      1. Collect all unique neighborhood hashes from training examples
      2. For each physics combination (vector, draw_color, stop_condition, include_seed):
         - Find ALL hashes that are valid for this physics
         - Simulate rays on all training examples
         - Validate: all ray pixels must match ground truth
         - If valid, emit ONE SchemaInstance per example with list of ALL valid hashes

    This groups seeds by physics behavior, reducing instances from ~760k to ~240.

    Args:
        task_context: TaskContext with training examples
        roles: Role mapping (unused but kept for API consistency)
        role_stats: Role statistics (unused but kept for API consistency)

    Returns:
        List of SchemaInstance objects for valid ray patterns
    """
    instances: List[SchemaInstance] = []

    # 1. Collect all unique neighborhood hashes from training examples
    all_hashes: Set[int] = set()
    for ex in task_context.train_examples:
        all_hashes.update(ex.neighborhood_hashes.values())

    if not all_hashes:
        return instances  # No hashes to mine

    # 2. Enumerate candidate combinations (PHYSICS-FIRST)
    # Loop over physics space: vector × color × stop_condition × include_seed
    for vector in VECTORS:
        for draw_color in range(task_context.C):
            for stop_condition in STOP_CONDITIONS:
                for include_seed in [False, True]:
                    # Find ALL hashes that are valid for this physics
                    valid_hashes: List[int] = []

                    for seed_hash in all_hashes:
                        # Validate pattern on all training examples
                        if validate_ray_pattern(
                            task_context,
                            seed_hash,
                            vector,
                            draw_color,
                            stop_condition,
                            include_seed
                        ):
                            valid_hashes.append(seed_hash)

                    # If we found valid hashes, emit ONE instance per example
                    if valid_hashes:
                        # Emit instances for TRAIN examples
                        for train_idx in range(len(task_context.train_examples)):
                            params = {
                                "example_type": "train",
                                "example_index": train_idx,
                                "rays": [{
                                    "seed_hashes": valid_hashes,  # LIST of all valid hashes
                                    "vector": str(vector),  # Convert to string for JSON compatibility
                                    "draw_color": draw_color,
                                    "stop_condition": stop_condition,
                                    "include_seed": include_seed
                                }]
                            }

                            instances.append(SchemaInstance(
                                family_id="S12",
                                params=params
                            ))

                        # Emit instances for TEST examples
                        for test_idx in range(len(task_context.test_examples)):
                            params = {
                                "example_type": "test",
                                "example_index": test_idx,
                                "rays": [{
                                    "seed_hashes": valid_hashes,  # LIST of all valid hashes
                                    "vector": str(vector),  # Convert to string for JSON compatibility
                                    "draw_color": draw_color,
                                    "stop_condition": stop_condition,
                                    "include_seed": include_seed
                                }]
                            }

                            instances.append(SchemaInstance(
                                family_id="S12",
                                params=params
                            ))

    return instances


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S12 miner with toy example...")
    print("=" * 70)

    # Create a 5x5 grid with diagonal ray pattern
    # Seed at (1,1), ray goes SE (1,1) with color 6
    input_grid = np.zeros((5, 5), dtype=int)
    input_grid[1, 1] = 1  # Seed pixel

    output_grid = np.zeros((5, 5), dtype=int)
    output_grid[2, 2] = 6  # Ray pixel 1
    output_grid[3, 3] = 6  # Ray pixel 2
    output_grid[4, 4] = 6  # Ray pixel 3

    ex_train = build_example_context(input_grid, output_grid)
    ex_train.neighborhood_hashes = {(1, 1): 12345}  # Manually set hash for seed

    # Create test example (same pattern, miner should apply to it)
    ex_test = build_example_context(input_grid, np.zeros((5, 5), dtype=int))
    ex_test.neighborhood_hashes = {(1, 1): 12345}  # Same seed hash

    ctx = TaskContext(train_examples=[ex_train], test_examples=[ex_test], C=10)

    print("Test 1: Should find diagonal SE ray")
    print("-" * 70)

    instances = mine_S12(ctx, {}, {})

    # Should find at least one instance with SE vector (1,1)
    found_se_ray = False
    for inst in instances:
        rays = inst.params.get("rays", [])
        if rays:
            ray = rays[0]
            if ray["vector"] == "(1, 1)" and ray["draw_color"] == 6:
                found_se_ray = True
                print(f"✓ Found SE ray: {ray}")
                break

    assert found_se_ray, "Expected to find SE ray pattern"

    print(f"\nTotal instances mined: {len(instances)}")
    print("=" * 70)
    print("✓ S12 miner self-test passed.")
