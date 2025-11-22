"""
S16 schema builder: Interaction / Chemistry (Lattice Engine).

This implements the S16 schema for local neighbor-based state transformations:
    "If pixel of color A is adjacent to color B, A becomes C"

S16 uses a Lattice Engine with Double Buffering to compute the Fixed Point:
- All pixels are scanned in parallel (read from current, write to next)
- Reactions continue until no more changes occur (convergence)
- Result is path-independent (same rules -> same Fixed Point)

S16 is geometry-preserving: output has same shape as input.
"""

from typing import Dict, Any
import numpy as np

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


# 4-connected neighbors (Up, Down, Left, Right)
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Maximum iterations for convergence (safety limit)
MAX_STEPS = 10


def parse_reaction_table(reaction_table: Dict[str, int]) -> Dict[tuple, int]:
    """
    Parse string-keyed reaction table to tuple-keyed.

    Args:
        reaction_table: {"(c_self,c_neigh)": c_out, ...}

    Returns:
        {(c_self, c_neigh): c_out, ...}
    """
    reactions = {}
    for key_str, c_out in reaction_table.items():
        # Parse "(c_self,c_neigh)" -> (int, int)
        key_str = key_str.strip()
        if key_str.startswith("(") and key_str.endswith(")"):
            inner = key_str[1:-1]
            parts = inner.split(",")
            if len(parts) == 2:
                c_self = int(parts[0].strip())
                c_neigh = int(parts[1].strip())
                reactions[(c_self, c_neigh)] = c_out
    return reactions


def simulate_to_fixed_point(
    input_grid: np.ndarray,
    reactions: Dict[tuple, int]
) -> np.ndarray:
    """
    Simulate chemical reactions to Fixed Point using Double Buffering.

    Double Buffering ensures path-independence:
    - Read from current_grid, write to next_grid
    - All pixels see the same input state each step
    - Swap buffers after each step

    Args:
        input_grid: Starting grid state
        reactions: {(c_self, c_neigh): c_out} reaction rules

    Returns:
        Fixed Point grid (no more reactions possible)
    """
    H, W = input_grid.shape
    current_grid = input_grid.copy()

    for step in range(MAX_STEPS):
        next_grid = current_grid.copy()  # Double buffer!
        changes = 0

        # Parallel scan: read from current, write to next
        for r in range(H):
            for c in range(W):
                c_self = int(current_grid[r, c])

                # Check neighbors for reaction trigger (priority: Up, Down, Left, Right)
                for dr, dc in NEIGHBORS_4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        c_neigh = int(current_grid[nr, nc])
                        key = (c_self, c_neigh)
                        if key in reactions:
                            c_new = reactions[key]
                            if c_new != c_self:
                                next_grid[r, c] = c_new
                                changes += 1
                                break  # One reaction per pixel per step

        current_grid = next_grid

        if changes == 0:
            break  # Fixed Point reached

    return current_grid


def build_S16_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Build S16 constraints: Lattice Engine for chemical reactions.

    Simulates reactions to Fixed Point, then constrains output pixels
    to match the computed Fixed Point.

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "reaction_table": {"(c_self,c_neigh)": c_out, ...}
        }

    Args:
        task_context: TaskContext with all grids and features
        schema_params: Parameters specifying reaction rules
        builder: ConstraintBuilder to add preferences to

    Example:
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "reaction_table": {"(0,1)": 1}  # White next to Red becomes Red
        ... }
        >>> build_S16_constraints(ctx, params, builder)
    """
    # 1. Parse reaction table
    reaction_table = schema_params.get("reaction_table", {})
    if not reaction_table:
        return

    reactions = parse_reaction_table(reaction_table)
    if not reactions:
        return

    # 2. Select example
    example_type = schema_params.get("example_type", "train")
    example_index = schema_params.get("example_index", 0)

    if example_type == "train":
        if example_index >= len(task_context.train_examples):
            return
        ex = task_context.train_examples[example_index]
    else:  # "test"
        if example_index >= len(task_context.test_examples):
            return
        ex = task_context.test_examples[example_index]

    # 3. Get grid dimensions
    input_grid = ex.input_grid
    H, W = input_grid.shape

    # Handle test examples (future geometry)
    output_H = ex.output_H
    output_W = ex.output_W
    if output_H is None or output_W is None:
        # S16 is geometry-preserving
        output_H, output_W = H, W

    # S16 requires geometry preservation
    if (output_H, output_W) != (H, W):
        return

    # 4. Simulate to Fixed Point
    fixed_point = simulate_to_fixed_point(input_grid, reactions)

    # 5. Apply constraints for changed pixels
    # Weight = 50.0 (Tier 2 - Object/Interaction)
    for r in range(H):
        for c in range(W):
            target = int(fixed_point[r, c])
            original = int(input_grid[r, c])
            if target != original:
                p_idx = r * W + c
                builder.prefer_pixel_color(p_idx, target, weight=50.0)


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("=" * 70)
    print("S16 Builder (Lattice Engine) self-test")
    print("=" * 70)

    # Test 1: Simple reaction - Blue(1) next to Red(2) becomes Green(3)
    print("\nTest 1: Blue(1) next to Red(2) -> Green(3)")
    print("-" * 70)

    in1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],  # Blue(1) next to Red(2)
        [0, 0, 0, 0, 0],
    ], dtype=int)

    out1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 2, 0, 0],  # Blue became Green
        [0, 0, 0, 0, 0],
    ], dtype=int)

    ex1 = build_example_context(in1, out1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=4)

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "reaction_table": {"(1,2)": 3}  # Blue next to Red -> Green
    }

    builder1 = ConstraintBuilder()
    build_S16_constraints(ctx1, params1, builder1)

    print(f"  Preferences: {len(builder1.preferences)}")
    # Should have 1 preference for pixel (1,1) -> color 3
    assert len(builder1.preferences) == 1
    pref = builder1.preferences[0]
    expected_idx = 1 * 5 + 1  # row 1, col 1, width 5
    assert pref[0] == expected_idx
    assert pref[1] == 3
    assert pref[2] == 50.0
    print(f"  Preference: pixel={pref[0]}, color={pref[1]}, weight={pref[2]}")
    print("  Correctly constrained Blue pixel to Green")

    # Test 2: Spreading pattern (multi-step convergence)
    print("\nTest 2: Spreading pattern (Red infects White)")
    print("-" * 70)

    in2 = np.array([
        [0, 0, 0],
        [0, 1, 0],  # Red in center
        [0, 0, 0],
    ], dtype=int)

    out2 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=int)

    ex2 = build_example_context(in2, out2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=2)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "reaction_table": {"(0,1)": 1}  # White next to Red -> Red
    }

    builder2 = ConstraintBuilder()
    build_S16_constraints(ctx2, params2, builder2)

    print(f"  Preferences: {len(builder2.preferences)}")
    # The Lattice Engine converges to Fixed Point:
    # Step 1: 4 adjacent whites become red
    # Step 2: 4 corner whites (now adjacent to red) become red
    # Total: 8 pixels changed (all whites became red)
    assert len(builder2.preferences) == 8
    print(f"  Correctly constrained all 8 white pixels to Red (chain reaction)")

    # Test 3: Fixed Point convergence (chain reaction)
    print("\nTest 3: Chain reaction (multi-step)")
    print("-" * 70)

    # Simulate: White(0) next to Red(1) -> Red(1)
    # Initial: [0, 1, 0, 0]
    # Step 1:  [1, 1, 1, 0]  (first and third 0s see 1)
    # Step 2:  [1, 1, 1, 1]  (fourth 0 now sees 1)

    reactions = {(0, 1): 1}
    in3 = np.array([[0, 1, 0, 0]], dtype=int)

    fixed = simulate_to_fixed_point(in3, reactions)
    print(f"  Input:  {in3}")
    print(f"  Fixed:  {fixed}")
    assert np.array_equal(fixed, np.array([[1, 1, 1, 1]]))
    print("  Chain reaction converged correctly")

    # Test 4: No reactions (empty table)
    print("\nTest 4: Empty reaction table")
    print("-" * 70)

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "reaction_table": {}
    }

    builder4 = ConstraintBuilder()
    build_S16_constraints(ctx1, params4, builder4)

    print(f"  Preferences: {len(builder4.preferences)}")
    assert len(builder4.preferences) == 0
    print("  Correctly returned no preferences for empty table")

    print("\n" + "=" * 70)
    print("S16 Builder (Lattice Engine) self-test passed.")
    print("=" * 70)
