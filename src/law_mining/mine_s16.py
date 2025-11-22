"""
S16 Miner: Interaction / Chemistry.

Mines local neighbor-based state transformations:
    "If pixel of color A is adjacent to color B, A becomes C"

This captures "chemical reactions" like:
- Fire spreading (red spreads to adjacent yellows)
- Infection (blue transforms adjacent whites)
- Color reactions (blue next to red becomes green)

The Lattice Law: Reactions must be consistent across all instances.
If (Blue, Red) -> Green in one place, it must be Green everywhere.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any

from src.schemas.context import TaskContext
from src.catalog.types import SchemaInstance


# 4-connected neighbors (Up, Down, Left, Right)
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def mine_S16(
    task_context: TaskContext,
    roles: Dict[Any, int],
    role_stats: Dict[int, Dict[str, Any]]
) -> List[SchemaInstance]:
    """
    Mine S16: Interaction/Chemistry reactions.

    Algorithm:
    1. Identify Interfaces: For each pixel, look at 4-neighbors
    2. Mine Reactions: (Self_Color, Neighbor_Color) -> Output_Color
    3. Consistency Check: Same trigger must always produce same output
    4. Emit Schema with reaction_table

    Args:
        task_context: TaskContext with train examples
        roles: Role assignments (unused, for signature compatibility)
        role_stats: Role statistics (unused, for signature compatibility)

    Returns:
        List of SchemaInstance with S16 reaction tables
    """
    instances: List[SchemaInstance] = []

    # Collect all observed reactions across training examples
    # reaction_observations: {(c_self, c_neigh): {c_out: count}}
    reaction_observations: Dict[tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # Track total occurrences of each (c_self, c_neigh) pair where c_self changed
    # This helps us verify consistency

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        # S16 requires geometry preservation (pixel-wise operation)
        if ex.input_grid.shape != ex.output_grid.shape:
            continue

        H, W = ex.input_grid.shape

        for r in range(H):
            for c in range(W):
                c_self = int(ex.input_grid[r, c])
                c_out = int(ex.output_grid[r, c])

                # Only track ACTIVE reactions (where color changed)
                if c_self == c_out:
                    continue

                # Check 4-neighbors for potential triggers
                for dr, dc in NEIGHBORS_4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        c_neigh = int(ex.input_grid[nr, nc])
                        reaction_observations[(c_self, c_neigh)][c_out] += 1

    # Filter for consistent reactions (Lattice Law)
    # A reaction is consistent if it ALWAYS produces the same output
    reaction_table: Dict[str, int] = {}

    for (c_self, c_neigh), outcomes in reaction_observations.items():
        if len(outcomes) == 1:
            # Consistent: always same output color
            c_out = list(outcomes.keys())[0]
            # Use string key for JSON serialization
            reaction_table[f"({c_self},{c_neigh})"] = c_out
        # else: inconsistent (multiple outputs), reject this reaction

    # Only emit if we found valid reactions
    if reaction_table:
        instances.append(SchemaInstance(
            family_id="S16",
            params={"reaction_table": reaction_table}
        ))

    return instances


if __name__ == "__main__":
    # Self-test with toy examples
    import numpy as np
    from src.schemas.context import build_example_context

    print("=" * 70)
    print("S16 Miner (Interaction/Chemistry) self-test")
    print("=" * 70)

    # Test 1: Simple reaction - Blue(1) next to Red(2) becomes Green(3)
    print("\nTest 1: Blue(1) next to Red(2) -> Green(3)")
    print("-" * 70)

    in1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],  # Blue(1) next to Red(2)
        [0, 0, 0, 0, 0],
        [0, 2, 1, 0, 0],  # Another Blue(1) next to Red(2)
        [0, 0, 0, 0, 0],
    ], dtype=int)

    out1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 2, 0, 0],  # Blue became Green
        [0, 0, 0, 0, 0],
        [0, 2, 3, 0, 0],  # Blue became Green
        [0, 0, 0, 0, 0],
    ], dtype=int)

    ex1 = build_example_context(in1, out1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=4)

    result1 = mine_S16(ctx1, {}, {})

    print(f"  Mined instances: {len(result1)}")
    if result1:
        print(f"  Reaction table: {result1[0].params['reaction_table']}")
        # Should have (1,2) -> 3 (Blue next to Red becomes Green)
        assert "(1,2)" in result1[0].params["reaction_table"]
        assert result1[0].params["reaction_table"]["(1,2)"] == 3
        print("  Found: (1,2) -> 3 (Blue next to Red becomes Green)")

    # Test 2: Inconsistent reaction - should NOT mine
    print("\nTest 2: Inconsistent reaction (should reject)")
    print("-" * 70)

    in2 = np.array([
        [0, 1, 2, 0],  # Blue next to Red
        [0, 0, 0, 0],
        [0, 1, 2, 0],  # Blue next to Red (again)
    ], dtype=int)

    out2 = np.array([
        [0, 3, 2, 0],  # Blue -> Green
        [0, 0, 0, 0],
        [0, 4, 2, 0],  # Blue -> Yellow (different!)
    ], dtype=int)

    ex2 = build_example_context(in2, out2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=5)

    result2 = mine_S16(ctx2, {}, {})

    print(f"  Mined instances: {len(result2)}")
    if result2:
        rt = result2[0].params["reaction_table"]
        # (1,2) should NOT be in the table (inconsistent)
        assert "(1,2)" not in rt, f"Should reject inconsistent reaction, got {rt}"
        print(f"  Reaction table: {rt}")
    else:
        print("  Correctly rejected inconsistent reactions")

    # Test 3: Spreading pattern - Red(1) spreads to adjacent White(0)
    print("\nTest 3: Spreading pattern - Red(1) infects White(0)")
    print("-" * 70)

    in3 = np.array([
        [0, 0, 0],
        [0, 1, 0],  # Red in center
        [0, 0, 0],
    ], dtype=int)

    out3 = np.array([
        [0, 1, 0],  # White above Red became Red
        [1, 1, 1],  # Whites beside Red became Red
        [0, 1, 0],  # White below Red became Red
    ], dtype=int)

    ex3 = build_example_context(in3, out3)
    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=2)

    result3 = mine_S16(ctx3, {}, {})

    print(f"  Mined instances: {len(result3)}")
    if result3:
        rt = result3[0].params["reaction_table"]
        print(f"  Reaction table: {rt}")
        # Should have (0,1) -> 1 (White next to Red becomes Red)
        assert "(0,1)" in rt
        assert rt["(0,1)"] == 1
        print("  Found: (0,1) -> 1 (White next to Red becomes Red)")

    # Test 4: No reactions (input == output)
    print("\nTest 4: No reactions (identity)")
    print("-" * 70)

    in4 = np.array([[1, 2], [3, 4]], dtype=int)
    out4 = np.array([[1, 2], [3, 4]], dtype=int)

    ex4 = build_example_context(in4, out4)
    ctx4 = TaskContext(train_examples=[ex4], test_examples=[], C=5)

    result4 = mine_S16(ctx4, {}, {})

    print(f"  Mined instances: {len(result4)}")
    assert len(result4) == 0, "Should not mine reactions when nothing changes"
    print("  Correctly returned empty (no active reactions)")

    print("\n" + "=" * 70)
    print("S16 Miner self-test passed.")
    print("=" * 70)
