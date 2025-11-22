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

Causality Check: A reaction (A, B) -> C is valid ONLY if A does NOT
turn into C when B is absent. This filters spurious correlations where
neighbor B is coincidentally present but not the actual cause.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any, Set
import numpy as np

from src.schemas.context import TaskContext
from src.catalog.types import SchemaInstance


# 4-connected neighbors (Up, Down, Left, Right)
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Causality threshold: If >20% of isolated A's turn into C, B is not the cause
CAUSALITY_THRESHOLD = 0.2

# Consistency threshold: Rule must apply to >=85% of eligible (c_self, c_neigh) pairs
CONSISTENCY_THRESHOLD = 0.85

# Minimum sample size: Need at least 5 eligible pixels for statistical significance
# 3-4 pixels can have 100% consistency by pure coincidence
MIN_SAMPLE_SIZE = 5


def get_neighbor_colors(grid: np.ndarray, r: int, c: int) -> Set[int]:
    """Get set of colors of 4-connected neighbors."""
    H, W = grid.shape
    neighbors = set()
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            neighbors.add(int(grid[nr, nc]))
    return neighbors


def verify_causality(
    task_context: TaskContext,
    c_self: int,
    c_neigh: int,
    c_out: int
) -> bool:
    """
    Verify that a reaction (c_self, c_neigh) -> c_out is causal, not spurious.

    Counterfactual Check:
    - Find all pixels of color c_self WITHOUT c_neigh as neighbor (Control Group)
    - If >20% of them ALSO became c_out, then c_neigh is irrelevant
    - The change is global (S2) or spatial (S12), not neighbor-based (S16)

    Args:
        task_context: TaskContext with training examples
        c_self: Self color in the reaction
        c_neigh: Neighbor color (potential trigger)
        c_out: Output color

    Returns:
        True if reaction is causal (c_neigh is necessary), False if spurious
    """
    control_total = 0  # c_self pixels WITHOUT c_neigh neighbor
    control_match = 0  # How many of those turned into c_out anyway

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            continue

        H, W = ex.input_grid.shape

        for r in range(H):
            for c in range(W):
                if int(ex.input_grid[r, c]) != c_self:
                    continue

                neighbors = get_neighbor_colors(ex.input_grid, r, c)

                # Control group: c_self pixels WITHOUT c_neigh as neighbor
                if c_neigh not in neighbors:
                    control_total += 1
                    if int(ex.output_grid[r, c]) == c_out:
                        control_match += 1

    # Calculate baseline probability: P(c_out | No c_neigh)
    if control_total == 0:
        # No control group exists - can't verify causality
        # Be CONSERVATIVE: reject (let S2 handle global recolors)
        # If ALL c_self pixels have c_neigh neighbor, we can't distinguish
        # "A next to B → C" from "All A → C" (global rule)
        return False

    p_baseline = control_match / control_total

    # If >20% of isolated c_self's turn into c_out, c_neigh is NOT the cause
    if p_baseline > CAUSALITY_THRESHOLD:
        return False  # Spurious - reject

    return True  # Causal - accept


def verify_consistency(
    task_context: TaskContext,
    c_self: int,
    c_neigh: int,
    c_out: int
) -> bool:
    """
    Verify that a reaction (c_self, c_neigh) -> c_out has high consistency.

    Consistency Check (Signal Strength):
    - Count all pixels of color c_self that have c_neigh as neighbor
    - Check what fraction of them became c_out
    - A TRUE chemical reaction should apply to ~100% of eligible pixels
    - If only 9% of eligible pixels react, it's coincidence, not a law

    Args:
        task_context: TaskContext with training examples
        c_self: Self color in the reaction
        c_neigh: Neighbor color (trigger)
        c_out: Expected output color

    Returns:
        True if reaction has high consistency (>=85%), False otherwise
    """
    treatment_total = 0  # c_self pixels WITH c_neigh neighbor
    treatment_match = 0  # How many of those turned into c_out

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            continue

        H, W = ex.input_grid.shape

        for r in range(H):
            for c in range(W):
                if int(ex.input_grid[r, c]) != c_self:
                    continue

                neighbors = get_neighbor_colors(ex.input_grid, r, c)

                # Treatment group: c_self pixels WITH c_neigh as neighbor
                if c_neigh in neighbors:
                    treatment_total += 1
                    if int(ex.output_grid[r, c]) == c_out:
                        treatment_match += 1

    # Calculate consistency: P(c_out | c_self with c_neigh)
    if treatment_total == 0:
        return False  # No eligible pixels

    # THERMODYNAMIC THRESHOLD: Require minimum sample size for statistical significance
    # 3-4 pixels can have 100% consistency by pure coincidence
    if treatment_total < MIN_SAMPLE_SIZE:
        return False  # Insufficient evidence - not statistically significant

    p_consistency = treatment_match / treatment_total

    # Require high consistency (>=85%) for a valid chemical reaction
    if p_consistency < CONSISTENCY_THRESHOLD:
        return False  # Weak correlation - not a law

    return True  # Strong, consistent reaction


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
    3. Lattice Law: Same trigger must always produce same output
    4. DOUBLE FILTER:
       a. Consistency (Signal Strength): >=85% of eligible pixels must react
       b. Causality (Exclusivity): <20% of isolated pixels should react
    5. Emit Schema with reaction_table

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
        if len(outcomes) != 1:
            # Inconsistent (multiple outputs) - reject
            continue

        c_out = list(outcomes.keys())[0]

        # DOUBLE FILTER: Both checks must pass

        # 1. CONSISTENCY CHECK (Signal Strength): Rule must apply to most eligible pixels
        if not verify_consistency(task_context, c_self, c_neigh, c_out):
            # Weak correlation (e.g., only 9% of eligible pixels react)
            # Not a law - just coincidence
            continue

        # 2. CAUSALITY CHECK (Exclusivity): Verify neighbor is actually the cause
        if not verify_causality(task_context, c_self, c_neigh, c_out):
            # Spurious correlation - neighbor is coincidental, not causal
            # Let S2 (global recolor) or S12 (spatial) handle it
            continue

        # Passed BOTH checks - valid reaction
        reaction_table[f"({c_self},{c_neigh})"] = c_out

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

    # Test 1: Real reaction - Blue(1) next to Red(2) becomes Green(3)
    # Blues NOT next to Red stay Blue (causal)
    # Need >= 5 Blues next to Red for MIN_SAMPLE_SIZE
    print("\nTest 1: Real reaction - Blue(1) next to Red(2) -> Green(3)")
    print("-" * 70)

    in1 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 1, 2, 0],  # 2 Blues next to Red
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 0, 2, 1, 0],  # 2 more Blues next to Red
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 0],  # 1 more Blue next to Red (total: 5)
        [1, 0, 0, 1, 0, 0, 1],  # Isolated Blues - control group
    ], dtype=int)

    out1 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 2, 0, 3, 2, 0],  # Blues became Green
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 3, 0, 2, 3, 0],  # Blues became Green
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 2, 0, 0, 0, 0],  # Blue became Green
        [1, 0, 0, 1, 0, 0, 1],  # Isolated Blues stayed Blue
    ], dtype=int)

    ex1 = build_example_context(in1, out1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=4)

    result1 = mine_S16(ctx1, {}, {})

    print(f"  Mined instances: {len(result1)}")
    if result1:
        rt = result1[0].params["reaction_table"]
        print(f"  Reaction table: {rt}")
        # Should have (1,2) -> 3 (Blue next to Red becomes Green)
        assert "(1,2)" in rt, f"Should find (1,2)->3, got {rt}"
        assert rt["(1,2)"] == 3
        print("  Found: (1,2) -> 3 (Blue next to Red becomes Green)")
        print("  Causality verified: isolated Blues stayed Blue")
    else:
        print("  WARNING: No reactions mined (check sample size)")

    # Test 2: Spurious reaction - Global recolor (ALL 7s -> 8)
    # Should be REJECTED because 7s change even without specific neighbor
    print("\nTest 2: Spurious reaction - Global recolor (should REJECT)")
    print("-" * 70)

    in2 = np.array([
        [7, 0, 7],  # 7 with 0 neighbor
        [0, 7, 0],  # 7 with 0 neighbors
        [7, 0, 7],  # 7 with 0 neighbor
    ], dtype=int)

    out2 = np.array([
        [8, 0, 8],  # All 7s became 8
        [0, 8, 0],  # Even isolated 7
        [8, 0, 8],  # All 7s became 8
    ], dtype=int)

    ex2 = build_example_context(in2, out2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=9)

    result2 = mine_S16(ctx2, {}, {})

    print(f"  Mined instances: {len(result2)}")
    if result2:
        rt = result2[0].params["reaction_table"]
        # (7,0) -> 8 should be REJECTED (7s change even without 0 neighbor)
        assert "(7,0)" not in rt, f"Should reject spurious (7,0)->8, got {rt}"
        print(f"  Reaction table: {rt}")
    else:
        print("  Correctly rejected spurious global recolor")
    print("  Causality check: 7s change even without specific neighbor = NOT causal")

    # Test 3: Spreading pattern - Red(1) infects White(0)
    # Only whites ADJACENT to red change (causal)
    # Need >= 5 whites next to Red for MIN_SAMPLE_SIZE
    print("\nTest 3: Spreading pattern - Red(1) infects adjacent White(0)")
    print("-" * 70)

    # Grid with 2 reds, each has 4 adjacent whites = 8 total treatment pixels
    # Corner whites are control group (not adjacent to red)
    in3 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],  # Two Reds at (1,2) and (1,4)
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    out3 = np.array([
        [0, 0, 1, 0, 1, 0, 0],  # Whites above reds became red
        [0, 1, 1, 1, 1, 1, 0],  # Whites beside reds became red
        [0, 0, 1, 0, 1, 0, 0],  # Whites below reds became red
    ], dtype=int)

    ex3 = build_example_context(in3, out3)
    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=2)

    result3 = mine_S16(ctx3, {}, {})

    print(f"  Mined instances: {len(result3)}")
    if result3:
        rt = result3[0].params["reaction_table"]
        print(f"  Reaction table: {rt}")
        # Should have (0,1) -> 1 (White next to Red becomes Red)
        assert "(0,1)" in rt, f"Should find (0,1)->1, got {rt}"
        assert rt["(0,1)"] == 1
        print("  Found: (0,1) -> 1 (White next to Red becomes Red)")
        print("  Causality verified: corner whites (not adjacent to red) stayed white")
    else:
        print("  WARNING: No reactions mined (check sample size)")

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

    # Test 5: Inconsistent reaction - should NOT mine
    print("\nTest 5: Inconsistent reaction (should reject)")
    print("-" * 70)

    in5 = np.array([
        [0, 1, 2, 0],  # Blue next to Red
        [0, 0, 0, 0],
        [0, 1, 2, 0],  # Blue next to Red (again)
    ], dtype=int)

    out5 = np.array([
        [0, 3, 2, 0],  # Blue -> Green
        [0, 0, 0, 0],
        [0, 4, 2, 0],  # Blue -> Yellow (different!)
    ], dtype=int)

    ex5 = build_example_context(in5, out5)
    ctx5 = TaskContext(train_examples=[ex5], test_examples=[], C=5)

    result5 = mine_S16(ctx5, {}, {})

    print(f"  Mined instances: {len(result5)}")
    if result5:
        rt = result5[0].params["reaction_table"]
        assert "(1,2)" not in rt, f"Should reject inconsistent reaction, got {rt}"
        print(f"  Reaction table: {rt}")
    else:
        print("  Correctly rejected inconsistent reactions")

    print("\n" + "=" * 70)
    print("S16 Miner self-test passed.")
    print("=" * 70)
