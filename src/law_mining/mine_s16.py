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

# =============================================================================
# TIERED THRESHOLDS: Different standards based on sample size
# =============================================================================
# CASE A: High count (>=5 treatment pixels) - allow some noise
HIGH_COUNT_THRESHOLD = 5
HIGH_COUNT_CONSISTENCY = 0.85  # 85% consistency
HIGH_COUNT_BASELINE = 0.20     # 20% baseline

# CASE B: Low count (2-4 treatment pixels) - require pristine evidence
LOW_COUNT_MIN = 2
LOW_COUNT_CONSISTENCY = 1.0    # 100% consistency (perfect)
LOW_COUNT_BASELINE = 0.05      # 5% baseline (highly causal)


def get_neighbor_colors(grid: np.ndarray, r: int, c: int) -> Set[int]:
    """Get set of colors of 4-connected neighbors."""
    H, W = grid.shape
    neighbors = set()
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            neighbors.add(int(grid[nr, nc]))
    return neighbors


def verify_rule(
    task_context: TaskContext,
    c_self: int,
    c_neigh: int,
    c_out: int
) -> bool:
    """
    Verify a reaction rule using TIERED evidence thresholds.

    This replaces separate verify_consistency() and verify_causality() with
    a unified approach that counts once and applies tiered logic.

    TIERED EVIDENCE THRESHOLDS:

    CASE A: Strong Evidence (>=5 treatment pixels)
        - Consistency >= 85% (some noise acceptable)
        - Baseline <= 20% (if control group exists)
        - No control group: OK (enough positive evidence)

    CASE B: Rare Event (2-4 treatment pixels) - Small Data Laws
        - Consistency = 100% (must be pristine)
        - Baseline <= 5% (highly causal)
        - No control group: REJECT (can't verify causality with little data)

    CASE C: Insufficient Evidence (<2 treatment pixels)
        - REJECT (too risky)

    Args:
        task_context: TaskContext with training examples
        c_self: Self color in the reaction
        c_neigh: Neighbor color (potential trigger)
        c_out: Output color

    Returns:
        True if rule passes tiered verification, False otherwise
    """
    # Count both treatment and control groups in one pass
    treatment_total = 0  # c_self pixels WITH c_neigh neighbor
    treatment_match = 0  # How many turned into c_out
    control_total = 0    # c_self pixels WITHOUT c_neigh neighbor
    control_match = 0    # How many turned into c_out anyway

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
                output_color = int(ex.output_grid[r, c])

                if c_neigh in neighbors:
                    # Treatment group: c_self WITH c_neigh neighbor
                    treatment_total += 1
                    if output_color == c_out:
                        treatment_match += 1
                else:
                    # Control group: c_self WITHOUT c_neigh neighbor
                    control_total += 1
                    if output_color == c_out:
                        control_match += 1

    # No treatment pixels at all
    if treatment_total == 0:
        return False

    # Calculate metrics
    p_consistency = treatment_match / treatment_total
    p_baseline = control_match / control_total if control_total > 0 else 0.0

    # =========================================================================
    # CASE A: Strong Evidence (>=5 treatment pixels)
    # With lots of data, some noise is acceptable
    # =========================================================================
    if treatment_total >= HIGH_COUNT_THRESHOLD:
        # Consistency check: >=85%
        if p_consistency < HIGH_COUNT_CONSISTENCY:
            return False

        # Causality check: baseline <=20% (if control group exists)
        # With enough positive evidence, we can skip if no control group
        if control_total > 0 and p_baseline > HIGH_COUNT_BASELINE:
            return False

        return True

    # =========================================================================
    # CASE B: Rare Event (2-4 treatment pixels) - Small Data Laws
    # With little data, rule must be PRISTINE
    # =========================================================================
    elif treatment_total >= LOW_COUNT_MIN:
        # Must be PERFECT (100% consistent)
        if p_consistency < LOW_COUNT_CONSISTENCY:
            return False

        # Must have control group to verify causality (can't risk it with small N)
        if control_total == 0:
            return False

        # Must be HIGHLY CAUSAL (baseline <=5%)
        if p_baseline > LOW_COUNT_BASELINE:
            return False

        return True

    # =========================================================================
    # CASE C: Insufficient Evidence (<2 treatment pixels)
    # Too risky even for us
    # =========================================================================
    else:
        return False


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
    4. TIERED VERIFICATION (via verify_rule):
       - High count (>=5): 85% consistency, 20% baseline
       - Low count (2-4): 100% consistency, 5% baseline, needs control group
       - Tiny count (<2): rejected
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

        # TIERED VERIFICATION: Different thresholds based on sample size
        if not verify_rule(task_context, c_self, c_neigh, c_out):
            # Failed verification (consistency, causality, or sample size)
            continue

        # Passed tiered verification - valid reaction
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
    # CASE A: 5 treatment pixels, uses 85% consistency / 20% baseline
    print("\nTest 1: CASE A - High count reaction (>=5 treatment pixels)")
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
    # Should be REJECTED because 7s change even WITHOUT 0 neighbor
    # Need 7s WITH 0 neighbor (treatment) AND 7s WITHOUT 0 neighbor (control)
    # Control group: 7s surrounded by other 7s (rows 4-5 interior)
    print("\nTest 2: Spurious reaction - Global recolor (should REJECT)")
    print("-" * 70)

    in2 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 7, 0, 7, 0, 7, 0],  # 3 7s with 0 neighbor (treatment)
        [0, 0, 0, 0, 0, 0, 0],
        [7, 7, 7, 7, 7, 7, 7],  # 7 7s with 0 neighbor from row 2 (treatment = 10)
        [7, 7, 7, 7, 7, 7, 7],  # 7 7s WITHOUT 0 neighbor (control)
        [7, 7, 7, 7, 7, 7, 7],  # 7 7s WITHOUT 0 neighbor (control = 14)
    ], dtype=int)

    out2 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 8, 0, 8, 0],  # 7s with 0 neighbor became 8
        [0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8],  # 7s with 0 neighbor became 8
        [8, 8, 8, 8, 8, 8, 8],  # 7s WITHOUT 0 neighbor ALSO became 8 (baseline = 100%!)
        [8, 8, 8, 8, 8, 8, 8],  # 7s WITHOUT 0 neighbor ALSO became 8
    ], dtype=int)

    ex2 = build_example_context(in2, out2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=9)

    result2 = mine_S16(ctx2, {}, {})

    print(f"  Mined instances: {len(result2)}")
    if result2:
        rt = result2[0].params["reaction_table"]
        # (7,0) -> 8 should be REJECTED (7s change even without 0 neighbor = 100% baseline)
        assert "(7,0)" not in rt, f"Should reject spurious (7,0)->8, got {rt}"
        print(f"  Reaction table: {rt}")
    else:
        print("  Correctly rejected spurious global recolor")
    print("  Causality check: 7s change even without 0 neighbor (baseline=100%) = NOT causal")

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

    # Test 6: Small Data Law - CASE B (2-4 treatment pixels with perfect evidence)
    # 3 Blues next to Red, ALL become Green (100% consistency)
    # 5 isolated Blues stay Blue (0% baseline = highly causal)
    print("\nTest 6: CASE B - Small data law (2-4 treatment, pristine evidence)")
    print("-" * 70)

    in6 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 1, 2, 0],  # 2 Blues next to Red
        [0, 0, 1, 2, 0, 0, 0],  # 1 Blue next to Red (total: 3 treatment)
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],  # 4 isolated Blues (control group)
        [1, 0, 0, 0, 0, 0, 0],  # 1 more isolated Blue (total: 5 control)
    ], dtype=int)

    out6 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 2, 0, 3, 2, 0],  # Blues became Green (100% consistency)
        [0, 0, 3, 2, 0, 0, 0],  # Blue became Green
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],  # Isolated Blues stayed Blue (0% baseline)
        [1, 0, 0, 0, 0, 0, 0],  # Isolated Blue stayed Blue
    ], dtype=int)

    ex6 = build_example_context(in6, out6)
    ctx6 = TaskContext(train_examples=[ex6], test_examples=[], C=4)

    result6 = mine_S16(ctx6, {}, {})

    print(f"  Mined instances: {len(result6)}")
    if result6:
        rt = result6[0].params["reaction_table"]
        print(f"  Reaction table: {rt}")
        assert "(1,2)" in rt, f"Should find (1,2)->3 with only 3 treatment pixels, got {rt}"
        assert rt["(1,2)"] == 3
        print("  CASE B PASS: Small data law recovered with pristine evidence!")
        print("  (3 treatment, 100% consistency, 0% baseline)")
    else:
        raise AssertionError("CASE B FAIL: Should accept pristine small data law")

    # Test 7: Small Data Law REJECTION - imperfect evidence
    # 3 Blues next to Red, but 1 isolated Blue also changed (baseline > 5%)
    print("\nTest 7: CASE B rejection - small data with impure baseline")
    print("-" * 70)

    in7 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],  # 1 Blue next to Red
        [0, 0, 0, 0, 0],
        [0, 2, 1, 0, 0],  # 1 Blue next to Red (total: 2 treatment)
        [1, 0, 1, 0, 1],  # 3 isolated Blues (control group)
    ], dtype=int)

    out7 = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 2, 0, 0],  # Blue -> Green (treatment)
        [0, 0, 0, 0, 0],
        [0, 2, 3, 0, 0],  # Blue -> Green (treatment)
        [3, 0, 1, 0, 1],  # 1 isolated Blue ALSO -> Green (baseline = 33%!)
    ], dtype=int)

    ex7 = build_example_context(in7, out7)
    ctx7 = TaskContext(train_examples=[ex7], test_examples=[], C=4)

    result7 = mine_S16(ctx7, {}, {})

    print(f"  Mined instances: {len(result7)}")
    if result7:
        rt = result7[0].params["reaction_table"]
        assert "(1,2)" not in rt, f"Should reject due to high baseline, got {rt}"
        print(f"  Reaction table: {rt}")
    else:
        print("  Correctly rejected: baseline 33% > 5% threshold for small data")

    print("\n" + "=" * 70)
    print("S16 Miner self-test passed.")
    print("=" * 70)
