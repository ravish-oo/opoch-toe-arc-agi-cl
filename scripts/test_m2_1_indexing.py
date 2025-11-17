#!/usr/bin/env python3
"""
Integration test for WO-M2.1 indexing.py

Tests indexing functions with real ARC grid data to ensure
H, W from actual tasks flow correctly through the indexing system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.arc_io import load_arc_training_challenges
from src.constraints.indexing import (
    flatten_index,
    unflatten_index,
    y_index,
    y_index_to_pc
)


def test_with_real_arc_grid():
    """Test indexing with a real ARC task grid"""
    print("Testing indexing with real ARC grid...")

    # Load ARC data
    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Get first task's first training input
    task_id = sorted(tasks.keys())[0]
    grid = tasks[task_id]["train"][0]
    H, W = grid.shape
    N = H * W
    C = 10  # ARC palette is 0-9

    print(f"  Task: {task_id}")
    print(f"  Grid shape: {H}x{W} (N={N} pixels)")
    print(f"  Palette: C={C} colors")

    # Test 1: Pixel index roundtrip with actual grid dimensions
    print("\n  Testing pixel index roundtrip...")
    test_coords = [
        (0, 0),           # Top-left
        (0, W-1),         # Top-right
        (H-1, 0),         # Bottom-left
        (H-1, W-1),       # Bottom-right
        (H//2, W//2),     # Center
    ]

    for (r, c) in test_coords:
        if r < H and c < W:  # Ensure coords are valid
            p_idx = flatten_index(r, c, W)
            rr, cc = unflatten_index(p_idx, W)
            assert (rr, cc) == (r, c), \
                f"Roundtrip failed for ({r},{c}): got ({rr},{cc})"

            # Also verify p_idx is in valid range
            assert 0 <= p_idx < N, \
                f"p_idx {p_idx} out of range [0, {N})"

    print(f"    ✓ Tested {len([c for c in test_coords if c[0] < H and c[1] < W])} positions")

    # Test 2: y-index roundtrip
    print("\n  Testing y-index roundtrip...")
    test_pixels = [0, N//2, N-1]  # First, middle, last pixel
    test_colors = [0, C//2, C-1]   # First, middle, last color

    for p_idx in test_pixels:
        for color in test_colors:
            y_idx = y_index(p_idx, color, C)
            p_back, c_back = y_index_to_pc(y_idx, C, W)

            assert p_back == p_idx, \
                f"p_idx mismatch: {p_idx} != {p_back}"
            assert c_back == color, \
                f"color mismatch: {color} != {c_back}"

            # Verify y_idx is in valid range
            assert 0 <= y_idx < N * C, \
                f"y_idx {y_idx} out of range [0, {N*C})"

    print(f"    ✓ Tested {len(test_pixels) * len(test_colors)} (p_idx, color) pairs")

    # Test 3: Verify actual grid color access aligns with indexing
    print("\n  Testing grid color access alignment...")
    sample_positions = [(0, 0), (H-1, W-1), (H//2, W//2)]

    for (r, c) in sample_positions:
        if r < H and c < W:
            # Get actual color from grid
            actual_color = int(grid[r, c])

            # Compute indices
            p_idx = flatten_index(r, c, W)
            y_idx = y_index(p_idx, actual_color, C)

            # Decode back
            p_back, color_back = y_index_to_pc(y_idx, C, W)
            r_back, c_back = unflatten_index(p_back, W)

            # Verify full roundtrip
            assert (r_back, c_back) == (r, c), \
                f"Position roundtrip failed: ({r},{c}) != ({r_back},{c_back})"
            assert color_back == actual_color, \
                f"Color roundtrip failed: {actual_color} != {color_back}"

    print(f"    ✓ Grid color access aligned for {len([p for p in sample_positions if p[0] < H and p[1] < W])} positions")

    # Test 4: Verify y-vector size
    print("\n  Testing y-vector size...")
    expected_size = N * C
    max_y_idx = y_index(N-1, C-1, C)
    assert max_y_idx == expected_size - 1, \
        f"Max y_idx {max_y_idx} != expected {expected_size - 1}"
    print(f"    ✓ y-vector size: {expected_size} (N={N} × C={C})")


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\nTesting edge cases...")

    # Test 1: Minimum grid (1x1)
    print("  Testing 1x1 grid...")
    H, W, C = 1, 1, 10
    p_idx = flatten_index(0, 0, W)
    assert p_idx == 0
    r, c = unflatten_index(0, W)
    assert (r, c) == (0, 0)
    y_idx = y_index(0, 5, C)
    assert y_idx == 5
    p_back, color_back = y_index_to_pc(5, C, W)
    assert (p_back, color_back) == (0, 5)
    print("    ✓ 1x1 grid passed")

    # Test 2: Large grid (30x30 - max ARC size)
    print("  Testing 30x30 grid (max ARC size)...")
    H, W, C = 30, 30, 10
    N = H * W
    # Test corners
    corners = [(0, 0), (0, 29), (29, 0), (29, 29)]
    for (r, c) in corners:
        p_idx = flatten_index(r, c, W)
        rr, cc = unflatten_index(p_idx, W)
        assert (rr, cc) == (r, c)
    # Test max y_idx
    max_y = y_index(N-1, C-1, C)
    assert max_y == N * C - 1
    print(f"    ✓ 30x30 grid passed (max y_idx={max_y})")

    # Test 3: Non-square grids
    print("  Testing non-square grids...")
    test_shapes = [(1, 30), (30, 1), (5, 20), (20, 5)]
    for (h, w) in test_shapes:
        n = h * w
        # Test first and last pixel
        p0 = flatten_index(0, 0, w)
        assert p0 == 0
        p_last = flatten_index(h-1, w-1, w)
        assert p_last == n - 1
        r_last, c_last = unflatten_index(p_last, w)
        assert (r_last, c_last) == (h-1, w-1)
    print(f"    ✓ Tested {len(test_shapes)} non-square grids")


def test_consistency_across_multiple_tasks():
    """Test indexing consistency across multiple ARC tasks"""
    print("\nTesting consistency across multiple ARC tasks...")

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Test on first 5 tasks
    sample_task_ids = sorted(tasks.keys())[:5]

    for task_id in sample_task_ids:
        task = tasks[task_id]

        if task["train"]:
            grid = task["train"][0]
            H, W = grid.shape
            N = H * W
            C = 10

            # Quick sanity check
            # Test a few positions
            for r in [0, H//2, H-1]:
                for c in [0, W//2, W-1]:
                    if r < H and c < W:
                        p_idx = flatten_index(r, c, W)
                        rr, cc = unflatten_index(p_idx, W)
                        assert (rr, cc) == (r, c), \
                            f"Task {task_id}: roundtrip failed at ({r},{c})"

    print(f"  ✓ Consistency verified across {len(sample_task_ids)} tasks")


def main():
    print("=" * 60)
    print("WO-M2.1 Integration Test - indexing.py")
    print("=" * 60)
    print()

    try:
        test_with_real_arc_grid()
        test_edge_cases()
        test_consistency_across_multiple_tasks()

        print()
        print("=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
