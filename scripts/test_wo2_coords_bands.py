#!/usr/bin/env python3
"""
Comprehensive test script for WO2 - coords_bands.py

Validates coordinate features, band labels, and border masks
for alignment with math kernel section 1.2.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.coords_bands import (
    coord_features,
    row_band_labels,
    col_band_labels,
    border_mask
)
from src.core.grid_types import Grid


def test_coord_features_structure():
    """Test that coord_features returns correct structure"""
    print("Testing coord_features structure...")

    grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    H, W = grid.shape  # 2x3

    features = coord_features(grid)

    # Check all pixels are present
    assert len(features) == H * W, f"Expected {H*W} pixels, got {len(features)}"

    # Check structure for each pixel
    for r in range(H):
        for c in range(W):
            pixel = (r, c)
            assert pixel in features, f"Pixel {pixel} not in features"

            f = features[pixel]
            assert "row" in f, f"Missing 'row' key for pixel {pixel}"
            assert "col" in f, f"Missing 'col' key for pixel {pixel}"
            assert "row_mod" in f, f"Missing 'row_mod' key for pixel {pixel}"
            assert "col_mod" in f, f"Missing 'col_mod' key for pixel {pixel}"

            # Check coordinate values
            assert f["row"] == r, f"Wrong row for pixel {pixel}"
            assert f["col"] == c, f"Wrong col for pixel {pixel}"

    print("  ✓ Structure correct for 2x3 grid")


def test_coord_features_modulo_keys():
    """Test that ALL modulo keys {2,3,4,5} are present (math kernel requirement)"""
    print("Testing coord_features has all modulo keys {2,3,4,5}...")

    grid = np.array([[0]], dtype=int)
    features = coord_features(grid)

    f = features[(0, 0)]

    # Check row_mod has all 4 keys
    assert isinstance(f["row_mod"], dict), "row_mod should be a dict"
    assert set(f["row_mod"].keys()) == {2, 3, 4, 5}, \
        f"row_mod should have keys {{2,3,4,5}}, got {set(f['row_mod'].keys())}"

    # Check col_mod has all 4 keys
    assert isinstance(f["col_mod"], dict), "col_mod should be a dict"
    assert set(f["col_mod"].keys()) == {2, 3, 4, 5}, \
        f"col_mod should have keys {{2,3,4,5}}, got {set(f['col_mod'].keys())}"

    print("  ✓ All modulo keys {2,3,4,5} present")


def test_coord_features_modulo_values():
    """Test that modulo values are computed correctly"""
    print("Testing coord_features modulo value correctness...")

    grid = np.zeros((10, 10), dtype=int)
    features = coord_features(grid)

    # Test various pixels
    test_cases = [
        ((0, 0), {2: 0, 3: 0, 4: 0, 5: 0}, {2: 0, 3: 0, 4: 0, 5: 0}),
        ((5, 7), {2: 1, 3: 2, 4: 1, 5: 0}, {2: 1, 3: 1, 4: 3, 5: 2}),
        ((9, 9), {2: 1, 3: 0, 4: 1, 5: 4}, {2: 1, 3: 0, 4: 1, 5: 4}),
        ((2, 3), {2: 0, 3: 2, 4: 2, 5: 2}, {2: 1, 3: 0, 4: 3, 5: 3}),
    ]

    for (r, c), expected_row_mod, expected_col_mod in test_cases:
        f = features[(r, c)]
        assert f["row_mod"] == expected_row_mod, \
            f"Pixel ({r},{c}): row_mod mismatch. Expected {expected_row_mod}, got {f['row_mod']}"
        assert f["col_mod"] == expected_col_mod, \
            f"Pixel ({r},{c}): col_mod mismatch. Expected {expected_col_mod}, got {f['col_mod']}"

    print(f"  ✓ Modulo values correct for {len(test_cases)} test cases")


def test_row_band_labels_edge_cases():
    """Test row_band_labels for all edge cases"""
    print("Testing row_band_labels edge cases...")

    # H=1: middle only
    bands = row_band_labels(1)
    assert bands == {0: 'middle'}, f"H=1: expected {{0: 'middle'}}, got {bands}"

    # H=2: top, bottom
    bands = row_band_labels(2)
    assert bands == {0: 'top', 1: 'bottom'}, \
        f"H=2: expected {{0: 'top', 1: 'bottom'}}, got {bands}"

    # H=3: top, middle, bottom
    bands = row_band_labels(3)
    assert bands == {0: 'top', 1: 'middle', 2: 'bottom'}, \
        f"H=3: expected {{0: 'top', 1: 'middle', 2: 'bottom'}}, got {bands}"

    # H=4: top, middle, middle, bottom
    bands = row_band_labels(4)
    assert bands == {0: 'top', 1: 'middle', 2: 'middle', 3: 'bottom'}, \
        f"H=4: expected {{0: 'top', 1: 'middle', 2: 'middle', 3: 'bottom'}}, got {bands}"

    # H=10: verify first=top, last=bottom, rest=middle
    bands = row_band_labels(10)
    assert bands[0] == 'top', "First row should be 'top'"
    assert bands[9] == 'bottom', "Last row should be 'bottom'"
    for r in range(1, 9):
        assert bands[r] == 'middle', f"Row {r} should be 'middle'"

    print("  ✓ Row band labels correct for H ∈ {1,2,3,4,10}")


def test_col_band_labels_edge_cases():
    """Test col_band_labels for all edge cases"""
    print("Testing col_band_labels edge cases...")

    # W=1: middle only
    bands = col_band_labels(1)
    assert bands == {0: 'middle'}, f"W=1: expected {{0: 'middle'}}, got {bands}"

    # W=2: left, right
    bands = col_band_labels(2)
    assert bands == {0: 'left', 1: 'right'}, \
        f"W=2: expected {{0: 'left', 1: 'right'}}, got {bands}"

    # W=3: left, middle, right
    bands = col_band_labels(3)
    assert bands == {0: 'left', 1: 'middle', 2: 'right'}, \
        f"W=3: expected {{0: 'left', 1: 'middle', 2: 'right'}}, got {bands}"

    # W=5: left, middle, middle, middle, right
    bands = col_band_labels(5)
    assert bands == {0: 'left', 1: 'middle', 2: 'middle', 3: 'middle', 4: 'right'}, \
        f"W=5: expected correct mapping, got {bands}"

    # W=10: verify first=left, last=right, rest=middle
    bands = col_band_labels(10)
    assert bands[0] == 'left', "First col should be 'left'"
    assert bands[9] == 'right', "Last col should be 'right'"
    for c in range(1, 9):
        assert bands[c] == 'middle', f"Col {c} should be 'middle'"

    print("  ✓ Col band labels correct for W ∈ {1,2,3,5,10}")


def test_band_labels_completeness():
    """Verify band labels cover all indices"""
    print("Testing band labels cover all indices...")

    for H in [1, 2, 3, 5, 10, 20, 30]:
        bands = row_band_labels(H)
        assert len(bands) == H, f"H={H}: should have {H} entries, got {len(bands)}"
        assert set(bands.keys()) == set(range(H)), \
            f"H={H}: should cover all indices 0..{H-1}"

    for W in [1, 2, 3, 5, 10, 20, 30]:
        bands = col_band_labels(W)
        assert len(bands) == W, f"W={W}: should have {W} entries, got {len(bands)}"
        assert set(bands.keys()) == set(range(W)), \
            f"W={W}: should cover all indices 0..{W-1}"

    print("  ✓ Band labels complete for various sizes")


def test_border_mask_structure():
    """Test border_mask returns correct structure"""
    print("Testing border_mask structure...")

    grid = np.zeros((5, 7), dtype=int)
    mask = border_mask(grid)

    # Check type and shape
    assert isinstance(mask, np.ndarray), "Mask should be ndarray"
    assert mask.dtype == bool, f"Mask dtype should be bool, got {mask.dtype}"
    assert mask.shape == grid.shape, \
        f"Mask shape should match grid {grid.shape}, got {mask.shape}"

    print("  ✓ Border mask structure correct")


def test_border_mask_edge_cases():
    """Test border_mask for various grid sizes"""
    print("Testing border_mask edge cases...")

    # 1x1 grid: all border
    grid = np.zeros((1, 1), dtype=int)
    mask = border_mask(grid)
    assert mask.all(), "1x1 grid: entire grid should be border"

    # 2x2 grid: all border
    grid = np.zeros((2, 2), dtype=int)
    mask = border_mask(grid)
    assert mask.all(), "2x2 grid: entire grid should be border"

    # 3x3 grid: only center is interior
    grid = np.zeros((3, 3), dtype=int)
    mask = border_mask(grid)
    expected = np.array([
        [True, True, True],
        [True, False, True],
        [True, True, True]
    ])
    assert np.array_equal(mask, expected), \
        f"3x3 grid: border mask mismatch.\nExpected:\n{expected}\nGot:\n{mask}"

    # 5x5 grid: 3x3 interior
    grid = np.zeros((5, 5), dtype=int)
    mask = border_mask(grid)
    # Count interior pixels (False values)
    interior_count = np.sum(~mask)
    expected_interior = 3 * 3  # (5-2) * (5-2)
    assert interior_count == expected_interior, \
        f"5x5 grid: should have {expected_interior} interior pixels, got {interior_count}"

    # Verify corners are always True
    assert mask[0, 0], "Top-left corner should be border"
    assert mask[0, 4], "Top-right corner should be border"
    assert mask[4, 0], "Bottom-left corner should be border"
    assert mask[4, 4], "Bottom-right corner should be border"

    # Verify center is False
    assert not mask[2, 2], "Center should not be border"

    print("  ✓ Border mask correct for various sizes")


def test_border_mask_logic():
    """Test border mask logic: r=0 or r=H-1 or c=0 or c=W-1"""
    print("Testing border mask logic...")

    H, W = 10, 12
    grid = np.zeros((H, W), dtype=int)
    mask = border_mask(grid)

    for r in range(H):
        for c in range(W):
            is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            assert mask[r, c] == is_border, \
                f"Pixel ({r},{c}): expected border={is_border}, got {mask[r,c]}"

    print("  ✓ Border logic correct for 10x12 grid")


def test_integration_with_arc_data():
    """Test integration with real ARC data"""
    print("Testing integration with real ARC data...")

    from src.core.arc_io import load_arc_training_challenges

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Test on first 5 tasks
    sample_task_ids = sorted(tasks.keys())[:5]

    for task_id in sample_task_ids:
        task = tasks[task_id]

        # Test on first train grid
        if task["train"]:
            grid = task["train"][0]
            H, W = grid.shape

            # coord_features should work
            features = coord_features(grid)
            assert len(features) == H * W, \
                f"Task {task_id}: coord_features should have {H*W} entries"

            # Check a random pixel has correct structure
            r, c = H // 2, W // 2
            f = features[(r, c)]
            assert set(f["row_mod"].keys()) == {2, 3, 4, 5}, \
                f"Task {task_id}: row_mod keys incorrect"
            assert set(f["col_mod"].keys()) == {2, 3, 4, 5}, \
                f"Task {task_id}: col_mod keys incorrect"

            # row_band_labels should work
            row_bands = row_band_labels(H)
            assert len(row_bands) == H, \
                f"Task {task_id}: row_band_labels should have {H} entries"

            # col_band_labels should work
            col_bands = col_band_labels(W)
            assert len(col_bands) == W, \
                f"Task {task_id}: col_band_labels should have {W} entries"

            # border_mask should work
            mask = border_mask(grid)
            assert mask.shape == grid.shape, \
                f"Task {task_id}: border_mask shape mismatch"
            assert mask.dtype == bool, \
                f"Task {task_id}: border_mask should be bool"

    print(f"  ✓ Integration test passed for {len(sample_task_ids)} ARC tasks")


def test_math_kernel_section_1_2_compliance():
    """
    Verify full compliance with math kernel section 1.2:
    Feature dictionary φ(p) - items 1-3
    """
    print("Testing math kernel section 1.2 compliance...")

    grid = np.random.randint(0, 10, size=(8, 10), dtype=int)
    H, W = grid.shape

    # 1. Coordinates (row, col)
    features = coord_features(grid)
    for r in range(H):
        for c in range(W):
            f = features[(r, c)]
            assert f["row"] == r, "Coordinate 'row' incorrect"
            assert f["col"] == c, "Coordinate 'col' incorrect"

    # 2. Residues / periodic indices: r mod k, c mod k for k ∈ {2,3,4,5}
    for r in range(H):
        for c in range(W):
            f = features[(r, c)]
            for k in [2, 3, 4, 5]:
                assert f["row_mod"][k] == r % k, \
                    f"Pixel ({r},{c}): row_mod[{k}] should be {r%k}, got {f['row_mod'][k]}"
                assert f["col_mod"][k] == c % k, \
                    f"Pixel ({r},{c}): col_mod[{k}] should be {c%k}, got {f['col_mod'][k]}"

    # 3. Bands & frames
    row_bands = row_band_labels(H)
    col_bands = col_band_labels(W)
    mask = border_mask(grid)

    # Verify deterministic behavior
    assert row_bands[0] in ['top', 'middle'], "First row should be top or middle"
    assert col_bands[0] in ['left', 'middle'], "First col should be left or middle"

    # Verify border flag exists and is boolean
    assert mask.dtype == bool, "Border mask should be boolean"

    print("  ✓ Math kernel section 1.2 fully compliant")


def main():
    print("=" * 60)
    print("WO2 Comprehensive Test Suite - coords_bands.py")
    print("=" * 60)
    print()

    tests = [
        test_coord_features_structure,
        test_coord_features_modulo_keys,
        test_coord_features_modulo_values,
        test_row_band_labels_edge_cases,
        test_col_band_labels_edge_cases,
        test_band_labels_completeness,
        test_border_mask_structure,
        test_border_mask_edge_cases,
        test_border_mask_logic,
        test_integration_with_arc_data,
        test_math_kernel_section_1_2_compliance,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print()
    print("=" * 60)
    print(f"✅ ALL {len(tests)} TEST SUITES PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
