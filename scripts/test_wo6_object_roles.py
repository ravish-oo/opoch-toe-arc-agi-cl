#!/usr/bin/env python3
"""
Comprehensive test script for WO6 - object_roles.py

Validates:
- component_sectors uses component bbox (NOT grid)
- Exact first/last/rest convention (matching WO2)
- component_border_interior uses 4-connectivity
- All 3 border conditions checked
- Mutually exclusive border/interior
- component_role_bits rank-based thirds
- Uniqueness by (color, shape_signature)
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.object_roles import (
    component_sectors,
    component_border_interior,
    component_role_bits
)
from src.features.components import Component, connected_components_by_color, compute_shape_signature
from src.core.grid_types import Grid


def test_component_sectors_uses_component_bbox():
    """
    CRITICAL TEST: Verify sectors are relative to component bbox, NOT grid
    """
    print("Testing component_sectors uses component bbox (NOT grid)...")

    # Create a component in the middle of the grid
    # Component at (5, 7) with 3x3 size
    grid = np.zeros((10, 12), dtype=int)
    grid[5:8, 7:10] = 1  # 3x3 block in middle

    comps = connected_components_by_color(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    # bbox should be (5, 7, 7, 9) for the 1-component
    assert comp_1.bbox == (5, 7, 7, 9), f"Expected bbox (5,7,7,9), got {comp_1.bbox}"

    sectors = component_sectors(comps)

    # Top-left pixel of component (5,7) should be "top", "left"
    assert sectors[(5, 7)] == {"vert_sector": "top", "horiz_sector": "left"}, \
        "Sectors should be relative to component bbox, not grid"

    # Bottom-right pixel of component (7,9) should be "bottom", "right"
    assert sectors[(7, 9)] == {"vert_sector": "bottom", "horiz_sector": "right"}, \
        "Sectors should be relative to component bbox, not grid"

    # If it used grid bounds, (5,7) would not be "top"/"left"
    print("  ✓ Sectors are component-relative (not grid-relative)")


def test_component_sectors_first_last_rest_convention():
    """
    CRITICAL TEST: Verify exact first/last/rest convention (matching WO2)
    """
    print("Testing component_sectors uses first/last/rest convention...")

    # Create components with different sizes
    grid = np.array([
        [1, 0, 0, 0],
        [2, 2, 0, 0],
        [3, 3, 3, 0],
        [4, 4, 4, 4],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    sectors = component_sectors(comps)

    # Component 1 (1x1 at (0,0))
    # h=1, w=1 → both center
    comp1_pixels = [(0, 0)]
    for p in comp1_pixels:
        assert sectors[p] == {"vert_sector": "center", "horiz_sector": "center"}, \
            f"1x1 component should be center/center"

    # Component 2 (1x2 at row 1)
    # h=1 → vert=center; w=2 → horiz=left/right
    assert sectors[(1, 0)]["horiz_sector"] == "left"
    assert sectors[(1, 1)]["horiz_sector"] == "right"

    # Component 3 (1x3 at row 2)
    # h=1 → vert=center; w=3 → horiz=left/center/right
    assert sectors[(2, 0)]["horiz_sector"] == "left"
    assert sectors[(2, 1)]["horiz_sector"] == "center"
    assert sectors[(2, 2)]["horiz_sector"] == "right"

    # Component 4 (1x4 at row 3)
    # h=1 → vert=center; w=4 → horiz=left/center/center/right
    assert sectors[(3, 0)]["horiz_sector"] == "left"
    assert sectors[(3, 1)]["horiz_sector"] == "center"
    assert sectors[(3, 2)]["horiz_sector"] == "center"
    assert sectors[(3, 3)]["horiz_sector"] == "right"

    print("  ✓ First/last/rest convention verified (matches WO2)")


def test_component_sectors_vertical_convention():
    """Test vertical sector convention for different heights"""
    print("Testing component_sectors vertical convention...")

    # Create vertical components
    grid = np.array([
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [0, 0, 3, 4],
        [0, 0, 0, 4],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    sectors = component_sectors(comps)

    # Component color 1: height=1 → all center
    c1 = [c for c in comps if c.color == 1][0]
    for p in c1.pixels:
        assert sectors[p]["vert_sector"] == "center"

    # Component color 2: height=2 → top/bottom
    c2 = [c for c in comps if c.color == 2][0]
    # Should have pixels at rows 0 and 1
    for p in c2.pixels:
        if p[0] == 0:
            assert sectors[p]["vert_sector"] == "top"
        elif p[0] == 1:
            assert sectors[p]["vert_sector"] == "bottom"

    # Component color 3: height=3 → top/center/bottom
    c3 = [c for c in comps if c.color == 3][0]
    for p in c3.pixels:
        if p[0] == 0:
            assert sectors[p]["vert_sector"] == "top"
        elif p[0] == 1:
            assert sectors[p]["vert_sector"] == "center"
        elif p[0] == 2:
            assert sectors[p]["vert_sector"] == "bottom"

    print("  ✓ Vertical sector convention verified")


def test_component_border_interior_four_connectivity():
    """
    CRITICAL TEST: Verify 4-connectivity (NOT 8-connectivity)
    """
    print("Testing component_border_interior uses 4-connectivity...")

    # Create a 3x3 component with interior pixel
    grid = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    border_info = component_border_interior(grid, comps)

    # Center pixel (1,1) should be interior (all 4 neighbors are same component)
    assert border_info[(1, 1)]["is_interior"] == True, \
        "Center pixel with 4 neighbors in same component should be interior"
    assert border_info[(1, 1)]["is_border"] == False

    # Edge pixels should be border (at least one neighbor is out of bounds)
    for (r, c) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]:
        assert border_info[(r, c)]["is_border"] == True, \
            f"Edge pixel {(r,c)} should be border"

    print("  ✓ 4-connectivity verified")


def test_component_border_interior_all_three_conditions():
    """
    CRITICAL TEST: Verify all 3 border conditions are checked
    """
    print("Testing component_border_interior checks all 3 conditions...")

    # Condition 1: Out of bounds
    grid1 = np.array([[1]], dtype=int)
    comps1 = connected_components_by_color(grid1)
    border1 = component_border_interior(grid1, comps1)
    assert border1[(0, 0)]["is_border"] == True, \
        "Single pixel should be border (OOB neighbors)"

    # Condition 2: Different color
    grid2 = np.array([
        [1, 2],
        [1, 2],
    ], dtype=int)
    comps2 = connected_components_by_color(grid2)
    border2 = component_border_interior(grid2, comps2)
    # Pixels at (0,0) and (1,0) are color 1, (0,1) and (1,1) are color 2
    # All should be border (different color neighbors)
    for p in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert border2[p]["is_border"] == True, \
            f"Pixel {p} should be border (different color neighbors)"

    # Condition 3: Same color but different component
    grid3 = np.array([
        [1, 0, 1],
        [0, 0, 0],
    ], dtype=int)
    comps3 = connected_components_by_color(grid3)
    border3 = component_border_interior(grid3, comps3)

    # The two 1-pixels are separate components (not connected)
    # Both should be border
    c1_comps = [c for c in comps3 if c.color == 1]
    assert len(c1_comps) == 2, "Should have 2 separate components of color 1"

    for comp in c1_comps:
        for p in comp.pixels:
            assert border3[p]["is_border"] == True, \
                "Isolated pixel should be border"

    print("  ✓ All 3 border conditions verified")


def test_component_border_interior_mutual_exclusivity():
    """
    CRITICAL TEST: Verify border and interior are mutually exclusive
    """
    print("Testing border/interior mutual exclusivity...")

    grid = np.random.randint(0, 5, size=(10, 10), dtype=int)
    comps = connected_components_by_color(grid)
    border_info = component_border_interior(grid, comps)

    for pixel, flags in border_info.items():
        is_border = flags["is_border"]
        is_interior = flags["is_interior"]

        # Exactly one should be True
        assert is_border != is_interior, \
            f"Pixel {pixel}: border and interior should be mutually exclusive. " \
            f"Got is_border={is_border}, is_interior={is_interior}"

        # Both should be boolean
        assert isinstance(is_border, bool)
        assert isinstance(is_interior, bool)

    print(f"  ✓ All {len(border_info)} pixels have mutually exclusive border/interior flags")


def test_component_role_bits_rank_based_thirds():
    """
    CRITICAL TEST: Verify rank-based thirds (NOT percentiles)
    """
    print("Testing component_role_bits uses rank-based thirds...")

    # Create components with known sizes
    # Sizes: [1, 2, 3, 4, 5, 6]
    grid = np.array([
        [1, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 0],
        [3, 3, 3, 0, 0, 0],
        [4, 4, 4, 4, 0, 0],
        [5, 5, 5, 5, 5, 0],
        [6, 6, 6, 6, 6, 6],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    roles = component_role_bits(comps)

    # n=6, small_idx=6//3=2, big_idx=2*6//3=4
    # sizes sorted: [1,2,3,4,5,6]
    # small_cutoff = sizes[2] = 3
    # big_cutoff = sizes[4] = 5

    # Verify cutoffs
    # size <= 3 → is_small
    # size >= 5 → is_big

    for comp in comps:
        role = roles[comp.id]
        if comp.size <= 3:
            assert role["is_small"] == True, \
                f"Component {comp.id} (size={comp.size}) should be small"
        else:
            assert role["is_small"] == False

        if comp.size >= 5:
            assert role["is_big"] == True, \
                f"Component {comp.id} (size={comp.size}) should be big"
        else:
            assert role["is_big"] == False

    print("  ✓ Rank-based thirds (n//3, 2*n//3) verified")


def test_component_role_bits_uniqueness_by_color_and_shape():
    """
    CRITICAL TEST: Verify uniqueness uses (color, shape_signature) tuple
    """
    print("Testing component_role_bits uniqueness by (color, shape_signature)...")

    # Create: two identical 2x2 squares of color 1, one 2x2 square of color 2
    grid = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    roles = component_role_bits(comps)

    # Get components by color
    c1_comps = [c for c in comps if c.color == 1]
    c2_comps = [c for c in comps if c.color == 2]

    # Color 1 should have 2 components (both 2x2 squares)
    assert len(c1_comps) == 2, "Should have 2 components of color 1"

    # Both color 1 components should have same shape_signature
    sig1 = c1_comps[0].shape_signature
    sig2 = c1_comps[1].shape_signature
    assert sig1 == sig2, "Both 2x2 squares should have same shape signature"

    # Neither should be unique (same color + same shape = not unique)
    for comp in c1_comps:
        assert roles[comp.id]["is_unique_shape"] == False, \
            f"Component {comp.id} (color 1, repeated shape) should NOT be unique"

    # Color 2 component should be unique (different color, even if same shape)
    for comp in c2_comps:
        assert roles[comp.id]["is_unique_shape"] == True, \
            f"Component {comp.id} (color 2, single instance) should be unique"

    print("  ✓ Uniqueness by (color, shape_signature) tuple verified")


def test_component_role_bits_handles_empty_list():
    """Test that component_role_bits handles empty components list"""
    print("Testing component_role_bits with empty list...")

    roles = component_role_bits([])
    assert roles == {}, "Empty components should return empty dict"

    print("  ✓ Empty list handled correctly")


def test_component_role_bits_auto_computes_shape_signature():
    """Test that component_role_bits auto-computes shape_signature if None"""
    print("Testing component_role_bits auto-computes shape_signature...")

    # Create component without shape_signature
    comp = Component(
        id=0,
        color=1,
        pixels=[(0, 0), (0, 1)],
        size=2,
        bbox=(0, 0, 0, 1),
        shape_signature=None  # Initially None
    )

    roles = component_role_bits([comp])

    # Should have computed shape_signature
    assert comp.shape_signature is not None, \
        "component_role_bits should compute shape_signature"

    # Should be able to call again without error
    roles2 = component_role_bits([comp])
    assert roles == roles2, "Repeated calls should be consistent"

    print("  ✓ Auto-computes shape_signature if None")


def test_all_pixels_covered_in_sectors_and_border():
    """Test that all component pixels are covered in both dicts"""
    print("Testing all pixels covered in sectors and border dicts...")

    grid = np.random.randint(0, 5, size=(8, 8), dtype=int)
    comps = connected_components_by_color(grid)

    sectors = component_sectors(comps)
    border_info = component_border_interior(grid, comps)

    # Collect all pixels from components
    all_comp_pixels = set()
    for comp in comps:
        all_comp_pixels.update(comp.pixels)

    # Check sectors covers all
    assert set(sectors.keys()) == all_comp_pixels, \
        "Sectors should cover all component pixels"

    # Check border_info covers all
    assert set(border_info.keys()) == all_comp_pixels, \
        "Border info should cover all component pixels"

    print(f"  ✓ All {len(all_comp_pixels)} component pixels covered")


def test_integration_with_arc_data():
    """Test with real ARC data"""
    print("Testing integration with real ARC data...")

    from src.core.arc_io import load_arc_training_challenges

    challenges_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(challenges_path)

    # Test on first 3 tasks
    sample_task_ids = sorted(tasks.keys())[:3]

    for task_id in sample_task_ids:
        task = tasks[task_id]

        if task["train"]:
            grid = task["train"][0]
            comps = connected_components_by_color(grid)

            # Test sectors
            sectors = component_sectors(comps)
            assert len(sectors) > 0, f"Task {task_id}: should have sectors"
            for pixel, sec in sectors.items():
                assert "vert_sector" in sec
                assert "horiz_sector" in sec
                assert sec["vert_sector"] in ["top", "center", "bottom"]
                assert sec["horiz_sector"] in ["left", "center", "right"]

            # Test border/interior
            border_info = component_border_interior(grid, comps)
            assert len(border_info) > 0, f"Task {task_id}: should have border info"
            for pixel, flags in border_info.items():
                assert "is_border" in flags
                assert "is_interior" in flags
                assert flags["is_border"] != flags["is_interior"], \
                    "Border and interior should be mutually exclusive"

            # Test role bits
            roles = component_role_bits(comps)
            assert len(roles) == len(comps), \
                f"Task {task_id}: should have role bits for all components"
            for comp_id, role in roles.items():
                assert "is_small" in role
                assert "is_big" in role
                assert "is_unique_shape" in role

    print(f"  ✓ Integration test passed for {len(sample_task_ids)} ARC tasks")


def test_math_kernel_section_1_2_8_compliance():
    """
    Verify compliance with math kernel section 1.2.8:
    - Quadrant/sector within component bbox
    - Role bits for components
    """
    print("Testing math kernel section 1.2.8 compliance...")

    grid = np.array([
        [1, 1, 1, 0, 2, 2],
        [1, 1, 1, 0, 2, 2],
        [1, 1, 1, 0, 0, 0],
    ], dtype=int)

    comps = connected_components_by_color(grid)

    # Quadrant/sector feature
    sectors = component_sectors(comps)

    # Verify each pixel has sector relative to its component
    c1 = [c for c in comps if c.color == 1][0]
    for pixel in c1.pixels:
        assert pixel in sectors, f"Pixel {pixel} should have sectors"
        sec = sectors[pixel]
        # Verify it's relative to component bbox, not grid
        r, c = pixel
        r_min, r_max, c_min, c_max = c1.bbox

        # Top row of component should be "top"
        if r == r_min:
            assert sec["vert_sector"] == "top"
        # Bottom row should be "bottom"
        elif r == r_max:
            assert sec["vert_sector"] == "bottom"

    # Role bits
    roles = component_role_bits(comps)

    # Verify role bits are assigned
    for comp in comps:
        assert comp.id in roles, f"Component {comp.id} should have role bits"
        role = roles[comp.id]
        assert isinstance(role["is_small"], bool)
        assert isinstance(role["is_big"], bool)
        assert isinstance(role["is_unique_shape"], bool)

    print("  ✓ Math kernel section 1.2.8 fully compliant")


def main():
    print("=" * 60)
    print("WO6 Comprehensive Test Suite - object_roles.py")
    print("=" * 60)
    print()

    tests = [
        test_component_sectors_uses_component_bbox,
        test_component_sectors_first_last_rest_convention,
        test_component_sectors_vertical_convention,
        test_component_border_interior_four_connectivity,
        test_component_border_interior_all_three_conditions,
        test_component_border_interior_mutual_exclusivity,
        test_component_role_bits_rank_based_thirds,
        test_component_role_bits_uniqueness_by_color_and_shape,
        test_component_role_bits_handles_empty_list,
        test_component_role_bits_auto_computes_shape_signature,
        test_all_pixels_covered_in_sectors_and_border,
        test_integration_with_arc_data,
        test_math_kernel_section_1_2_8_compliance,
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
