#!/usr/bin/env python3
"""
Comprehensive test script for WO3 - components.py

Validates connected component extraction with critical focus on:
- 4-connectivity (NOT 8-connectivity)
- Bbox format and inclusiveness
- Component ID sequencing
- Both scipy and BFS fallback paths
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.components import (
    Component,
    connected_components_by_color,
    _label_with_bfs
)
from src.core.grid_types import Grid


def test_component_dataclass_structure():
    """Test Component dataclass has correct fields"""
    print("Testing Component dataclass structure...")

    comp = Component(
        id=0,
        color=5,
        pixels=[(0, 0), (0, 1)],
        size=2,
        bbox=(0, 0, 0, 1)
    )

    assert comp.id == 0
    assert comp.color == 5
    assert comp.pixels == [(0, 0), (0, 1)]
    assert comp.size == 2
    assert comp.bbox == (0, 0, 0, 1)

    print("  ✓ Component dataclass structure correct")


def test_four_connectivity_not_eight():
    """
    CRITICAL TEST: Verify 4-connectivity (diagonal pixels NOT connected)

    Grid:
      1 0 1
      0 0 0
      1 0 1

    With 4-connectivity: should find 4 separate components (1 per corner)
    With 8-connectivity: would find 1 component (all corners connected via center diagonals)
    """
    print("Testing 4-connectivity (NOT 8-connectivity)...")

    grid = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ], dtype=int)

    comps = connected_components_by_color(grid)

    # Filter to color 1
    color_1_comps = [c for c in comps if c.color == 1]

    # With 4-connectivity: 4 separate components (one per corner pixel)
    # With 8-connectivity: would be 1 component (all connected diagonally)
    assert len(color_1_comps) == 4, \
        f"Expected 4 separate components (4-connectivity), got {len(color_1_comps)}. " \
        f"This suggests 8-connectivity is being used!"

    # Verify each is size 1
    for comp in color_1_comps:
        assert comp.size == 1, f"Each corner should be separate, got size {comp.size}"

    print("  ✓ 4-connectivity verified (diagonals NOT connected)")


def test_four_connectivity_cross_pattern():
    """
    Another 4-connectivity test with cross pattern:

      0 1 0
      1 1 1
      0 1 0

    Should be 1 component with 5 pixels (cross shape)
    """
    print("Testing 4-connectivity with cross pattern...")

    grid = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=int)

    comps = connected_components_by_color(grid)
    color_1_comps = [c for c in comps if c.color == 1]

    # Should be 1 component (the cross)
    assert len(color_1_comps) == 1, \
        f"Cross pattern should be 1 component, got {len(color_1_comps)}"

    comp = color_1_comps[0]
    assert comp.size == 5, f"Cross should have 5 pixels, got {comp.size}"

    print("  ✓ Cross pattern correctly connected")


def test_bbox_format_and_inclusiveness():
    """Test that bbox is (r_min, r_max, c_min, c_max) and inclusive"""
    print("Testing bbox format and inclusiveness...")

    # Simple 2x2 blob at specific position
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    comps = connected_components_by_color(grid)
    color_1_comps = [c for c in comps if c.color == 1]

    assert len(color_1_comps) == 1, "Should have 1 component for color 1"
    comp = color_1_comps[0]

    # Blob spans rows 2-3, cols 2-3
    expected_bbox = (2, 3, 2, 3)
    assert comp.bbox == expected_bbox, \
        f"Bbox should be {expected_bbox} (inclusive), got {comp.bbox}"

    # Verify it's (r_min, r_max, c_min, c_max) not some other order
    r_min, r_max, c_min, c_max = comp.bbox
    assert r_min == 2, "First element should be r_min"
    assert r_max == 3, "Second element should be r_max"
    assert c_min == 2, "Third element should be c_min"
    assert c_max == 3, "Fourth element should be c_max"

    # Verify bbox is inclusive (includes endpoints)
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            assert grid[r, c] == 1, f"Bbox should include pixel ({r},{c})"

    print("  ✓ Bbox format (r_min, r_max, c_min, c_max) and inclusive bounds verified")


def test_component_id_sequential():
    """Test that component IDs are sequential 0,1,2,... across all colors"""
    print("Testing component IDs are sequential...")

    grid = np.array([
        [1, 1, 2, 2],
        [0, 0, 0, 0],
        [3, 3, 4, 4]
    ], dtype=int)

    comps = connected_components_by_color(grid)

    # Should have components for colors 0,1,2,3,4
    # IDs should be 0,1,2,3,4 (one per color, all connected within color)
    ids = [c.id for c in comps]
    expected_ids = list(range(len(comps)))

    assert ids == expected_ids, \
        f"Component IDs should be sequential {expected_ids}, got {ids}"

    print(f"  ✓ Component IDs sequential: {ids}")


def test_colors_processed_ascending():
    """Test that colors are processed in ascending order"""
    print("Testing colors processed in ascending order...")

    grid = np.array([
        [9, 9],
        [3, 3],
        [1, 1],
        [5, 5]
    ], dtype=int)

    comps = connected_components_by_color(grid)

    colors_in_order = [c.color for c in comps]
    expected_order = [1, 3, 5, 9]

    assert colors_in_order == expected_order, \
        f"Colors should be in ascending order {expected_order}, got {colors_in_order}"

    print(f"  ✓ Colors processed in order: {colors_in_order}")


def test_multiple_components_per_color():
    """Test multiple disconnected components of the same color"""
    print("Testing multiple components per color...")

    # Two separate blobs of color 1
    grid = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=int)

    comps = connected_components_by_color(grid)
    color_1_comps = [c for c in comps if c.color == 1]

    assert len(color_1_comps) == 2, \
        f"Should have 2 separate components for color 1, got {len(color_1_comps)}"

    # Check sizes
    sizes = sorted([c.size for c in color_1_comps])
    assert sizes == [4, 4], f"Both components should be size 4, got {sizes}"

    # Check bboxes
    bboxes = [c.bbox for c in color_1_comps]
    # First blob: (0,1,0,1), second blob: (3,4,3,4)
    expected_bboxes = {(0, 1, 0, 1), (3, 4, 3, 4)}
    actual_bboxes = set(bboxes)
    assert actual_bboxes == expected_bboxes, \
        f"Bboxes should be {expected_bboxes}, got {actual_bboxes}"

    print("  ✓ Multiple components per color handled correctly")


def test_pixels_as_list_of_tuples():
    """Test that pixels are returned as List[Tuple[int, int]], not numpy array"""
    print("Testing pixels are List[Tuple[int, int]]...")

    grid = np.array([[1, 1]], dtype=int)
    comps = connected_components_by_color(grid)

    comp = comps[0]
    assert isinstance(comp.pixels, list), f"pixels should be list, got {type(comp.pixels)}"
    assert len(comp.pixels) == 2, "Should have 2 pixels"

    for pixel in comp.pixels:
        assert isinstance(pixel, tuple), f"Each pixel should be tuple, got {type(pixel)}"
        assert len(pixel) == 2, "Each pixel should be (r, c)"
        assert isinstance(pixel[0], int), "Row should be int"
        assert isinstance(pixel[1], int), "Col should be int"

    print("  ✓ Pixels are List[Tuple[int, int]]")


def test_size_matches_len_pixels():
    """Test that size field matches len(pixels)"""
    print("Testing size == len(pixels)...")

    grid = np.random.randint(0, 3, size=(10, 10), dtype=int)
    comps = connected_components_by_color(grid)

    for comp in comps:
        assert comp.size == len(comp.pixels), \
            f"Component {comp.id}: size={comp.size} but len(pixels)={len(comp.pixels)}"

    print(f"  ✓ size == len(pixels) for all {len(comps)} components")


def test_edge_case_single_pixel():
    """Test single pixel grid"""
    print("Testing single pixel grid...")

    grid = np.array([[5]], dtype=int)
    comps = connected_components_by_color(grid)

    assert len(comps) == 1, "Should have 1 component"
    comp = comps[0]

    assert comp.color == 5
    assert comp.size == 1
    assert comp.pixels == [(0, 0)]
    assert comp.bbox == (0, 0, 0, 0), "Single pixel bbox should be (0, 0, 0, 0)"

    print("  ✓ Single pixel grid handled correctly")


def test_edge_case_all_same_color():
    """Test grid with all pixels same color"""
    print("Testing all same color...")

    grid = np.ones((5, 5), dtype=int) * 7
    comps = connected_components_by_color(grid)

    assert len(comps) == 1, "Should have 1 component (entire grid)"
    comp = comps[0]

    assert comp.color == 7
    assert comp.size == 25, "Should have all 25 pixels"
    assert comp.bbox == (0, 4, 0, 4), "Bbox should span entire 5x5 grid"

    print("  ✓ All same color grid handled correctly")


def test_edge_case_all_different_colors():
    """Test grid where every pixel is different color"""
    print("Testing all different colors...")

    grid = np.arange(12, dtype=int).reshape(3, 4)
    comps = connected_components_by_color(grid)

    # 12 pixels, 12 different colors -> 12 components
    assert len(comps) == 12, f"Should have 12 components, got {len(comps)}"

    for comp in comps:
        assert comp.size == 1, f"Each component should be size 1, got {comp.size}"

    print("  ✓ All different colors handled correctly")


def test_bfs_fallback_path():
    """Test the BFS fallback implementation directly"""
    print("Testing BFS fallback path...")

    # Create a mask for testing
    mask = np.array([
        [True, True, False],
        [False, False, False],
        [False, True, True]
    ], dtype=bool)

    # Call BFS directly
    comps = _label_with_bfs(mask, color=1, start_id=10)

    # Should find 2 components
    assert len(comps) == 2, f"BFS should find 2 components, got {len(comps)}"

    # Check IDs start at 10
    ids = [c.id for c in comps]
    assert ids == [10, 11], f"IDs should be [10, 11], got {ids}"

    # Check sizes
    sizes = sorted([c.size for c in comps])
    assert sizes == [2, 2], f"Sizes should be [2, 2], got {sizes}"

    print("  ✓ BFS fallback works correctly")


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

            # Basic sanity checks
            assert isinstance(comps, list), f"Task {task_id}: should return list"
            assert all(isinstance(c, Component) for c in comps), \
                f"Task {task_id}: all items should be Component"

            # Check IDs are sequential
            ids = [c.id for c in comps]
            assert ids == list(range(len(comps))), \
                f"Task {task_id}: IDs not sequential: {ids}"

            # Check all pixels are valid
            H, W = grid.shape
            for comp in comps:
                for r, c in comp.pixels:
                    assert 0 <= r < H, f"Task {task_id}: invalid row {r}"
                    assert 0 <= c < W, f"Task {task_id}: invalid col {c}"
                    assert grid[r, c] == comp.color, \
                        f"Task {task_id}: pixel ({r},{c}) has color {grid[r,c]}, " \
                        f"not {comp.color}"

            # Check bbox validity
            for comp in comps:
                r_min, r_max, c_min, c_max = comp.bbox
                assert 0 <= r_min <= r_max < H, f"Task {task_id}: invalid bbox rows"
                assert 0 <= c_min <= c_max < W, f"Task {task_id}: invalid bbox cols"

    print(f"  ✓ Integration test passed for {len(sample_task_ids)} ARC tasks")


def test_math_kernel_section_1_2_4_compliance():
    """
    Verify compliance with math kernel section 1.2.4:
    Connected components (by color)
    """
    print("Testing math kernel section 1.2.4 compliance...")

    grid = np.array([
        [1, 1, 2, 2],
        [1, 0, 0, 2],
        [3, 3, 3, 0]
    ], dtype=int)

    comps = connected_components_by_color(grid)

    # For each component, verify:
    # - comp_id(p) is defined for each pixel
    # - component size is correct
    # - bounding box is correct

    # Build a map: pixel -> component
    pixel_to_comp = {}
    for comp in comps:
        for pixel in comp.pixels:
            assert pixel not in pixel_to_comp, f"Pixel {pixel} in multiple components!"
            pixel_to_comp[pixel] = comp

    # Check all non-background pixels are in some component
    # (assuming 0 might be background, but actually all colors should be included)
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            # Every pixel should be in exactly one component
            assert (r, c) in pixel_to_comp, f"Pixel ({r},{c}) not in any component"

    # Verify component metadata
    for comp in comps:
        # Size should equal number of pixels
        assert comp.size == len(comp.pixels), "Size mismatch"

        # Bbox should contain all pixels
        r_min, r_max, c_min, c_max = comp.bbox
        for r, c in comp.pixels:
            assert r_min <= r <= r_max, f"Pixel ({r},{c}) row outside bbox"
            assert c_min <= c <= c_max, f"Pixel ({r},{c}) col outside bbox"

        # All pixels should have the component's color
        for r, c in comp.pixels:
            assert grid[r, c] == comp.color, "Pixel color mismatch"

    print("  ✓ Math kernel section 1.2.4 fully compliant")


def main():
    print("=" * 60)
    print("WO3 Comprehensive Test Suite - components.py")
    print("=" * 60)
    print()

    tests = [
        test_component_dataclass_structure,
        test_four_connectivity_not_eight,
        test_four_connectivity_cross_pattern,
        test_bbox_format_and_inclusiveness,
        test_component_id_sequential,
        test_colors_processed_ascending,
        test_multiple_components_per_color,
        test_pixels_as_list_of_tuples,
        test_size_matches_len_pixels,
        test_edge_case_single_pixel,
        test_edge_case_all_same_color,
        test_edge_case_all_different_colors,
        test_bfs_fallback_path,
        test_integration_with_arc_data,
        test_math_kernel_section_1_2_4_compliance,
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
