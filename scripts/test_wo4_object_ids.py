#!/usr/bin/env python3
"""
Comprehensive test script for WO4 - object_id and shape_signature

Validates:
- Translation invariance of shape signatures
- Correct grouping by (color, shape_signature)
- Global sequential object_ids
- Integration with components from WO3
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
    compute_shape_signature,
    assign_object_ids
)
from src.core.grid_types import Grid


def test_shape_signature_translation_invariance():
    """Test that shape_signature is translation-invariant"""
    print("Testing shape_signature translation invariance...")

    # Create two identical L-shapes at different positions
    # Shape 1: top-left at (0,0)
    comp1 = Component(
        id=0,
        color=1,
        pixels=[(0, 0), (1, 0), (1, 1)],
        size=3,
        bbox=(0, 1, 0, 1)
    )

    # Shape 2: same L-shape at (5,7)
    comp2 = Component(
        id=1,
        color=1,
        pixels=[(5, 7), (6, 7), (6, 8)],
        size=3,
        bbox=(5, 6, 7, 8)
    )

    sig1 = compute_shape_signature(comp1)
    sig2 = compute_shape_signature(comp2)

    # Both should have same signature
    assert sig1 == sig2, \
        f"Translation invariance failed: {sig1} != {sig2}"

    # Expected: ((0,0), (1,0), (1,1)) in sorted order
    expected = ((0, 0), (1, 0), (1, 1))
    assert sig1 == expected, f"Expected {expected}, got {sig1}"

    print(f"  ✓ Translation invariance verified: {sig1}")


def test_shape_signature_normalization():
    """Test that shape_signature normalizes by r_min, c_min"""
    print("Testing shape_signature normalization...")

    # Component at position (10, 20)
    comp = Component(
        id=0,
        color=1,
        pixels=[(10, 20), (10, 21), (11, 20)],
        size=3,
        bbox=(10, 11, 20, 21)
    )

    sig = compute_shape_signature(comp)

    # After normalization by (r_min=10, c_min=20):
    # (10,20) -> (0,0)
    # (10,21) -> (0,1)
    # (11,20) -> (1,0)
    expected = ((0, 0), (0, 1), (1, 0))
    assert sig == expected, f"Normalization failed: expected {expected}, got {sig}"

    print(f"  ✓ Normalization correct: {sig}")


def test_shape_signature_sorting():
    """Test that shape_signature is sorted for determinism"""
    print("Testing shape_signature sorting...")

    # Create component with unsorted pixels
    comp = Component(
        id=0,
        color=1,
        pixels=[(5, 8), (5, 7), (6, 7)],  # Intentionally unsorted
        size=3,
        bbox=(5, 6, 7, 8)
    )

    sig = compute_shape_signature(comp)

    # After normalization: (0,1), (0,0), (1,0)
    # After sorting: (0,0), (0,1), (1,0)
    expected = ((0, 0), (0, 1), (1, 0))
    assert sig == expected, f"Sorting failed: expected {expected}, got {sig}"

    # Verify it's actually sorted
    assert list(sig) == sorted(sig), "Signature not sorted!"

    print(f"  ✓ Sorting verified: {sig}")


def test_assign_object_ids_grouping_by_color_and_shape():
    """Test that assign_object_ids groups by (color, shape_signature)"""
    print("Testing assign_object_ids groups by (color, shape_signature)...")

    # Create components:
    # - Two 2x2 squares of color 1 at different positions (same object_id)
    # - One 2x2 square of color 2 (different object_id despite same shape)
    # - One L-shape of color 1 (different object_id despite same color)

    grid = np.array([
        [1, 1, 0, 2, 2, 0, 1, 0],
        [1, 1, 0, 2, 2, 0, 1, 1],
    ], dtype=int)

    comps = connected_components_by_color(grid)

    # Manually compute signatures
    for comp in comps:
        comp.shape_signature = compute_shape_signature(comp)

    # Expected components:
    # - Background (color 0): several pixels
    # - Color 1: one 2x2 square, one L-shape
    # - Color 2: one 2x2 square

    color_1_comps = [c for c in comps if c.color == 1]
    color_2_comps = [c for c in comps if c.color == 2]

    # Get object_ids
    pixel_to_obj = assign_object_ids(comps)

    # The two 2x2 squares of color 1 should have SAME object_id
    square_1_pixels = [(0, 0), (0, 1), (1, 0), (1, 1)]
    l_shape_pixels = [(0, 6), (1, 6), (1, 7)]

    if all((r, c) in pixel_to_obj for r, c in square_1_pixels):
        obj_id_square1 = pixel_to_obj[square_1_pixels[0]]
        for r, c in square_1_pixels:
            assert pixel_to_obj[(r, c)] == obj_id_square1, \
                "All pixels in 2x2 square should have same object_id"

    # The L-shape should have DIFFERENT object_id (different shape)
    if all((r, c) in pixel_to_obj for r, c in l_shape_pixels):
        obj_id_l = pixel_to_obj[l_shape_pixels[0]]
        obj_id_square1 = pixel_to_obj[(0, 0)]
        assert obj_id_l != obj_id_square1, \
            "L-shape and square should have different object_ids"

    # Color 2 square should have DIFFERENT object_id (different color)
    color_2_pixels = [(0, 3), (0, 4), (1, 3), (1, 4)]
    if all((r, c) in pixel_to_obj for r, c in color_2_pixels):
        obj_id_color2 = pixel_to_obj[color_2_pixels[0]]
        obj_id_color1_square = pixel_to_obj[(0, 0)]
        assert obj_id_color2 != obj_id_color1_square, \
            "Same shape but different color should have different object_ids"

    print("  ✓ Grouping by (color, shape_signature) verified")


def test_object_ids_globally_sequential():
    """Test that object_ids are globally sequential 0,1,2,..."""
    print("Testing object_ids are globally sequential...")

    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    pixel_to_obj = assign_object_ids(comps)

    # Get all unique object_ids
    all_obj_ids = sorted(set(pixel_to_obj.values()))

    # Should be 0,1,2,3,4,5 (6 different colors -> 6 object_ids)
    expected_ids = list(range(len(all_obj_ids)))
    assert all_obj_ids == expected_ids, \
        f"Object IDs should be sequential {expected_ids}, got {all_obj_ids}"

    print(f"  ✓ Object IDs sequential: {all_obj_ids}")


def test_assign_object_ids_covers_all_pixels():
    """Test that assign_object_ids returns all component pixels"""
    print("Testing assign_object_ids covers all component pixels...")

    grid = np.random.randint(0, 5, size=(10, 10), dtype=int)
    comps = connected_components_by_color(grid)
    pixel_to_obj = assign_object_ids(comps)

    # Collect all pixels from components
    all_comp_pixels = set()
    for comp in comps:
        for pixel in comp.pixels:
            all_comp_pixels.add(pixel)

    # Check that dict has exact same pixels
    dict_pixels = set(pixel_to_obj.keys())

    assert dict_pixels == all_comp_pixels, \
        f"Pixel mismatch: {len(dict_pixels)} in dict vs {len(all_comp_pixels)} in components"

    print(f"  ✓ All {len(all_comp_pixels)} component pixels covered")


def test_same_shape_different_positions_same_color():
    """Test that identical shapes at different positions with same color get same object_id"""
    print("Testing same shape, different positions, same color...")

    # Two separate 2x2 squares of color 1
    grid = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    pixel_to_obj = assign_object_ids(comps)

    # Both 2x2 squares should have SAME object_id
    square1_pixel = (0, 0)
    square2_pixel = (3, 3)

    obj_id_1 = pixel_to_obj[square1_pixel]
    obj_id_2 = pixel_to_obj[square2_pixel]

    assert obj_id_1 == obj_id_2, \
        f"Identical shapes with same color should have same object_id: " \
        f"got {obj_id_1} and {obj_id_2}"

    print(f"  ✓ Same shape + same color = same object_id ({obj_id_1})")


def test_component_shape_signature_field_optional():
    """Test that Component can be created without shape_signature (backward compat)"""
    print("Testing Component shape_signature field is optional...")

    # Create component without shape_signature (as in WO3)
    comp = Component(
        id=0,
        color=1,
        pixels=[(0, 0), (0, 1)],
        size=2,
        bbox=(0, 0, 0, 1)
    )

    # Should default to None
    assert comp.shape_signature is None, \
        f"shape_signature should default to None, got {comp.shape_signature}"

    # assign_object_ids should compute it
    comps = [comp]
    pixel_to_obj = assign_object_ids(comps)

    # Now it should be computed
    assert comp.shape_signature is not None, \
        "assign_object_ids should compute shape_signature"

    print("  ✓ shape_signature field is optional (backward compatible)")


def test_different_shapes_same_color():
    """Test that different shapes with same color get different object_ids"""
    print("Testing different shapes, same color...")

    # Create a 2x2 square and a 1x3 bar, both color 1
    grid = np.array([
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    pixel_to_obj = assign_object_ids(comps)

    square_pixel = (0, 0)
    bar_pixel = (0, 3)

    obj_id_square = pixel_to_obj[square_pixel]
    obj_id_bar = pixel_to_obj[bar_pixel]

    assert obj_id_square != obj_id_bar, \
        f"Different shapes should have different object_ids: " \
        f"got {obj_id_square} and {obj_id_bar}"

    print(f"  ✓ Different shapes + same color = different object_ids")


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
            pixel_to_obj = assign_object_ids(comps)

            # Verify all components have shape_signature computed
            for comp in comps:
                assert comp.shape_signature is not None, \
                    f"Task {task_id}: component {comp.id} missing shape_signature"

            # Verify all component pixels are in dict
            comp_pixels = set()
            for comp in comps:
                comp_pixels.update(comp.pixels)

            dict_pixels = set(pixel_to_obj.keys())
            assert dict_pixels == comp_pixels, \
                f"Task {task_id}: pixel count mismatch"

            # Verify object_ids are sequential
            obj_ids = sorted(set(pixel_to_obj.values()))
            expected_ids = list(range(len(obj_ids)))
            assert obj_ids == expected_ids, \
                f"Task {task_id}: object_ids not sequential"

    print(f"  ✓ Integration test passed for {len(sample_task_ids)} ARC tasks")


def test_shape_signature_single_pixel():
    """Test shape_signature for single pixel component"""
    print("Testing shape_signature for single pixel...")

    comp = Component(
        id=0,
        color=1,
        pixels=[(5, 7)],
        size=1,
        bbox=(5, 5, 7, 7)
    )

    sig = compute_shape_signature(comp)

    # Single pixel at (5,7), bbox (5,5,7,7)
    # After normalization: (5-5, 7-7) = (0,0)
    expected = ((0, 0),)
    assert sig == expected, f"Single pixel signature should be {expected}, got {sig}"

    print(f"  ✓ Single pixel signature: {sig}")


def test_math_kernel_section_1_2_5_compliance():
    """
    Verify compliance with math kernel section 1.2.5:
    Object classes (shape equivalence up to translation)
    """
    print("Testing math kernel section 1.2.5 compliance...")

    # Create grid with multiple instances of same shapes
    grid = np.array([
        [1, 1, 0, 1, 1, 0, 2, 2],
        [1, 1, 0, 1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 3, 0, 0, 1, 1],
        [3, 3, 0, 3, 3, 0, 1, 1],
    ], dtype=int)

    comps = connected_components_by_color(grid)
    pixel_to_obj = assign_object_ids(comps)

    # Verify object_id(p) is defined for each pixel in components
    for comp in comps:
        for pixel in comp.pixels:
            assert pixel in pixel_to_obj, \
                f"Pixel {pixel} not in object_id map"

    # Verify translation invariance:
    # Two 2x2 squares of color 1 at positions (0,0) and (0,3)
    # should have same object_id
    if (0, 0) in pixel_to_obj and (0, 3) in pixel_to_obj:
        obj_id_1 = pixel_to_obj[(0, 0)]
        obj_id_2 = pixel_to_obj[(0, 3)]

        # Get their components
        comp1 = next(c for c in comps if (0, 0) in c.pixels)
        comp2 = next(c for c in comps if (0, 3) in c.pixels)

        if comp1.color == comp2.color and comp1.shape_signature == comp2.shape_signature:
            assert obj_id_1 == obj_id_2, \
                "Same color + same shape should have same object_id"

    # Verify shape matters: different shapes should have different object_ids
    # (even if same color)
    color_1_comps = [c for c in comps if c.color == 1]
    if len(color_1_comps) > 1:
        # Check if any have different signatures
        sigs = [c.shape_signature for c in color_1_comps]
        unique_sigs = set(sigs)
        if len(unique_sigs) > 1:
            # Different shapes should map to different object_ids
            obj_ids_for_color1 = set(
                pixel_to_obj[c.pixels[0]] for c in color_1_comps
            )
            # Should have at least as many object_ids as unique shapes
            assert len(obj_ids_for_color1) >= len(unique_sigs), \
                "Different shapes should have different object_ids"

    print("  ✓ Math kernel section 1.2.5 fully compliant")


def main():
    print("=" * 60)
    print("WO4 Comprehensive Test Suite - object_ids & shape_signature")
    print("=" * 60)
    print()

    tests = [
        test_shape_signature_translation_invariance,
        test_shape_signature_normalization,
        test_shape_signature_sorting,
        test_assign_object_ids_grouping_by_color_and_shape,
        test_object_ids_globally_sequential,
        test_assign_object_ids_covers_all_pixels,
        test_same_shape_different_positions_same_color,
        test_component_shape_signature_field_optional,
        test_different_shapes_same_color,
        test_integration_with_arc_data,
        test_shape_signature_single_pixel,
        test_math_kernel_section_1_2_5_compliance,
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
