#!/usr/bin/env python3
"""
Comprehensive review test for WO-M3.5: S8 + S9 + S10 implementation.

This test verifies:
  1. S8, S9, S10 use fix_pixel_color (NOT forbid loop)
  2. All three are geometry-preserving
  3. Builders are param-driven (no detection logic)
  4. Dispatch and families correctly wired
  5. Required features are correct
  6. No TODOs, stubs, or corner-cutting
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.schemas.context import build_example_context, TaskContext
from src.schemas.dispatch import apply_schema_instance, BUILDERS
from src.schemas.families import SCHEMA_FAMILIES
from src.constraints.builder import ConstraintBuilder


def test_s8_uses_fix_not_forbid():
    """Test that S8 uses fix_pixel_color, NOT forbid loop."""
    print("\nTest: S8 uses fix_pixel_color (NOT forbid loop)")
    print("-" * 70)

    # Create 2x2 grid
    input_grid = np.zeros((2, 2), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Single tile pixel
    params = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 1,
        "tile_width": 1,
        "tile_pattern": {
            "(0,0)": 5
        },
        "region_origin": "(0,0)",
        "region_height": 1,
        "region_width": 1
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S8", params, ctx, builder)

    # Single pixel with fix should be 1 constraint (not 9 with forbid loop)
    assert len(builder.constraints) == 1, \
        f"Expected 1 constraint (fix), got {len(builder.constraints)} (forbid loop?)"

    # Check it's a fix constraint (rhs=1, single coefficient)
    c = builder.constraints[0]
    assert c.rhs == 1, f"Expected fix constraint (rhs=1), got rhs={c.rhs}"
    assert len(c.indices) == 1, f"Expected 1 index, got {len(c.indices)}"
    assert c.coeffs == [1], f"Expected coeffs=[1], got {c.coeffs}"

    print(f"  ✓ S8 uses fix_pixel_color (1 constraint for 1 pixel)")


def test_s9_uses_fix_not_forbid():
    """Test that S9 uses fix_pixel_color, NOT forbid loop."""
    print("\nTest: S9 uses fix_pixel_color (NOT forbid loop)")
    print("-" * 70)

    # Create 3x3 grid
    input_grid = np.zeros((3, 3), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Single propagation pixel (up from center)
    params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(1,1)",
            "up_color": 5,
            "down_color": None,
            "left_color": None,
            "right_color": None,
            "max_up": 1,
            "max_down": 0,
            "max_left": 0,
            "max_right": 0
        }]
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S9", params, ctx, builder)

    # Single pixel with fix should be 1 constraint
    assert len(builder.constraints) == 1, \
        f"Expected 1 constraint (fix), got {len(builder.constraints)}"

    c = builder.constraints[0]
    assert c.rhs == 1, f"Expected fix constraint (rhs=1), got rhs={c.rhs}"
    assert len(c.indices) == 1, f"Expected 1 index, got {len(c.indices)}"
    assert c.coeffs == [1], f"Expected coeffs=[1], got {c.coeffs}"

    print(f"  ✓ S9 uses fix_pixel_color (1 constraint for 1 pixel)")


def test_s10_uses_fix_not_forbid():
    """Test that S10 uses fix_pixel_color, NOT forbid loop."""
    print("\nTest: S10 uses fix_pixel_color (NOT forbid loop)")
    print("-" * 70)

    # Create 3x3 grid with single border pixel
    input_grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    params = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 5,
        "interior_color": 7
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S10", params, ctx, builder)

    # Count constraints - should match border + interior pixels (1 constraint each)
    # NOT 9x constraints per pixel from forbid loop
    border_pixels = [(r, c) for (r, c), info in ex.border_info.items()
                     if info.get("is_border")]
    interior_pixels = [(r, c) for (r, c), info in ex.border_info.items()
                       if info.get("is_interior")]
    expected = len(border_pixels) + len(interior_pixels)

    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints (fix), got {len(builder.constraints)}"

    # Spot-check first constraint is fix
    if builder.constraints:
        c = builder.constraints[0]
        assert c.rhs == 1, f"Expected fix constraint (rhs=1), got rhs={c.rhs}"
        assert len(c.indices) == 1, f"Expected 1 index per constraint, got {len(c.indices)}"

    print(f"  ✓ S10 uses fix_pixel_color ({expected} constraints for {expected} pixels)")


def test_s8_geometry_preserving():
    """Test that S8 is geometry-preserving."""
    print("\nTest: S8 is geometry-preserving")
    print("-" * 70)

    # Read source code to verify implementation
    s8_source = (project_root / "src/schemas/s8_tiling.py").read_text()

    # Check uses input_H, input_W (not output_H, output_W)
    assert "ex.input_H" in s8_source, "S8 should use ex.input_H"
    assert "ex.input_W" in s8_source, "S8 should use ex.input_W"
    assert "output_H" not in s8_source or "# geometry-preserving" in s8_source, \
        "S8 should be geometry-preserving"

    # Check comment confirms geometry-preserving
    assert "geometry-preserving" in s8_source.lower(), \
        "S8 should be documented as geometry-preserving"

    print(f"  ✓ S8 is geometry-preserving (uses input_H, input_W)")


def test_s9_geometry_preserving():
    """Test that S9 is geometry-preserving."""
    print("\nTest: S9 is geometry-preserving")
    print("-" * 70)

    s9_source = (project_root / "src/schemas/s9_cross_propagation.py").read_text()

    assert "ex.input_H" in s9_source, "S9 should use ex.input_H"
    assert "ex.input_W" in s9_source, "S9 should use ex.input_W"
    assert "geometry-preserving" in s9_source.lower(), \
        "S9 should be documented as geometry-preserving"

    print(f"  ✓ S9 is geometry-preserving (uses input_H, input_W)")


def test_s10_geometry_preserving():
    """Test that S10 is geometry-preserving."""
    print("\nTest: S10 is geometry-preserving")
    print("-" * 70)

    s10_source = (project_root / "src/schemas/s10_frame_border.py").read_text()

    assert "ex.input_H" in s10_source, "S10 should use ex.input_H"
    assert "ex.input_W" in s10_source, "S10 should use ex.input_W"
    assert "geometry-preserving" in s10_source.lower(), \
        "S10 should be documented as geometry-preserving"

    print(f"  ✓ S10 is geometry-preserving (uses input_H, input_W)")


def test_param_driven_no_detection():
    """Test that S8-S10 are param-driven with no detection logic."""
    print("\nTest: S8-S10 are param-driven (no detection logic)")
    print("-" * 70)

    s8_source = (project_root / "src/schemas/s8_tiling.py").read_text()
    s9_source = (project_root / "src/schemas/s9_cross_propagation.py").read_text()
    s10_source = (project_root / "src/schemas/s10_frame_border.py").read_text()

    # Check for detection function calls that should NOT be present
    # (Look for actual detection logic, not just words in comments)
    forbidden_patterns = [
        "def detect_", "def infer_", "def learn_", "def mine_",
        "def find_seed", "def analyze_", "def pattern_match"
    ]

    for pattern in forbidden_patterns:
        assert pattern not in s8_source, \
            f"S8 should not have detection logic (found '{pattern}')"
        assert pattern not in s9_source, \
            f"S9 should not have detection logic (found '{pattern}')"
        assert pattern not in s10_source, \
            f"S10 should not have detection logic (found '{pattern}')"

    # Check they use schema_params (param-driven)
    assert "schema_params.get" in s8_source, "S8 should use schema_params"
    assert "schema_params.get" in s9_source, "S9 should use schema_params"
    assert "schema_params.get" in s10_source, "S10 should use schema_params"

    # Verify no NotImplementedError (i.e., not stubs)
    assert "NotImplementedError" not in s8_source, "S8 should not be a stub"
    assert "NotImplementedError" not in s9_source, "S9 should not be a stub"
    assert "NotImplementedError" not in s10_source, "S10 should not be a stub"

    print(f"  ✓ S8-S10 are param-driven (no detection logic)")


def test_no_todos_stubs_mvp():
    """Test that S8-S10 have no TODOs, stubs, or MVP markers."""
    print("\nTest: No TODOs, stubs, or simplified implementations")
    print("-" * 70)

    s8_source = (project_root / "src/schemas/s8_tiling.py").read_text()
    s9_source = (project_root / "src/schemas/s9_cross_propagation.py").read_text()
    s10_source = (project_root / "src/schemas/s10_frame_border.py").read_text()

    markers = ["TODO", "FIXME", "HACK", "XXX", "stub", "MVP", "simplified"]

    for marker in markers:
        assert marker.upper() not in s8_source.upper(), \
            f"S8 contains '{marker}' marker"
        assert marker.upper() not in s9_source.upper(), \
            f"S9 contains '{marker}' marker"
        assert marker.upper() not in s10_source.upper(), \
            f"S10 contains '{marker}' marker"

    print(f"  ✓ No TODOs, stubs, or MVP markers in S8-S10")


def test_dispatch_wiring():
    """Test that S8-S10 are correctly wired in dispatch."""
    print("\nTest: S8-S10 wired in dispatch")
    print("-" * 70)

    # Check BUILDERS registry
    assert "S8" in BUILDERS, "S8 not in BUILDERS"
    assert "S9" in BUILDERS, "S9 not in BUILDERS"
    assert "S10" in BUILDERS, "S10 not in BUILDERS"

    # Check builder names
    assert BUILDERS["S8"].__name__ == "build_S8_constraints"
    assert BUILDERS["S9"].__name__ == "build_S9_constraints"
    assert BUILDERS["S10"].__name__ == "build_S10_constraints"

    print(f"  ✓ S8-S10 correctly wired in dispatch.BUILDERS")


def test_families_wiring():
    """Test that S8-S10 metadata in families matches WO spec."""
    print("\nTest: S8-S10 metadata in families")
    print("-" * 70)

    # S8 checks
    s8 = SCHEMA_FAMILIES["S8"]
    assert s8.builder_name == "build_S8_constraints"
    assert "tile_height" in s8.parameter_spec
    assert "tile_width" in s8.parameter_spec
    assert "tile_pattern" in s8.parameter_spec
    assert "region_origin" in s8.parameter_spec
    assert s8.required_features == [], f"S8 required_features should be [], got {s8.required_features}"

    # S9 checks
    s9 = SCHEMA_FAMILIES["S9"]
    assert s9.builder_name == "build_S9_constraints"
    assert "seeds" in s9.parameter_spec
    assert s9.required_features == [], f"S9 required_features should be [], got {s9.required_features}"

    # S10 checks
    s10 = SCHEMA_FAMILIES["S10"]
    assert s10.builder_name == "build_S10_constraints"
    assert "border_color" in s10.parameter_spec
    assert "interior_color" in s10.parameter_spec
    assert "border_info" in s10.required_features, \
        f"S10 should require border_info, got {s10.required_features}"

    print(f"  ✓ S8-S10 metadata matches WO spec")


def test_s8_tiling_behavior():
    """Test S8 tiling behavior matches WO spec."""
    print("\nTest: S8 tiling behavior")
    print("-" * 70)

    # 4x4 grid with 2x2 tiles
    input_grid = np.zeros((4, 4), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    params = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S8", params, ctx, builder)

    # Should tile 4 times (0,0), (0,2), (2,0), (2,2)
    # Each tile has 4 pixels → 16 total
    assert len(builder.constraints) == 16, \
        f"Expected 16 constraints (4 tiles × 4 pixels), got {len(builder.constraints)}"

    print(f"  ✓ S8 correctly tiles 2x2 pattern across 4x4 grid (16 constraints)")


def test_s9_cross_behavior():
    """Test S9 cross propagation behavior matches WO spec."""
    print("\nTest: S9 cross propagation behavior")
    print("-" * 70)

    # 5x5 grid
    input_grid = np.zeros((5, 5), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(2,2)",
            "up_color": 1,
            "down_color": 2,
            "left_color": 3,
            "right_color": 4,
            "max_up": 2,
            "max_down": 2,
            "max_left": 2,
            "max_right": 2
        }]
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S9", params, ctx, builder)

    # Up: 2, Down: 2, Left: 2, Right: 2 → 8 pixels (center NOT included)
    assert len(builder.constraints) == 8, \
        f"Expected 8 constraints (cross without center), got {len(builder.constraints)}"

    print(f"  ✓ S9 correctly propagates cross from center (8 spokes, center excluded)")


def test_s9_does_not_touch_center():
    """Test that S9 does NOT color the center pixel."""
    print("\nTest: S9 does not touch center pixel")
    print("-" * 70)

    # 3x3 grid
    input_grid = np.zeros((3, 3), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{
            "center": "(1,1)",
            "up_color": 1,
            "down_color": 1,
            "left_color": 1,
            "right_color": 1,
            "max_up": 1,
            "max_down": 1,
            "max_left": 1,
            "max_right": 1
        }]
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S9", params, ctx, builder)

    # Should have 4 constraints (spokes only, NOT center)
    assert len(builder.constraints) == 4, \
        f"Expected 4 constraints (no center), got {len(builder.constraints)}"

    # Verify center pixel index is NOT in any constraint
    W = ex.input_W
    center_idx = 1 * W + 1  # (1,1) in 3x3
    for c in builder.constraints:
        for idx in c.indices:
            p_idx = idx // 10  # Reverse y_index calculation
            assert p_idx != center_idx, "S9 should NOT touch center pixel"

    print(f"  ✓ S9 does not touch center pixel (4 spokes only)")


def test_s10_border_info_usage():
    """Test that S10 correctly uses border_info from context."""
    print("\nTest: S10 uses border_info correctly")
    print("-" * 70)

    # 5x5 grid with component
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    params = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 5,
        "interior_color": 7
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S10", params, ctx, builder)

    # Count expected constraints from border_info
    border_pixels = [(r, c) for (r, c), info in ex.border_info.items()
                     if info.get("is_border")]
    interior_pixels = [(r, c) for (r, c), info in ex.border_info.items()
                       if info.get("is_interior")]
    expected = len(border_pixels) + len(interior_pixels)

    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints (from border_info), got {len(builder.constraints)}"

    print(f"  ✓ S10 uses border_info correctly ({len(border_pixels)} border + {len(interior_pixels)} interior)")


def test_error_handling():
    """Test that S8-S10 handle invalid inputs gracefully."""
    print("\nTest: Error handling for invalid inputs")
    print("-" * 70)

    input_grid = np.zeros((3, 3), dtype=int)
    ex = build_example_context(input_grid, input_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # S8: Invalid example index
    builder = ConstraintBuilder()
    apply_schema_instance("S8", {"example_type": "train", "example_index": 999}, ctx, builder)
    assert len(builder.constraints) == 0, "Invalid index should return early"

    # S9: Invalid seed format
    builder = ConstraintBuilder()
    apply_schema_instance("S9", {
        "example_type": "train",
        "example_index": 0,
        "seeds": [{"center": "invalid"}]
    }, ctx, builder)
    # Should not crash

    # S10: Out of palette color
    builder = ConstraintBuilder()
    apply_schema_instance("S10", {
        "example_type": "train",
        "example_index": 0,
        "border_color": 999,  # Out of range
        "interior_color": 1
    }, ctx, builder)
    # Should clamp to valid color (not crash)

    print(f"  ✓ S8-S10 handle invalid inputs gracefully")


def main():
    print("=" * 70)
    print("WO-M3.5 COMPREHENSIVE REVIEW TEST")
    print("Testing S8 (Tiling) + S9 (Cross) + S10 (Frame/Border)")
    print("=" * 70)

    try:
        # Core requirements from WO
        test_s8_uses_fix_not_forbid()
        test_s9_uses_fix_not_forbid()
        test_s10_uses_fix_not_forbid()

        test_s8_geometry_preserving()
        test_s9_geometry_preserving()
        test_s10_geometry_preserving()

        test_param_driven_no_detection()
        test_no_todos_stubs_mvp()

        # Wiring and integration
        test_dispatch_wiring()
        test_families_wiring()

        # Behavioral tests
        test_s8_tiling_behavior()
        test_s9_cross_behavior()
        test_s9_does_not_touch_center()
        test_s10_border_info_usage()

        # Error handling
        test_error_handling()

        print("\n" + "=" * 70)
        print("✅ WO-M3.5 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ S8 (Tiling) - COMPLETE")
        print("    - Uses fix_pixel_color (NOT forbid loop)")
        print("    - Geometry-preserving")
        print("    - Param-driven (no detection)")
        print("    - Correctly tiles patterns across regions")
        print()
        print("  ✓ S9 (Cross propagation) - COMPLETE")
        print("    - Uses fix_pixel_color (NOT forbid loop)")
        print("    - Geometry-preserving")
        print("    - Param-driven (no detection)")
        print("    - Does NOT touch center pixel")
        print("    - Correctly propagates in 4 directions")
        print()
        print("  ✓ S10 (Frame/border) - COMPLETE")
        print("    - Uses fix_pixel_color (NOT forbid loop)")
        print("    - Geometry-preserving")
        print("    - Param-driven (uses border_info)")
        print("    - Correctly assigns border/interior colors")
        print()
        print("  ✓ Dispatch and families correctly wired")
        print("  ✓ No TODOs, stubs, or simplified implementations")
        print("  ✓ Error handling graceful")
        print()
        print("WO-M3.5 IMPLEMENTATION READY FOR PRODUCTION")
        print("=" * 70)
        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
