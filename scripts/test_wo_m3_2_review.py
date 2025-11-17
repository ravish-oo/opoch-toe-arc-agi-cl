#!/usr/bin/env python3
"""
WO-M3.2 Comprehensive Review Test

Tests all aspects of WO-M3.2 implementation:
  - S3 schema builder (band/stripe laws)
  - S4 schema builder (residue-class coloring)
  - Catalog infrastructure (types.py)
  - Kernel runner (kernel.py)
  - Dispatch wiring

Critical checks:
  - S4 uses fix_pixel_color (NOT forbid loop)
  - No feature computation in builders (param-driven only)
  - Kernel has minimal scope (NO solver)
  - Correct param structures
  - Geometry-preserving (use input_H, input_W)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.schemas.context import build_example_context, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance, BUILDERS
from src.schemas.families import SCHEMA_FAMILIES
from src.catalog.types import SchemaInstance, TaskLawConfig
from src.runners.kernel import solve_arc_task


def test_catalog_types_structure():
    """Test catalog/types.py has correct dataclass structure."""
    print("Testing catalog/types.py structure...")

    # Test SchemaInstance
    instance = SchemaInstance(
        family_id="S3",
        params={"row_classes": [[0, 1]]}
    )
    assert instance.family_id == "S3"
    assert instance.params["row_classes"] == [[0, 1]]

    # Test TaskLawConfig
    config = TaskLawConfig(schema_instances=[instance])
    assert len(config.schema_instances) == 1
    assert config.schema_instances[0].family_id == "S3"

    print("  ✓ SchemaInstance and TaskLawConfig dataclasses correct")


def test_s3_param_structure():
    """Test S3 accepts correct param structure."""
    print("Testing S3 param structure...")

    input_grid = np.array([[1, 2], [3, 4], [1, 2]], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Test all param keys
    params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [[0, 2]],
        "col_classes": [],
        "col_period_K": None,
        "row_period_K": None
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", params, ctx, builder)

    # Should add constraints for tying rows 0 and 2
    # 2 columns × 5 colors = 10 constraints
    assert len(builder.constraints) == 10

    print("  ✓ S3 accepts correct param structure")
    print(f"    Added {len(builder.constraints)} constraints for row bands")


def test_s3_row_band_ties():
    """Test S3 row band ties work correctly."""
    print("Testing S3 row band ties...")

    # 4x3 grid
    input_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 2, 3]
    ], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Tie rows 0, 2, 3 (three rows in same band)
    params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [[0, 2, 3]],
        "col_classes": [],
        "col_period_K": None,
        "row_period_K": None
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", params, ctx, builder)

    # Three rows in band: (0,2), (0,3), (2,3) = 3 pairs
    # 3 pairs × 3 columns × 10 colors = 90 constraints
    expected = 3 * 3 * 10
    assert len(builder.constraints) == expected

    print(f"  ✓ S3 row bands: {len(builder.constraints)} constraints")
    print(f"    (3 row pairs × 3 columns × 10 colors)")


def test_s3_col_band_ties():
    """Test S3 column band ties work correctly."""
    print("Testing S3 column band ties...")

    # 3x4 grid
    input_grid = np.array([
        [1, 2, 3, 1],
        [4, 5, 6, 4],
        [7, 8, 9, 7]
    ], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Tie columns 0 and 3
    params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [],
        "col_classes": [[0, 3]],
        "col_period_K": None,
        "row_period_K": None
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", params, ctx, builder)

    # 1 col pair × 3 rows × 10 colors = 30 constraints
    expected = 1 * 3 * 10
    assert len(builder.constraints) == expected

    print(f"  ✓ S3 col bands: {len(builder.constraints)} constraints")
    print(f"    (1 col pair × 3 rows × 10 colors)")


def test_s3_col_periodicity():
    """Test S3 column periodicity (col_period_K)."""
    print("Testing S3 column periodicity...")

    # 3x6 grid
    input_grid = np.array([
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [5, 6, 5, 6, 5, 6]
    ], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Column period = 2 (tie col 0 to col 2 to col 4, etc.)
    params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [],
        "col_classes": [],
        "col_period_K": 2,
        "row_period_K": None
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", params, ctx, builder)

    # For each row, tie: (c, c+2) for c in range(4)
    # Row 0: (0,2), (1,3), (2,4), (3,5) = 4 pairs
    # 3 rows × 4 pairs × 10 colors = 120 constraints
    expected = 3 * 4 * 10
    assert len(builder.constraints) == expected

    print(f"  ✓ S3 col periodicity: {len(builder.constraints)} constraints")
    print(f"    (3 rows × 4 col pairs × 10 colors)")


def test_s3_row_periodicity():
    """Test S3 row periodicity (row_period_K)."""
    print("Testing S3 row periodicity...")

    # 6x3 grid
    input_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Row period = 2 (tie row 0 to row 2 to row 4, etc.)
    params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [],
        "col_classes": [],
        "col_period_K": None,
        "row_period_K": 2
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", params, ctx, builder)

    # For each col, tie: (r, r+2) for r in range(4)
    # Col 0: (0,2), (1,3), (2,4), (3,5) = 4 pairs
    # 3 cols × 4 pairs × 10 colors = 120 constraints
    expected = 3 * 4 * 10
    assert len(builder.constraints) == expected

    print(f"  ✓ S3 row periodicity: {len(builder.constraints)} constraints")
    print(f"    (3 cols × 4 row pairs × 10 colors)")


def test_s4_param_structure():
    """Test S4 accepts correct param structure."""
    print("Testing S4 param structure...")

    input_grid = np.array([[1, 2], [3, 4]], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Test all param keys
    params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {"0": 1, "1": 3}
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", params, ctx, builder)

    # 2x2 grid = 4 pixels, one fix per pixel
    assert len(builder.constraints) == 4

    print("  ✓ S4 accepts correct param structure")
    print(f"    Added {len(builder.constraints)} constraints")


def test_s4_uses_fix_not_forbid():
    """CRITICAL: Test S4 uses fix_pixel_color, NOT forbid loop."""
    print("Testing S4 uses fix_pixel_color (NOT forbid)...")

    input_grid = np.array([[1, 2], [3, 4]], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {"0": 1, "1": 3}
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", params, ctx, builder)

    # Should have 4 constraints (one fix per pixel)
    # NOT 4 * (C-1) = 16 constraints (which would be forbid loop)
    assert len(builder.constraints) == 4, \
        f"S4 should use fix (4 constraints), not forbid loop ({4 * (5-1)} constraints)"

    # Verify constraint structure: each should be a unit constraint (rhs=1)
    for c in builder.constraints:
        assert c.rhs == 1, "fix constraint should have rhs=1"
        assert len(c.indices) == 1, "fix constraint should have 1 index"
        assert c.coeffs == [1], "fix constraint should have coeff=1"

    print("  ✓ S4 correctly uses fix_pixel_color")
    print("    (4 fix constraints, NOT 16 forbid constraints)")


def test_s4_col_residue_mapping():
    """Test S4 column residue mapping."""
    print("Testing S4 column residue mapping...")

    # 3x4 grid
    input_grid = np.array([
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Even cols → color 2, odd cols → color 4
    params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {
            "0": 2,  # even cols
            "1": 4   # odd cols
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", params, ctx, builder)

    # 3x4 grid = 12 pixels, one fix per pixel
    assert len(builder.constraints) == 12

    print(f"  ✓ S4 col residue mapping: {len(builder.constraints)} constraints")


def test_s4_row_residue_mapping():
    """Test S4 row residue mapping."""
    print("Testing S4 row residue mapping...")

    # 4x3 grid
    input_grid = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Even rows → color 2, odd rows → color 4
    params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "row",
        "K": 2,
        "residue_to_color": {
            "0": 2,  # even rows
            "1": 4   # odd rows
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", params, ctx, builder)

    # 4x3 grid = 12 pixels, one fix per pixel
    assert len(builder.constraints) == 12

    print(f"  ✓ S4 row residue mapping: {len(builder.constraints)} constraints")


def test_s4_partial_mapping():
    """Test S4 partial residue mapping (some residues unmapped)."""
    print("Testing S4 partial mapping...")

    # 4x4 grid
    input_grid = np.zeros((4, 4), dtype=int)
    output_grid = input_grid.copy()
    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Map only residues 0 and 2 (leave 1 and 3 unconstrained)
    params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 4,
        "residue_to_color": {
            "0": 1,  # col 0
            "2": 3   # col 2
            # cols 1 and 3 not mapped
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", params, ctx, builder)

    # Only cols 0 and 2 mapped: 4 rows × 2 cols = 8 constraints
    assert len(builder.constraints) == 8

    print(f"  ✓ S4 partial mapping: {len(builder.constraints)} constraints")
    print("    (only cols 0,2 mapped, cols 1,3 unconstrained)")


def test_kernel_integration():
    """Test kernel.py integrates S3/S4 correctly."""
    print("Testing kernel.py integration...")

    task_id = "00576224"

    # Create config with S3 and S4
    config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S3",
            params={
                "example_type": "train",
                "example_index": 0,
                "row_classes": [[0, 1]],
                "col_classes": [],
                "col_period_K": None,
                "row_period_K": None
            }
        ),
        SchemaInstance(
            family_id="S4",
            params={
                "example_type": "train",
                "example_index": 0,
                "axis": "col",
                "K": 2,
                "residue_to_color": {"0": 1, "1": 3}
            }
        )
    ])

    # Run kernel
    builder = solve_arc_task(task_id, config)

    # Should have constraints from both S3 and S4
    assert len(builder.constraints) > 0

    print(f"  ✓ Kernel integration: {len(builder.constraints)} constraints")
    print("    (from S3 + S4 schemas)")


def test_kernel_no_solver():
    """Test kernel.py has minimal scope (NO solver)."""
    print("Testing kernel.py has NO solver...")

    # Read kernel.py source
    kernel_path = project_root / "src/runners/kernel.py"
    source = kernel_path.read_text()

    # Should NOT import solver libraries
    forbidden_imports = [
        "scipy.optimize",
        "cvxpy",
        "gurobipy",
        "pulp"
    ]

    for imp in forbidden_imports:
        assert imp not in source, \
            f"kernel.py should NOT import '{imp}' (solver integration comes later)"

    # Should NOT call solver functions
    forbidden_calls = [
        "linprog(",
        "minimize(",
        "solve(",
        ".solve()"
    ]

    for call in forbidden_calls:
        assert call not in source, \
            f"kernel.py should NOT call '{call}' (solver integration comes later)"

    # Should return ConstraintBuilder (not grids)
    assert "return builder" in source

    # Should NOT return predicted grids
    assert "return " in source
    assert "predicted_grid" not in source
    assert "output_grid" not in source.split("return")[-1]  # Check what's being returned

    print("  ✓ kernel.py has minimal scope (NO solver)")


def test_no_feature_computation():
    """Test S3/S4 do NOT compute features (param-driven only)."""
    print("Testing S3/S4 are param-driven (NO feature computation)...")

    # Read S3 source
    s3_path = project_root / "src/schemas/s3_bands.py"
    s3_source = s3_path.read_text()

    # Read S4 source
    s4_path = project_root / "src/schemas/s4_residue_color.py"
    s4_source = s4_path.read_text()

    # Should NOT contain feature mining keywords
    forbidden_keywords = [
        "mine_",
        "detect_",
        "find_pattern",
        "discover_",
        "infer_"
    ]

    for keyword in forbidden_keywords:
        assert keyword not in s3_source, \
            f"S3 should NOT contain '{keyword}' (param-driven, not feature mining)"
        assert keyword not in s4_source, \
            f"S4 should NOT contain '{keyword}' (param-driven, not feature mining)"

    print("  ✓ S3/S4 are param-driven (NO feature mining)")


def test_dispatch_wiring():
    """Test S3/S4 are correctly wired into dispatch."""
    print("Testing dispatch/families wiring...")

    # Check BUILDERS has S3/S4
    assert "S3" in BUILDERS
    assert "S4" in BUILDERS

    # Check builder functions are callable
    assert callable(BUILDERS["S3"])
    assert callable(BUILDERS["S4"])

    # Check families match
    assert "S3" in SCHEMA_FAMILIES
    assert "S4" in SCHEMA_FAMILIES
    assert SCHEMA_FAMILIES["S3"].builder_name == "build_S3_constraints"
    assert SCHEMA_FAMILIES["S4"].builder_name == "build_S4_constraints"

    # Check builder function names match
    assert BUILDERS["S3"].__name__ == "build_S3_constraints"
    assert BUILDERS["S4"].__name__ == "build_S4_constraints"

    print("  ✓ S3/S4 correctly wired into dispatch/families")


def test_geometry_preserving():
    """Test S3/S4 use input_H, input_W (geometry-preserving)."""
    print("Testing S3/S4 are geometry-preserving...")

    # Read S3 source
    s3_path = project_root / "src/schemas/s3_bands.py"
    s3_source = s3_path.read_text()

    # Read S4 source
    s4_path = project_root / "src/schemas/s4_residue_color.py"
    s4_source = s4_path.read_text()

    # Should use input_H, input_W (not output_H, output_W)
    assert "ex.input_H" in s3_source
    assert "ex.input_W" in s3_source
    assert "ex.input_H" in s4_source
    assert "ex.input_W" in s4_source

    # Should NOT use output_H, output_W
    assert "ex.output_H" not in s3_source
    assert "ex.output_W" not in s3_source
    assert "ex.output_H" not in s4_source
    assert "ex.output_W" not in s4_source

    print("  ✓ S3/S4 use input_H, input_W (geometry-preserving)")


def main():
    print("=" * 70)
    print("WO-M3.2 COMPREHENSIVE REVIEW TEST")
    print("=" * 70)
    print()
    print("Testing:")
    print("  - S3 schema builder (band/stripe laws)")
    print("  - S4 schema builder (residue-class coloring)")
    print("  - Catalog infrastructure (types.py)")
    print("  - Kernel runner (kernel.py)")
    print("  - Dispatch wiring")
    print()
    print("=" * 70)
    print()

    try:
        # Catalog infrastructure
        test_catalog_types_structure()

        # S3 tests
        test_s3_param_structure()
        test_s3_row_band_ties()
        test_s3_col_band_ties()
        test_s3_col_periodicity()
        test_s3_row_periodicity()

        # S4 tests
        test_s4_param_structure()
        test_s4_uses_fix_not_forbid()  # CRITICAL
        test_s4_col_residue_mapping()
        test_s4_row_residue_mapping()
        test_s4_partial_mapping()

        # Kernel tests
        test_kernel_integration()
        test_kernel_no_solver()

        # Design checks
        test_no_feature_computation()
        test_dispatch_wiring()
        test_geometry_preserving()

        print()
        print("=" * 70)
        print("✅ WO-M3.2 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ S3 builder (band/stripe laws) - COMPLETE")
        print("  ✓ S4 builder (residue-class coloring) - COMPLETE")
        print("  ✓ S4 uses fix_pixel_color (NOT forbid loop) - VERIFIED")
        print("  ✓ Catalog types (SchemaInstance, TaskLawConfig) - COMPLETE")
        print("  ✓ Kernel runner (solve_arc_task) - COMPLETE")
        print("  ✓ Kernel has minimal scope (NO solver) - VERIFIED")
        print("  ✓ Builders are param-driven (NO feature mining) - VERIFIED")
        print("  ✓ Dispatch/families wiring - CORRECT")
        print("  ✓ Geometry-preserving (use input_H, input_W) - VERIFIED")
        print()
        print("WO-M3.2 IMPLEMENTATION READY FOR PRODUCTION")
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
