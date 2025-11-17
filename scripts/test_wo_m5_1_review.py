#!/usr/bin/env python3
"""
Comprehensive review test for WO-M5.1: Result & diagnostics struct.

This test verifies:
  1. No TODOs, stubs, or simplified implementations
  2. Shape check happens FIRST (before cell comparison)
  3. NO sentinel -1 values (clean bifurcation)
  4. NO overlapping area logic
  5. Exact field names match spec
  6. Mutable default handled correctly (field(default_factory=list))
  7. Filter in compute_train_mismatches works
  8. All test cases pass (identical, mismatch, shape mismatch)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.runners.results import (
    SolveDiagnostics,
    SolveStatus,
    compute_grid_mismatches,
    compute_train_mismatches
)
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.core.grid_types import Grid


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Check for common markers
    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError",
               "stub", "Stub", "simplified", "Simplified", "MVP"]

    for marker in markers:
        assert marker not in source, \
            f"Found '{marker}' in results.py"

    print("  ✓ No TODOs, stubs, or markers found")


def test_no_sentinel_values():
    """Test that NO sentinel -1 values are used."""
    print("\nTest: No sentinel -1 values")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Should NOT have -1 sentinel logic
    assert "-1" not in source, \
        "Found sentinel -1 in code (should use clean bifurcation)"

    print("  ✓ No sentinel -1 values found")
    print("  ✓ Clean bifurcation: shape mismatch OR cell diffs, not mixed")


def test_no_overlapping_area_logic():
    """Test that NO overlapping area comparison logic exists."""
    print("\nTest: No overlapping area logic")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Should NOT have overlapping area logic
    forbidden = ["min(Ht", "min(Hp", "min(Wt", "min(Wp", "overlapping", "overlap"]
    for pattern in forbidden:
        assert pattern not in source, \
            f"Found '{pattern}' - overlapping area logic should be removed"

    print("  ✓ No overlapping area comparison logic found")
    print("  ✓ Clean bifurcation confirmed")


def test_shape_check_first():
    """Test that shape check happens BEFORE cell comparison."""
    print("\nTest: Shape check happens FIRST")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Find compute_grid_mismatches function
    func_start = source.find("def compute_grid_mismatches")
    assert func_start != -1, "compute_grid_mismatches not found"

    # Check that shape check happens before any cell comparison
    shape_check_pos = source.find("if (Ht, Wt) != (Hp, Wp):", func_start)
    cell_comp_pos = source.find("mismatch_mask = (true_grid != pred_grid)", func_start)

    assert shape_check_pos != -1, "Shape check not found"
    assert cell_comp_pos != -1, "Cell comparison not found"
    assert shape_check_pos < cell_comp_pos, \
        "Shape check must happen BEFORE cell comparison"

    print("  ✓ Shape check happens FIRST in compute_grid_mismatches")
    print("  ✓ Immediate return on shape mismatch")


def test_exact_field_names():
    """Test that exact field names match spec."""
    print("\nTest: Exact field names match spec")
    print("-" * 70)

    # Test shape mismatch record
    true_grid = np.array([[0, 1]], dtype=int)
    pred_grid = np.array([[0, 1], [2, 3]], dtype=int)

    shape_diff = compute_grid_mismatches(true_grid, pred_grid)
    assert len(shape_diff) == 1, "Expected single shape mismatch record"

    record = shape_diff[0]
    assert "shape_mismatch" in record, "Missing 'shape_mismatch' field"
    assert "true_shape" in record, "Missing 'true_shape' field"
    assert "pred_shape" in record, "Missing 'pred_shape' field"
    assert record["shape_mismatch"] == True
    assert record["true_shape"] == (1, 2)
    assert record["pred_shape"] == (2, 2)

    print("  ✓ Shape mismatch record has exact fields: shape_mismatch, true_shape, pred_shape")

    # Test cell diff record
    true_grid2 = np.array([[0, 1]], dtype=int)
    pred_grid2 = np.array([[0, 9]], dtype=int)

    cell_diff = compute_grid_mismatches(true_grid2, pred_grid2)
    assert len(cell_diff) == 1, "Expected single cell diff"

    cell_record = cell_diff[0]
    assert "r" in cell_record, "Missing 'r' field"
    assert "c" in cell_record, "Missing 'c' field"
    assert "true" in cell_record, "Missing 'true' field"
    assert "pred" in cell_record, "Missing 'pred' field"
    assert cell_record == {"r": 0, "c": 1, "true": 1, "pred": 9}

    print("  ✓ Cell diff record has exact fields: r, c, true, pred")


def test_solve_diagnostics_structure():
    """Test SolveDiagnostics dataclass structure."""
    print("\nTest: SolveDiagnostics structure")
    print("-" * 70)

    # Create minimal law config
    law_config = TaskLawConfig(
        schema_instances=[SchemaInstance(family_id="S1", params={})]
    )

    # Test all status values
    for status in ["ok", "infeasible", "mismatch", "error"]:
        diag = SolveDiagnostics(
            task_id="test_task",
            law_config=law_config,
            status=status,  # type: ignore
            solver_status="Optimal",
            num_constraints=10,
            num_variables=20,
            schema_ids_used=["S1", "S2"],
            train_mismatches=[],
            error_message=None
        )

        assert diag.task_id == "test_task"
        assert diag.status == status
        assert diag.solver_status == "Optimal"
        assert diag.num_constraints == 10
        assert diag.num_variables == 20
        assert diag.schema_ids_used == ["S1", "S2"]
        assert diag.train_mismatches == []
        assert diag.error_message is None

    print("  ✓ SolveDiagnostics has all required fields")
    print("  ✓ All status values work: ok, infeasible, mismatch, error")

    # Verify mutable default uses field(default_factory=list)
    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()
    assert "field(default_factory=list)" in source, \
        "Must use field(default_factory=list) for mutable defaults"

    print("  ✓ Uses field(default_factory=list) for mutable defaults")


def test_filter_in_compute_train_mismatches():
    """Test that compute_train_mismatches filters empty diffs."""
    print("\nTest: Filter in compute_train_mismatches")
    print("-" * 70)

    # Three grids: first matches, second has diff, third matches
    g1 = np.array([[0, 1]], dtype=int)
    g2 = np.array([[0, 1]], dtype=int)  # matches g1
    g3 = np.array([[0, 9]], dtype=int)  # differs from g1
    g4 = np.array([[0, 1]], dtype=int)  # matches g1

    true_grids = [g1, g1, g1]
    pred_grids = [g2, g3, g4]  # Only middle one differs

    train_mm = compute_train_mismatches(true_grids, pred_grids)

    # Should only include example 1 (the one with diff)
    assert len(train_mm) == 1, \
        f"Expected only 1 mismatch record, got {len(train_mm)}"
    assert train_mm[0]["example_idx"] == 1, \
        f"Expected example_idx=1, got {train_mm[0]['example_idx']}"
    assert len(train_mm[0]["diff_cells"]) == 1, \
        "Expected 1 diff cell"

    print("  ✓ Only includes examples with actual differences")
    print("  ✓ Filters out empty diff_cells correctly")

    # All match case
    all_match = compute_train_mismatches([g1, g1], [g2, g4])
    assert all_match == [], "Expected empty list when all examples match"

    print("  ✓ Returns empty list when all examples match")


def test_identical_grids():
    """Test identical grids return empty mismatches."""
    print("\nTest: Identical grids (no mismatches)")
    print("-" * 70)

    grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    grid_copy = grid.copy()

    diff = compute_grid_mismatches(grid, grid_copy)
    assert diff == [], f"Expected empty list, got {diff}"

    print("  ✓ Identical grids return empty list")


def test_cell_mismatches():
    """Test per-cell mismatch detection."""
    print("\nTest: Cell mismatches (equal shapes)")
    print("-" * 70)

    true_grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    pred_grid = np.array([[0, 9, 2], [8, 4, 5]], dtype=int)

    diff = compute_grid_mismatches(true_grid, pred_grid)

    # Should have exactly 2 mismatches: (0,1) and (1,0)
    assert len(diff) == 2, f"Expected 2 mismatches, got {len(diff)}"

    # Sort for deterministic checking
    diff_sorted = sorted(diff, key=lambda x: (x["r"], x["c"]))

    assert diff_sorted[0] == {"r": 0, "c": 1, "true": 1, "pred": 9}
    assert diff_sorted[1] == {"r": 1, "c": 0, "true": 3, "pred": 8}

    print("  ✓ Cell mismatches detected correctly")
    print("  ✓ Correct coordinates and values")


def test_shape_mismatch():
    """Test shape mismatch detection."""
    print("\nTest: Shape mismatch detection")
    print("-" * 70)

    # Different shapes
    true_grid = np.array([[0, 1, 2]], dtype=int)  # 1x3
    pred_grid = np.array([[0, 1], [2, 3]], dtype=int)  # 2x2

    diff = compute_grid_mismatches(true_grid, pred_grid)

    assert len(diff) == 1, "Expected single shape mismatch record"
    assert diff[0]["shape_mismatch"] == True
    assert diff[0]["true_shape"] == (1, 3)
    assert diff[0]["pred_shape"] == (2, 2)

    # Verify NO cell-level diffs in this record
    assert "r" not in diff[0], "Shape mismatch record should not have 'r' field"
    assert "c" not in diff[0], "Shape mismatch record should not have 'c' field"

    print("  ✓ Shape mismatch returns single high-level record")
    print("  ✓ No cell-level diffs when shapes differ")
    print("  ✓ Clean bifurcation confirmed")


def test_code_organization():
    """Test code organization and quality."""
    print("\nTest: Code organization")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Check has module docstring
    assert '"""' in source[:200], "Module should have docstring"
    print("  ✓ Module has docstring")

    # Check functions have docstrings
    assert "Args:" in source, "Functions should document Args"
    assert "Returns:" in source, "Functions should document Returns"
    print("  ✓ Functions have docstrings with Args/Returns")

    # Check imports organized
    assert "from __future__ import annotations" in source
    assert "from dataclasses import dataclass, field" in source
    assert "from typing import Literal, Optional, List, Dict" in source
    assert "import numpy as np" in source
    print("  ✓ Imports organized correctly")

    print("  ✓ Code organization is clean")


def main():
    print("=" * 70)
    print("WO-M5.1 COMPREHENSIVE REVIEW TEST")
    print("Testing result & diagnostics structures")
    print("=" * 70)

    try:
        # Core implementation checks
        test_no_todos_stubs()
        test_no_sentinel_values()
        test_no_overlapping_area_logic()
        test_shape_check_first()
        test_exact_field_names()
        test_solve_diagnostics_structure()
        test_filter_in_compute_train_mismatches()
        test_code_organization()

        # Functional tests
        test_identical_grids()
        test_cell_mismatches()
        test_shape_mismatch()

        print("\n" + "=" * 70)
        print("✅ WO-M5.1 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Implementation quality - EXCELLENT")
        print("    - No TODOs, stubs, or simplified implementations")
        print("    - No sentinel -1 values")
        print("    - No overlapping area logic")
        print("    - Clean bifurcation (shape mismatch OR cell diffs)")
        print()
        print("  ✓ Design requirements - ALL MET")
        print("    - Shape check happens FIRST")
        print("    - Exact field names match spec")
        print("    - SolveDiagnostics structure correct")
        print("    - Mutable defaults handled correctly")
        print("    - Filter in compute_train_mismatches works")
        print()
        print("  ✓ Functional tests - ALL PASSED")
        print("    - Identical grids: empty mismatches")
        print("    - Cell mismatches: correct detection")
        print("    - Shape mismatches: single high-level record")
        print()
        print("WO-M5.1 IMPLEMENTATION READY FOR M5.2 INTEGRATION")
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
