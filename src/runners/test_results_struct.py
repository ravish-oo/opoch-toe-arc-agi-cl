"""
Smoke test for SolveDiagnostics and mismatch computation.

This test validates that the results.py module works correctly without
requiring the full kernel integration. Tests:
  - compute_grid_mismatches with known differences
  - compute_train_mismatches wrapper
  - SolveDiagnostics structure creation

No kernel dependencies - pure diagnostics testing.
"""

import numpy as np

from src.runners.results import (
    SolveDiagnostics,
    compute_grid_mismatches,
    compute_train_mismatches
)
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.core.grid_types import Grid


def _dummy_law_config() -> TaskLawConfig:
    """
    Create a minimal TaskLawConfig for testing.

    Returns:
        TaskLawConfig with one fake S1 schema instance
    """
    schema = SchemaInstance(family_id="S1", params={})
    return TaskLawConfig(schema_instances=[schema])


def smoke_test_results_struct():
    """
    Main smoke test: validate mismatch computation and diagnostics structure.

    Tests three scenarios:
      1. Grid with known mismatches at specific cells
      2. Train mismatches wrapper for multiple examples
      3. SolveDiagnostics construction with all fields
    """
    print("\n" + "=" * 70)
    print("SMOKE TEST: results.py diagnostics structures")
    print("=" * 70)

    # Build two small 3x3 grids with a couple of mismatches
    true_grid: Grid = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ], dtype=int)

    pred_grid: Grid = np.array([
        [0, 1, 2],
        [0, 9, 2],   # mismatch at (1,1): true=1, pred=9
        [5, 1, 2],   # mismatch at (2,0): true=0, pred=5
    ], dtype=int)

    print("\nTest 1: Single grid mismatches")
    print("-" * 70)
    print("True grid:")
    print(true_grid)
    print("\nPredicted grid:")
    print(pred_grid)

    mismatches = compute_grid_mismatches(true_grid, pred_grid)
    print(f"\nSingle-grid mismatches: {mismatches}")

    # Verify we got exactly 2 mismatches
    assert len(mismatches) == 2, f"Expected 2 mismatches, got {len(mismatches)}"

    # Check they're at the right locations with right values
    # Sort by (r, c) for deterministic checking
    mismatches_sorted = sorted(mismatches, key=lambda x: (x["r"], x["c"]))

    assert mismatches_sorted[0] == {"r": 1, "c": 1, "true": 1, "pred": 9}, \
        f"Mismatch at (1,1) incorrect: {mismatches_sorted[0]}"
    assert mismatches_sorted[1] == {"r": 2, "c": 0, "true": 0, "pred": 5}, \
        f"Mismatch at (2,0) incorrect: {mismatches_sorted[1]}"

    print("✓ Single-grid mismatches correct")

    print("\nTest 2: Train mismatches wrapper")
    print("-" * 70)

    train_mismatches = compute_train_mismatches([true_grid], [pred_grid])
    print(f"Train mismatches: {train_mismatches}")

    # Verify structure
    assert len(train_mismatches) == 1, f"Expected 1 example with mismatches"
    assert train_mismatches[0]["example_idx"] == 0, "Wrong example index"
    assert len(train_mismatches[0]["diff_cells"]) == 2, "Wrong number of diff cells"

    print("✓ Train mismatches wrapper correct")

    print("\nTest 3: SolveDiagnostics structure")
    print("-" * 70)

    diag = SolveDiagnostics(
        task_id="dummy_task",
        law_config=_dummy_law_config(),
        status="mismatch",
        solver_status="Optimal",
        num_constraints=10,
        num_variables=9,
        schema_ids_used=["S1"],
        train_mismatches=train_mismatches,
        error_message=None,
    )

    print(f"SolveDiagnostics created successfully:")
    print(f"  task_id: {diag.task_id}")
    print(f"  status: {diag.status}")
    print(f"  solver_status: {diag.solver_status}")
    print(f"  num_constraints: {diag.num_constraints}")
    print(f"  num_variables: {diag.num_variables}")
    print(f"  schema_ids_used: {diag.schema_ids_used}")
    print(f"  train_mismatches: {len(diag.train_mismatches)} example(s)")
    print(f"  error_message: {diag.error_message}")

    # Verify fields
    assert diag.task_id == "dummy_task"
    assert diag.status == "mismatch"
    assert diag.solver_status == "Optimal"
    assert diag.schema_ids_used == ["S1"]
    assert len(diag.train_mismatches) == 1

    print("✓ SolveDiagnostics structure correct")

    print("\nTest 4: Exact match case (no mismatches)")
    print("-" * 70)

    # Make pred_grid identical to true_grid
    pred_grid_exact = true_grid.copy()

    mismatches_exact = compute_grid_mismatches(true_grid, pred_grid_exact)
    print(f"Mismatches for identical grids: {mismatches_exact}")
    assert mismatches_exact == [], "Expected empty list for identical grids"

    train_mismatches_exact = compute_train_mismatches([true_grid], [pred_grid_exact])
    print(f"Train mismatches for identical grids: {train_mismatches_exact}")
    assert train_mismatches_exact == [], "Expected empty list when all examples match"

    # Create diagnostics with status="ok"
    diag_ok = SolveDiagnostics(
        task_id="dummy_task",
        law_config=_dummy_law_config(),
        status="ok",
        solver_status="Optimal",
        num_constraints=10,
        num_variables=9,
        schema_ids_used=["S1"],
        train_mismatches=[],  # empty for exact match
        error_message=None,
    )

    assert diag_ok.status == "ok"
    assert diag_ok.train_mismatches == []

    print("✓ Exact match case correct")

    print("\nTest 5: Shape mismatch case")
    print("-" * 70)

    # Different shapes
    true_small = np.array([[0, 1]], dtype=int)  # 1x2
    pred_large = np.array([[0, 1], [2, 3]], dtype=int)  # 2x2

    shape_diff = compute_grid_mismatches(true_small, pred_large)
    print(f"Shape mismatch result: {shape_diff}")

    assert len(shape_diff) == 1, "Expected single shape mismatch record"
    assert shape_diff[0]["shape_mismatch"] == True
    assert shape_diff[0]["true_shape"] == (1, 2)
    assert shape_diff[0]["pred_shape"] == (2, 2)

    print("✓ Shape mismatch detection correct")

    print("\n" + "=" * 70)
    print("✓ ALL SMOKE TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - Single-grid mismatch detection: ✓")
    print("  - Train mismatches wrapper: ✓")
    print("  - SolveDiagnostics structure: ✓")
    print("  - Exact match handling: ✓")
    print("  - Shape mismatch detection: ✓")
    print("\nDiagnostics structures are ready for kernel integration (M5.2)")
    print()


if __name__ == "__main__":
    smoke_test_results_struct()
