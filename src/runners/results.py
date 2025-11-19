"""
Result and diagnostics structures for ARC-AGI solver.

This module defines SolveDiagnostics, the single structured object that captures
everything about a solve attempt - especially failures. This is what Pi-agents
will use to debug and refine law configs.

Key components:
  - SolveDiagnostics: Complete solve attempt record (status, constraints, mismatches)
  - compute_grid_mismatches: Per-cell diff between true and predicted grids
  - compute_train_mismatches: Wrapper for multiple training examples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Tuple

import numpy as np

from src.catalog.types import TaskLawConfig
from src.core.grid_types import Grid


# Status type for solve attempts
SolveStatus = Literal["ok", "infeasible", "mismatch", "mismatch_train", "mismatch_test", "error"]


@dataclass
class ExampleSummary:
    """
    Summary of a single ARC example (train or test).

    Provides high-level structural information about the example without
    full grid data. Useful for Pi-agents to understand task characteristics.

    Attributes:
        input_shape: (H, W) shape of input grid
        output_shape: (H', W') shape of output grid, or None for test examples
        components_per_color: Dict mapping color -> number of connected components
                             of that color in the input grid
    """
    input_shape: Tuple[int, int]
    output_shape: Optional[Tuple[int, int]]
    components_per_color: Dict[int, int]


@dataclass
class SolveDiagnostics:
    """
    Complete diagnostics for a single solve attempt.

    This structure is what Pi-agents see when they try a law config on a task.
    It captures everything: success/failure status, solver details, and
    detailed mismatch information for training and test examples.

    Attributes:
        task_id: ARC task identifier
        law_config: The TaskLawConfig that was applied
        status: Solve outcome - one of:
            - "ok": solver succeeded and all outputs match (train and test if checked)
            - "infeasible": ILP had no solution
            - "mismatch": solver succeeded but outputs don't match (legacy, being phased out)
            - "mismatch_train": solver succeeded but train outputs don't match
            - "mismatch_test": solver succeeded, train matches, but test outputs don't match
            - "error": unexpected error during solving
        solver_status: Raw status string from pulp solver (e.g. "Optimal", "Infeasible")
        num_constraints: Total number of constraints in the ILP
        num_variables: Total number of variables (typically N*C for grid)
        schema_ids_used: List of schema family IDs applied (e.g. ["S1", "S2"])
        train_mismatches: List of per-example mismatch records (only when use_training_labels=True)
            Each element: {
                "example_idx": int,
                "diff_cells": List of cell-level diffs or shape mismatch record
            }
        test_mismatches: List of per-example mismatch records (only when use_test_labels=True)
            Same structure as train_mismatches
        error_message: Optional error message for status="error"
    """
    task_id: str
    law_config: TaskLawConfig

    status: SolveStatus              # "ok", "infeasible", "mismatch", "error"
    solver_status: str               # raw status string from pulp/solver

    num_constraints: int
    num_variables: int
    schema_ids_used: List[str]       # e.g. ["S1", "S2"]

    # Only for training tasks (when ground truth is available)
    train_mismatches: List[Dict] = field(default_factory=list)
    # Each element: {
    #   "example_idx": int,
    #   "diff_cells": List[{"r": int, "c": int, "true": int, "pred": int}]
    #                or [{"shape_mismatch": True, "true_shape": tuple, "pred_shape": tuple}]
    # }

    # Only for test validation (when use_test_labels=True)
    test_mismatches: List[Dict] = field(default_factory=list)
    # Same structure as train_mismatches

    # Per-schema constraint counts (M5.X)
    schema_constraint_counts: Dict[str, int] = field(default_factory=dict)
    # key: schema family ID (e.g. "S1", "S2", ..., "S11")
    # value: total number of constraints contributed by that schema

    # Example-level summaries (M5.X)
    example_summaries: List[ExampleSummary] = field(default_factory=list)
    # One ExampleSummary per train+test example, in order

    # Debug / error information
    error_message: Optional[str] = None


def compute_grid_mismatches(
    true_grid: Grid,
    pred_grid: Grid
) -> List[Dict[str, int]]:
    """
    Compute per-cell mismatches between true and predicted grids.

    This function handles two cases:

    1. Shapes match: Returns detailed per-cell differences
       - Uses numpy to find all cells where colors differ
       - Each mismatch: {"r": row, "c": col, "true": color, "pred": color}

    2. Shapes differ: Returns single high-level shape mismatch record
       - No per-cell comparison (doesn't make sense with different dimensions)
       - Single record: {"shape_mismatch": True, "true_shape": (H,W), "pred_shape": (H,W)}

    Args:
        true_grid: Ground truth grid (H x W numpy array)
        pred_grid: Predicted grid (H' x W' numpy array)

    Returns:
        List of mismatch records. Either:
          - Empty list if grids are identical
          - List of per-cell diffs if shapes match but values differ
          - Single shape-mismatch record if dimensions don't match

    Example (matching shapes):
        >>> true = np.array([[0, 1], [2, 3]])
        >>> pred = np.array([[0, 9], [2, 3]])
        >>> compute_grid_mismatches(true, pred)
        [{"r": 0, "c": 1, "true": 1, "pred": 9}]

    Example (different shapes):
        >>> true = np.array([[0, 1]])
        >>> pred = np.array([[0, 1], [2, 3]])
        >>> compute_grid_mismatches(true, pred)
        [{"shape_mismatch": True, "true_shape": (1, 2), "pred_shape": (2, 2)}]
    """
    Ht, Wt = true_grid.shape
    Hp, Wp = pred_grid.shape

    # Case 1: Shape mismatch - return high-level error record
    if (Ht, Wt) != (Hp, Wp):
        return [{
            "shape_mismatch": True,
            "true_shape": (Ht, Wt),
            "pred_shape": (Hp, Wp),
        }]

    # Case 2: Shapes match - do per-cell comparison
    # Find all cells where true != pred
    mismatch_mask = (true_grid != pred_grid)

    # If no mismatches, return empty list
    if not mismatch_mask.any():
        return []

    # Get coordinates of all mismatched cells
    mismatch_coords = np.argwhere(mismatch_mask)

    # Build list of mismatch records
    diff_cells = []
    for coord in mismatch_coords:
        r, c = int(coord[0]), int(coord[1])
        diff_cells.append({
            "r": r,
            "c": c,
            "true": int(true_grid[r, c]),
            "pred": int(pred_grid[r, c])
        })

    return diff_cells


def compute_train_mismatches(
    true_grids: List[Grid],
    pred_grids: List[Grid]
) -> List[Dict]:
    """
    Compare lists of true and predicted training grids example-wise.

    For each training example, computes detailed mismatches and packages them
    with the example index. Only includes examples that have at least one
    mismatch (either shape or cell-level).

    Args:
        true_grids: List of ground truth output grids for training examples
        pred_grids: List of predicted output grids from solver

    Returns:
        List of mismatch records, one per example with differences:
          [
            {
              "example_idx": i,
              "diff_cells": [{"r":..., "c":..., "true":..., "pred":...}, ...]
                            or [{"shape_mismatch": True, ...}]
            },
            ...
          ]
        Empty list if all examples match perfectly.

    Raises:
        AssertionError: If true_grids and pred_grids have different lengths

    Example:
        >>> true = [np.array([[0, 1]]), np.array([[2, 3]])]
        >>> pred = [np.array([[0, 1]]), np.array([[2, 9]])]
        >>> compute_train_mismatches(true, pred)
        [{"example_idx": 1, "diff_cells": [{"r": 0, "c": 1, "true": 3, "pred": 9}]}]
    """
    # Safety check: must have same number of examples
    assert len(true_grids) == len(pred_grids), \
        f"Mismatch in example counts: {len(true_grids)} true vs {len(pred_grids)} pred"

    train_mismatches = []

    # Compare each example
    for i, (true_grid, pred_grid) in enumerate(zip(true_grids, pred_grids)):
        diff_cells = compute_grid_mismatches(true_grid, pred_grid)

        # Only include examples with actual differences
        if diff_cells:
            train_mismatches.append({
                "example_idx": i,
                "diff_cells": diff_cells
            })

    return train_mismatches


if __name__ == "__main__":
    # Quick self-test
    print("Testing results.py diagnostics structures...")
    print("=" * 70)

    # Test 1: Identical grids
    g1 = np.array([[0, 1], [2, 3]], dtype=int)
    g2 = np.array([[0, 1], [2, 3]], dtype=int)
    diff = compute_grid_mismatches(g1, g2)
    print(f"Test 1 (identical grids): {diff}")
    assert diff == [], "Expected no mismatches"

    # Test 2: One cell differs
    g3 = np.array([[0, 9], [2, 3]], dtype=int)
    diff = compute_grid_mismatches(g1, g3)
    print(f"Test 2 (one cell diff): {diff}")
    assert len(diff) == 1
    assert diff[0] == {"r": 0, "c": 1, "true": 1, "pred": 9}

    # Test 3: Shape mismatch
    g4 = np.array([[0, 1, 2]], dtype=int)
    diff = compute_grid_mismatches(g1, g4)
    print(f"Test 3 (shape mismatch): {diff}")
    assert len(diff) == 1
    assert diff[0]["shape_mismatch"] == True

    # Test 4: Train mismatches
    true_grids = [g1, g1]
    pred_grids = [g2, g3]  # First matches, second has diff
    train_mm = compute_train_mismatches(true_grids, pred_grids)
    print(f"Test 4 (train mismatches): {train_mm}")
    assert len(train_mm) == 1
    assert train_mm[0]["example_idx"] == 1

    print("\n✓ All self-tests passed")
    print("=" * 70)
