"""
TaskContext and ExampleContext dataclasses for schema builders.

This module defines the unified context objects passed into all build_Sk_constraints
functions. It aggregates all φ(p) features from M1 into a single coherent structure.

Each ExampleContext represents one train or test example (input/output pair).
Each TaskContext represents one full ARC task (train + test examples).

All features are precomputed using M1 feature operators.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import json

import numpy as np

from src.core.grid_types import Grid, Pixel
from src.core.arc_io import load_arc_training_challenges

from src.features.coords_bands import (
    coord_features,
    row_band_labels,
    col_band_labels,
)
from src.features.components import (
    Component,
    connected_components_by_color,
    assign_object_ids,
)
from src.features.neighborhoods import (
    row_nonzero_flags,
    col_nonzero_flags,
    neighborhood_hashes,
)
from src.features.object_roles import (
    component_sectors,
    component_border_interior,
    component_role_bits,
)


@dataclass
class ExampleContext:
    """
    Feature context for a single train or test example.

    Contains all φ(p) features computed on the input grid, plus the output grid
    (if available, i.e., for training examples).

    Attributes:
        input_grid: Input grid (H, W)
        output_grid: Output grid (H', W') or None for test examples
        input_H: Input grid height
        input_W: Input grid width
        output_H: Output grid height (None for test)
        output_W: Output grid width (None for test)
        components: List of Component objects from input grid
        object_ids: Mapping (r,c) -> object_id for pixels in input
        role_bits: Mapping component.id -> {"is_small": bool, "is_big": bool, ...}
        sectors: Mapping (r,c) -> {"vert_sector": str, "horiz_sector": str}
        border_info: Mapping (r,c) -> {"is_border": bool, "is_interior": bool}
        row_bands: Mapping row -> "top"/"middle"/"bottom"
        col_bands: Mapping col -> "left"/"middle"/"right"
        row_nonzero: Mapping row -> bool (True if row has any non-zero)
        col_nonzero: Mapping col -> bool (True if col has any non-zero)
        neighborhood_hashes: Mapping (r,c) -> hash of 3×3 neighborhood
        coords: Mapping (r,c) -> (row, col) coordinates
        row_residues: Mapping row -> {k: row%k} for k in {2,3,4,5}
        col_residues: Mapping col -> {k: col%k} for k in {2,3,4,5}
    """
    # Raw grids
    input_grid: Grid
    output_grid: Optional[Grid]

    # Shapes
    input_H: int
    input_W: int
    output_H: Optional[int]
    output_W: Optional[int]

    # Component-level features on input_grid
    components: List[Component]
    object_ids: Dict[Pixel, int]
    role_bits: Dict[int, Dict[str, bool]]

    # Per-pixel features on input_grid
    sectors: Dict[Pixel, Dict[str, str]]           # vert/horiz sectors in component bbox
    border_info: Dict[Pixel, Dict[str, bool]]      # is_border / is_interior
    row_bands: Dict[int, str]
    col_bands: Dict[int, str]
    row_nonzero: Dict[int, bool]
    col_nonzero: Dict[int, bool]
    neighborhood_hashes: Dict[Pixel, int]

    # Coord/residue features
    coords: Dict[Pixel, Tuple[int, int]]           # (r,c)
    row_residues: Dict[int, Dict[int, int]]        # row -> {k -> r%k}
    col_residues: Dict[int, Dict[int, int]]        # col -> {k -> c%k}


@dataclass
class TaskContext:
    """
    Feature context for a full ARC task (train + test examples).

    Attributes:
        train_examples: List of ExampleContext for training pairs
        test_examples: List of ExampleContext for test inputs
        C: Palette size (max color + 1 across all grids in task)
    """
    train_examples: List[ExampleContext]
    test_examples: List[ExampleContext]
    C: int


def build_example_context(
    input_grid: Grid,
    output_grid: Optional[Grid]
) -> ExampleContext:
    """
    Build an ExampleContext for a single input/output pair.

    Computes all φ(p) features on the input grid using M1 feature operators.

    Args:
        input_grid: Input grid (H, W)
        output_grid: Output grid or None for test examples

    Returns:
        ExampleContext with all features populated

    Example:
        >>> input_grid = np.array([[0, 1], [2, 3]], dtype=int)
        >>> output_grid = np.array([[3, 2], [1, 0]], dtype=int)
        >>> ctx = build_example_context(input_grid, output_grid)
        >>> ctx.input_H, ctx.input_W
        (2, 2)
        >>> len(ctx.components)
        4
    """
    # 1. Compute shapes
    input_H, input_W = input_grid.shape
    if output_grid is not None:
        output_H, output_W = output_grid.shape
    else:
        output_H = output_W = None

    # 2. Components & object ids
    components = connected_components_by_color(input_grid)
    object_ids = assign_object_ids(components)
    role_bits = component_role_bits(components)

    # 3. Per-pixel roles & sectors
    sectors = component_sectors(components)
    border_info = component_border_interior(input_grid, components)

    # 4. Bands
    row_bands = row_band_labels(input_H)
    col_bands = col_band_labels(input_W)

    # 5. Row/col flags
    row_nonzero = row_nonzero_flags(input_grid)
    col_nonzero = col_nonzero_flags(input_grid)

    # 6. Neighborhood hashes
    nbh_hashes = neighborhood_hashes(input_grid, radius=1)

    # 7. Coord & residues
    # coord_features returns: (r,c) -> {"row": r, "col": c, "row_mod": {k:...}, "col_mod": {k:...}}
    coord_feats = coord_features(input_grid)

    coords: Dict[Pixel, Tuple[int, int]] = {}
    row_residues: Dict[int, Dict[int, int]] = {}
    col_residues: Dict[int, Dict[int, int]] = {}

    for (r, c), feats in coord_feats.items():
        # Store coordinates
        coords[(r, c)] = (feats["row"], feats["col"])

        # Store row residues (deduplicate across columns)
        if r not in row_residues:
            row_residues[r] = {}
            for k, val in feats["row_mod"].items():
                row_residues[r][k] = val

        # Store col residues (deduplicate across rows)
        if c not in col_residues:
            col_residues[c] = {}
            for k, val in feats["col_mod"].items():
                col_residues[c][k] = val

    # 8. Return ExampleContext
    return ExampleContext(
        input_grid=input_grid,
        output_grid=output_grid,
        input_H=input_H,
        input_W=input_W,
        output_H=output_H,
        output_W=output_W,
        components=components,
        object_ids=object_ids,
        role_bits=role_bits,
        sectors=sectors,
        border_info=border_info,
        row_bands=row_bands,
        col_bands=col_bands,
        row_nonzero=row_nonzero,
        col_nonzero=col_nonzero,
        neighborhood_hashes=nbh_hashes,
        coords=coords,
        row_residues=row_residues,
        col_residues=col_residues,
    )


def build_task_context_from_raw(
    task_data: Dict[str, Any]
) -> TaskContext:
    """
    Build a TaskContext from raw task data.

    Expected task_data format:
      {
        "train": [{"input": Grid, "output": Grid}, ...],
        "test": [{"input": Grid}, ...]
      }

    This is the format returned by load_arc_task().

    Args:
        task_data: Task data dictionary with train/test examples

    Returns:
        TaskContext with all examples and palette size C

    Example:
        >>> task_data = {
        ...     "train": [
        ...         {"input": np.array([[0, 1]]), "output": np.array([[1, 0]])}
        ...     ],
        ...     "test": [{"input": np.array([[0, 2]])}]
        ... }
        >>> ctx = build_task_context_from_raw(task_data)
        >>> len(ctx.train_examples)
        1
        >>> len(ctx.test_examples)
        1
        >>> ctx.C
        3
    """
    # 1. Build train_examples
    train_examples = []
    for pair in task_data["train"]:
        ex = build_example_context(pair["input"], pair["output"])
        train_examples.append(ex)

    # 2. Build test_examples
    test_examples = []
    for item in task_data["test"]:
        ex = build_example_context(item["input"], output_grid=None)
        test_examples.append(ex)

    # 3. Compute palette size C = max color + 1
    all_grids: List[Grid] = []

    # Collect input grids
    all_grids.extend([ex.input_grid for ex in train_examples])
    all_grids.extend([ex.input_grid for ex in test_examples])

    # Collect train output grids
    all_grids.extend([ex.output_grid for ex in train_examples if ex.output_grid is not None])

    max_color = max(int(grid.max()) for grid in all_grids)
    C = max_color + 1

    # 4. Return TaskContext
    return TaskContext(
        train_examples=train_examples,
        test_examples=test_examples,
        C=C,
    )


def load_arc_task(task_id: str, challenges_path: Path) -> Dict[str, Any]:
    """
    Load a single ARC task from challenges file and convert to standard format.

    This is a helper that adapts the arc_io output format to the format expected
    by build_task_context_from_raw().

    arc_io returns:
      {
        task_id: {
          "train": [Grid, ...],           # inputs only
          "train_outputs": [Grid, ...],   # outputs separate
          "test": [Grid, ...]
        }
      }

    This function converts it to:
      {
        "train": [{"input": Grid, "output": Grid}, ...],
        "test": [{"input": Grid}, ...]
      }

    Args:
        task_id: ARC task identifier
        challenges_path: Path to arc-agi_training_challenges.json

    Returns:
        Task data in standard format

    Raises:
        KeyError: If task_id not found in challenges file

    Example:
        >>> path = Path("data/arc-agi_training_challenges.json")
        >>> task_data = load_arc_task("00d62c1b", path)
        >>> "train" in task_data and "test" in task_data
        True
    """
    # Load all tasks
    all_tasks = load_arc_training_challenges(challenges_path)

    if task_id not in all_tasks:
        raise KeyError(f"Task '{task_id}' not found in {challenges_path}")

    raw_task = all_tasks[task_id]

    # Convert to standard format
    train_examples = []
    for input_grid, output_grid in zip(raw_task["train"], raw_task["train_outputs"]):
        train_examples.append({
            "input": input_grid,
            "output": output_grid
        })

    test_examples = []
    for input_grid in raw_task["test"]:
        test_examples.append({
            "input": input_grid
        })

    return {
        "train": train_examples,
        "test": test_examples
    }


if __name__ == "__main__":
    # Sanity check with a tiny hand-crafted grid
    print("Testing ExampleContext with toy grid...")
    print("=" * 70)

    input_grid = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [2, 2, 0]
    ], dtype=int)

    output_grid = np.array([
        [3, 3],
        [3, 3]
    ], dtype=int)

    ex = build_example_context(input_grid, output_grid)

    print(f"Input shape: {ex.input_H}x{ex.input_W}")
    print(f"Output shape: {ex.output_H}x{ex.output_W}")
    print(f"Num components: {len(ex.components)}")
    print(f"Num pixels with object_ids: {len(ex.object_ids)}")
    print(f"Num neighborhood hashes: {len(ex.neighborhood_hashes)}")

    # Check coordinate features
    print("\nSample coordinate features:")
    pixel = (0, 0)
    print(f"  coords[{pixel}] = {ex.coords.get(pixel)}")
    print(f"  row_residues[0] = {ex.row_residues.get(0)}")
    print(f"  col_residues[0] = {ex.col_residues.get(0)}")

    # Check bands
    print("\nBand labels:")
    print(f"  row_bands = {ex.row_bands}")
    print(f"  col_bands = {ex.col_bands}")

    # Check row/col flags
    print("\nNonzero flags:")
    print(f"  row_nonzero = {ex.row_nonzero}")
    print(f"  col_nonzero = {ex.col_nonzero}")

    # Check residues for a sample row
    print("\nRow 1 residues:")
    for k in [2, 3, 4, 5]:
        expected = 1 % k
        actual = ex.row_residues[1][k]
        assert actual == expected, f"Row residue mismatch: {actual} != {expected}"
        print(f"  1 % {k} = {actual} ✓")

    print("\n" + "=" * 70)
    print("✓ ExampleContext sanity checks passed.")

    print("\n" + "=" * 70)
    print("Testing TaskContext with toy task...")
    print("=" * 70)

    task_data = {
        "train": [
            {"input": input_grid, "output": output_grid}
        ],
        "test": [
            {"input": np.array([[1, 2], [3, 4]], dtype=int)}
        ]
    }

    ctx = build_task_context_from_raw(task_data)

    print(f"Num train examples: {len(ctx.train_examples)}")
    print(f"Num test examples: {len(ctx.test_examples)}")
    print(f"Palette size C: {ctx.C}")

    # Verify C = max color + 1
    # input has colors {0,1,2}, output has {3}, test has {1,2,3,4}
    # max = 4, so C = 5
    assert ctx.C == 5, f"Expected C=5, got C={ctx.C}"
    print(f"  ✓ C = {ctx.C} (correct)")

    print("\n" + "=" * 70)
    print("✓ TaskContext sanity checks passed.")
