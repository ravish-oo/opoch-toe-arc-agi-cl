## 🔹 Expanded WO-M5.1 – Result & diagnostics struct

### File 1: `src/runners/results.py`

**Goal:** Define a single, precise diagnostics object + a helper to compute train mismatches. No wiggle room.

#### 1. Imports and dependencies

Use only standard libraries and existing project modules:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Tuple

import numpy as np

from src.catalog.types import TaskLawConfig
from src.core.grid_types import Grid
```

* `Literal` for status type.
* `TaskLawConfig` from your existing M2 code.
* `Grid` (np.ndarray[int]) from `grid_types.py`.
* No custom algorithms; use numpy for diffing grids.

#### 2. Define `SolveDiagnostics`

Define exactly this dataclass (field names and types matter, Pi-agent will rely on them):

```python
SolveStatus = Literal["ok", "infeasible", "mismatch", "error"]

@dataclass
class SolveDiagnostics:
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
    # }

    # Debug / error information
    error_message: Optional[str] = None
```

**Constraints:**

* `status` must be *one of* `"ok" | "infeasible" | "mismatch" | "error"`.
* `train_mismatches` must be an empty list when no training comparison is done.
* No optional fields besides `error_message`.

This structure is what a Pi-agent will see; do not rename fields without updating the rest of the system later.

#### 3. Helper: compute mismatches between grids

Still in `results.py`, implement:

```python
def compute_grid_mismatches(
    true_grid: Grid,
    pred_grid: Grid
) -> List[Dict[str, int]]:
    """
    Compute per-cell mismatches between true and predicted grids.

    Returns:
        A list of dicts, each:
          { "r": row_index, "c": col_index, "true": true_color, "pred": pred_color }
        If shapes differ, this function should handle it gracefully by:
          - limiting comparison to the overlapping area, and
          - treating extra cells on either side as mismatches as well.
    """
```

**Implementation guidance (be explicit to Claude):**

* Use numpy to compare:

  * If shapes are equal:

    * `mask = (true_grid != pred_grid)`
    * `mismatch_coords = np.argwhere(mask)`
    * For each `(r, c)` in `mismatch_coords`:

      * append `{ "r": int(r), "c": int(c), "true": int(true_grid[r,c]), "pred": int(pred_grid[r,c]) }`.

* If shapes differ:

  * Let `Ht, Wt = true_grid.shape`, `Hp, Wp = pred_grid.shape`.
  * Overlapping area:

    * `H = min(Ht, Hp)`, `W = min(Wt, Wp)`.
    * Compare as above for `[0:H, 0:W]`.
  * Extra rows/cols in true_grid:

    * For `r >= H` or `c >= W` in true_grid:

      * treat predicted color as -1 (or some sentinel) in the mismatch record.
  * Extra rows/cols in pred_grid:

    * For `r >= H` or `c >= W` in pred_grid:

      * treat true color as -1 in the mismatch record.
  * The sentinel -1 is acceptable, but document it in a comment.

Be deterministic and simple; don’t invent clever diff formats.

#### 4. Helper: build `train_mismatches` for multiple examples

Add another helper:

```python
def compute_train_mismatches(
    true_grids: List[Grid],
    pred_grids: List[Grid]
) -> List[Dict]:
    """
    Compare lists of true and predicted training grids example-wise.

    Returns:
      A list of mismatch records:
        [
          {
            "example_idx": i,
            "diff_cells": [ { "r":..., "c":..., "true":..., "pred":... }, ... ]
          },
          ...
        ]
      Only examples with at least one diff_cells entry are included.
    """
```

Implementation:

* Assume `len(true_grids) == len(pred_grids)` (assert for safety).
* For each index `i`, call `compute_grid_mismatches(true_grids[i], pred_grids[i])`.
* If result is non-empty, append:

  ```python
  {
      "example_idx": i,
      "diff_cells": diff_cells
  }
  ```

---

### File 2: `src/runners/test_results_struct.py` (thin runner/test)

**Goal:** Provide a **simple, concrete test/runner** to validate `SolveDiagnostics` and mismatch helpers without touching the whole kernel yet.

#### 1. Contents

```python
from src.runners.results import SolveDiagnostics, compute_grid_mismatches, compute_train_mismatches
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.core.grid_types import Grid
import numpy as np

def _dummy_law_config() -> TaskLawConfig:
    # Minimal TaskLawConfig with one fake schema instance
    schema = SchemaInstance(family_id="S1", params={})
    return TaskLawConfig(schema_instances=[schema])

def smoke_test_results_struct():
    # Build two small 3x3 grids with a couple of mismatches
    true_grid: Grid = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ], dtype=int)

    pred_grid: Grid = np.array([
        [0, 1, 2],
        [0, 9, 2],   # mismatch at (1,1)
        [5, 1, 2],   # mismatch at (2,0)
    ], dtype=int)

    mismatches = compute_grid_mismatches(true_grid, pred_grid)
    print("Single-grid mismatches:", mismatches)

    train_mismatches = compute_train_mismatches([true_grid], [pred_grid])
    print("Train mismatches:", train_mismatches)

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

    print("SolveDiagnostics:", diag)

if __name__ == "__main__":
    smoke_test_results_struct()
```

---

## Reviewer + Tester instructions

**For implementer:**

* Implement `SolveDiagnostics`, `compute_grid_mismatches`, `compute_train_mismatches` in `results.py` exactly as specified.
* Implement `test_results_struct.py` and ensure it runs without importing anything else from the kernel.

**For reviewer/tester:**

1. Run the smoke test:

   ```bash
   python -m src.runners.test_results_struct
   ```

2. Check that:

   * The “Single-grid mismatches” output shows **exactly** the mismatched cells with correct `r, c, true, pred`.
   * “Train mismatches” is a list with one entry, `"example_idx": 0`, and the same `diff_cells` inside.
   * The printed `SolveDiagnostics` instance:

     * has `status="mismatch"`,
     * `schema_ids_used=["S1"]`,
     * `train_mismatches` populated as expected,
     * no exceptions raised.

3. Optionally, add a second quick check:

   * Make `pred_grid` identical to `true_grid`,
   * Verify `compute_grid_mismatches` and `compute_train_mismatches` return empty lists,
   * Construct a `SolveDiagnostics` with `status="ok"` and empty `train_mismatches`.

**No changes to `kernel.py` yet** in this WO. That will come in the next WO when we wire diagnostics into the kernel runner.

---
#clarification
Clear Implementation Plan

  compute_grid_mismatches() behavior:

  Case 1: Shapes match
  if true_grid.shape == pred_grid.shape:
      # Per-cell diff using numpy
      # Return: [{"r": int, "c": int, "true": int, "pred": int}, ...]

  Case 2: Shapes differ
  if true_grid.shape != pred_grid.shape:
      # Single shape-mismatch record
      # Return: [{"shape_mismatch": True, "true_shape": (Ht, Wt), "pred_shape": (Hp, Wp)}]

  No sentinels, no -1, no fake colors. Clean binary: either detailed per-cell diffs OR high-level shape error.

  Why this works:

  - Pi-agent sees: "wrong output dimensions = bad law config"
  - No need to fake per-cell comparisons when shapes don't even match
  - Simpler logic, clearer semantics
