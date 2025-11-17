## 🚧 WO-M3.0 – Implement TaskContext & ExampleContext

**Files**

* `src/schemas/context.py`  ← **main**
* `src/runners/build_context_for_task.py`  ← thin runner for testing

**Goal**

Define concrete dataclasses `ExampleContext` and `TaskContext` plus a helper function that builds a `TaskContext` for a single ARC task using **only** the M1 feature operators and grid IO.

This will be the **only object** passed into `build_Sk_constraints` later.

---

## 1️⃣ `src/schemas/context.py`

### 1.1 Imports & types

Use only standard libs + your own modules. No new algorithms.

**Imports:**

```python
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any

import numpy as np

from src.core.grid_types import Grid, Pixel, Component
from src.core.arc_io import load_arc_task  # if you need it later, but context.py itself doesn’t have to call it

from src.features.coords_bands import (
    coord_features,          # you defined this in M1
    row_band_labels,
    col_band_labels,
)
from src.features.components import (
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
```

Assume:

* `Pixel = tuple[int, int]` is defined in `grid_types.py`.
* `Grid` is `np.ndarray` of shape `(H,W), dtype=int`.
* The feature functions all exist per M1.

### 1.2 Dataclasses

#### `ExampleContext`

One per **example** (train or test), capturing all φ features for that grid.

```python
@dataclass
class ExampleContext:
    # raw grids
    input_grid: Grid
    output_grid: Optional[Grid]  # None for test examples

    # shapes
    input_H: int
    input_W: int
    output_H: Optional[int]
    output_W: Optional[int]

    # component-level features on input_grid
    components: List[Component]
    object_ids: Dict[Pixel, int]
    role_bits: Dict[int, Dict[str, bool]]

    # per-pixel features on input_grid
    sectors: Dict[Pixel, Dict[str, str]]           # vert/horiz sectors in component bbox
    border_info: Dict[Pixel, Dict[str, bool]]      # is_border / is_interior
    row_bands: Dict[int, str]
    col_bands: Dict[int, str]
    row_nonzero: Dict[int, bool]
    col_nonzero: Dict[int, bool]
    neighborhood_hashes: Dict[Pixel, int]

    # coord/residue features
    coords: Dict[Pixel, Tuple[int, int]]           # (r,c)
    row_residues: Dict[int, Dict[int, int]]        # row -> {k -> r%k}
    col_residues: Dict[int, Dict[int, int]]        # col -> {k -> c%k}
```

#### `TaskContext`

One per **ARC task**:

```python
@dataclass
class TaskContext:
    train_examples: List[ExampleContext]   # each has input_grid + output_grid
    test_examples: List[ExampleContext]    # each has input_grid, output_grid=None
    C: int                                 # palette size (max color + 1 over all grids)
```

> 🔒 No “H/W” at task level: they may differ per example. Builders will look at `example.input_H`, etc., when they need shapes.

### 1.3 Helper: `build_example_context`

Implement a function that builds an `ExampleContext` for a single input/output pair.

**Signature:**

```python
def build_example_context(
    input_grid: Grid,
    output_grid: Optional[Grid]
) -> ExampleContext:
    ...
```

**Logic (no algorithms, just calls):**

1. Compute shapes:

   ```python
   input_H, input_W = input_grid.shape
   if output_grid is not None:
       output_H, output_W = output_grid.shape
   else:
       output_H = output_W = None
   ```

2. Components & object ids:

   ```python
   components = connected_components_by_color(input_grid)
   object_ids = assign_object_ids(components)  # (r,c) -> object_id
   role_bits = component_role_bits(components) # comp.id -> flags
   ```

3. Per-pixel roles & sectors:

   ```python
   sectors = component_sectors(components)  # (r,c) -> {"vert_sector":..., "horiz_sector":...}
   border_info = component_border_interior(input_grid, components)
   ```

4. Bands:

   ```python
   row_bands = row_band_labels(input_H)
   col_bands = col_band_labels(input_W)
   ```

5. Row/col flags:

   ```python
   row_nonzero = row_nonzero_flags(input_grid)
   col_nonzero = col_nonzero_flags(input_grid)
   ```

6. Neighborhood hashes:

   ```python
   nbh_hashes = neighborhood_hashes(input_grid, radius=1)
   ```

7. Coord & residues:

   * Use `coord_features` you wrote in M1.

   Example pattern (adapt to your actual return structure):

   ```python
   coord_feats = coord_features(input_grid)  # (r,c) -> {"row":r, "col":c, "row_mod":{k:...}, "col_mod":{k:...}}
   coords = {}
   row_residues: Dict[int, Dict[int,int]] = {}
   col_residues: Dict[int, Dict[int,int]] = {}
   for (r,c), feats in coord_feats.items():
       coords[(r,c)] = (feats["row"], feats["col"])
       # row residues
       if r not in row_residues:
           row_residues[r] = {}
       for k, val in feats["row_mod"].items():
           row_residues[r][k] = val
       # col residues
       if c not in col_residues:
           col_residues[c] = {}
       for k, val in feats["col_mod"].items():
           col_residues[c][k] = val
   ```

8. Return:

   ```python
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
   ```

### 1.4 Helper: `build_task_context_from_raw`

Implement:

```python
def build_task_context_from_raw(
    task_data: Dict[str, Any]
) -> TaskContext:
    """
    task_data is the structure returned by arc_io.load_arc_task(...):
      {
        "train": [{"input": Grid, "output": Grid}, ...],
        "test":  [{"input": Grid}, ...]
      }
    Build a TaskContext by:
      - creating ExampleContext for each train pair,
      - creating ExampleContext for each test input (output_grid=None),
      - computing C = max color + 1 over all grids.
    """
```

Steps:

1. Build train_examples:

   ```python
   train_examples = []
   for pair in task_data["train"]:
       ex = build_example_context(pair["input"], pair["output"])
       train_examples.append(ex)
   ```

2. Build test_examples:

   ```python
   test_examples = []
   for item in task_data["test"]:
       ex = build_example_context(item["input"], output_grid=None)
       test_examples.append(ex)
   ```

3. Compute palette size `C`:

   ```python
   all_grids = [ex.input_grid for ex in train_examples + test_examples]
   all_grids += [ex.output_grid for ex in train_examples if ex.output_grid is not None]

   max_color = max(int(grid.max()) for grid in all_grids)
   C = max_color + 1
   ```

4. Return:

   ```python
   return TaskContext(
       train_examples=train_examples,
       test_examples=test_examples,
       C=C,
   )
   ```

---

## 2️⃣ `src/runners/build_context_for_task.py` (thin runner)

**Goal:** a small script to test `TaskContext` construction on a real ARC-AGI training task.

**Imports:**

```python
import json
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
```

**Logic:**

1. Let it take a `task_id` or just hardcode one for now (e.g. first in `arc-agi_training_challenges.json`).
2. Use `load_arc_task` to fetch that task’s train/test grids.
3. Call `build_task_context_from_raw(task_data)`.
4. Print some diagnostics, for example:

   ```python
   print("Num train examples:", len(ctx.train_examples))
   print("Num test examples:", len(ctx.test_examples))
   print("Palette size C:", ctx.C)

   ex0 = ctx.train_examples[0]
   print("Example 0 input shape:", ex0.input_grid.shape)
   print("Example 0 #components:", len(ex0.components))
   print("Example 0 #neighborhood hashes:", len(ex0.neighborhood_hashes))
   ```

No solver, no schemas here — just verifying we can build φ for real tasks.

---

## 3️⃣ Reviewer + Tester instructions

For the **implementer**:

* Follow the file structure exactly:

  * `src/schemas/context.py` only defines dataclasses + builder functions above.
  * No extra algorithms beyond calling existing feature functions.
* Use imports from `src.features.*` and `src.core.*`; do **not** re-implement components, bands, or hashes.

For the **reviewer/tester**:

1. Pick 1–2 tasks from `data/arc-agi_training_challenges.json`.
2. Use `load_arc_task(task_id)` (or whatever helper you have) to get `task_data`.
3. Call `build_task_context_from_raw(task_data)`.
4. Check:

   * `len(ctx.train_examples) == len(task_data["train"])`
   * `len(ctx.test_examples) == len(task_data["test"])`
   * `ctx.C` is ≥ max color seen in any of the grids.
5. For one `ExampleContext`, assert:

   * `input_grid.shape` equals original input grid shape,
   * `output_grid.shape` matches original output grid (for train),
   * `len(components) > 0` if there are non-zero pixels,
   * `coords` includes all pixels in the input grid,
   * `row_residues[r][k] == r % k` for a few sample rows and k in {2,3,4,5}.

Optionally:

* For a tiny hand-crafted grid (3×3 or 4×4), you can manually verify:

  * border vs interior labels,
  * sector labels (top/middle/bottom, left/center/right).

---
