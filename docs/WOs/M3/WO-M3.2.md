## 🚧 WO-M3.2 – Implement S3 + S4 (bands/stripes + residue coloring), wire dispatch & kernel

### Overview

* **S3 (bands/stripes)**: tie rows/columns that should share the same color patterns (band equivalence and/or periodic structure).
* **S4 (periodic residue coloring)**: enforce that color depends only on row/col residue mod K.

Both are **geometry-preserving**: output grid has the same shape as input for the examples they apply to. So we can safely use `input_H` / `input_W` for y indexing, just like S1/S2.

The builders will be **param-driven only**: they do not discover band classes or residue-to-color maps; those will be supplied in `params` (Pi-agent later).

---

## 1️⃣ S3 – Band / stripe laws

**File:** `src/schemas/s3_bands.py`

### 1.1 Imports

```python
from typing import Dict, Any, List

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.constraints.indexing import y_index
from src.core.grid_types import Pixel
```

Assume you already have:

* `builder.tie_pixel_colors(p_idx, q_idx, C)` from M2.
* `ExampleContext` in `context.py` with:

  * `input_H`, `input_W`,
  * `row_bands`, `col_bands`, `row_nonzero`, `col_nonzero`.

### 1.2 Param format for S3

We define one clear schema:

```python
# params for S3
{
  "example_type": "train" | "test",
  "example_index": int,

  # optional grouping of rows into band-classes
  # each row_classes[i] is a list of row indices that should share a pattern
  "row_classes": [
    [0, 2],     # class 0: rows 0 and 2 tied
    [1, 3, 4],  # class 1: rows 1,3,4 tied
    ...
  ],

  # optional grouping of columns into band-classes
  "col_classes": [
    [0, 2, 4],
    [1, 3, 5]
  ],

  # optional periodicity along columns for each row
  # if set: pixels (r, c) and (r, c+K) must share color when both in range
  "col_period_K": int | null,

  # optional periodicity along rows for each column
  "row_period_K": int | null
}
```

Notes:

* `row_classes` and `col_classes` are **mutually independent**; you can use one, both, or neither.
* `col_period_K` / `row_period_K` are optional; if `None` or missing, no periodic tying is done.
* Builders will **not** interpret empty or missing lists as errors; they just skip that part.

### 1.3 `build_S3_constraints` implementation

Signature:

```python
def build_S3_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

1. Select example:

   ```python
   ex_list = context.train_examples if params["example_type"] == "train" else context.test_examples
   ex = ex_list[params["example_index"]]
   H, W = ex.input_H, ex.input_W
   C = context.C
   ```

2. **Row band ties** (using provided classes):

   ```python
   for row_class in params.get("row_classes", []):
       # tie all rows in this class pairwise
       for i in range(len(row_class)):
           r1 = row_class[i]
           for j in range(i+1, len(row_class)):
               r2 = row_class[j]
               # for each column
               for c in range(W):
                   p_idx1 = r1 * W + c
                   p_idx2 = r2 * W + c
                   builder.tie_pixel_colors(p_idx1, p_idx2, C)
   ```

3. **Column band ties** (similar for columns):

   ```python
   for col_class in params.get("col_classes", []):
       for i in range(len(col_class)):
           c1 = col_class[i]
           for j in range(i+1, len(col_class)):
               c2 = col_class[j]
               # for each row
               for r in range(H):
                   p_idx1 = r * W + c1
                   p_idx2 = r * W + c2
                   builder.tie_pixel_colors(p_idx1, p_idx2, C)
   ```

4. **Column periodicity** (optional):

   ```python
   col_K = params.get("col_period_K")
   if col_K is not None:
       K = int(col_K)
       for r in range(H):
           for c in range(W):
               c2 = c + K
               if c2 < W:
                   p_idx1 = r * W + c
                   p_idx2 = r * W + c2
                   builder.tie_pixel_colors(p_idx1, p_idx2, C)
   ```

5. **Row periodicity** (optional):

   ```python
   row_K = params.get("row_period_K")
   if row_K is not None:
       K = int(row_K)
       for c in range(W):
           for r in range(H):
               r2 = r + K
               if r2 < H:
                   p_idx1 = r * W + c
                   p_idx2 = r2 * W + c
                   builder.tie_pixel_colors(p_idx1, p_idx2, C)
   ```

No feature mining, no band detection – we just apply what `params` say.

---

## 2️⃣ S4 – Periodicity & residue-class coloring

**File:** `src/schemas/s4_residue_color.py`

### 2.1 Imports

```python
from typing import Dict, Any

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Pixel
```

We will use `ExampleContext.row_residues` and `ExampleContext.col_residues` populated in M3.0.

### 2.2 Param format for S4

We define:

```python
# params for S4
{
  "example_type": "train" | "test",
  "example_index": int,

  # axis along which residue is computed: "row" or "col"
  "axis": "row" | "col",

  # modulus K
  "K": int,

  # residue -> color map (as strings for JSON)
  # e.g. for K=2, even/odd pattern:
  "residue_to_color": {
    "0": 1,   # residue 0 -> color 1
    "1": 3    # residue 1 -> color 3
  }
}
```

If a residue is not found in `residue_to_color`, we do **not** constrain that pixel (i.e., builder skips it).

### 2.3 `build_S4_constraints` implementation

Signature:

```python
def build_S4_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

1. Select example:

   ```python
   ex_list = context.train_examples if params["example_type"] == "train" else context.test_examples
   ex = ex_list[params["example_index"]]
   H, W = ex.input_H, ex.input_W
   C = context.C
   ```

2. Extract params:

   ```python
   axis = params["axis"]       # "row" or "col"
   K = int(params["K"])
   residue_to_color = params["residue_to_color"]  # dict[str,int]
   ```

3. For each pixel `(r,c)`:

   * Compute residue:

     * If `axis == "row"`: `res = ex.row_residues[r].get(K)`
     * If `axis == "col"`: `res = ex.col_residues[c].get(K)`
   * If `res is None`: skip.
   * Convert `res` to str: `res_str = str(res)`.
   * If `res_str` not in `residue_to_color`: skip.
   * Otherwise `color = int(residue_to_color[res_str])`.
   * Compute `p_idx = r * W + c`.
   * Forbid all colors ≠ `color`:

     ```python
     for c_out in range(C):
         if c_out == color:
             continue
         builder.forbid_pixel_color(p_idx, c_out)
     ```

No law discovery; purely applying the parameterized mapping.

---

## 3️⃣ Wire S3 + S4 into dispatch

**File:** `src/schemas/dispatch.py`

* Import:

  ```python
  from src.schemas.s3_bands import build_S3_constraints
  from src.schemas.s4_residue_color import build_S4_constraints
  ```

* Update `BUILDERS` mapping:

  ```python
  BUILDERS = {
      "S1": build_S1_constraints,
      "S2": build_S2_constraints,
      "S3": build_S3_constraints,
      "S4": build_S4_constraints,
      # S5..S11 as stubs or later implementations
  }
  ```

**File:** `src/schemas/families.py`

* Ensure `SchemaFamily` entries for S3 and S4 have:

  ```python
  SchemaFamily(
      id="S3",
      name="Band / stripe laws",
      builder_name="build_S3_constraints",
      parameter_spec={
          "example_type": "str",
          "example_index": "int",
          "row_classes": "list[list[int]]",
          "col_classes": "list[list[int]]",
          "col_period_K": "int?|None",
          "row_period_K": "int?|None",
      },
      required_features=["row_bands", "col_bands", "row_nonzero", "col_nonzero"]
  )
  SchemaFamily(
      id="S4",
      name="Residue-class coloring",
      builder_name="build_S4_constraints",
      parameter_spec={
          "example_type": "str",
          "example_index": "int",
          "axis": "str",   # "row" or "col"
          "K": "int",
          "residue_to_color": "dict[str,int]"
      },
      required_features=["row_residues", "col_residues"]
  )
  ```

Adjust to the exact type strings you’ve been using.

---

## 4️⃣ Kernel runner touch (thin integration step)

**File:** `src/runners/kernel.py` (if not present, create it now)

Add/ensure:

```python
from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import TaskLawConfig  # assuming you defined this in M2

def solve_arc_task(task_id: str, law_config: TaskLawConfig):
    """
    Core kernel entrypoint.
    For now:
      - load task
      - build TaskContext
      - build constraints for all schema instances in law_config
      - (solver integration will come later)
    """
    task_data = load_arc_task(task_id)
    ctx = build_task_context_from_raw(task_data)

    builder = ConstraintBuilder()

    for schema_instance in law_config.schema_instances:
        apply_schema_instance(
            family_id=schema_instance.family_id,
            params=schema_instance.params,
            task_context=ctx,
            builder=builder,
        )

    # TODO: integrate LP solver in next milestone.
    # For now, just return the builder so tests can inspect constraints.
    return builder
```

This keeps the kernel entrypoint **real** and evolving, but doesn’t force you to wire a solver yet.

---

## 5️⃣ Reviewer + tester instructions

**For implementer:**

* Implement `build_S3_constraints` and `build_S4_constraints` exactly per the param formats above.
* Do **not**:

  * infer row/col classes,
  * infer residue-to-color maps,
  * or assume anything about solver.
* Assume output grids have the same shape as input for S3/S4 (geometry-preserving).

**For reviewer/tester:**

1. Create a tiny toy task in a test script (can be inside `test_context_dispatch_integration.py` or a new small test):

   * Example grid 4×4, where:

     * rows 0 and 2 should be tied,
     * cols 1 and 3 should be tied,
     * or simple even/odd column residue coloring.

2. Build a `TaskContext` for that toy task using `build_task_context_from_raw`.

3. Construct a `TaskLawConfig` (or local stub) with:

   * One S3 instance with:

     * `"example_type": "train"`,
     * `"example_index": 0`,
     * `"row_classes": [[0, 2]]`,
     * `"col_classes": []`,
     * `"col_period_K": None`, `"row_period_K": None`.

   * One S4 instance with:

     * `"example_type": "train"`,
     * `"example_index": 0`,
     * `"axis": "col"`,
     * `"K": 2`,
     * `"residue_to_color": {"0": 1, "1": 3}`.

4. Call:

   ```python
   from src.runners.kernel import solve_arc_task
   builder = solve_arc_task(task_id, law_config)
   print(len(builder.constraints))
   ```

5. Verify:

   * No exceptions,
   * `len(builder.constraints) > 0`,
   * Optional: inspect a few constraints to see that ties and forbids match the intended rows/cols/residues.

If this passes, S3 and S4 are correctly wired into the kernel and ready for Pi-agent usage later.

---
