## 🚧 WO-M3.1 – Implement S1 + S2 schema builders (param-driven), wire dispatch, add thin test runner

**Goal:**

* Implement **S1** (copy/equality ties) and **S2** (component recolor) as real builder functions that:

  * take a `TaskContext`,
  * take a **schema instance’s params** (`dict`),
  * and emit constraints via `ConstraintBuilder`.
* Wire them into the `dispatch` module.
* Add a small runner/test so we know they actually work on toy cases.

No law-mining here yet — builders only **apply** law instances given explicit parameters. Pi-agent (later) will decide those params.

---

## 1️⃣ S1: copy/equality ties

**File:** `src/schemas/s1_copy_tie.py`

### 1.1 Imports

```python
from typing import Dict, Any, List, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.constraints.indexing import y_index  # or appropriate helper
```

We do **not** re-implement features; we just use coordinates coming from `ExampleContext`.

### 1.2 Param format for S1

We fix a clear param structure so there’s no ambiguity.

For v0:

```python
# params for S1
{
  "ties": [
    {
      "example_type": "train" | "test",
      "example_index": int,          # index into context.train_examples or context.test_examples
      "pairs": [                     # list of pixel pairs to tie in that example's output grid
        ((r1, c1), (r2, c2)),
        ...
      ]
    },
    ...
  ]
}
```

* We assume **per-example** ties only (no cross-example ties for now).
* Builder will **ignore** `params` fields not matching this schema.

### 1.3 `build_S1_constraints` implementation

Signature:

```python
def build_S1_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

For each entry in `params["ties"]`:

1. Select the right `ExampleContext`:

   ```python
   if entry["example_type"] == "train":
       ex = context.train_examples[entry["example_index"]]
   else:
       ex = context.test_examples[entry["example_index"]]
   ```

2. For each pixel pair `((r1,c1), (r2,c2))`:

   * Compute pixel indices `p_idx1`, `p_idx2` in flattened grid.
     Use `p_idx = r * ex.input_W + c` (or a helper from `grid_types`).
   * For each color `c` in `0..context.C-1`, compute the y indices using `y_index(...)` for this example grid:

     * If your indexing currently assumes a single grid, just use `N = ex.input_H * ex.input_W` and index **per-example**; we can generalize later if needed.
   * Call `builder.tie_pixel_colors(p_idx1, p_idx2, C=context.C)`.

In pseudocode:

```python
for tie in params.get("ties", []):
    ex = context.train_examples[...] or context.test_examples[...]
    H, W = ex.input_H, ex.input_W
    for (r1, c1), (r2, c2) in tie["pairs"]:
        p_idx1 = r1 * W + c1
        p_idx2 = r2 * W + c2
        builder.tie_pixel_colors(p_idx1, p_idx2, context.C)
```

No mining; the pairs are given.

---

## 2️⃣ S2: component-wise recolor

**File:** `src/schemas/s2_component_recolor.py`

### 2.1 Imports

```python
from typing import Dict, Any

from src.schemas.context import TaskContext, ExampleContext
from src.constraints.builder import ConstraintBuilder
from src.constraints.indexing import y_index
from src.core.grid_types import Pixel
```

### 2.2 Param format for S2

We fix this schema:

```python
# params for S2
{
  "example_type": "train" | "test",
  "example_index": int,            # which example to apply to
  "input_color": int,              # c_in
  "size_to_color": {               # map from component size to output color
    "1": 3,
    "2": 2,
    "else": 1
  }
}
```

Notes:

* S2 is *per-example* in v0. If we want to apply the same mapping to multiple examples, we invoke S2 multiple times with different `example_index`.
* Keys in `size_to_color` are strings when coming from JSON; builder will `.get(str(size))` and fall back to `"else"`.

### 2.3 `build_S2_constraints` implementation

Signature:

```python
def build_S2_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

1. Select example:

   ```python
   if params["example_type"] == "train":
       ex = context.train_examples[params["example_index"]]
   else:
       ex = context.test_examples[params["example_index"]]
   ```

2. Extract:

   ```python
   input_color = int(params["input_color"])
   size_to_color = params["size_to_color"]  # dict[str, int]
   H, W = ex.input_H, ex.input_W
   ```

3. For each component in `ex.components`:

   * If `comp.color != input_color`, skip.

   * Determine output color:

     ```python
     size_str = str(comp.size)
     if size_str in size_to_color:
         out_color = int(size_to_color[size_str])
     else:
         out_color = int(size_to_color.get("else", 0))  # default 0 if no mapping
     ```

   * For each pixel `(r,c)` in `comp.pixels`:

     * compute `p_idx = r * W + c`.
     * call `builder.fix_pixel_color(p_idx, out_color, context.C)`.

No extra logic; no law mining.

---

## 3️⃣ Wire into `dispatch.py` and `families.py`

**File:** `src/schemas/dispatch.py`

* Import:

  ```python
  from src.schemas.s1_copy_tie import build_S1_constraints
  from src.schemas.s2_component_recolor import build_S2_constraints
  ```

* Update `BUILDERS` mapping:

  ```python
  BUILDERS = {
      "S1": build_S1_constraints,
      "S2": build_S2_constraints,
      # S3..S11 remain pointing to stubs or to be filled in later
  }
  ```

**File:** `src/schemas/families.py`

* Make sure `SchemaFamily.builder_name` for S1 and S2 matches:

  ```python
  SchemaFamily(
      id="S1",
      name="Direct pixel color tie",
      builder_name="build_S1_constraints",
      ...
  )
  SchemaFamily(
      id="S2",
      name="Component-wise recolor map",
      builder_name="build_S2_constraints",
      ...
  )
  ```

This gives Pi-agent & tooling a consistent naming.

---

## 4️⃣ Thin runner / test wiring for M3.1

We DON’T create a new runner file; we **extend an existing one** to test these builders.

**File:** `src/runners/test_context_dispatch_integration.py`

Add a small test function, e.g.:

```python
from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.constraints.builder import ConstraintBuilder, add_one_hot_constraints
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import SchemaInstance, TaskLawConfig  # if not yet defined, use simple local stubs

def test_s1_s2_on_toy_task():
    # 1. load a simple ARC training task (or a tiny hand-crafted one)
    task_id = "simple_toy"  # or real ID if you have one
    task_data = load_arc_task(task_id)
    ctx = build_task_context_from_raw(task_data)

    # 2. create a simple law config:
    #    - S2: recolor size-1 components of color 0 to color 3 in train example 0
    s2_params = {
        "example_type": "train",
        "example_index": 0,
        "input_color": 0,
        "size_to_color": {"1": 3, "else": 0},
    }

    # 3. build constraints
    builder = ConstraintBuilder()
    # Optionally add one-hot per pixel for example 0
    ex0 = ctx.train_examples[0]
    N0 = ex0.input_H * ex0.input_W
    from src.constraints.indexing import add_one_hot_constraints_for_example  # or inline
    # add_one_hot_constraints(builder, N0, ctx.C)

    # 4. apply S2
    apply_schema_instance(
        family_id="S2",
        params=s2_params,
        task_context=ctx,
        builder=builder,
    )

    # 5. assert we have some constraints
    assert len(builder.constraints) > 0
    print("S2 constraints count:", len(builder.constraints))
```

You don’t have to solve the LP here; just ensure:

* `build_task_context_from_raw` runs,
* `build_S2_constraints` runs,
* constraints are added (no crash, non-empty).

You can add a tiny S1 test similarly with a hand-coded tie.

---

## 5️⃣ Reviewer + tester checklist

For **implementer**:

* Don’t compute features inside S1/S2; only use what `TaskContext` provides.
* Don’t try to “figure out” ties or recolor maps algorithmically — only apply what `params` says.
* Keep S1 and S2 under ~150 LOC each.
* Use existing helpers from `grid_types`, `indexing`, and `features`.

For **reviewer/tester**:

1. Check `params` structure is followed exactly (keys, types).

2. In `test_context_dispatch_integration.py`, run:

   ```bash
   python -m src.runners.test_context_dispatch_integration
   ```

   or equivalent.

3. Confirm:

   * No import errors from `dispatch` or `families`.
   * `build_S1_constraints` and `build_S2_constraints` run without exceptions on a toy task.
   * `builder.constraints` becomes non-empty after applying S2 with a mapping that targets at least one component.

Optional:

* For a tiny 3×3 grid, manually verify that a size-1 component recolors as expected by inspecting constraints (if you want deeper assurance).

---
