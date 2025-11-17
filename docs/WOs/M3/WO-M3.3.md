## 🚧 WO-M3.3 – Implement S5 + S11 (template stamping & local codebook), wire dispatch, update kernel tests

### High-level behavior (fixed assumptions for v0)

* Both S5 and S11 are **geometry-preserving**: they write into output grids of the same shape as input.
* Builders **do not mine** templates from training; they **only apply** templates given in `params`.
* Templates are specified relative to a center pixel (seed/center), as a dict of `(dr, dc) → color`.

---

## 1️⃣ S5 – Template stamping (seed → template)

**File:** `src/schemas/s5_template_stamping.py`

### 1.1 Imports

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Pixel
```

We rely on:

* `ExampleContext.neighborhood_hashes` from M1/M3.0,
* `ExampleContext.input_H`, `input_W` for shape,
* `ConstraintBuilder.fix_pixel_color`.

No new algorithms, just matching and stamping.

### 1.2 Param format for S5

We fix a precise param schema:

```python
# params for S5
{
  "example_type": "train" | "test",
  "example_index": int,

  # mapping from seed neighborhood hash to a template
  # Each template is a dict[(dr, dc)] -> color
  # meaning: at relative offset (dr,dc) from the seed center, set this color.
  "seed_templates": {
    "123456": { "(0,0)": 5, "(0,1)": 5, "(1,0)": 5, "(1,1)": 5 },
    "987654": { "(0,0)": 2, "(-1,0)": 2, "(1,0)": 2, "(0,-1)": 2, "(0,1)": 2 }
  }
}
```

Conventions:

* Keys of `seed_templates` are **stringified hashes** of 3×3 neighborhoods (from `neighborhood_hashes`).
* Template offsets are string `"(<dr>,<dc>)"`, e.g. `"(0,1)"`, `"(-1,0)"`.
* Colors are ints.

Builder will:

* For each pixel whose `neighborhood_hashes[(r,c)]` matches a seed hash,
* Stamp the corresponding template around `(r,c)`.

### 1.3 `build_S5_constraints` implementation

Signature:

```python
def build_S5_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

1. **Select example:**

```python
if params["example_type"] == "train":
    ex = context.train_examples[params["example_index"]]
else:
    ex = context.test_examples[params["example_index"]]

H, W = ex.input_H, ex.input_W
C = context.C
nbh = ex.neighborhood_hashes  # dict[(r,c)] -> hash_value (int or similar)
```

2. **Parse templates:**

Convert the stringified offset keys into `(dr,dc)` tuples:

```python
raw_templates = params.get("seed_templates", {})
parsed_templates: Dict[int, Dict[Tuple[int,int], int]] = {}

for hash_str, offset_map in raw_templates.items():
    h_val = int(hash_str)
    tmpl: Dict[Tuple[int,int], int] = {}
    for offset_str, color in offset_map.items():
        # offset_str like "(0,1)" or "(-1,0)"
        dr_dc = offset_str.strip("()").split(",")
        dr = int(dr_dc[0])
        dc = int(dr_dc[1])
        tmpl[(dr, dc)] = int(color)
    parsed_templates[h_val] = tmpl
```

3. **Stamp templates for each matching seed:**

For each pixel `(r,c)`:

* Get its neighborhood hash `h = nbh[(r,c)]` (if not present, skip).
* If `h` in `parsed_templates`, get `tmpl = parsed_templates[h]`.
* For each `(dr,dc), color` in `tmpl`:

  * Compute target position: `rr = r + dr`, `cc = c + dc`.
  * If `(rr,cc)` is inside the grid (`0 ≤ rr < H`, `0 ≤ cc < W`):

    * Compute `p_idx = rr * W + cc`.
    * Call `builder.fix_pixel_color(p_idx, color, C)`.

Pseudo:

```python
for (r, c), h_val in nbh.items():
    if h_val not in parsed_templates:
        continue
    tmpl = parsed_templates[h_val]
    for (dr, dc), color in tmpl.items():
        rr = r + dr
        cc = c + dc
        if 0 <= rr < H and 0 <= cc < W:
            p_idx = rr * W + cc
            builder.fix_pixel_color(p_idx, color, C)
```

No law discovery, just stamping.

---

## 2️⃣ S11 – Local codebook (hash → template)

**File:** `src/schemas/s11_local_codebook.py`

S11 is structurally similar to S5, but instead of “seed types”, it’s “every neighborhood hash is a symbol with a template”.

### 2.1 Imports

Same style:

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Pixel
```

### 2.2 Param format for S11

We define:

```python
# params for S11
{
  "example_type": "train" | "test",
  "example_index": int,

  # mapping from any neighborhood hash H to its output template P(H)
  "hash_templates": {
    "123456": { "(0,0)": 5, "(0,1)": 5, "(1,0)": 5, "(1,1)": 5 },
    "222222": { "(0,0)": 0 },  # maybe overwrite center only
    ...
  }
}
```

Same conventions as S5:

* Keys of `hash_templates` are stringified hashes.
* Offsets are `"(dr,dc)"` strings.
* Colors are ints.

### 2.3 `build_S11_constraints` implementation

Signature:

```python
def build_S11_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

1. **Select example & get hashes:**

Same as S5:

```python
if params["example_type"] == "train":
    ex = context.train_examples[params["example_index"]]
else:
    ex = context.test_examples[params["example_index"]]

H, W = ex.input_H, ex.input_W
C = context.C
nbh = ex.neighborhood_hashes
```

2. **Parse templates:**

Same decoding logic, just different param key:

```python
raw_templates = params.get("hash_templates", {})
parsed_templates: Dict[int, Dict[Tuple[int,int], int]] = {}

for hash_str, offset_map in raw_templates.items():
    h_val = int(hash_str)
    tmpl: Dict[Tuple[int,int], int] = {}
    for offset_str, color in offset_map.items():
        dr_dc = offset_str.strip("()").split(",")
        dr = int(dr_dc[0])
        dc = int(dr_dc[1])
        tmpl[(dr, dc)] = int(color)
    parsed_templates[h_val] = tmpl
```

3. **Apply templates for each pixel hash:**

For each `(r,c)`:

* Get `h = nbh[(r,c)]`.
* If `h` in `parsed_templates`, apply `tmpl` exactly as in S5:

```python
for (r, c), h_val in nbh.items():
    if h_val not in parsed_templates:
        continue
    tmpl = parsed_templates[h_val]
    for (dr, dc), color in tmpl.items():
        rr = r + dr
        cc = c + dc
        if 0 <= rr < H and 0 <= cc < W:
            p_idx = rr * W + cc
            builder.fix_pixel_color(p_idx, color, C)
```

Again, no mining — just codebook application.

---

## 3️⃣ Wire S5 + S11 into dispatch & families

**File:** `src/schemas/dispatch.py`

* Import:

```python
from src.schemas.s5_template_stamping import build_S5_constraints
from src.schemas.s11_local_codebook import build_S11_constraints
```

* Update `BUILDERS`:

```python
BUILDERS = {
    "S1": build_S1_constraints,
    "S2": build_S2_constraints,
    "S3": build_S3_constraints,
    "S4": build_S4_constraints,
    "S5": build_S5_constraints,
    # S6..S10
    "S11": build_S11_constraints,
}
```

**File:** `src/schemas/families.py`

* Ensure entries for S5/S11 specify these builders and param specs explicitly, e.g.:

```python
SchemaFamily(
    id="S5",
    name="Template stamping (seed → template)",
    builder_name="build_S5_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "seed_templates": "dict[str, dict[str,int]]"
    },
    required_features=["neighborhood_hashes"]
)

SchemaFamily(
    id="S11",
    name="Local neighborhood codebook",
    builder_name="build_S11_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "hash_templates": "dict[str, dict[str,int]]"
    },
    required_features=["neighborhood_hashes"]
)
```

---

## 4️⃣ Kernel touch (no API change, just readiness)

We already have `solve_arc_task(task_id, law_config)` in `src/runners/kernel.py`.

For M3.3, **no API change** needed; just ensure we can include S5/S11 instances in `TaskLawConfig` and they will be applied through `apply_schema_instance` and these new builders.

Nothing extra to add here beyond keeping imports valid.

---

## 5️⃣ Reviewer + tester instructions

**For implementer:**

* Implement S5 and S11 *exactly* as per the param formats above.
* Do not:

  * try to identify seeds from training,
  * infer templates from outputs,
  * or modify shapes.
* Use `fix_pixel_color` to enforce template colors (not `forbid` loops).

**For reviewer / tester:**

1. Create a tiny toy task (can be a fake one in a test script):

   * Input grid 5×5.
   * Pick one pixel `(2,2)` with a distinctive 3×3 pattern (so it gets a known hash).
   * Handcraft `neighborhood_hashes` or use the existing feature functions.

2. Build `TaskContext` for that toy task with `build_task_context_from_raw`.

3. Construct a `TaskLawConfig` with one S5 instance:

   ```python
   from src.catalog.types import SchemaInstance, TaskLawConfig

   s5_params = {
       "example_type": "train",
       "example_index": 0,
       "seed_templates": {
           str(hash_center): { "(0,0)": 5, "(0,1)": 5 }  # choose a hash_center from ex.neighborhood_hashes[(2,2)]
       }
   }

   law_config = TaskLawConfig(
       schema_instances=[SchemaInstance(family_id="S5", params=s5_params)]
   )
   ```

4. Call:

   ```python
   from src.runners.kernel import solve_arc_task
   builder = solve_arc_task(task_id, law_config)
   print(len(builder.constraints))
   ```

5. Verify:

   * No exceptions.
   * `len(builder.constraints) > 0`.
   * Optionally, inspect a few constraints corresponding to pixels `(2,2)` and `(2,3)` to confirm they are fixed to color 5.

Repeat with S11 by using `hash_templates` instead of `seed_templates` and applying to all occurrences of that hash.

---
