## 🚧 WO-M3.4 – Implement S6 + S7 (cropping & summary) in a param-driven way

### High-level behavior

* **S6 (crop ROI)**:

  * For a given example:

    * output grid has shape `(output_H, output_W)`.
    * each output pixel `(r_out, c_out)` either:

      * maps to some input pixel `(r_in, c_in)` inside an ROI → output color = `input_grid[r_in, c_in]`,
      * or is background → output color = `background_color`.
    * Builder only **fixes** output colors; no search.

* **S7 (summary grid)**:

  * For a given example:

    * output grid is a K×L “summary” grid.
    * each summary cell `(r_out, c_out)` has a **precomputed summary color**.
  * Builder just fixes each output pixel to its summary color.

The Pi-agent / law discovery later will decide *which* ROI or summary colors to use and fill the `params`.

---

## 1️⃣ S6 – Cropping to ROI

**File:** `src/schemas/s6_crop_roi.py`

### 1.1 Imports

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Grid
```

We reuse:

* `TaskContext` / `ExampleContext` to access `input_grid`.
* `ConstraintBuilder.fix_pixel_color` to constrain outputs.

### 1.2 Param format for S6

Define a clear, explicit schema:

```python
# params for S6
{
  "example_type": "train" | "test",   # which side we’re applying to
  "example_index": int,

  # dimensions of the output grid for this example
  "output_height": int,
  "output_width": int,

  # background color for pixels not belonging to the ROI
  "background_color": int,

  # mapping from output coordinates -> input coordinates
  # for cropped pixels
  "out_to_in": {
    "(0,0)": "(2,3)",
    "(0,1)": "(2,4)",
    ...
  }
}
```

Conventions:

* Keys of `out_to_in` are string `"(r_out,c_out)"`.
* Values are string `"(r_in,c_in)"`.
* If an output coordinate `(r_out, c_out)` is **not** in `out_to_in`, builder will set it to `background_color`.

### 1.3 `build_S6_constraints` implementation

Signature:

```python
def build_S6_constraints(
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

input_grid: Grid = ex.input_grid
input_H, input_W = ex.input_H, ex.input_W
C = context.C
```

2. **Get output grid shape & mapping:**

```python
out_H = int(params["output_height"])
out_W = int(params["output_width"])
background = int(params["background_color"])
out_to_in_raw = params.get("out_to_in", {})
```

3. **Parse mapping into tuple keys:**

```python
from ast import literal_eval  # stdlib, safe for simple tuples

out_to_in: Dict[Tuple[int,int], Tuple[int,int]] = {}
for k_str, v_str in out_to_in_raw.items():
    r_out, c_out = literal_eval(k_str)  # e.g. "(0,1)" -> (0,1)
    r_in, c_in  = literal_eval(v_str)
    out_to_in[(r_out, c_out)] = (r_in, c_in)
```

4. **For each output position `(r_out,c_out)`**:

* Compute the flattened index `p_idx_out = r_out * out_W + c_out`.
* If `(r_out,c_out)` in `out_to_in`:

  * get `(r_in,c_in)`,
  * **read input color**: `color = int(input_grid[r_in, c_in])`,
  * set `builder.fix_pixel_color(p_idx_out, color, C)`.
* Else:

  * set `builder.fix_pixel_color(p_idx_out, background, C)`.

Pseudo:

```python
for r_out in range(out_H):
    for c_out in range(out_W):
        p_idx_out = r_out * out_W + c_out
        key = (r_out, c_out)
        if key in out_to_in:
            r_in, c_in = out_to_in[key]
            # safe-guard in case mapping is out of bounds
            if 0 <= r_in < input_H and 0 <= c_in < input_W:
                color = int(input_grid[r_in, c_in])
            else:
                color = background
        else:
            color = background
        builder.fix_pixel_color(p_idx_out, color, C)
```

No inference; all ROI/select logic lives in `params`.

---

## 2️⃣ S7 – Aggregation / summary grids

**File:** `src/schemas/s7_aggregation.py`

### 2.1 Imports

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
```

We assume Pi-agent or offline code already computed the summary colors.

### 2.2 Param format for S7

Schema:

```python
# params for S7
{
  "example_type": "train" | "test",
  "example_index": int,

  "output_height": int,
  "output_width": int,

  # mapping from output cell -> summary color
  "summary_colors": {
    "(0,0)": 3,
    "(0,1)": 0,
    "(1,0)": 2,
    "(1,1)": 5
  }
}
```

Conventions:

* Keys are `"(r_out,c_out)"`.
* Values are ints (colors).
* If an output cell does **not** appear in `summary_colors`, builder does nothing for that cell (we can choose to either:

  * leave it unconstrained, or
  * force it to background; for now, we **leave it unconstrained** so higher-level logic can decide).

### 2.3 `build_S7_constraints` implementation

Signature:

```python
def build_S7_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior:

1. **Select example and get dims:**

```python
if params["example_type"] == "train":
    ex = context.train_examples[params["example_index"]]
else:
    ex = context.test_examples[params["example_index"]]

out_H = int(params["output_height"])
out_W = int(params["output_width"])
C = context.C
```

2. **Parse summary colors:**

```python
from ast import literal_eval

raw_summaries = params.get("summary_colors", {})
summary_colors: Dict[Tuple[int,int], int] = {}

for k_str, color in raw_summaries.items():
    r_out, c_out = literal_eval(k_str)
    summary_colors[(r_out, c_out)] = int(color)
```

3. **Apply constraints:**

For each `(r_out, c_out), color` in `summary_colors`:

* if `0 ≤ r_out < out_H` and `0 ≤ c_out < out_W`:

  * `p_idx_out = r_out * out_W + c_out`
  * `builder.fix_pixel_color(p_idx_out, color, C)`

Pseudo:

```python
for (r_out, c_out), color in summary_colors.items():
    if 0 <= r_out < out_H and 0 <= c_out < out_W:
        p_idx_out = r_out * out_W + c_out
        builder.fix_pixel_color(p_idx_out, color, C)
```

(We do **not** constrain other cells; they can be handled by other schemas or left to solver.)

---

## 3️⃣ Wire S6 + S7 into dispatch & families

**File:** `src/schemas/dispatch.py`

* Add imports:

```python
from src.schemas.s6_crop_roi import build_S6_constraints
from src.schemas.s7_aggregation import build_S7_constraints
```

* Extend `BUILDERS` mapping:

```python
BUILDERS = {
    "S1": build_S1_constraints,
    "S2": build_S2_constraints,
    "S3": build_S3_constraints,
    "S4": build_S4_constraints,
    "S5": build_S5_constraints,
    "S6": build_S6_constraints,
    "S7": build_S7_constraints,
    # S8..S10
    "S11": build_S11_constraints,
}
```

**File:** `src/schemas/families.py`

* Make sure S6 and S7 entries reflect param spec:

```python
SchemaFamily(
    id="S6",
    name="Crop to ROI / dominant object",
    builder_name="build_S6_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "output_height": "int",
        "output_width": "int",
        "background_color": "int",
        "out_to_in": "dict[str,str]"
    },
    required_features=["input_grid"]  # really just need access to input_grid colors
)

SchemaFamily(
    id="S7",
    name="Aggregation / summary grid",
    builder_name="build_S7_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "output_height": "int",
        "output_width": "int",
        "summary_colors": "dict[str,int]"
    },
    required_features=[]  # uses only params for colors; input features used upstream
)
```

---

## 4️⃣ Kernel & runner notes for this WO

No change in kernel API:

* `solve_arc_task(task_id, law_config)` already:

  * loads task,
  * builds `TaskContext`,
  * iterates over `schema_instances` and calls `apply_schema_instance`,
  * returns `ConstraintBuilder`.

Now, if `law_config` includes S6/S7 instances, they’ll be applied just like S1–S5.

We don’t add a new runner file here; testers will call `solve_arc_task` with tailored `TaskLawConfig` as below.

---

## 5️⃣ Reviewer + tester instructions

**For implementer:**

* Stick exactly to the param formats above for S6 & S7.
* Don’t infer bounding boxes or summary colors; use what’s in `params`.
* Use only:

  * `ex.input_grid` for reading input colors (S6),
  * `ConstraintBuilder.fix_pixel_color` for constraints.
* No new math or algorithms beyond mapping and bounds checks.

**For reviewer/tester:**

Create a small **toy task** harness (e.g., in `test_context_dispatch_integration.py` or a dedicated test):

1. **Toy crop test (S6)**

   * Define a one-task structure in memory:

     ```python
     grid = np.array([
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
     ], dtype=int)
     task_data = {
         "train": [{"input": grid, "output": None}],  # output not needed to build context
         "test":  []
     }
     ```

   * Build `TaskContext`:

     ```python
     ctx = build_task_context_from_raw(task_data)
     ```

   * Define a `TaskLawConfig` with one S6 instance that crops the central 2×2 square:

     ```python
     from src.catalog.types import SchemaInstance, TaskLawConfig

     s6_params = {
         "example_type": "train",
         "example_index": 0,
         "output_height": 2,
         "output_width": 2,
         "background_color": 0,
         "out_to_in": {
             "(0,0)": "(1,1)",
             "(0,1)": "(1,2)",
             "(1,0)": "(2,1)",
             "(1,1)": "(2,2)",
         },
     }

     law_config = TaskLawConfig(
         schema_instances=[SchemaInstance(family_id="S6", params=s6_params)]
     )
     ```

   * Call kernel:

     ```python
     builder = solve_arc_task("toy_crop", law_config)
     print(len(builder.constraints))
     assert len(builder.constraints) > 0
     ```

   * (Optional) Inspect constraints to ensure they fix 4 pixels and nothing else.

2. **Toy summary test (S7)**

   * Use `output_height=2, output_width=2`, and `summary_colors`:

     ```python
     s7_params = {
         "example_type": "train",
         "example_index": 0,
         "output_height": 2,
         "output_width": 2,
         "summary_colors": {
             "(0,0)": 1,
             "(0,1)": 2,
             "(1,0)": 3,
             "(1,1)": 4,
         },
     }
     law_config = TaskLawConfig(
         schema_instances=[SchemaInstance(family_id="S7", params=s7_params)]
     )
     builder = solve_arc_task("toy_summary", law_config)
     assert len(builder.constraints) >= 4
     ```

If these tests pass (no crashes, constraints present and plausible), S6/S7 are correctly wired and ready for Pi-agent use later.

---
