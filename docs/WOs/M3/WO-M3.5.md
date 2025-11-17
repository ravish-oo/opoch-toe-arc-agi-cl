## 🚧 WO-M3.5 – Implement S8 + S9 + S10 (tiling, cross propagation, frame/border)

**Files:**

* `src/schemas/s8_tiling.py`
* `src/schemas/s9_cross_propagation.py`
* `src/schemas/s10_frame_border.py`

**Goal:** implement these schemas as **param-driven builders** that:

* operate on a single `ExampleContext` at a time,
* are geometry-preserving (output shape = input_H × input_W),
* only use existing features + `ConstraintBuilder`,
* and are reachable via `solve_arc_task` with a `TaskLawConfig`.

No law discovery inside; that’s for Pi-agents later.

---

## 1️⃣ S8 – Tiling / replication

**File:** `src/schemas/s8_tiling.py`

### 1.1 Imports

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Grid
```

We rely on:

* ExampleContext.input_H, input_W, input_grid,
* ConstraintBuilder.fix_pixel_color.

### 1.2 Param format for S8

We define a clear, explicit schema:

```python
# params for S8
{
  "example_type": "train" | "test",
  "example_index": int,

  # base tile dimensions
  "tile_height": int,
  "tile_width": int,

  # base tile pattern: offsets from tile origin -> color
  "tile_pattern": {
    "(0,0)": 1,
    "(0,1)": 2,
    "(1,0)": 3,
    "(1,1)": 4
  },

  # tiling region in output grid coordinates
  "region_origin": "(r0,c0)",      # top-left of tiling region
  "region_height": int,
  "region_width": int
}
```

Conventions:

* Tile is defined relative to origin (0,0) offsets.
* `region_origin` defines top-left of tiling area in the output grid.
* Output grid shape is the same as `ex.input_H × ex.input_W`.

### 1.3 `build_S8_constraints`

Signature:

```python
def build_S8_constraints(
    context: TaskContext,
    params: Dict[str, Any],
    builder: ConstraintBuilder,
) -> None:
    ...
```

Behavior (geometry-preserving):

1. Select example:

```python
if params["example_type"] == "train":
    ex = context.train_examples[params["example_index"]]
else:
    ex = context.test_examples[params["example_index"]]

H, W = ex.input_H, ex.input_W
C = context.C
```

2. Parse params:

```python
tile_h = int(params["tile_height"])
tile_w = int(params["tile_width"])

from ast import literal_eval
region_origin = literal_eval(params["region_origin"])   # "(r0,c0)" -> (r0,c0)
r0, c0 = region_origin
region_h = int(params["region_height"])
region_w = int(params["region_width"])

raw_pattern = params.get("tile_pattern", {})
tile_pattern: Dict[Tuple[int,int], int] = {}
for k_str, color in raw_pattern.items():
    dr, dc = literal_eval(k_str)
    tile_pattern[(dr, dc)] = int(color)
```

3. For each tile position in region, stamp base tile:

Loop over all tile origins inside region:

```python
for tr in range(r0, r0 + region_h, tile_h):
    for tc in range(c0, c0 + region_w, tile_w):
        # stamp tile pattern at (tr,tc)
        for (dr, dc), color in tile_pattern.items():
            rr = tr + dr
            cc = tc + dc
            if 0 <= rr < H and 0 <= cc < W:
                p_idx = rr * W + cc
                builder.fix_pixel_color(p_idx, color, C)
```

No reference to input_grid content here — this is a pure pattern tiling.

---

## 2️⃣ S9 – Cross / plus propagation

**File:** `src/schemas/s9_cross_propagation.py`

### 2.1 Imports

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
```

We **do not** detect seeds or directions here; they are given in params.

### 2.2 Param format for S9

We define seeds directly:

```python
# params for S9
{
  "example_type": "train" | "test",
  "example_index": int,

  "seeds": [
    {
      "center": "(r,c)",

      # colors along four directions; null or missing means "do not propagate"
      "up_color": int | null,
      "down_color": int | null,
      "left_color": int | null,
      "right_color": int | null,

      # max steps in each direction (inclusive)
      "max_up": int,
      "max_down": int,
      "max_left": int,
      "max_right": int
    },
    ...
  ]
}
```

Conventions:

* Directions propagate along 4-connected axes.
* We stop either at max steps or at grid boundary.

### 2.3 `build_S9_constraints`

Signature:

```python
def build_S9_constraints(
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

H, W = ex.input_H, ex.input_W
C = context.C
```

2. Loop over seeds:

```python
from ast import literal_eval

seeds = params.get("seeds", [])
for seed in seeds:
    r_center, c_center = literal_eval(seed["center"])

    up_color = seed.get("up_color")
    down_color = seed.get("down_color")
    left_color = seed.get("left_color")
    right_color = seed.get("right_color")

    max_up = int(seed.get("max_up", 0))
    max_down = int(seed.get("max_down", 0))
    max_left = int(seed.get("max_left", 0))
    max_right = int(seed.get("max_right", 0))

    # propagate up
    if up_color is not None:
        color = int(up_color)
        for step in range(1, max_up + 1):
            rr = r_center - step
            cc = c_center
            if rr < 0:
                break
            p_idx = rr * W + cc
            builder.fix_pixel_color(p_idx, color, C)

    # propagate down
    if down_color is not None:
        color = int(down_color)
        for step in range(1, max_down + 1):
            rr = r_center + step
            cc = c_center
            if rr >= H:
                break
            p_idx = rr * W + cc
            builder.fix_pixel_color(p_idx, color, C)

    # propagate left
    if left_color is not None:
        color = int(left_color)
        for step in range(1, max_left + 1):
            rr = r_center
            cc = c_center - step
            if cc < 0:
                break
            p_idx = rr * W + cc
            builder.fix_pixel_color(p_idx, color, C)

    # propagate right
    if right_color is not None:
        color = int(right_color)
        for step in range(1, max_right + 1):
            rr = r_center
            cc = c_center + step
            if cc >= W:
                break
            p_idx = rr * W + cc
            builder.fix_pixel_color(p_idx, color, C)
```

We don’t touch the center itself; that can be handled via S5/S11 or another schema if needed.

---

## 3️⃣ S10 – Frame / border vs interior

**File:** `src/schemas/s10_frame_border.py`

### 3.1 Imports

```python
from typing import Dict, Any, Tuple

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder
from src.core.grid_types import Pixel
```

We will use:

* `ExampleContext.border_info[(r,c)]["is_border"] / ["is_interior"]` from `component_border_interior`.

### 3.2 Param format for S10

We start with a simple global frame model:

```python
# params for S10
{
  "example_type": "train" | "test",
  "example_index": int,

  # colors to use for border vs interior pixels
  "border_color": int,
  "interior_color": int
}
```

This covers many “frame around object” and “fill interior differently” tasks when applied to the relevant component. (We can refine to per-component later by adding IDs in params.)

### 3.3 `build_S10_constraints`

Signature:

```python
def build_S10_constraints(
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

H, W = ex.input_H, ex.input_W
C = context.C

border_info = ex.border_info  # (r,c) -> {"is_border": bool, "is_interior": bool}
```

2. Extract colors:

```python
border_color = int(params["border_color"])
interior_color = int(params["interior_color"])
```

3. For each pixel `(r,c)`:

* Compute `p_idx = r * W + c`.
* Look up `info = border_info.get((r,c), {})`.
* If `info.get("is_border")`:

  * `builder.fix_pixel_color(p_idx, border_color, C)`
* elif `info.get("is_interior")`:

  * `builder.fix_pixel_color(p_idx, interior_color, C)`
* else:

  * do nothing (e.g., background or outside all components).

Pseudo:

```python
for r in range(H):
    for c in range(W):
        p_idx = r * W + c
        info = border_info.get((r,c), {})
        if info.get("is_border"):
            builder.fix_pixel_color(p_idx, border_color, C)
        elif info.get("is_interior"):
            builder.fix_pixel_color(p_idx, interior_color, C)
```

---

## 4️⃣ Wire S8 + S9 + S10 into dispatch & families

**File:** `src/schemas/dispatch.py`

Add imports:

```python
from src.schemas.s8_tiling import build_S8_constraints
from src.schemas.s9_cross_propagation import build_S9_constraints
from src.schemas.s10_frame_border import build_S10_constraints
```

Extend `BUILDERS`:

```python
BUILDERS = {
    "S1": build_S1_constraints,
    "S2": build_S2_constraints,
    "S3": build_S3_constraints,
    "S4": build_S4_constraints,
    "S5": build_S5_constraints,
    "S6": build_S6_constraints,
    "S7": build_S7_constraints,
    "S8": build_S8_constraints,
    "S9": build_S9_constraints,
    "S10": build_S10_constraints,
    "S11": build_S11_constraints,
}
```

**File:** `src/schemas/families.py`

Ensure S8, S9, S10 entries match:

```python
SchemaFamily(
    id="S8",
    name="Tiling / replication",
    builder_name="build_S8_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "tile_height": "int",
        "tile_width": "int",
        "tile_pattern": "dict[str,int]",
        "region_origin": "str",
        "region_height": "int",
        "region_width": "int",
    },
    required_features=[]  # uses only params; no need for extra φ in builder
)

SchemaFamily(
    id="S9",
    name="Cross / plus propagation",
    builder_name="build_S9_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "seeds": "list[dict]",
    },
    required_features=[]  # seeds fully defined in params
)

SchemaFamily(
    id="S10",
    name="Frame / border vs interior",
    builder_name="build_S10_constraints",
    parameter_spec={
        "example_type": "str",
        "example_index": "int",
        "border_color": "int",
        "interior_color": "int",
    },
    required_features=["border_info"]
)
```

---

## 5️⃣ Kernel / runner integration

We already have:

```python
def solve_arc_task(task_id: str, law_config: TaskLawConfig) -> ConstraintBuilder:
    ...
```

No API change needed for M3.5; adding S8/S9/S10 just means:

* a `TaskLawConfig` can now contain `SchemaInstance` with `family_id="S8"`/`"S9"`/`"S10"`,
* and `apply_schema_instance` will route correctly through `dispatch.BUILDERS`.

No new runner files; testers will just call `solve_arc_task`.

---

## 6️⃣ Reviewer + tester instructions

**For implementer:**

* Implement S8, S9, S10 exactly per the param formats given.
* Do **not**:

  * detect tiles,
  * detect crosses,
  * infer border colors, etc.
* Use only:

  * `ExampleContext.input_grid`, `input_H`, `input_W`,
  * `border_info` for S10,
  * `ConstraintBuilder.fix_pixel_color` for all three.

**For reviewer/tester:**

Create small toy tests (can be in an existing integration test file) that:

### S8 toy test

* Construct a 4×4 dummy task (e.g., just a placeholder grid).
* Define an S8 law that tiles a 2×2 pattern of colors 1,2,3,4 over the whole 4×4 grid.
* `TaskLawConfig` with one S8 instance.
* Call `solve_arc_task("toy_s8", law_config)`.
* Check `len(builder.constraints) > 0` and spot-check that constraints fix 16 pixels.

### S9 toy test

* For a 5×5 grid, define one seed at center `(2,2)`, with:

  * `up_color = 1`, `max_up = 2`,
  * `down_color = 2`, `max_down = 2`,
  * `left_color = 3`, `max_left = 2`,
  * `right_color = 4`, `max_right = 2`.
* `TaskLawConfig` with one S9 instance.
* Call `solve_arc_task("toy_s9", law_config)`, ensure constraints exist.

### S10 toy test

* For a 3×3 grid with a single component (non-zero in middle), ensure `border_info` marks border vs interior.
* Use S10 with `border_color=5`, `interior_color=7`.
* Call `solve_arc_task("toy_s10", law_config)`, check that:

  * number of constraints ≥ number of pixels,
  * constraints correspond to fixing border/interior pixels.

You don’t need the LP solver yet — just confirm that:

* builders execute without error,
* constraints are emitted,
* the shapes/indices used are consistent with H, W.

---
