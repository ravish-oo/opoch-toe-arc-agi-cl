## 🚧 WO-M3.6 – Sanity test harness for all schemas (S1–S11)

**File:** `src/runners/test_schemas_smoke.py`

**Goal:**

A small script that:

* constructs tiny toy tasks in memory (no JSON files),
* builds a `TaskContext` for each toy task,
* applies one `SchemaInstance` per schema S1–S11,
* checks that:

  * the builder runs without crashing,
  * some constraints are added,
  * constraints look structurally consistent (indices, coeffs).

No LP solver integration yet.

---

## 1️⃣ File structure & imports

**File:** `src/runners/test_schemas_smoke.py`

Imports:

```python
import numpy as np
from typing import Dict, Any

from src.schemas.context import build_task_context_from_raw, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import SchemaInstance, TaskLawConfig
```

Use only standard libs (`numpy`) + your own modules. No new libraries.

---

## 2️⃣ Helper: build a minimal TaskContext from in-memory grids

Inside `test_schemas_smoke.py`, define:

```python
def make_toy_task_context(
    train_inputs,
    train_outputs=None,
    test_inputs=None,
) -> TaskContext:
    """
    Utility to build a TaskContext from in-memory numpy grids.

    Args:
        train_inputs: list[Grid]
        train_outputs: list[Grid] | None
        test_inputs: list[Grid] | None

    Returns:
        TaskContext for a synthetic 'toy' task.
    """
    if train_outputs is None:
        # default: no outputs
        train = [{"input": g, "output": None} for g in train_inputs]
    else:
        assert len(train_inputs) == len(train_outputs)
        train = [{"input": g_in, "output": g_out}
                 for g_in, g_out in zip(train_inputs, train_outputs)]

    if test_inputs is None:
        test = []
    else:
        test = [{"input": g, "output": None} for g in test_inputs]

    task_data: Dict[str, Any] = {
        "train": train,
        "test": test,
    }
    return build_task_context_from_raw(task_data)
```

This uses your existing `build_task_context_from_raw` logic from M3.0.

---

## 3️⃣ One smoke-test function per schema

Define a function per schema: `smoke_S1()`, ..., `smoke_S11()`.
Each function:

* constructs a tiny grid (3×3, 4×4, etc.),
* builds `TaskContext`,
* defines `TaskLawConfig` with a single `SchemaInstance`,
* runs `apply_schema_instance` via a `ConstraintBuilder`,
* prints/returns the number of constraints.

Below I’ll outline each test with exact param shapes; exact grid values can be simple.

### 3.1 S1 – Direct pixel tie

```python
def smoke_S1():
    grid = np.array([
        [1, 2],
        [3, 4],
    ], dtype=int)

    ctx = make_toy_task_context([grid], [None])  # output not needed for smoke

    # Tie (0,0) and (1,1) in train example 0
    s1_params = {
        "example_type": "train",
        "example_index": 0,
        "ties": [
            {
                "example_type": "train",
                "example_index": 0,
                "pairs": [((0, 0), (1, 1))],
            }
        ],
    }

    law_config = TaskLawConfig(
        schema_instances=[
            SchemaInstance(family_id="S1", params=s1_params)
        ]
    )

    builder = ConstraintBuilder()
    apply_schema_instance("S1", s1_params, ctx, builder)

    print("S1 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

(You can either go through `apply_schema_instance` directly as above, or loop through `law_config.schema_instances`.)

### 3.2 S2 – Component recolor

```python
def smoke_S2():
    grid = np.array([
        [0, 0],
        [0, 0],
    ], dtype=int)

    ctx = make_toy_task_context([grid], [None])

    s2_params = {
        "example_type": "train",
        "example_index": 0,
        "input_color": 0,
        "size_to_color": {"4": 3}  # all 4 pixels form one comp, recolor to 3
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S2", s2_params, ctx, builder)

    print("S2 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.3 S3 – Bands/rows

```python
def smoke_S3():
    grid = np.zeros((4, 4), dtype=int)
    ctx = make_toy_task_context([grid], [None])

    # tie rows 0 and 2
    s3_params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [[0, 2]],
        "col_classes": [],
        "col_period_K": None,
        "row_period_K": None,
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", s3_params, ctx, builder)

    print("S3 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.4 S4 – Residue coloring

```python
def smoke_S4():
    grid = np.zeros((2, 4), dtype=int)
    ctx = make_toy_task_context([grid], [None])

    # color even columns -> 1, odd -> 2
    s4_params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {"0": 1, "1": 2},
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", s4_params, ctx, builder)

    print("S4 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.5 S5 – Template stamping

```python
def smoke_S5():
    grid = np.zeros((5, 5), dtype=int)
    ctx = make_toy_task_context([grid], [None])

    # pick some hash key that exists in ctx.train_examples[0].neighborhood_hashes
    ex = ctx.train_examples[0]
    any_pixel, any_hash = next(iter(ex.neighborhood_hashes.items()))

    s5_params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(any_hash): {
                "(0,0)": 5,  # stamp a single 5 at the seed center
            }
        },
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S5", s5_params, ctx, builder)

    print("S5 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.6 S6 – Crop ROI

```python
def smoke_S6():
    grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=int)

    ctx = make_toy_task_context([grid], [None])

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

    builder = ConstraintBuilder()
    apply_schema_instance("S6", s6_params, ctx, builder)

    print("S6 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.7 S7 – Summary

```python
def smoke_S7():
    grid = np.zeros((3, 3), dtype=int)
    ctx = make_toy_task_context([grid], [None])

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

    builder = ConstraintBuilder()
    apply_schema_instance("S7", s7_params, ctx, builder)

    print("S7 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.8 S8 – Tiling

```python
def smoke_S8():
    grid = np.zeros((4, 4), dtype=int)
    ctx = make_toy_task_context([grid], [None])

    s8_params = {
        "example_type": "train",
        "example_index": 0,
        "tile_height": 2,
        "tile_width": 2,
        "tile_pattern": {
            "(0,0)": 1,
            "(0,1)": 2,
            "(1,0)": 3,
            "(1,1)": 4,
        },
        "region_origin": "(0,0)",
        "region_height": 4,
        "region_width": 4,
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S8", s8_params, ctx, builder)

    print("S8 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.9 S9 – Cross propagation

```python
def smoke_S9():
    grid = np.zeros((5, 5), dtype=int)
    ctx = make_toy_task_context([grid], [None])

    s9_params = {
        "example_type": "train",
        "example_index": 0,
        "seeds": [
            {
                "center": "(2,2)",
                "up_color": 1,
                "down_color": 2,
                "left_color": 3,
                "right_color": 4,
                "max_up": 1,
                "max_down": 1,
                "max_left": 1,
                "max_right": 1,
            }
        ],
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S9", s9_params, ctx, builder)

    print("S9 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.10 S10 – Frame/border

```python
def smoke_S10():
    # cross-shaped component should give some border/interior pixels
    grid = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=int)

    ctx = make_toy_task_context([grid], [None])
    s10_params = {
        "example_type": "train",
        "example_index": 0,
        "border_color": 5,
        "interior_color": 7,
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S10", s10_params, ctx, builder)

    print("S10 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

### 3.11 S11 – Local codebook

```python
def smoke_S11():
    grid = np.zeros((5, 5), dtype=int)
    ctx = make_toy_task_context([grid], [None])

    ex = ctx.train_examples[0]
    any_pixel, any_hash = next(iter(ex.neighborhood_hashes.items()))

    s11_params = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(any_hash): {
                "(0,0)": 9
            }
        },
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S11", s11_params, ctx, builder)

    print("S11 constraints:", len(builder.constraints))
    assert len(builder.constraints) > 0
```

---

## 4️⃣ Main entrypoint in test_schemas_smoke.py

At bottom of `test_schemas_smoke.py`:

```python
if __name__ == "__main__":
    smoke_S1()
    smoke_S2()
    smoke_S3()
    smoke_S4()
    smoke_S5()
    smoke_S6()
    smoke_S7()
    smoke_S8()
    smoke_S9()
    smoke_S10()
    smoke_S11()
    print("All schema smoke tests ran successfully.")
```

This is the **thin runner** for schema sanity: run once, see that all builders work and produce constraints.

---

## 5️⃣ Reviewer + tester instructions

For **implementer**:

* Follow the param formats exactly as defined earlier for each S_k.
* Reuse `make_toy_task_context` to avoid duplicating context logic.
* Do *not* introduce any new feature calculations; only use what `build_task_context_from_raw` gives you.
* Keep `test_schemas_smoke.py` under ~250 LOC by avoiding over-commenting.

For **reviewer/tester**:

1. Run:

   ```bash
   python -m src.runners.test_schemas_smoke
   ```

2. Check:

   * No imports errors.
   * No exceptions in any `smoke_Sk`.
   * Each printed `Sk constraints: N` is `N > 0`.

3. Optionally, inspect one or two `builder.constraints` entries by adding a debug print in one smoke test to ensure indices and coeffs look reasonable (e.g. no negative pixel index, correct RHS).

This WO gives you a single place to quickly verify that **every schema builder S1–S11 is integrated, wired, and constraint-producing**, without depending on real ARC files or the LP solver yet.
