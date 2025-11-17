## WO-M4.3 – Integrate solver into kernel runner

### Files

* `src/runners/kernel.py`  (augment / refactor)
* `src/runners/test_kernel_smoke.py`  (new, thin runner)

---

## A. `src/runners/kernel.py`

### A.1 Imports (be explicit)

At the top of `kernel.py`, ensure you have:

```python
from typing import Dict, List

import numpy as np

from src.core.grid_types import Grid
from src.core.arc_io import load_arc_task_by_id  # you may need to add this if not present
from src.schemas.context import build_task_context_from_raw, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import TaskLawConfig
from src.solver.lp_solver import solve_constraints_for_grid
from src.solver.decoding import y_to_grid
```

> **Note:** If `load_arc_task_by_id` doesn’t exist yet, the implementer should add it in `arc_io.py` as:
>
> ```python
> def load_arc_task_by_id(task_id: str) -> dict:
>     # open arc-agi_training_challenges.json, find the task with this ID, return its raw dict
> ```

Keep `kernel.py` focused: no JSON parsing in here.

---

### A.2 Define/Refactor `solve_arc_task`

We want a **single, clear API**:

```python
def solve_arc_task(
    task_id: str,
    law_config: TaskLawConfig
) -> Dict[str, List[Grid]]:
    """
    Solve an ARC-AGI task using a given law configuration.

    Args:
        task_id: the ID of the task in arc-agi_training_challenges.json.
        law_config: TaskLawConfig listing which schema instances to apply.

    Returns:
        A dict with:
          {
            "train_outputs_pred": [Grid, ...],  # same order as training examples in the JSON
            "test_outputs_pred":  [Grid, ...]   # same order as test inputs in the JSON
          }
    """
```

### A.3 Implementation steps (per our agreed flow)

Inside `solve_arc_task`:

1. **Load raw task**

   ```python
   raw_task = load_arc_task_by_id(task_id)
   ```

   This should give you a dict with `"train"` and `"test"` entries similar to the ARC JSON format.

2. **Build TaskContext**

   ```python
   context: TaskContext = build_task_context_from_raw(raw_task)
   ```

   We assume `TaskContext` includes:

   * `train_examples: list[ExampleContext]`
   * `test_examples: list[ExampleContext]`
   * `C: int` (palette size)

3. **Prepare result containers**

   ```python
   train_outputs_pred: List[Grid] = []
   test_outputs_pred: List[Grid] = []
   ```

4. **Solve for each TRAIN example (optional but recommended)**

   This is valuable for validation even if you only care about test outputs later.

   ```python
   for ex in context.train_examples:
       H_out = ex.output_H if ex.output_H is not None else ex.input_H
       W_out = ex.output_W if ex.output_W is not None else ex.input_W
       num_pixels = H_out * W_out
       num_colors = context.C

       builder = ConstraintBuilder()

       # Apply all schema instances to this example.
       # M3 builder design: schemas may look at all train_examples to infer params,
       # but they should emit constraints for this example's output only.
       for schema_instance in law_config.schema_instances:
           apply_schema_instance(
               family_id=schema_instance.family_id,
               params=schema_instance.params,
               task_context=context,
               builder=builder,
               # if your apply_schema_instance supports it, pass the current example explicitly
               # example=ex
           )

       # Solve ILP for this grid
       y = solve_constraints_for_grid(
           builder,
           num_pixels=num_pixels,
           num_colors=num_colors,
           objective="min_sum"
       )

       # Decode to grid
       grid_pred = y_to_grid(y, H_out, W_out, num_colors)
       train_outputs_pred.append(grid_pred)
   ```

   If for some reason your current `apply_schema_instance` API requires an explicit example index, update the call accordingly (e.g. `apply_schema_instance(..., example_index=i)`).

5. **Solve for each TEST example**

   Similar to train, except `output_H`/`output_W` come from schema params or are equal to `input_H`/`input_W` for geometry-preserving schemas:

   ```python
   for ex in context.test_examples:
       H_out = ex.output_H if ex.output_H is not None else ex.input_H
       W_out = ex.output_W if ex.output_W is not None else ex.input_W
       num_pixels = H_out * W_out
       num_colors = context.C

       builder = ConstraintBuilder()

       for schema_instance in law_config.schema_instances:
           apply_schema_instance(
               family_id=schema_instance.family_id,
               params=schema_instance.params,
               task_context=context,
               builder=builder,
               # example=ex  # if supported
           )

       y = solve_constraints_for_grid(
           builder,
           num_pixels=num_pixels,
           num_colors=num_colors,
           objective="min_sum"
       )

       grid_pred = y_to_grid(y, H_out, W_out, num_colors)
       test_outputs_pred.append(grid_pred)
   ```

6. **Return results**

   ```python
   return {
       "train_outputs_pred": train_outputs_pred,
       "test_outputs_pred":  test_outputs_pred,
   }
   ```

> **Note:** We are explicitly doing **one ILP per example** (per output grid).
> The schema builders can still consider **all** train examples when building constraints; we’re only keeping the numeric solve step per-grid for simplicity.

---

## B. `src/runners/test_kernel_smoke.py` – Thin runner

This is a small script to check that:

* `solve_arc_task` runs end-to-end *without crashing*,
* it returns the right number of grids.

We deliberately keep the expectations weak here, because real correctness depends on law_config and schema logic. This is a **smoke test**, not a full ARC judge.

### B.1 Imports

```python
from src.runners.kernel import solve_arc_task
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.core.arc_io import list_available_task_ids  # if you have it, else hardcode one
```

If `list_available_task_ids` doesn’t exist, you can skip it and manually specify a known `task_id` that exists in your training JSON.

### B.2 Minimal law_config (dummy)

We assume you already have at least one schema family implemented (e.g. S1 or S2). For a smoke test, you can:

* pick a simple training task where that schema is applicable (your team will know one), and
* a simple `TaskLawConfig` with a single `SchemaInstance` using default-ish params.

Example stub (you’ll adjust params as needed):

```python
def make_dummy_law_config() -> TaskLawConfig:
    # Example: use S1 with some basic parameters;
    # this depends on your S1 parameter_spec.
    return TaskLawConfig(
        schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={
                    # Fill with minimal params accepted by build_S1_constraints
                    # e.g.: "mode": "copy_input_to_output"
                }
            )
        ]
    )
```

### B.3 Test function

```python
def test_kernel_smoke():
    # Replace with a real task_id from arc-agi_training_challenges.json
    task_id = "some_known_simple_task_id"

    law_config = make_dummy_law_config()
    result = solve_arc_task(task_id, law_config)

    assert "train_outputs_pred" in result
    assert "test_outputs_pred" in result

    train_out = result["train_outputs_pred"]
    test_out  = result["test_outputs_pred"]

    # Basic structural checks:
    print(f"Train outputs predicted: {len(train_out)}")
    print(f"Test outputs predicted:  {len(test_out)}")

    # You can add more checks once you know the expected grid shapes.
    print("test_kernel_smoke: OK")
```

### B.4 Main

```python
if __name__ == "__main__":
    test_kernel_smoke()
```

### B.5 Reviewer/tester instructions

* Make sure at least one schema family (say S1) is implemented and its parameters are understood.

* Pick a **real training task_id** from `arc-agi_training_challenges.json` that is compatible with that schema.

* Update `task_id` and `make_dummy_law_config()` accordingly.

* Then run:

  ```bash
  python -m src.runners.test_kernel_smoke
  ```

* Expected:

  * No exceptions from solver / decoding / kernel,
  * Printed counts for predicted train/test outputs,
  * A final `test_kernel_smoke: OK`.

If this fails:

* Check:

  * `load_arc_task_by_id` returns a reasonable structure,
  * `build_task_context_from_raw` builds the expected number of examples,
  * `apply_schema_instance` isn’t raising due to missing params,
  * `solve_constraints_for_grid` and `y_to_grid` are wired correctly.

---
# clarificaitons
These are good catches from your implementer. I’ll answer them one by one and tie them back to the math kernel, and I’ll be explicit about what should change in the kernel design so nothing is silent or fuzzy.

---

## 1. `load_arc_task_by_id` vs existing `load_arc_task`

> There’s already `load_arc_task(task_id, challenges_path)` in `context.py`.

✅ Then we **do not** introduce a new `load_arc_task_by_id`. The kernel should use the existing function:

* In `kernel.py`:

```python
from src.schemas.context import load_arc_task, build_task_context_from_raw

raw_task = load_arc_task(task_id, challenges_path=Path("data/arc-agi_training_challenges.json"))
context = build_task_context_from_raw(raw_task)
```

So: **reuse existing IO**; no duplication.

---

## 2. Output dimensions for test (non-geometry-preserving schemas)

You’re right: my v0 code snippet

```python
H_out = ex.output_H if ex.output_H is not None else ex.input_H
W_out = ex.output_W if ex.output_W is not None else ex.input_W
```

assumes geometry-preserving **as a fallback**, which is not enough for S6/S7 (crop/summary).

Here’s the clean way to think about it, consistent with the math kernel:

* **Output shape is part of the law**, not something the kernel guesses.

  * For training examples:

    * `ExampleContext.output_H/output_W` are known from the ground truth → the law learning phase must match them.
  * For test examples:

    * S6/S7 (crop/summary) schemas must already encode the target output shape in their **params**, learned from training.
    * Law learning / Pi-agent step should set those params (e.g. `{"output_height": 3, "output_width": 3}`).

So in `solve_arc_task` for **test**:

* Kernel should do something like:

```python
# Pseudocode
H_out, W_out = infer_output_shape_from_law_config(law_config, example_index, example_type="test")
if H_out is None or W_out is None:
    # fallback: geometry-preserving
    H_out, W_out = ex.input_H, ex.input_W
```

Key points:

* For *geometry-preserving* schemas only, the fallback `output = input` is fine.
* For crop/summary schemas (S6/S7), output dims **must** come from law_config schema params that the Pi-agent learned from train pairs.
* Kernel itself **does not invent shapes**; it just reads them from `ExampleContext` (train) or `schema_params` (test).

For now, in a hand-run flow, you can keep:

* train: `H_out/W_out` from `ExampleContext` (since true outputs known),
* test: law_config must include output dims in params for S6/S7 tasks.

---

## 3. Per-example vs global constraints (example_index / example_type in schema params)

You’re absolutely right: many schemas **look at all train examples** to mine parameters, but constraints for a given solve should target **one example’s output**.

Right now you have:

* Some builders reading `example_type` / `example_index` from `schema_params`.

### What the math kernel wants

Conceptually:

1. **Law mining** (across all train examples):

   * Learn *global* parameters: e.g. “color 0 components: size1→3, size2→2, else→1”, or “crop to bbox of largest object”.
   * This is cross-example.

2. **Constraint emission** (per example):

   * For each example, use the *same* global params to emit constraints on that example’s y variables.
   * No need to rewrite params per example; the call knows which example it’s for.

### Clean design for builders

Instead of burying `example_index` inside `schema_params`, do this:

* Change `apply_schema_instance` (and builder signatures) to something like:

```python
def apply_schema_instance(
    family_id: str,
    params: dict,
    task_context: TaskContext,
    builder: ConstraintBuilder,
    example_type: str,      # "train" or "test"
    example_index: int      # which example within that type
):
    ...
```

* Then each builder sees:

  * `task_context` (all examples, to mine invariants),
  * `example_type/example_index` (which example’s y to constrain),
  * `params` (global law params, e.g. size→color map, crop rule).

That matches the math spec:

* B(T) is global at the conceptual level, but we’re free to emit rows “per example” as long as they use the same parameters.

So **my correction** to the kernel flow:

```python
for i, ex in enumerate(context.train_examples):
    builder = ConstraintBuilder()
    for schema_instance in law_config.schema_instances:
        apply_schema_instance(
            family_id=schema_instance.family_id,
            params=schema_instance.params,
            task_context=context,
            builder=builder,
            example_type="train",
            example_index=i,
        )
    y = solve_constraints_for_grid(...)
    ...
```

And similarly for test examples. No need to make multiple law_configs or mutate `schema_params` per example; the example index is passed explicitly.

---

## 4. Infeasibility handling (and “no silent fail”)

Math kernel assumes: if laws are correct, the constraint system is feasible and yields a unique one-hot y.

In reality:

* Pi-agent may pick wrong schema/params → constraints become infeasible or inconsistent.

M4.1 already defines:

* `InfeasibleModelError` from `lp_solver`.

We should explicitly define in `kernel.solve_arc_task`:

* For now (v0), **do not swallow** this:

  * Either:

    * let `InfeasibleModelError` propagate up,
    * or catch it and re-raise a `TaskSolveError` that includes:

      * `task_id`,
      * `example_type`,
      * `example_index`,
      * maybe a message about which schema families were active.

Example:

```python
try:
    y = solve_constraints_for_grid(...)
except InfeasibleModelError as e:
    raise TaskSolveError(
        f"Infeasible for task {task_id}, example_type={example_type}, example_index={i}, reason={e}"
    )
```

Later, when a Pi-agent wraps `solve_arc_task`, *that* agent will:

* catch `TaskSolveError`,
* examine its message / payload,
* and treat it as a signal to:

  * refine schema/params for that task,
  * or flag the task as “not yet solved by current law_config”.

Key point:
**No silent fallbacks.** Every failure should be:

* either a thrown error,
* or a clearly tagged result (if you later choose to return an error structure).

At the kernel layer, I’d strongly favor raising an exception with rich context.

---

### TL;DR of what to adjust

1. **Reuse** `load_arc_task` from `context.py` in `kernel.py`.
2. For **output dims**:

   * train: use `ExampleContext.output_H/W` (true labels),
   * test: read from law_config schema params (for S6/S7), or fall back to input H/W only for geometry-preserving tasks.
3. Refactor schema execution to pass `example_type` / `example_index` **as arguments to `apply_schema_instance`**, not as persistent params.
4. In `solve_arc_task`, **never swallow solver infeasibility**:

   * raise a structured exception (e.g. `TaskSolveError`), so a Pi-agent can see exactly what failed and where.

All of that is fully consistent with the math kernel your friend wrote; we’re just making the implementation-level wiring match that structure and be Pi-agent friendly.
