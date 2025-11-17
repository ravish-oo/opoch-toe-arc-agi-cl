## 🔹 Expanded WO-M5.2 – Extend kernel to return diagnostics + outputs

### File: `src/runners/kernel.py` (augment / refactor)

**Goal:** evolve the kernel so that **one public entrypoint** returns:

* predicted grids (train + test), and
* a fully-populated `SolveDiagnostics` object.

---

### 1. Imports (be explicit)

At the top of `src/runners/kernel.py`, ensure these imports exist:

```python
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.core.grid_types import Grid
from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import TaskLawConfig
from src.runners.results import (
    SolveDiagnostics,
    compute_train_mismatches,
)
from src.solver.lp_solver import solve_constraints_for_grid  # or your actual function
from src.solver.decoding import y_to_grid
```

> ✅ Use only these existing modules; no new libraries or algorithm implementations here.

If your actual solver function name differs, adapt the import and usage accordingly, but keep everything consistent.

---

### 2. Define the new public API: `solve_arc_task_with_diagnostics`

Add this function to `kernel.py` (or refactor existing `solve_arc_task` into this shape):

```python
def solve_arc_task_with_diagnostics(
    task_id: str,
    law_config: TaskLawConfig,
    use_training_labels: bool = False,
) -> Tuple[Dict[str, List[Grid]], SolveDiagnostics]:
    """
    High-level entrypoint for solving an ARC task with a given law_config.

    Args:
        task_id: ID key from arc-agi_training_challenges.json, etc.
        law_config: TaskLawConfig containing the schema instances to apply.
        use_training_labels: If True, compare predicted train outputs with
                             ground truth labels and populate mismatch info.

    Returns:
        outputs: {
            "train": [Grid, ...],  # predicted outputs for each train example (in order)
            "test":  [Grid, ...],  # predicted outputs for each test input
        }
        diagnostics: SolveDiagnostics with status, solver_status, counts, mismatches, etc.
    """
```

Then implement the body as follows.

---

### 3. Implementation steps (no wiggle room)

Inside `solve_arc_task_with_diagnostics`:

#### Step 3.1 – Load raw task and build TaskContext

```python
    # 1) Load raw task JSON (train inputs/outputs, test inputs)
    raw_task = load_arc_task(task_id)

    # 2) Build TaskContext from raw_task
    task_context: TaskContext = build_task_context_from_raw(raw_task)
```

Assume `build_task_context_from_raw` already knows how to:

* create `TaskContext.train_examples` and `TaskContext.test_examples`,
* compute all features from M1.

#### Step 3.2 – Initialize containers

```python
    train_outputs_pred: List[Grid] = []
    test_outputs_pred: List[Grid] = []

    # For diagnostics
    total_constraints = 0
    total_variables = 0
    schema_ids_used = [inst.family_id for inst in law_config.schema_instances]

    # We will set these based on solver and mismatch results
    status: str = "ok"
    solver_status_str: str = "Unknown"
    train_mismatches = []
    error_message: str | None = None
```

---

#### Step 3.3 – Solve for every train example

We’ll do **one ILP per example** (as per M4 design).

For each `ExampleContext` in `task_context.train_examples`:

1. Build constraints via schemas into a fresh `ConstraintBuilder`.
2. Call the ILP solver for that example.
3. Decode `y` to a Grid and append to `train_outputs_pred`.
4. Accumulate `num_constraints`, `num_variables`.

Example:

```python
    try:
        # 3) Solve for each train example
        for ex_idx, ex_ctx in enumerate(task_context.train_examples):
            builder = ConstraintBuilder()

            # Apply all schema instances to this example
            for schema_inst in law_config.schema_instances:
                apply_schema_instance(
                    family_id=schema_inst.family_id,
                    params=schema_inst.params,
                    task_context=task_context,
                    example_index=ex_idx,
                    builder=builder,
                )

            # Count constraints
            num_constraints_for_example = len(builder.constraints)
            total_constraints += num_constraints_for_example

            # Call solver for this example.
            # solve_constraints_for_grid is assumed to:
            #   - create binary y[p,c] vars,
            #   - add one-hot constraints,
            #   - add all LinearConstraints,
            #   - return (y_flat, solver_status_str_single).
            y_flat, solver_status_str_single = solve_constraints_for_grid(
                builder=builder,
                example_context=ex_ctx,
                num_colors=task_context.C,
            )
            solver_status_str = solver_status_str_single  # keep last, or combine later

            # Decode y_flat to a grid of the correct output shape
            H_out = ex_ctx.output_H if ex_ctx.output_H is not None else ex_ctx.input_H
            W_out = ex_ctx.output_W if ex_ctx.output_W is not None else ex_ctx.input_W

            grid_pred: Grid = y_to_grid(
                y=y_flat,
                H=H_out,
                W=W_out,
                C=task_context.C,
            )
            train_outputs_pred.append(grid_pred)

            # num_variables for this example = H_out * W_out * C
            total_variables += H_out * W_out * task_context.C
```

**Notes:**

* `apply_schema_instance` must support an `example_index` argument now (if it doesn’t yet, update its signature and internals accordingly).
* `solve_constraints_for_grid` must accept `builder`, `example_context`, and `num_colors`, and return `(y_flat, solver_status_str)`.

---

#### Step 3.4 – Solve for every test example

Same as training, but with `output_grid` in `ExampleContext` usually `None`, and results go into `test_outputs_pred`.

```python
        # 4) Solve for each test example
        for ex_idx, ex_ctx in enumerate(task_context.test_examples):
            builder = ConstraintBuilder()

            for schema_inst in law_config.schema_instances:
                apply_schema_instance(
                    family_id=schema_inst.family_id,
                    params=schema_inst.params,
                    task_context=task_context,
                    example_index=len(task_context.train_examples) + ex_idx,
                    builder=builder,
                )

            num_constraints_for_example = len(builder.constraints)
            total_constraints += num_constraints_for_example

            y_flat, solver_status_str_single = solve_constraints_for_grid(
                builder=builder,
                example_context=ex_ctx,
                num_colors=task_context.C,
            )
            solver_status_str = solver_status_str_single

            H_out = ex_ctx.output_H if ex_ctx.output_H is not None else ex_ctx.input_H
            W_out = ex_ctx.output_W if ex_ctx.output_W is not None else ex_ctx.input_W

            grid_pred: Grid = y_to_grid(
                y=y_flat,
                H=H_out,
                W=W_out,
                C=task_context.C,
            )
            test_outputs_pred.append(grid_pred)

            total_variables += H_out * W_out * task_context.C
```

> 🔧 Implementation note:
> Whether you count examples by a global index or separate indices for train/test is up to you; just be consistent with how `apply_schema_instance` expects `example_index`.

---

#### Step 3.5 – Compare with training labels (if requested)

If `use_training_labels=True`, compare predicted train outputs with ground truth and set status accordingly.

You already have ground truth in `TaskContext` (from `build_task_context_from_raw`).

```python
        if use_training_labels:
            true_train_outputs: List[Grid] = [
                ex_ctx.output_grid for ex_ctx in task_context.train_examples
                if ex_ctx.output_grid is not None
            ]

            train_mismatches = compute_train_mismatches(true_train_outputs, train_outputs_pred)
            if len(train_mismatches) > 0:
                status = "mismatch"
            else:
                status = "ok"
        else:
            status = "ok"

    except Exception as e:
        # Catch any unexpected error during building or solving
        status = "error"
        error_message = str(e)
        # solver_status_str may remain "Unknown"
```

If the solver itself raises an explicit “infeasible” condition (e.g. `solve_constraints_for_grid` throws a custom exception), you can distinguish that and set `status="infeasible"` instead of `"error"`.

---

### 4. Construct `SolveDiagnostics` and return

After try/except:

```python
    diagnostics = SolveDiagnostics(
        task_id=task_id,
        law_config=law_config,
        status=status,                           # "ok", "mismatch", "infeasible", "error"
        solver_status=solver_status_str,
        num_constraints=total_constraints,
        num_variables=total_variables,
        schema_ids_used=schema_ids_used,
        train_mismatches=train_mismatches,
        error_message=error_message,
    )

    outputs = {
        "train": train_outputs_pred,
        "test": test_outputs_pred,
    }

    return outputs, diagnostics
```

---

### 5. Optional: keep old `solve_arc_task` as thin wrapper

If something external already calls `solve_arc_task(task_id, law_config)`, keep it as a wrapper:

```python
def solve_arc_task(task_id: str, law_config: TaskLawConfig) -> Dict[str, List[Grid]]:
    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=False,
    )
    # Optionally, you can log diagnostics or ignore it here.
    return outputs
```

---

## Thin runner for testing this WO

### File: `src/runners/test_kernel_with_diagnostics.py`

**Goal:** sanity-check that:

* `solve_arc_task_with_diagnostics` runs end-to-end,
* returns outputs and a non-crashing `SolveDiagnostics`.

**Contents:**

```python
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.catalog.types import TaskLawConfig, SchemaInstance

def dummy_law_config() -> TaskLawConfig:
    # Minimal empty or trivial config to test plumbing
    # If empty config leads to infeasible, that's fine; we just want no crash.
    return TaskLawConfig(schema_instances=[])

def main():
    task_id = "0"  # or a real task id from your training json

    law_config = dummy_law_config()

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
    )

    print("=== Outputs ===")
    print("Train outputs predicted:", len(outputs["train"]))
    print("Test outputs predicted:", len(outputs["test"]))

    print("=== Diagnostics ===")
    print("Status:", diagnostics.status)
    print("Solver status:", diagnostics.solver_status)
    print("Num constraints:", diagnostics.num_constraints)
    print("Num variables:", diagnostics.num_variables)
    print("Schema IDs used:", diagnostics.schema_ids_used)
    print("Train mismatches:", diagnostics.train_mismatches)
    print("Error message:", diagnostics.error_message)

if __name__ == "__main__":
    main()
```

> For now, we don’t expect correctness — just that diagnostics are populated and no silent failure.

---

## Reviewer + Tester instructions

1. **Static review:**

   * Check that:

     * `solve_arc_task_with_diagnostics` follows the spec exactly.
     * It uses:

       * `load_arc_task`,
       * `build_task_context_from_raw`,
       * `ConstraintBuilder`,
       * `apply_schema_instance`,
       * `solve_constraints_for_grid`,
       * `y_to_grid`,
       * `SolveDiagnostics`,
       * `compute_train_mismatches`.
     * No custom algorithms or non-standard libs are introduced.

2. **Run the test runner:**

   ```bash
   python -m src.runners.test_kernel_with_diagnostics
   ```

   Confirm:

   * The script prints:

     * some number of train/test outputs (even if zero),
     * diagnostics with a sensible `status` ("infeasible" or "error" is acceptable for dummy law_config),
     * no uncaught exceptions.

3. **Optional deeper test:**

   * If you have a hand-crafted `TaskLawConfig` for a simple task where the math works:

     * replace `dummy_law_config()` with that config,
     * rerun,
     * verify that:

       * `status` is `"ok"` or `"mismatch"` as expected,
       * `train_mismatches` reflect actual differences.

---
