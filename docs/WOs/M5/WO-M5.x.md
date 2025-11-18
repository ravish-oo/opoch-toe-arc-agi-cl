## WO-M5.X – Enrich diagnostics with per-schema stats & example summaries

### High-level goal

Augment the existing solver diagnostics so that a Pi-agent (or a human) can see:

1. **How many constraints each schema family contributed** (S1–S11).
2. **Per-example summaries**:

   * input / output shape,
   * components per color.

Everything is *read-only* instrumentation. We do **not** change the LP or constraint math itself.

---

## Part A – Per-schema constraint counts

### A.1 Data model change: `SolveDiagnostics`

**File:** `src/runners/results.py`

1. Open the file and locate `SolveDiagnostics` (likely a `@dataclass`).

2. Add a new field:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class SolveDiagnostics:
    # existing fields...
    # e.g.
    # constraints_count: int
    # solve_status: str
    # ...

    schema_constraint_counts: Dict[str, int] = field(default_factory=dict)
    # key: schema family ID, e.g. "S1", "S2", ..., "S11"
    # value: total number of constraints contributed by that schema family
```

3. Make sure any existing construction of `SolveDiagnostics` either:

* passes `schema_constraint_counts` explicitly, or
* lets it default to `{}` (preferred).

We’re **only adding a field**, no breaking change to existing fields.

---

### A.2 Instrument `apply_schema_instance` to count constraints

**File:** `src/schemas/dispatch.py`

We assume you already have something like:

```python
def apply_schema_instance(
    builder: ConstraintBuilder,
    family_id: str,
    params: dict,
    context: TaskContext,
) -> None:
    ...
```

If the signature differs, adapt accordingly, but keep these points:

1. Ensure `family_id: str` is available (e.g., `"S1"`, `"S2"`, …).

2. Ensure `builder` exposes `constraints` as a list (as we spec’d earlier: `builder.constraints: List[LinearConstraint]`).

3. **Add an optional dict parameter** to accumulate counts:

```python
from typing import Optional, Dict

def apply_schema_instance(
    builder: ConstraintBuilder,
    family_id: str,
    params: dict,
    context: TaskContext,
    schema_constraint_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Build constraints for a single schema instance and optionally update
    per-schema constraint counts.
    """
    before = len(builder.constraints)

    # Existing logic:
    # 1) Look up SchemaFamily by `family_id`
    # 2) Call its builder: family.builder(builder, context, params)
    family = SCHEMA_FAMILIES[family_id]
    family.builder(builder, context, params)

    after = len(builder.constraints)

    if schema_constraint_counts is not None:
        added = after - before
        if added < 0:
            # This should never happen; sanity check.
            raise RuntimeError(
                f"Schema builder {family_id} reduced constraint count: {added}"
            )
        if added > 0:
            schema_constraint_counts[family_id] = (
                schema_constraint_counts.get(family_id, 0) + added
            )
```

4. **IMPORTANT:** do *not* mutate or reorder constraints beyond what existing builders do. We’re only measuring `before` and `after`.

---

### A.3 Wire counts from `kernel.solve_arc_task_with_diagnostics`

**File:** `src/runners/kernel.py`

We assume there is a function approximately like:

```python
def solve_arc_task_with_diagnostics(task: ArcTask) -> SolveDiagnostics:
    ...
```

1. Inside this function, before applying any schema instances:

```python
from typing import Dict
from src.schemas.dispatch import apply_schema_instance
from src.runners.results import SolveDiagnostics

def solve_arc_task_with_diagnostics(task: ArcTask) -> SolveDiagnostics:
    builder = ConstraintBuilder()
    schema_constraint_counts: Dict[str, int] = {}

    # existing logic to build constraints from schemas:
    # for instance:
    # for schema_instance in task_config.schema_instances:
    #    apply_schema_instance(builder, schema_instance.family_id, schema_instance.params, context)
```

2. Modify the calls to `apply_schema_instance` to pass the dict:

```python
    for schema_instance in task_config.schema_instances:
        apply_schema_instance(
            builder=builder,
            family_id=schema_instance.family_id,
            params=schema_instance.params,
            context=context,
            schema_constraint_counts=schema_constraint_counts,
        )
```

3. After all constraints are built and the LP is solved, build `SolveDiagnostics`:

```python
    # builder.constraints now has the final list of LinearConstraint
    # solve the LP as before, get y*, status, etc.

    diagnostics = SolveDiagnostics(
        # existing fields...
        # e.g.
        # constraints_count=len(builder.constraints),
        # solve_status=solve_status,
        # ...
        schema_constraint_counts=schema_constraint_counts,
        example_summaries=example_summaries,  # Part B, see below
    )
    return diagnostics
```

If `SolveDiagnostics` is created somewhere else (e.g., in `results.py` or another helper), adapt that call site to include `schema_constraint_counts`.

---

## Part B – Example-level summaries

### B.1 Add `ExampleSummary` dataclass

**File:** `src/runners/results.py` **or** new `src/schemas/summaries.py`

Given you already have `SolveDiagnostics` in `results.py`, I’d put `ExampleSummary` there to keep diagnostics in one place.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class ExampleSummary:
    input_shape: Tuple[int, int]
    output_shape: Optional[Tuple[int, int]]  # None for test-only examples
    components_per_color: Dict[int, int]    # color -> number of connected components
```

Then extend `SolveDiagnostics`:

```python
@dataclass
class SolveDiagnostics:
    # existing fields...
    schema_constraint_counts: Dict[str, int] = field(default_factory=dict)
    example_summaries: List[ExampleSummary] = field(default_factory=list)
```

---

### B.2 Compute summaries in `kernel.solve_arc_task_with_diagnostics`

**File:** `src/runners/kernel.py`

We assume `TaskContext` or similar exists and gives access to train/test examples:

```python
# Pseudocode: adapt to your actual context types
train_examples: List[ExampleContext] = task_context.train_examples
test_examples: List[ExampleContext]  = task_context.test_examples
```

Each `ExampleContext` should expose:

* `input_grid: Grid`
* `output_grid: Optional[Grid]` (None for test examples if you’re mimicking Kaggle)

**Key requirement:** use existing connected components logic. Do *not* re-implement. If you have `connected_components(grid, color=None) -> List[Component]`, use that.

Implementation sketch:

```python
from src.runners.results import ExampleSummary
from src.schemas.components import connected_components  # hypothetical import

def solve_arc_task_with_diagnostics(task: ArcTask) -> SolveDiagnostics:
    ...
    example_summaries: List[ExampleSummary] = []

    # Helper function: turns a grid into color -> #components map
    def components_per_color(grid: Grid) -> Dict[int, int]:
        comps = connected_components(grid, color=None)
        counts: Dict[int, int] = {}
        for comp in comps:
            counts[comp.color] = counts.get(comp.color, 0) + 1
        return counts

    # Process train examples
    for ex in task_context.train_examples:
        input_grid = ex.input_grid
        output_grid = ex.output_grid  # should exist for training tasks
        input_shape = input_grid.shape  # (H, W)
        output_shape = output_grid.shape if output_grid is not None else None

        cpc_input = components_per_color(input_grid)
        # Optionally: cpc_output if needed later; for now spec says input grids only,
        # but you can easily expand ExampleSummary if desired.

        example_summaries.append(
            ExampleSummary(
                input_shape=input_shape,
                output_shape=output_shape,
                components_per_color=cpc_input,
            )
        )

    # Process test examples
    for ex in task_context.test_examples:
        input_grid = ex.input_grid
        input_shape = input_grid.shape
        output_shape = None  # no label in Kaggle/eval setting
        cpc_input = components_per_color(input_grid)

        example_summaries.append(
            ExampleSummary(
                input_shape=input_shape,
                output_shape=output_shape,
                components_per_color=cpc_input,
            )
        )

    # ... then build constraints, solve LP, etc. as in Part A

    diagnostics = SolveDiagnostics(
        # existing fields...
        schema_constraint_counts=schema_constraint_counts,
        example_summaries=example_summaries,
    )
    return diagnostics
```

If your `TaskContext` structure is slightly different, keep semantics the same: we must produce one `ExampleSummary` per example (train+test) with:

* shapes (input, output or None),
* `components_per_color` computed via your existing component machinery.

---

## Part C – Thin diagnostics runner

To make this WO testable and Pi-agent-friendly, add a tiny runner.

**File:** `scripts/diagnose_task.py` (new) or `src/runners/diagnose.py`

Purpose:

* Given a task ID (or path to ARC JSON),
* Run `solve_arc_task_with_diagnostics`,
* Print:

  * schema_constraint_counts,
  * example_summaries (input/output shapes & components_per_color),
  * maybe solve status.

Example stub:

```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.data.arc_loader import load_arc_task  # adjust to your loader

def main():
    if len(sys.argv) != 2:
        print("Usage: diagnose_task.py TASK_ID_OR_PATH", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    if Path(arg).exists():
        task = load_arc_task_from_file(Path(arg))
    else:
        task = load_arc_task(arg)  # by task ID

    diagnostics = solve_arc_task_with_diagnostics(task)

    # Simple JSON-ish output
    out = {
        "schema_constraint_counts": diagnostics.schema_constraint_counts,
        "example_summaries": [
            {
                "input_shape": es.input_shape,
                "output_shape": es.output_shape,
                "components_per_color": es.components_per_color,
            }
            for es in diagnostics.example_summaries
        ],
        # you can add solve_status, etc.
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

This will be invaluable for Pi-agents and humans to inspect behavior on real ARC tasks.

---

## Kernel implications

We **do not** change the core kernel semantics:

* B(T)y = 0,
* y ∈ {0,1}^{NC},
* one-hot constraints per pixel,
* TU matrix → unique vertex solution.

We only:

* track how many constraints each schema’s builder added,
* compute example summaries using existing `connected_components`.

If your kernel currently returns only the solution grid and a minimal diagnostics object, we evolve it by:

* enriching `SolveDiagnostics`,
* ensuring `solve_arc_task_with_diagnostics` (or equivalent) is the single entry point that:

  * builds constraints,
  * runs LP,
  * returns `(solution, diagnostics)` or `diagnostics` that contains the solution/metadata.

---

## Reviewer instructions

When reviewing this WO, check:

1. **Data model changes:**

   * `SolveDiagnostics` has:

     * `schema_constraint_counts: Dict[str, int]`,
     * `example_summaries: List[ExampleSummary]`.
   * No existing callers are broken (defaults used where needed).

2. **Constraint counting logic:**

   * `apply_schema_instance`:

     * computes `before = len(builder.constraints)` and `after = ...`,
     * updates `schema_constraint_counts[family_id] += (after - before)` for valid (≥0) diff.
     * doesn’t modify constraints outside of calling the existing schema builder.

3. **Summaries:**

   * `ExampleSummary` is correctly populated:

     * `input_shape` equals `grid.shape`,
     * `output_shape` is `(H,W)` for train examples, `None` for test,
     * `components_per_color` matches what `connected_components` yields:

       * e.g. for a simple grid with one red and two blue components.

4. **Thin runner:**

   * `diagnose_task.py` (or equivalent):

     * correctly loads a task,
     * calls `solve_arc_task_with_diagnostics`,
     * prints JSON with the new fields.

5. **No new algorithmic code for components:**

   * `components_per_color` uses your existing `connected_components` util,
   * no manually re-coded DFS/BFS.

---

## Tester instructions

1. **Unit-style tests (if you have a test suite):**

   * Create a small synthetic task with:

     * known schemas used (e.g., only S2 and S6),
     * trivial grids.

   * Run `solve_arc_task_with_diagnostics`.

   * Assert:

     * `schema_constraint_counts["S2"] > 0`,
     * `schema_constraint_counts["S6"] > 0`,
     * other schemas are either 0 or absent.

   * For `ExampleSummary`:

     * Construct a grid with known components:

       * e.g., two 0-components, one 1-component.
     * Assert `components_per_color == {0: 2, 1: 1}`.

2. **Integration tests with real ARC tasks:**

   Using `diagnose_task.py`:

   * Pick a few known tasks (e.g. `9344f635`, `91714a58`, etc.).

   * Run:

     ```bash
     ./scripts/diagnose_task.py 9344f635
     ```

   * Verify:

     * There are as many `example_summaries` entries as train+test examples.
     * Shapes make sense (input vs output).
     * `components_per_color` values are plausible (non-zero where expected).
     * `schema_constraint_counts` is non-empty and includes families you know are relevant (e.g. S2 for component recolor; S6 for crop tasks).

3. **Regression safety:**

   * Run your existing LP solver on a few tasks before and after this change:

     * Ensure solutions (output grids) are unchanged.
     * Only diagnostics are enriched.

---
