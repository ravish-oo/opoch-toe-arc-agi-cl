## 🔹 WO-M6.3A – Miners for S1, S2, S10

### File: `src/law_mining/mine_s1_s2_s10.py`

**Goal:** For a single `TaskContext` and corresponding `RolesMapping` + `RoleStats`, mine all **always-true** instances of:

* **S1** – “role → constant color” (copy/tie/fix)
* **S2** – component-wise recolor
* **S10** – border vs interior recolor

Each miner returns a list of `SchemaInstance`s whose `params` match the **actual existing** schema builders (`build_S1_constraints`, `build_S2_constraints`, `build_S10_constraints`).

No “best effort”, no defaults, no “most frequent” — only invariants that are exactly true across all training examples.

---

### 1. Imports & types

At the top of `mine_s1_s2_s10.py`:

```python
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from collections import defaultdict

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance

from src.features.components import (
    connected_components_by_color,
    compute_shape_signature,
)
from src.features.object_roles import (
    component_border_interior,
)
```

> 🔎 We only use existing φ operators and datastructures. No new algorithms besides simple aggregation.

You will need to **inspect** your existing `build_S1_constraints`, `build_S2_constraints`, `build_S10_constraints` to know the exact `schema_params` format they expect. The miner must **adapt to that**, not change the builders.

---

### 2. Miner for S1 – `mine_S1`

**Signature:**

```python
def mine_S1(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

#### 2.1 Concept

We want to detect roles where all *training outputs* agree on a single color. For each such role, we can safely say:

> “Any pixel with this role must be color c_out in the final output.”

We then translate that into whatever param format your S1 builder uses (e.g. fixed colors per coordinate, or equivalence ties).

#### 2.2 Logic (no wiggle room)

Implementation outline:

1. Initialize a structure to collect per-role constant colors:

   ```python
   role_to_color: Dict[int, int] = {}
   ```

2. For each `role_id, stats` in `role_stats.items()`:

   ```python
   stats = role_stats[role_id]
   colors_out = { color for (_ex, _r, _c, color) in stats.train_out }
   ```

   Cases:

   * `len(colors_out) == 0` → no evidence in outputs → **skip** (do not mine S1 for this role).
   * `len(colors_out) > 1` → contradiction on this role → **skip** (this role is not governed by a simple constant-color S1 law).
   * `len(colors_out) == 1` → let `c_out = colors_out.pop()` and record:

     ```python
     role_to_color[role_id] = c_out
     ```

3. If `role_to_color` is empty → return `[]` (no S1 instances for this task).

4. Translate roles into coordinates / S1 params:

   You must match your existing builder. Two common patterns:

   * **Pattern A: fixed colors per pixel**

     If your `build_S1_constraints` understands something like:

     ```python
     params = {
         "fixed_pixels": [((kind, ex_idx, r, c), color), ...]
     }
     ```

     then:

     ```python
     fixed_pixels = []
     for (kind, ex_idx, r, c), role_id in roles.items():
         if role_id in role_to_color:
             fixed_pixels.append(((kind, ex_idx, r, c), role_to_color[role_id]))
     params = { "fixed_pixels": fixed_pixels }
     instances = [SchemaInstance(family_id="S1", params=params)]
     ```

   * **Pattern B: tie pixels with same role**

     If your builder expects `"ties": [{"pairs": [ ((r1,c1),(r2,c2)), ... ]}]`, you can:

     * For each `role_id` in `role_to_color`, gather all pixels with that `role_id` in train_out/test_in, pick one canonical pixel and tie all others to it.
     * Additionally, fix the canonical pixel’s color to `c_out`.

   **The actual pattern must match your already-implemented S1 builder.**

5. Return `instances` (usually 0 or 1 elements, depending on how you batch them).

**Constraints:**

* We only emit S1 instances for roles where **train_out is 100% consistent**.
* We never decide a color for roles not seen in outputs.
* No default choices, no “tie everything with same role even if outputs disagree”.

---

### 3. Miner for S2 – `mine_S2`

**Signature:**

```python
def mine_S2(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

#### 3.1 Concept

S2 is “component-wise recolor map”:

> For each connected component class (e.g. same input color + shape), if all its occurrences in train outputs have the same color, we can define a recolor mapping for that class.

We mine such mappings, then encode them as S2 schema_params.

#### 3.2 Logic

1. For each training example:

   ```python
   for ex_idx, ex in enumerate(task_context.train_examples):
       grid_in = ex.input_grid
       grid_out = ex.output_grid   # must not be None for training
       comps = connected_components_by_color(grid_in)
   ```

2. For each `Component` in `comps`:

   * Compute its `shape_signature` via `compute_shape_signature`.

   * Define a **component class key**. A good starting point:

     ```python
     key = (comp.color, comp.shape_signature)
     ```

     You may decide to include size explicitly or derive it from `shape_signature`; but be consistent with what `build_S2_constraints` expects.

   * For each pixel `(r,c)` in `comp.pixels`:

     * get `color_out = grid_out[r,c]`.

   * Check that all pixels in this component have the **same** `color_out` (otherwise this component is not uniformly recolored → S2 not applicable to this comp).

   * Record `(key, color_out)` for this example.

3. Aggregate across all training examples:

   * Build:

     ```python
     class_to_colors: Dict[key, set[int]] = defaultdict(set)
     ```

   * For each component class instance, add its single output color into `class_to_colors[key]`.

4. Decide which classes are valid S2 mappings:

   * For each `key, color_set` in `class_to_colors.items()`:

     * If `len(color_set) == 0` → no evidence → skip.
     * If `len(color_set) > 1` → inconsistent across examples → **skip** (do not include in S2).
     * If `len(color_set) == 1` → let `c_out` be the single color; this class can be recolored to `c_out`.

   * Build a mapping:

     ```python
     class_to_color_out: Dict[key, int]
     ```

5. Translate into schema_params:

   This depends on your S2 builder. Two common patterns:

   * **Pattern A: color_in + size_to_color**

     If `build_S2_constraints` expects something like:

     ```python
     params = {
         "color_in": <int>,
         "size_to_color": { size1: col1, size2: col2, ... }
     }
     ```

     then:

     * For a fixed `color_in`, group classes by component `size`:

       * `size = len(comp.pixels)`
       * `size_to_color[size] = c_out` if those classes are consistent.
     * For multiple input colors, you may need multiple SchemaInstances (one per `color_in`).

   * **Pattern B: direct class map**

     If your S2 builder supports something like:

     ```python
     params = {
         "class_to_color": {
             (color_in, shape_signature): color_out,
             ...
         }
     }
     ```

     then fill that directly from `class_to_color_out`.

   Again, the implementer must adapt this step to the **actual S2 param format**.

6. If no valid mappings are found → return `[]`. Otherwise, return a list of one or more `SchemaInstance(family_id="S2", params=params)`.

**Constraints:**

* S2 miner **never** invents a mapping for classes without evidence.
* It **never** merges conflicting colors by picking “majority”.
* Only classes with exactly one consistent color across all training examples get included.

---

### 4. Miner for S10 – `mine_S10`

**Signature:**

```python
def mine_S10(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

#### 4.1 Concept

S10 is “frame / border vs interior”:

> For each object (or component class), if border pixels and interior pixels always get fixed colors in training outputs, we can enforce that as a law.

#### 4.2 Logic

1. For each training example:

   ```python
   for ex_idx, ex in enumerate(task_context.train_examples):
       grid_in = ex.input_grid
       grid_out = ex.output_grid
       comps = connected_components_by_color(grid_in)
       border_info = component_border_interior(grid_in, comps)
       # border_info: dict[(r,c)] -> {"is_border": bool, "is_interior": bool}
   ```

2. For each component, define a **component class key**. Similar to S2:

   ```python
   key = (comp.color, compute_shape_signature(comp))
   ```

3. For each pixel `(r,c)` in `comp.pixels`:

   * from `border_info[(r,c)]`, determine:

     * if `is_border` or `is_interior`.
   * get:

     * `color_out = grid_out[r,c]`.

4. Aggregate colors by `(key, role_type)` where `role_type ∈ {"border", "interior"}`:

   ```python
   border_colors: Dict[Tuple[key, str], set[int]] = defaultdict(set)
   ```

   * For each border pixel:

     * `border_colors[(key, "border")].add(color_out)`
   * For each interior pixel:

     * `border_colors[(key, "interior")].add(color_out)`

5. Decide valid frame rules:

   For each `(key, role_type), color_set`:

   * If `len(color_set) == 0` → no evidence → skip.
   * If `len(color_set) > 1` → conflicting border (or interior) color across training examples → **skip**.
   * If `len(color_set) == 1`:

     * Let `c_out =` unique color.
     * This says: “for class `key`, role `role_type` must be color `c_out`.”

   Combine border + interior for that class only if **both** are determined (optional, depending on builder expectations).

6. Translate into `schema_params`:

   Suppose your S10 builder expects something like:

   ```python
   params = {
       "rules": [
           {
               "component_class": key,
               "border_color": b,
               "interior_color": i,
           },
           ...
       ]
   }
   ```

   Then:

   * Build `rules` list from `border_colors` where both roles have unique colors.
   * If no rules → return `[]`.
   * Else → return `[SchemaInstance(family_id="S10", params=params)]`.

**Constraints:**

* S10 miner never picks “most common” border color; only “always same” counts.
* If a class has only border or only interior determined, you can either:

  * create a rule for just that role_type **if** your builder supports partial rules, or
  * require both sides; just be explicit and consistent.
* No defaults for unobserved roles.

---

## Thin smoke test runner

### File: `src/law_mining/test_mine_s1_s2_s10_smoke.py`

**Goal:** sanity-check that the miners run on a simple training task and produce some `SchemaInstance`s without crashing.

Contents:

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s1_s2_s10 import mine_S1, mine_S2, mine_S10

def main():
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "0"  # or some other small, known-easy task

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s1_instances = mine_S1(task_context, roles, role_stats)
    s2_instances = mine_S2(task_context, roles, role_stats)
    s10_instances = mine_S10(task_context, roles, role_stats)

    print("S1 instances:", s1_instances)
    print("S2 instances:", s2_instances)
    print("S10 instances:", s10_instances)

if __name__ == "__main__":
    main()
```

> This is just plumbing: we’re not checking correctness yet, just that miners run and produce structured data.

---

## Reviewer + tester instructions

**For implementer:**

* Implement `mine_S1`, `mine_S2`, `mine_S10` exactly per spec:

  * only mine invariants that are **always true** on training IO,
  * never invent mappings for roles or components without output evidence,
  * adapt your `schema_params` to the exact formats expected by `build_S1_constraints`, `build_S2_constraints`, `build_S10_constraints` (inspect those builders).
* Use only:

  * `TaskContext`, `roles`, `role_stats`,
  * φ operators (`components`, `object_roles`),
  * `collections.defaultdict` and simple loops.

**For reviewer/tester:**

1. **Static review:**

   * Check that:

     * all candidates are checked against **all** training examples,
     * any conflicting color evidence leads to candidate discard,
     * no “most common color” or “fallbacks” are used,
     * unobserved roles/components are simply not mined.

2. **Smoke test run:**

   ```bash
   python -m src.law_mining.test_mine_s1_s2_s10_smoke
   ```

   * Confirm:

     * script runs without errors,
     * returns a list of `SchemaInstance`s (possibly empty if the chosen task doesn’t fit these schemas).

3. **Optional deeper test:**

   * Pick a task that you **know** is S2/S10-like from your manual analysis.
   * Run:

     * build `TaskContext` → `compute_roles` → `compute_role_stats` → `mine_S1/S2/S10`,
     * assemble a `TaskLawConfig` from these instances,
     * call `solve_arc_task_with_diagnostics(use_training_labels=True)`,
     * expect `diagnostics.status == "ok"` for that task.

