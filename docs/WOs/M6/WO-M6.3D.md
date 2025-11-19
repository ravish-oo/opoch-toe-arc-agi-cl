## 🔹 M6.3D – S1 tie miner

### File: `src/law_mining/mine_s1_ties.py`

**Goal:** Implement `mine_S1` that:

* uses `TaskContext`, `RolesMapping`, and `RoleStats`,
* finds roles whose **training outputs** are **always** the same color,
* and for those roles, emits S1 `SchemaInstance`s that only contain **tie constraints** (no color fixes),
* matching the existing S1 builder’s expected params:

```python
params = {
    "ties": [
        {
            "example_type": "train" | "test",
            "example_index": int,
            "pairs": [((r1, c1), (r2, c2)), ...]
        },
        ...
    ]
}
```

No defaults, no fallbacks, no fix_pixel_color.

---

### 1. Imports & types

At the top of `mine_s1_ties.py`:

```python
from __future__ import annotations

from typing import Dict, List, Tuple
from collections import defaultdict

from src.schemas.context import TaskContext
from src.law_mining.roles import RolesMapping, NodeKind
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance
```

We assume:

* `RolesMapping` = `Dict[(kind, example_idx, r, c), role_id]`,
* `NodeKind` = `Literal["train_in", "train_out", "test_in"]`.

---

### 2. `mine_S1` – tie roles with homogeneous train_out colors

**Signature:**

```python
def mine_S1(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

**Core idea:**

* For a given `role_id`, if **all** its `train_out` appearances have the **same color**, that role is a good candidate to tie:

  * we tie all positions with that `role_id` in the **train outputs** (per example),
  * and optionally in the **test outputs** (i.e., test_in positions, since test outputs aren’t known yet).
* We **never** fix the color; we only enforce equality of y across those positions.

**Algorithm (no wiggle room):**

1. Initialize:

   ```python
   ties_by_example: Dict[Tuple[str, int], List[Tuple[Tuple[int,int], Tuple[int,int]]]] = defaultdict(list)
   ```

   This will map `(example_type, example_index)` to a list of tie pairs `((r1,c1),(r2,c2))`.

2. For each `role_id, stats` in `role_stats.items()`:

   * Extract all output colors for this role in training outputs:

     ```python
     colors_out = {color for (_ex, _r, _c, color) in stats.train_out}
     ```

   * Cases:

     * `len(colors_out) == 0` → This role never appears in train outputs → skip this role for S1. We have no evidence about its output behavior.
     * `len(colors_out) > 1` → This role has **conflicting** output colors in train_out → skip this role completely (we must not tie them; they’re not guaranteed equal).
     * `len(colors_out) == 1` → training outputs are homogeneous for this role → we can tie all positions of that role.

3. For each such **homogeneous** `role_id`:

   For each training example index `ex_idx`:

   * Collect all **train_out** positions with this role:

     ```python
     train_out_positions: List[Tuple[int,int]] = []
     for (kind, k, r, c), rid in roles.items():
         if rid != role_id:
             continue
         if kind == "train_out" and k == ex_idx:
             train_out_positions.append((r, c))
     ```

   * If `len(train_out_positions) ≥ 2`:

     * We build a tie chain to a canonical pixel (e.g. first in list):

       ```python
       anchor = train_out_positions[0]
       for pos in train_out_positions[1:]:
           ties_by_example[("train", ex_idx)].append((anchor, pos))
       ```

   Similarly, for each test example index `test_idx`:

   * Collect all **test_in** positions with this role:

     ```python
     test_in_positions: List[Tuple[int,int]] = []
     for (kind, k, r, c), rid in roles.items():
         if rid != role_id:
             continue
         if kind == "test_in" and k == test_idx:
             test_in_positions.append((r, c))
     ```

   * If `len(test_in_positions) ≥ 2`, tie them:

     ```python
     anchor = test_in_positions[0]
     for pos in test_in_positions[1:]:
         ties_by_example[("test", test_idx)].append((anchor, pos))
     ```

   * We do **not** tie train_in positions here; S1 is about equalities in outputs (train_out + test_out), and tying test_in positions ensures they share a color once any one of them is fixed by some other law.

4. Build `SchemaInstance`s from `ties_by_example`:

   ```python
   instances: List[SchemaInstance] = []
   ties_param_list: List[Dict] = []

   for (example_type, ex_idx), pairs in ties_by_example.items():
       if not pairs:
           continue
       ties_param_list.append({
           "example_type": example_type,
           "example_index": ex_idx,
           "pairs": pairs,
       })

   if ties_param_list:
       params = { "ties": ties_param_list }
       instances.append(SchemaInstance(family_id="S1", params=params))
   ```

5. Return `instances`. If `ties_by_example` is empty, return `[]`.

**Constraints:**

* We **never** introduce a tie for a role whose `train_out` colors are not homogeneous.
* We **never** pick a color; we only tie positions.
* No defaults for roles not seen in outputs; they’re ignored for S1.

---

### 3. Thin smoke test runner

**File:** `src/law_mining/test_mine_s1_ties_smoke.py`

Goal: check that `mine_S1` runs and produces `SchemaInstance`s with the expected shape.

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s1_ties import mine_S1

def main():
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "0"  # pick a simple task id present in training json

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s1_instances = mine_S1(task_context, roles, role_stats)

    print("S1 instances:", s1_instances)

if __name__ == "__main__":
    main()
```

---

### 4. Reviewer + tester instructions

**For implementer:**

* Implement `mine_S1` exactly as above:

  * Use `role_stats[role_id].train_out` to decide homogeneity of output colors.
  * For homogeneous roles, tie **only** positions with that role:

    * within each `train_out` grid,
    * within each `test_in` grid.
  * Do **not** fix colors; do **not** tie roles with conflicting train_out colors.
  * Match the S1 builder’s param shape: `"ties": [ { "example_type", "example_index", "pairs": [...] }, ... ]`.

* Use only:

  * standard Python (collections, typing),
  * existing `TaskContext`, `RolesMapping`, `RoleStats`.

**For reviewer/tester:**

1. **Static check:**

   * Confirm that:

     * `mine_S1` never calls any color-fixing builder (e.g. no `fix_pixel_color` logic),
     * only produces tie pairs for roles where `stats.train_out` colors are homogeneous.
     * roles with 0 or >1 distinct train_out colors are ignored.

2. **Smoke test:**

   ```bash
   python -m src.law_mining.test_mine_s1_ties_smoke
   ```

   * Confirm it runs without exceptions.
   * Inspect printed `SchemaInstance`:

     * `family_id == "S1"`,
     * `params["ties"]` is a list of dicts with `"example_type"`, `"example_index"`, `"pairs"`.

3. **Optional deeper test:**

   * For a task where you know some rows/regions play identical roles (e.g. symmetric outputs),:

     * run `mine_S1`,
     * then assemble a `TaskLawConfig` including S1 and S2/S10 etc.,
     * call `solve_arc_task_with_diagnostics(..., use_training_labels=True)`,
     * check that adding S1 does not change correctness but may simplify constraints/propagation.

