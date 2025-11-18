## 🔹 Expanded WO-M6.2 – Role statistics aggregator

### File: `src/law_mining/role_stats.py`

**Goal:** For a single ARC task (`TaskContext` + `RolesMapping`), aggregate all appearances of each `role_id` in train_in, train_out, and test_in, including their colors. This is pure factual compression, no heuristics or defaults.

---

### 1. Imports & types

At the top of `role_stats.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from collections import defaultdict

from src.schemas.context import TaskContext, ExampleContext
from src.core.grid_types import Grid
from src.law_mining.roles import RolesMapping, NodeKind  # from M6.1
```

We assume `RolesMapping` and `NodeKind` are defined in `roles.py` as:

```python
# NodeKind = Literal["train_in", "train_out", "test_in"]
# RolesMapping = Dict[tuple[NodeKind, int, int, int], int]
```

---

### 2. Define `RoleStats`

Add:

```python
@dataclass
class RoleStats:
    """
    Aggregated appearances of a single role_id across all grids of a task.

    train_in  entries are (example_idx, r, c, color_in_train_input)
    train_out entries are (example_idx, r, c, color_in_train_output)
    test_in   entries are (example_idx, r, c, color_in_test_input)
    """
    train_in: List[Tuple[int, int, int, int]] = field(default_factory=list)
    train_out: List[Tuple[int, int, int, int]] = field(default_factory=list)
    test_in: List[Tuple[int, int, int, int]] = field(default_factory=list)
```

* No extra fields.
* No defaults beyond empty lists.

This is just raw evidence for miners; they will compute sets, maps, etc. on top.

---

### 3. Implement `compute_role_stats`

Add:

```python
def compute_role_stats(
    task_context: TaskContext,
    roles: RolesMapping,
) -> Dict[int, RoleStats]:
    """
    For each role_id, collect:
      - all its appearances in train_in, train_out, test_in,
      - with associated colors.

    Args:
        task_context: TaskContext containing train_examples and test_examples.
        roles: mapping (kind, example_idx, r, c) -> role_id as returned by compute_roles.

    Returns:
        dict: role_id -> RoleStats
    """
```

Implementation details (no wiggle room):

1. **Initialize storage**

```python
    role_stats: Dict[int, RoleStats] = defaultdict(RoleStats)
```

2. **Iterate over all role assignments**

The `roles` dict already contains all node keys. For each `(kind, example_idx, r, c), role_id`:

* Look up the corresponding grid and color:

  * If `kind == "train_in"`:

    * `ex = task_context.train_examples[example_idx]`
    * `grid = ex.input_grid`
    * `color = int(grid[r, c])`
    * append `(example_idx, r, c, color)` to `role_stats[role_id].train_in`.
  * If `kind == "train_out"`:

    * `ex = task_context.train_examples[example_idx]`
    * `grid = ex.output_grid`
    * If `grid is None`, **skip** (should not happen for proper training tasks, but we fail-safe).
    * Else:

      * `color = int(grid[r, c])`
      * append `(example_idx, r, c, color)` to `role_stats[role_id].train_out`.
  * If `kind == "test_in"`:

    * `ex = task_context.test_examples[example_idx]`
    * `grid = ex.input_grid`
    * `color = int(grid[r, c])`
    * append `(example_idx, r, c, color)` to `role_stats[role_id].test_in`.

Concrete code:

```python
    for (kind, ex_idx, r, c), role_id in roles.items():
        if kind == "train_in":
            ex: ExampleContext = task_context.train_examples[ex_idx]
            grid: Grid = ex.input_grid
            color = int(grid[r, c])
            role_stats[role_id].train_in.append((ex_idx, r, c, color))

        elif kind == "train_out":
            ex: ExampleContext = task_context.train_examples[ex_idx]
            grid: Grid | None = ex.output_grid
            if grid is None:
                # For training tasks this should not happen, but we fail-safe by skipping.
                continue
            color = int(grid[r, c])
            role_stats[role_id].train_out.append((ex_idx, r, c, color))

        elif kind == "test_in":
            ex: ExampleContext = task_context.test_examples[ex_idx]
            grid: Grid = ex.input_grid
            color = int(grid[r, c])
            role_stats[role_id].test_in.append((ex_idx, r, c, color))

        else:
            # Unknown kind -> this should never happen if RolesMapping is well-formed.
            raise ValueError(f"Unknown node kind in roles mapping: {kind!r}")
```

3. **Return a normal dict**

`defaultdict` is fine internally, but return a plain `dict`:

```python
    return dict(role_stats)
```

No ordering assumptions. No filtering of roles; if a role appears only in train_in or only in test_in, we still include it — that’s just a fact. Miners will decide what to do with that.

> 🔒 No defaults, no heuristics — this function only passes through true structural data.

---

## Thin smoke test runner

### File: `src/law_mining/test_role_stats_smoke.py`

**Goal:** run `compute_roles` + `compute_role_stats` on a real training task and print a few role stats to validate wiring.

Contents:

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats

def main():
    # Use a real training challenges file
    challenges_path = Path("data/arc-agi_training_challenges.json")

    # Pick a simple known task id; adjust as needed
    task_id = "0"

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    print("Number of distinct roles:", len(role_stats))

    # Print stats for first few roles
    for role_id, stats in list(role_stats.items())[:5]:
        print(f"Role {role_id}:")
        print("  train_in count :", len(stats.train_in))
        print("  train_out count:", len(stats.train_out))
        print("  test_in count  :", len(stats.test_in))

if __name__ == "__main__":
    main()
```

This is just a sanity check:

* does not assert correctness of roles,
* just checks that role_stats is populated and consistent.

---

## Reviewer + tester instructions

**For implementer:**

* Implement `RoleStats` and `compute_role_stats` exactly as specified:

  * use `roles.items()` as the single source of truth for which nodes exist,
  * use `TaskContext` and `ExampleContext` to get the right grids,
  * correctly map `(kind, ex_idx, r, c)` to the right grid and color.
* Use only:

  * `collections.defaultdict` for aggregation,
  * numpy for grid indexing,
  * no extra algorithms or defaults.

**For reviewer/tester:**

1. **Static review:**

   * Confirm:

     * `RoleStats` matches the spec.
     * `compute_role_stats`:

       * never makes up colors or positions,
       * doesn’t filter out roles except for the “train_out grid is None” guard,
       * raises on unknown `kind`, which is correct.

2. **Smoke run:**

   ```bash
   python -m src.law_mining.test_role_stats_smoke
   ```

   Validate:

   * It runs without exceptions.
   * Reports a reasonable number of roles (not 0; not insane).
   * For each printed role:

     * `train_in` + `train_out` + `test_in` counts are non-negative integers.
     * For training tasks with outputs, most roles should have some `train_out` entries.

3. **Optional deeper check:**

   * Pick a tiny ARC task with 1 train example.
   * Manually inspect one role_id:

     * verify that `train_in` entries correspond to pixels that share the same role from `compute_roles`,
     * verify colors match the actual grids.

---
