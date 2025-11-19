## 🔹 WO-M6.4 – Law miner orchestrator: `mine_law_config`

### File: `src/law_mining/mine_law_config.py`

**Goal:** For a single task (`TaskContext`), run the full law-miner stack:

* compute roles,
* compute role stats,
* run all per-schema miners `mine_Sk`,
* and assemble a `TaskLawConfig` (list of `SchemaInstance`s).

No coverage guarantees, no fallbacks. Whatever we mine is what we get; the kernel + diagnostics will tell us if it solves training or not.

---

### 1. Imports

At the top of `mine_law_config.py`:

```python
from __future__ import annotations

from typing import List

from src.schemas.context import TaskContext
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats, RoleStats
from src.catalog.types import TaskLawConfig, SchemaInstance

from src.law_mining.mine_s1_ties import mine_S1
from src.law_mining.mine_s1_s2_s10 import mine_S2, mine_S10
from src.law_mining.mine_s3_s4_s8_s9 import mine_S3, mine_S4, mine_S8, mine_S9
from src.law_mining.mine_s5_s6_s7_s11 import mine_S5, mine_S6, mine_S7, mine_S11
```

> 🔎 If your miners are split into slightly different modules than above, adapt the imports accordingly. The important part is: we explicitly call all miners that currently exist.

---

### 2. API

```python
def mine_law_config(task_context: TaskContext) -> TaskLawConfig:
    """
    High-level law miner for a single ARC task.

    Steps:
      - compute structural roles for all pixels across all grids,
      - aggregate role-level statistics,
      - invoke miners for all schema families S1..S11,
      - concatenate all mined SchemaInstance objects into a TaskLawConfig.

    This function does NOT try to judge coverage or correctness; it only
    encodes what the miners infer as always-true laws from training data.
    Any mismatch/infeasibility will be exposed later by the kernel +
    diagnostics when this TaskLawConfig is used.
    """
```

---

### 3. Implementation (no wiggle room)

Inside `mine_law_config`:

1. Compute roles and role_stats:

```python
    # 1) Structural role labels (WL/q) for all pixels
    roles = compute_roles(task_context)

    # 2) Aggregate role-level statistics across train_in, train_out, test_in
    role_stats = compute_role_stats(task_context, roles)
```

2. Initialize schema instance list:

```python
    schema_instances: List[SchemaInstance] = []
```

3. Call each miner in a **fixed, explicit order**, just concatenating their results:

```python
    # S1: tie/equality only (may return [] in this milestone)
    schema_instances.extend(mine_S1(task_context, roles, role_stats))

    # S2, S10: component recolor / frame
    schema_instances.extend(mine_S2(task_context, roles, role_stats))
    schema_instances.extend(mine_S10(task_context, roles, role_stats))

    # S3, S4, S8, S9: bands, residues, tiling, plus-propagation
    schema_instances.extend(mine_S3(task_context, roles, role_stats))
    schema_instances.extend(mine_S4(task_context, roles, role_stats))
    schema_instances.extend(mine_S8(task_context, roles, role_stats))
    schema_instances.extend(mine_S9(task_context, roles, role_stats))

    # S5, S6, S7, S11: template stamping, crop, summary, local codebook
    schema_instances.extend(mine_S5(task_context, roles, role_stats))
    schema_instances.extend(mine_S6(task_context, roles, role_stats))
    schema_instances.extend(mine_S7(task_context, roles, role_stats))
    schema_instances.extend(mine_S11(task_context, roles, role_stats))
```

4. Return the `TaskLawConfig`:

```python
    return TaskLawConfig(schema_instances=schema_instances)
```

That’s it. No “if empty, add fallback”, no filtering, no ranking, no heuristics.

---

### 4. Thin smoke test runner

**File:** `src/law_mining/test_mine_law_config_smoke.py`

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config

def main():
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "0"  # or any known training task id

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    law_config = mine_law_config(task_context)
    print("Number of schema instances mined:", len(law_config.schema_instances))
    for inst in law_config.schema_instances[:10]:
        print(inst)

if __name__ == "__main__":
    main()
```

> This just shows that the wiring works; correctness is judged later via `solve_arc_task_with_diagnostics`.

---
