## 🔹 WO-M6.5 – Training sweep with miner: end-to-end validation

### File: `src/runners/sweep_training_with_miner.py`

**Goal:** For all tasks in `arc-agi_training_challenges.json`:

* build `TaskContext`,
* mine laws via `mine_law_config`,
* run the kernel via `solve_arc_task_with_diagnostics(..., use_training_labels=True)`,
* store successful `TaskLawConfig`s in the catalog,
* and log failures (status ≠ "ok") with diagnostics for later inspection.

This file does **no** law logic, **no** fallbacks. It just orchestrates.

---

### 1. Imports

At the top:

```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw, TaskContext
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.catalog.types import TaskLawConfig
from src.catalog.store import save_task_law_config  # implement if not present
from src.runners.results import SolveDiagnostics
```

> If `save_task_law_config` doesn’t exist yet, the implementer should add it in `src/catalog/store.py` as a simple writer (e.g. to a JSON/YAML catalog file).

---

### 2. Helper: iterate over training task_ids

We don’t want to guess about how to get task ids; we know the training challenges file is JSON with task ids as keys.

```python
def load_training_task_ids(challenges_path: Path) -> list[str]:
    """
    Load all task_ids from the ARC-AGI training challenges JSON.
    """
    with challenges_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    # keys are the task ids; keep as strings
    return sorted(list(data.keys()))
```

We do **not** attempt to reuse internal arc_io code for this; just use `json.load` from stdlib.

---

### 3. Main sweep function

```python
def sweep_training_with_miner(
    challenges_path: Path,
    failures_log_path: Path,
) -> None:
    """
    Run law mining + kernel validation over all training tasks.

    For each task_id:
      - build TaskContext,
      - mine TaskLawConfig,
      - run kernel with training labels,
      - store law_config in catalog if status == "ok",
      - otherwise log diagnostics for later inspection.

    This function does NOT adjust or patch law_config; it only records ground truth
    about which tasks are solved under current miners.
    """
```

Implementation outline:

1. Get all task_ids:

```python
    task_ids = load_training_task_ids(challenges_path)
```

2. Open a failures log (JSON Lines) for appending:

```python
    failures_log_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_log_path.open("w", encoding="utf-8") as log_f:
        for task_id in task_ids:
            ...
```

3. For each `task_id`:

```python
            try:
                # 1) Load raw task and build TaskContext
                raw_task = load_arc_task(task_id, challenges_path)
                task_context: TaskContext = build_task_context_from_raw(raw_task)

                # 2) Mine laws
                law_config: TaskLawConfig = mine_law_config(task_context)

                # 3) Validate with kernel (using training labels)
                outputs, diagnostics = solve_arc_task_with_diagnostics(
                    task_id=task_id,
                    law_config=law_config,
                    use_training_labels=True,
                    challenges_path=challenges_path,
                )

                # 4) Branch on diagnostics.status
                if diagnostics.status == "ok":
                    # Store successful law_config in catalog
                    save_task_law_config(task_id, law_config)
                else:
                    # Log failure diagnostics
                    failure_record = {
                        "task_id": task_id,
                        "status": diagnostics.status,
                        "solver_status": diagnostics.solver_status,
                        "num_constraints": diagnostics.num_constraints,
                        "num_variables": diagnostics.num_variables,
                        "schema_ids_used": diagnostics.schema_ids_used,
                        "train_mismatches": diagnostics.train_mismatches,
                        "error_message": diagnostics.error_message,
                    }
                    log_f.write(json.dumps(failure_record) + "\n")

            except Exception as e:
                # Hard failure (unexpected exception) – log and continue
                failure_record = {
                    "task_id": task_id,
                    "status": "error",
                    "solver_status": "EXCEPTION",
                    "num_constraints": 0,
                    "num_variables": 0,
                    "schema_ids_used": [],
                    "train_mismatches": [],
                    "error_message": str(e),
                }
                log_f.write(json.dumps(failure_record) + "\n")
```

* If some miner or the solver blows up, we mark that task as `"error"` and keep going.
* We do **not** “fix” anything here; it’s pure reporting.

4. Add `if __name__ == "__main__":` stub for CLI run:

```python
if __name__ == "__main__":
    challenges_path = Path("data/arc-agi_training_challenges.json")
    failures_log_path = Path("logs/miner_training_failures.jsonl")
    sweep_training_with_miner(challenges_path, failures_log_path)
```

---
