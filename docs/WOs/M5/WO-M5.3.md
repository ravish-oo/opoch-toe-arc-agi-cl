## 🔹 Expanded WO-M5.3 – Training sweep + catalog builder script

### File: `src/runners/build_catalog_from_training.py`

**Goal:** Provide a script that can be run by a human or Pi-agent to:

* loop over all training tasks,
* for each task with an existing `TaskLawConfig`, run `solve_arc_task_with_diagnostics`,
* update the catalog if status is `"ok"`,
* log failures (mismatch/infeasible/error) in a Pi-agent-friendly JSON log.

> ❗️ This script **does no law discovery**.
> It only executes/validates existing law configs.

---

### 1. Imports (only standard libs + existing modules)

At top of `build_catalog_from_training.py`:

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.core.grid_types import Grid
from src.core.arc_io import load_arc_task_ids  # see note below
from src.catalog.types import TaskLawConfig
from src.catalog.store import (
    load_task_law_config,
    save_task_law_config,
)
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.runners.results import SolveDiagnostics
```

> 🔧 **Note:** if you don’t yet have `load_arc_task_ids`, see below (Section 2).
> Otherwise, just import whatever function you already have that returns all training task_ids.

---

### 2. Helper: iterate training task_ids

If not already present, **implement** this helper in `src/core/arc_io.py` (or ensure an existing equivalent is used):

```python
def load_arc_task_ids(challenges_path: Path) -> List[str]:
    """
    Read the ARC training challenges JSON and return a list of task_id strings.

    Assumes the JSON is a list of objects, each with an "id" field.
    """
    import json  # standard lib

    with challenges_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # data is expected to be a list of {"id": "...", ...}
    return [item["id"] for item in data]
```

In **this WO**, we assume `load_arc_task_ids` exists and works.

---

### 3. Logging setup

Inside `build_catalog_from_training.py`, declare a logger:

```python
logger = logging.getLogger(__name__)
```

We’ll configure basic logging in `main()`.

---

### 4. Core sweep function

Implement:

```python
def sweep_training_tasks(
    challenges_path: Path,
    failure_log_path: Path,
    only_with_configs: bool = True,
    max_tasks: Optional[int] = None,
) -> None:
    """
    Sweep over training tasks, run the kernel with existing law configs,
    and update the catalog / log failures.

    Args:
        challenges_path: Path to arc-agi_training_challenges.json.
        failure_log_path: Path to a JSONL file where failures will be appended.
        only_with_configs: If True, skip tasks that have no stored TaskLawConfig.
        max_tasks: If not None, limit to the first max_tasks task_ids (for testing).
    """
```

**Implementation details (no wiggle room):**

1. Get all task_ids:

   ```python
   task_ids = load_arc_task_ids(challenges_path)
   if max_tasks is not None:
       task_ids = task_ids[:max_tasks]
   ```

2. Open failure log for appending (JSONL):

   ```python
   failure_log_path.parent.mkdir(parents=True, exist_ok=True)
   failure_log_file = failure_log_path.open("a", encoding="utf-8")
   ```

3. Loop over task_ids:

   ```python
   for task_id in task_ids:
       logger.info("Processing task_id=%s", task_id)

       # Try to load existing law config
       law_config: Optional[TaskLawConfig] = load_task_law_config(task_id)

       if law_config is None:
           if only_with_configs:
               logger.info("  No law_config for task_id=%s, skipping.", task_id)
               continue
           else:
               logger.info("  No law_config for task_id=%s, using empty config.", task_id)
               law_config = TaskLawConfig(schema_instances=[])
   ```

4. Call the kernel with diagnostics:

   ```python
       try:
           outputs, diagnostics = solve_arc_task_with_diagnostics(
               task_id=task_id,
               law_config=law_config,
               use_training_labels=True,
               challenges_path=challenges_path,
           )
       except Exception as e:
           logger.exception("Error while solving task_id=%s: %s", task_id, e)
           # Write a failure record with status="error"
           record = {
               "task_id": task_id,
               "status": "error",
               "error_message": str(e),
           }
           failure_log_file.write(json.dumps(record) + "\n")
           continue
   ```

5. Handle status:

   ```python
       if diagnostics.status == "ok":
           logger.info("  OK for task_id=%s", task_id)
           # Mark this config as valid / up-to-date in the catalog:
           save_task_law_config(task_id, law_config)
           # No failure log entry for this task
       else:
           logger.warning(
               "  Failure for task_id=%s: status=%s, solver_status=%s",
               task_id,
               diagnostics.status,
               diagnostics.solver_status,
           )
           # Serialize diagnostics into a JSON-friendly dict
           failure_record = {
               "task_id": diagnostics.task_id,
               "status": diagnostics.status,
               "solver_status": diagnostics.solver_status,
               "num_constraints": diagnostics.num_constraints,
               "num_variables": diagnostics.num_variables,
               "schema_ids_used": diagnostics.schema_ids_used,
               "train_mismatches": diagnostics.train_mismatches,
               "error_message": diagnostics.error_message,
           }
           failure_log_file.write(json.dumps(failure_record) + "\n")
   ```

6. Close file at end:

   ```python
   failure_log_file.close()
   ```

> 📌 This function never mutates `law_config`. It just runs it, updates catalog when successful, and logs failure diagnostics.

---

### 5. CLI entrypoint (thin runner)

At bottom of `build_catalog_from_training.py`:

```python
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweep ARC training tasks and build/update law catalog.")
    parser.add_argument(
        "--challenges-path",
        type=Path,
        default=Path("data/arc-agi_training_challenges.json"),
        help="Path to ARC training challenges JSON.",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("logs/training_failures.jsonl"),
        help="Path to JSONL file where failure diagnostics will be logged.",
    )
    parser.add_argument(
        "--only-with-configs",
        action="store_true",
        help="If set, skip tasks without existing law configs.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional limit on number of tasks to process (for quick tests).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    sweep_training_tasks(
        challenges_path=args.challenges_path,
        failure_log_path=args.failure_log,
        only_with_configs=args.only_with_configs,
        max_tasks=args.max_tasks,
    )

if __name__ == "__main__":
    main()
```

This is the **thin runner** for this WO.

---

## Reviewer + Tester Instructions

**For implementer:**

* Implement `build_catalog_from_training.py` exactly as described:

  * use `load_arc_task_ids` to get task_ids,
  * use `load_task_law_config` / `save_task_law_config` from `catalog.store`,
  * use `solve_arc_task_with_diagnostics` (M5.2),
  * write failures as JSONL records.

**For reviewer/tester:**

1. **Smoke test with no configs:**

   ```bash
   python -m src.runners.build_catalog_from_training --max-tasks 3 --only-with-configs
   ```

   * Expect:

     * log messages saying tasks are “skipping, no law_config,”
     * script exits cleanly,
     * no `logs/training_failures.jsonl` or an empty file.

2. **Test with a dummy config for one simple task:**

   * Manually create a very trivial `TaskLawConfig` for a known easy task and save it via `save_task_law_config(task_id, config)`. For now, even an empty config is acceptable (it should likely result in `"infeasible"` or `"mismatch"`).

   * Run:

     ```bash
     python -m src.runners.build_catalog_from_training --max-tasks 5
     ```

   * Check:

     * script runs without uncaught exceptions,
     * `logs/training_failures.jsonl` contains JSON lines for tasks where `status != "ok"`,
     * each JSON line has keys: `"task_id"`, `"status"`, `"solver_status"`, `"schema_ids_used"`, etc.

3. **No algorithmic law discovery:**

   * Confirm by code inspection that nowhere in this file:

     * are φ features analyzed,
     * are schemas inferred or modified.
   * It should only:

     * read configs,
     * call the kernel,
     * write results to catalog or failure log.

---
