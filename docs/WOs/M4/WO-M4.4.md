## WO-M4.4 – Training-set validation runner

### Libraries & dependencies

Use only standard libs + your own modules:

```python
import argparse
from pathlib import Path

import numpy as np

from src.runners.kernel import solve_arc_task
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.schemas.context import load_arc_task  # already exists per implementer
from src.core.arc_io import load_training_solutions  # we’ll specify this below
```

No custom parsing logic beyond `json` in `arc_io`.

---

## A. Ensure solution loading exists in `arc_io.py`

Before writing the runner, we need a helper to read `arc-agi_training_solutions.json`.

Add this to `src/core/arc_io.py` (if not already present):

```python
import json
from pathlib import Path
from typing import Dict, Any

def load_training_solutions(solutions_path: Path) -> Dict[str, Any]:
    """
    Load arc-agi_training_solutions.json.

    Returns:
        A dict mapping task_id -> solution info.
        Exact structure depends on ARC-AGI format, but we assume:
          solutions[task_id]["test"] is a list of output grids (lists of lists of ints)
        or similar.

    Implementer:
      - Use json.load(open(solutions_path)).
      - Do NOT reinvent parsing; just return the loaded dict.
    """
    with solutions_path.open("r") as f:
        data = json.load(f)
    return data
```

This function just loads the JSON into a dict. The runner will adapt to whatever the exact structure is.

---

## B. `src/runners/validate_on_training.py`

### B.1 Purpose

Command-line script:

* takes a `task_id` and a serialized law_config (for now, hardcoded or simple),

* runs the full kernel (schemas + constraints + solver + decoding),

* compares:

  * predicted train outputs vs true train outputs (from challenges file),
  * predicted test outputs vs known test solutions (from solutions file), **if available**,

* prints mismatches in a way that a Pi-agent (or human) can see what went wrong.

### B.2 CLI setup

At top of the file:

```python
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.runners.kernel import solve_arc_task
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.schemas.context import load_arc_task
from src.core.arc_io import load_training_solutions
from src.core.grid_types import Grid
```

Then define:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate law config on ARC-AGI training task."
    )
    parser.add_argument(
        "task_id",
        type=str,
        help="Task ID from arc-agi_training_challenges.json"
    )
    parser.add_argument(
        "--challenges_path",
        type=Path,
        default=Path("data/arc-agi_training_challenges.json"),
        help="Path to arc-agi_training_challenges.json"
    )
    parser.add_argument(
        "--solutions_path",
        type=Path,
        default=Path("data/arc-agi_training_solutions.json"),
        help="Path to arc-agi_training_solutions.json"
    )
    return parser.parse_args()
```

### B.3 Constructing a law_config (for now)

For now, we assume you hardcode or import a `TaskLawConfig` appropriate for the task. Eventually, a Pi-agent will generate this. For the WO, we just specify a stub the implementer can adjust:

```python
def make_law_config_for_task(task_id: str) -> TaskLawConfig:
    """
    Placeholder: construct a TaskLawConfig for this task.

    Implementer:
      - For now, hardcode or look up a law config that you know should apply.
      - Later, this will be provided by a Pi-agent or a catalog lookup.
    """
    # Example: simple S1-only config; adjust params to match your S1 spec
    return TaskLawConfig(
        schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={
                    # fill in required S1 parameters, e.g. "mode": "copy_input_to_output"
                }
            )
        ]
    )
```

Reviewer/implementer must **fill in** the params sensibly for a chosen test task.

### B.4 Helper: convert ARC JSON grids to numpy Grids

Add a small helper for readability:

```python
def list_of_lists_to_grid(grid_ll: List[List[int]]) -> Grid:
    return np.array(grid_ll, dtype=int)
```

And conversely:

```python
def grids_from_challenge_train(raw_task: Dict[str, Any]) -> List[Grid]:
    """
    Extract true training output grids from the raw challenge JSON for this task.
    Assumes raw_task["train"] is a list of dicts with "output" keys.
    """
    train = raw_task.get("train", [])
    outputs = []
    for pair in train:
        out_ll = pair["output"]
        outputs.append(list_of_lists_to_grid(out_ll))
    return outputs
```

For test solutions:

```python
def grids_from_training_solutions(
    task_id: str,
    solutions: Dict[str, Any]
) -> List[Grid]:
    """
    Extract true test output grids from training_solutions for this task, if present.
    Implementer must adapt to the exact format in arc-agi_training_solutions.json.
    """
    if task_id not in solutions:
        return []

    task_solutions = solutions[task_id]
    # Adjust this according to the real structure:
    # e.g. maybe task_solutions["test"] is a list of grids.
    test_solutions_ll = task_solutions.get("test", [])

    return [list_of_lists_to_grid(g) for g in test_solutions_ll]
```

Implementer will need to inspect `arc-agi_training_solutions.json` once to confirm the exact keys.

### B.5 Core validation logic

Now define:

```python
def compare_grids(pred: Grid, true: Grid) -> Dict[str, Any]:
    """
    Compare two grids and return a summary dict:
      - match: bool
      - diff_coords: list of (r, c) where they differ
    """
    if pred.shape != true.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: pred {pred.shape}, true {true.shape}",
            "diff_coords": []
        }

    equal_mask = (pred == true)
    if equal_mask.all():
        return {
            "match": True,
            "reason": "exact_match",
            "diff_coords": []
        }

    diff_coords = [(int(r), int(c)) for r, c in zip(*np.where(~equal_mask))]
    return {
        "match": False,
        "reason": "value_mismatch",
        "diff_coords": diff_coords
    }
```

And the main function:

```python
def validate_on_training(task_id: str, challenges_path: Path, solutions_path: Path) -> None:
    # Load raw challenge and solutions
    raw_task = load_arc_task(task_id, challenges_path)
    solutions = load_training_solutions(solutions_path)

    # Build true grids
    true_train_grids = grids_from_challenge_train(raw_task)
    true_test_grids  = grids_from_training_solutions(task_id, solutions)

    # Build law config for this task
    law_config = make_law_config_for_task(task_id)

    # Run solver
    from src.runners.kernel import solve_arc_task
    try:
        result = solve_arc_task(task_id, law_config)
    except Exception as e:
        print(f"[ERROR] solve_arc_task failed for task {task_id}: {e}")
        return

    pred_train = result.get("train_outputs_pred", [])
    pred_test  = result.get("test_outputs_pred", [])

    print(f"Task {task_id}:")
    print(f"  Train examples: {len(true_train_grids)}, predicted: {len(pred_train)}")
    print(f"  Test examples:  {len(true_test_grids)}, predicted: {len(pred_test)}")

    # Compare train grids
    for i, true_grid in enumerate(true_train_grids):
        if i >= len(pred_train):
            print(f"  [TRAIN {i}] No prediction produced.")
            continue
        summary = compare_grids(pred_train[i], true_grid)
        if summary["match"]:
            print(f"  [TRAIN {i}] OK (exact match)")
        else:
            print(f"  [TRAIN {i}] MISMATCH: {summary['reason']}, differing cells: {summary['diff_coords'][:10]}")

    # Compare test grids if we have true solutions
    if true_test_grids:
        for j, true_grid in enumerate(true_test_grids):
            if j >= len(pred_test):
                print(f"  [TEST {j}] No prediction produced.")
                continue
            summary = compare_grids(pred_test[j], true_grid)
            if summary["match"]:
                print(f"  [TEST {j}] OK (exact match)")
            else:
                print(f"  [TEST {j}] MISMATCH: {summary['reason']}, differing cells: {summary['diff_coords'][:10]}")
    else:
        print("  No test solutions found in arc-agi_training_solutions.json for this task.")
```

This ensures:

* **No silent failures**:

  * if `solve_arc_task` throws e.g. `TaskSolveError` or `InfeasibleModelError`, we print error and bail.
* Later, a Pi-agent can hook into this behavior (call `validate_on_training` with a law_config, see mismatches/errors, update laws).

### B.6 CLI entry point

Finally:

```python
def main():
    args = parse_args()
    validate_on_training(args.task_id, args.challenges_path, args.solutions_path)

if __name__ == "__main__":
    main()
```

---

## Reviewer/Tester Instructions

1. **Pick a simple training task** that you have a plausible `TaskLawConfig` for.

   * Update `make_law_config_for_task(task_id)` to return the right schemas/params.

2. Run:

   ```bash
   python -m src.runners.validate_on_training <TASK_ID>
   ```

3. Observe output:

   * Should print counts of train/test examples and OK/MISMATCH per example.
   * If solver is infeasible or kernel has a bug, you’ll see `[ERROR] solve_arc_task failed...`.

4. Once you have multiple tasks with valid law configs, you can:

   * Run this script for each one and collect stats,
   * Use it as the core evaluation loop when Pi-agents start proposing law configs.
# clarifications
Nice, your implementer is asking exactly the right questions. Let’s clarify each one cleanly.

---

## Issue 1 – `load_training_solutions` vs `load_arc_training_solutions`

> arc_io.py already has `load_arc_training_solutions()` with same purpose.

Use the **existing** function. Don’t add a duplicate.

So in `validate_on_training.py`:

```python
from src.core.arc_io import load_arc_training_solutions  # use this
```

And call it as-is:

```python
solutions = load_arc_training_solutions(solutions_path)
```

If you want, you *can* add a simple alias in `arc_io.py`:

```python
load_training_solutions = load_arc_training_solutions
```

—but it’s optional. Reusing `load_arc_training_solutions` is enough.

---

## Issue 2 – Raw task structure vs helper expectations

You’re right: my helper sketch assumed a JSON-like structure, but your real loader already normalizes it.

Per your note, `load_arc_task()` (in `context.py`) returns:

```python
{
  "train": [{"input": Grid, "output": Grid}],
  "test": [{"input": Grid}]
}
```

So adjust the helpers to this shape.

Concretely, in `validate_on_training.py`:

```python
def get_true_train_grids(raw_task: Dict[str, Any]) -> list[Grid]:
    """
    Extract true training output grids from the normalized raw_task.
    raw_task["train"] is a list of dicts with 'output' as a Grid already.
    """
    return [pair["output"] for pair in raw_task.get("train", [])]
```

For test solutions via `load_arc_training_solutions`:

```python
def get_true_test_grids(task_id: str, solutions: Dict[str, list[Grid]]) -> list[Grid]:
    """
    Given task_id and solutions mapping task_id -> list[Grid],
    return the list of true test grids, or [] if not present.
    """
    return solutions.get(task_id, [])
```

So:

* `raw_task` from `load_arc_task()` → use its `"output"` Grids directly.
* `solutions` from `load_arc_training_solutions()` → mapping task_id → list[Grid].

No extra JSON structure assumptions.

---

## Issue 3 – Placeholder law config: what should we actually do?

For this runner, we want it to be **immediately runnable**, not just a skeleton.

So yes: use the **same minimal working S1 config** you used in `test_kernel_smoke.py` (or whatever you already know works for at least one simple training task).

So in `validate_on_training.py`:

```python
def make_law_config_for_task(task_id: str) -> TaskLawConfig:
    """
    For now, return a minimal, working TaskLawConfig for this task.
    Implementer:
      - Use the same config you used in test_kernel_smoke.py, or
      - If you already have a hand-crafted law config for this task, return that.
    """
    return TaskLawConfig(
        schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={
                    # same params you used in your smoke test for S1
                }
            )
        ]
    )
```

That way:

* The script is runnable **right now** for at least one task,
* And when you start learning richer law configs, you can replace this function with:

  * a catalog lookup,
  * or Pi-agent–generated configs.

---

## Extra: failure handling (no silent fails)

As you mentioned: **never silent fail**.

So in `validate_on_training.py`:

```python
from src.solver.lp_solver import InfeasibleModelError

...

try:
    result = solve_arc_task(task_id, law_config)
except InfeasibleModelError as e:
    print(f"[ERROR] Infeasible ILP for task {task_id}: {e}")
    return
except Exception as e:
    print(f"[ERROR] solve_arc_task failed for task {task_id}: {e}")
    return
```

This ensures:

* If the law_config is bad, you **see** it immediately,
* A future Pi-agent can catch the same error and refine laws/params.

---

So:

* ✅ Reuse `load_arc_training_solutions`, no duplication.
* ✅ Adjust helpers to the **real** `load_arc_task` + solutions shapes.
* ✅ Seed `make_law_config_for_task` with a real S1 config so the script runs.
* ✅ Surface infeasibility/errors explicitly.

That keeps everything aligned with the math kernel *and* makes the validation runner useful immediately.
