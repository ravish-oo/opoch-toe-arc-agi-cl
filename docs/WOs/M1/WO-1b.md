## 🔹 Work Order 1b – ARC JSON IO + tiny runner

**Files:**

* `src/core/arc_io.py`
* `src/runners/print_sample_task.py`

**Goal:**
Load ARC-AGI training challenges from JSON into our `Grid` type, and have a tiny runner that prints a sample task to ensure everything is wired.

**Libraries to use:**

* `json`
* `pathlib.Path`
* `typing` (`Dict`, `List`, `Tuple`)
* `numpy`
* from `src.core.grid_types` import `Grid`

**NOTE:**
We only care about:

* `arc-agi_training_challenges.json`
* optionally `arc-agi_training_solutions.json` for tests.

We will **not** assume the exact JSON schema by guess; instead we define an output contract and tell Claude to inspect the file to conform to it.

### 1b.1 Implement `load_arc_training_challenges`

In `src/core/arc_io.py`:

```python
from pathlib import Path
from typing import Dict, List
import json
import numpy as np

from src.core.grid_types import Grid
```

Implement:

```python
def load_arc_training_challenges(path: Path) -> Dict[str, Dict[str, List[Grid]]]:
    """
    Load ARC-AGI training challenges from the given JSON file.

    Output contract (must be followed):

      returns a dict:
        {
          task_id: {
            "train": [Grid, Grid, ...],  # each is an input grid
            "train_outputs": [Grid, Grid, ...],  # matching outputs (if present)
            "test": [Grid, Grid, ...]    # test input grids
          },
          ...
        }

    - task_id is a string.
    - All Grids must be numpy arrays with dtype=int.
    - If the challenges JSON does not contain train outputs (only inputs),
      then "train_outputs" may be an empty list for now.
    """

    # Implementation must:
    #   - open the JSON file,
    #   - inspect its structure,
    #   - adapt to the ARC-AGI format,
    #   - but always produce the above output shape.
    ...
```

Explicit instructions:

* Convert every list-of-lists grid in JSON to `np.ndarray(dtype=int)`.
* Ensure `train` and `train_outputs` lists are the same length if outputs are present.
* If training outputs are not in this file (they’re in `arc-agi_training_solutions.json`), just:

  * set `"train_outputs": []` here,
  * we’ll join with solutions later.

### 1b.2 (Optional but good) solutions loader

You can define:

```python
def load_arc_training_solutions(path: Path) -> Dict[str, Dict[str, List[Grid]]]:
    """
    Load ARC-AGI training solutions from JSON.

    Expected contract:

      {
        task_id: {
          "test_outputs": [Grid, Grid, ...]
        },
        ...
      }

    - Inspect the actual JSON structure and adapt, but make sure to
      produce this contract.
    """
    ...
```

We may use this later, but don’t overcomplicate now.

### 1b.3 Thin runner: print a sample task

Create `src/runners/print_sample_task.py`:

```python
from pathlib import Path

from src.core.arc_io import load_arc_training_challenges
from src.core.grid_types import print_grid

def main() -> None:
    data_path = Path("data/arc-agi_training_challenges.json")
    tasks = load_arc_training_challenges(data_path)

    # Pick the first task_id
    task_ids = sorted(tasks.keys())
    if not task_ids:
        print("No tasks loaded.")
        return

    task_id = task_ids[0]
    t = tasks[task_id]

    print(f"Task: {task_id}")
    print(f"  #train examples: {len(t['train'])}")
    print(f"  #test inputs:    {len(t['test'])}")

    if t["train"]:
        print("\nFirst train input grid:")
        print_grid(t["train"][0])

        if t["train_outputs"]:
            print("\nFirst train output grid:")
            print_grid(t["train_outputs"][0])

    if t["test"]:
        print("\nFirst test input grid:")
        print_grid(t["test"][0])

if __name__ == "__main__":
    main()
```

### Reviewer/tester instructions:

1. Ensure repo structure matches:

   ```text
   src/core/grid_types.py
   src/core/arc_io.py
   src/runners/print_sample_task.py
   data/arc-agi_training_challenges.json
   ```

2. Run:

   ```bash
   python -m src.runners.print_sample_task
   ```

3. Check:

   * No stack trace.
   * It prints something like:

     ```text
     Task: <some_id>
       #train examples: N
       #test inputs:    M

     First train input grid:
     ...

     First train output grid:
     ...   # if present

     First test input grid:
     ...
     ```

4. Verify:

   * `train` and `test` grids look like valid ARC grids (small integer arrays).
   * Grids are `numpy.ndarray` with `dtype=int` (you can inspect with a quick `type()` and `grid.dtype` if you want).

