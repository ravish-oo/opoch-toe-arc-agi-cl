## 🔹 Expanded WO-M6.1 – Role labeller (WL/q) over TaskContext

### File: `src/law_mining/roles.py`

**Goal:** For a single ARC task (one `TaskContext`), assign a **role_id** to each pixel in each grid (`train_in`, `train_out`, `test_in`) using φ + WL-style refinement. Mining will use these roles instead of raw pixels.

We do **pure structural refinement**, no heuristics, no defaults.

---

### 1. Imports & type aliases

Use only standard libs + existing project modules.

At the top of `roles.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal

from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.core.grid_types import Grid
from src.features.coords_bands import (
    row_band_labels,
    col_band_labels,
)
from src.features.components import (
    connected_components_by_color,
    compute_shape_signature,
)
```

> 🔎 Note: we **re-use** existing φ operators. No new custom algorithms beyond the WL loop itself.

Define the key type:

```python
# kind ∈ {"train_in", "train_out", "test_in"}
NodeKind = Literal["train_in", "train_out", "test_in"]

# Node key: (kind, example_idx, r, c)
RolesMapping = Dict[Tuple[NodeKind, int, int, int], int]
```

---

### 2. Internal representation of a “node”

To simplify WL, define a simple struct:

```python
@dataclass(frozen=True)
class Node:
    kind: NodeKind
    example_idx: int   # index into TaskContext.train_examples or test_examples
    r: int
    c: int
```

We’ll maintain:

* `nodes: List[Node]` – all nodes in this task.
* `labels: Dict[Node, Tuple]` – current label (tuple) per node.

---

### 3. Build nodes and initial labels

Implement a function:

```python
def compute_roles(task_context: TaskContext, wl_iters: int = 3) -> RolesMapping:
    """
    Use φ (coords, bands, components, etc.) + WL-style refinement
    to assign stable role_ids per pixel across all grids in this task.

    Returns:
        roles: mapping (kind, example_idx, r, c) -> role_id (0..R-1)
    """
```

Inside `compute_roles`:

#### 3.1 Collect nodes

* For each `ex_idx, ex_ctx` in `enumerate(task_context.train_examples)`:

  * For each pixel `(r,c)` in `ex_ctx.input_grid`:

    * add `Node("train_in", ex_idx, r, c)`.
  * If `ex_ctx.output_grid` is not `None`:

    * for each `(r,c)` in `ex_ctx.output_grid.shape`:

      * add `Node("train_out", ex_idx, r, c)`.

* For each `ex_idx, ex_ctx` in `enumerate(task_context.test_examples)`:

  * For each `(r,c)` in `ex_ctx.input_grid`:

    * add `Node("test_in", ex_idx, r, c)`.

We **do not** create nodes for test outputs (they don’t exist).

#### 3.2 Initial labels (pure structure)

For each node, we define a **tuple label** from:

* `kind` (train_in/train_out/test_in),
* the raw color at that pixel,
* row/col “bands” within its own grid,
* grid-border flag,
* (optionally) shape-signature of its component (for train_in / train_out only).

Implementation sketch:

1. For each `ExampleContext` (train_in, train_out, test_in) separately:

   * Extract `grid = input_grid` or `output_grid` accordingly.
   * Let `H, W = grid.shape`.
   * Precompute:

     * `row_bands = row_band_labels(H)`  # row → "top"/"middle"/"bottom"
     * `col_bands = col_band_labels(W)`  # col → "left"/"middle"/"right"
     * grid-border flag: `is_border = (r == 0 or r == H-1 or c == 0 or c == W-1)`
   * For **component-based shape**, only for *inputs and outputs*:

     * call `connected_components_by_color(grid)` and `compute_shape_signature` for each component.
     * build map `(r,c) -> (color, shape_signature)` for that grid.

2. For each node:

   * Get the right grid + precomputed info based on:

     * `kind` and `example_idx`.
   * Define initial label as a tuple, e.g.:

     ```python
     label = (
         node.kind,
         int(color),                          # color from the grid
         row_bands[r],
         col_bands[c],
         bool(is_border_flag),
         shape_sig_for_pixel or None,         # if we computed it, else None
     )
     ```

No special treatment for color `0` vs others.
No defaults. Just structural descriptors.

Store into `labels[node] = label`.

---

### 4. Neighborhood definition

Define a helper function inside `roles.py`:

```python
def _neighbors(node: Node, task_context: TaskContext) -> List[Node]:
    """
    Return 4-connected neighbors (up/down/left/right) of this node,
    restricted to the same kind and same example_idx.
    """
```

Implementation details:

* Get the appropriate grid shape from `TaskContext`:

  * for `"train_in"`: `task_context.train_examples[node.example_idx].input_H / input_W`
  * for `"train_out"`: `output_H/output_W`
  * for `"test_in"`: `task_context.test_examples[node.example_idx].input_H / input_W`
* For each of the four offsets `(dr, dc) ∈ {(-1,0),(1,0),(0,-1),(0,1)}`:

  * Compute `(rr, cc) = (node.r + dr, node.c + dc)`.
  * If `0 <= rr < H` and `0 <= cc < W`, create `Node(node.kind, node.example_idx, rr, cc)` and include it.

We **don’t** cross-link input ↔ output at WL level; WL structural refinement is per-grid-kind. Cross-input/output invariants will be mined by schemas using train_out.

---

### 5. WL refinement loop

Implement WL refinement inside `compute_roles` after initial labels:

```python
    nodes = list(labels.keys())
    for it in range(wl_iters):
        new_labels: Dict[Node, Tuple] = {}
        for node in nodes:
            base_label = labels[node]
            neighs = _neighbors(node, task_context)
            neigh_labels = [labels[n] for n in neighs]
            # Canonical multiset: sort by tuple
            neigh_labels_sorted = tuple(sorted(neigh_labels))
            new_label = (base_label, neigh_labels_sorted)
            new_labels[node] = new_label

        # Early exit if stable
        if all(new_labels[n] == labels[n] for n in nodes):
            labels = new_labels
            break

        labels = new_labels
```

* `wl_iters` default can be 3 or 4; no need to be huge for tiny ARC grids.
* No randomness, no heuristics. Pure deterministic refinement.

---

### 6. Map final labels to role_ids

After WL converges:

* We want to map **distinct label tuples** to consecutive integers `[0..R-1]`.

Implementation:

```python
    # Map label tuple -> role_id
    label_to_role: Dict[Tuple, int] = {}
    roles: RolesMapping = {}
    next_role_id = 0

    for node in nodes:
        lab = labels[node]
        if lab not in label_to_role:
            label_to_role[lab] = next_role_id
            next_role_id += 1
        role_id = label_to_role[lab]
        roles[(node.kind, node.example_idx, node.r, node.c)] = role_id

    return roles
```

This `roles` mapping is the only output of `compute_roles`.

No default colors, no “special treatment” for any role.
It’s purely structural information for the miners.

---

## Thin smoke test runner

### File: `src/law_mining/test_roles_smoke.py`

**Goal:** verify `compute_roles` runs on a real ARC training task, produces a reasonable number of roles, and is deterministic.

Contents:

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles

def main():
    # Pick a small, simple task id that you know exists in arc-agi_training_challenges.json
    task_id = "0"  # adjust to a valid id present in your data
    challenges_path = Path("data/arc-agi_training_challenges.json")

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)

    print("Total assigned roles:", len(set(roles.values())))
    # Print a small sample
    sample_items = list(roles.items())[:20]
    for (key, role_id) in sample_items:
        print("Node:", key, "-> role_id:", role_id)

if __name__ == "__main__":
    main()
```

> This is just a smoke test: we’re not checking correctness, only that:
>
> * it runs without error,
> * produces role_ids,
> * doesn’t explode in count (e.g. not 1 role per pixel unless the task is totally structureless).

---

## Reviewer + tester instructions

**For implementer:**

* Follow `roles.py` spec literally:

  * Use `TaskContext` and `ExampleContext` as they exist,
  * Recompute any needed φ from existing `src/features/*` modules,
  * Implement `_neighbors` as 4-connected, kind-local,
  * Implement WL with deterministic tuple sorting, no random hashes.
* Do **not** introduce:

  * special-casing “background 0”,
  * defaults,
  * any use of training outputs beyond including their colors in initial labels.

**For reviewer/tester:**

1. **Static review:**

   * Confirm:

     * `compute_roles` only depends on structure (kind, colors, φ, neighbors).
     * No ad-hoc thresholds or defaults.
     * WL loop is deterministic and has a fixed max iteration count.

2. **Smoke test run:**

   ```bash
   python -m src.law_mining.test_roles_smoke
   ```

   Check that:

   * Script runs without exceptions.
   * It prints a sensible number of roles (e.g. not 0, not an absurd number).
   * Re-running gives the same mapping (deterministic behavior).

3. **Optional deeper check:**

   * Run `compute_roles` twice on the same `TaskContext` and verify:

     ```python
     roles1 = compute_roles(task_context)
     roles2 = compute_roles(task_context)
     assert roles1 == roles2
     ```

   * (This can be added as a small unit test.)

---
