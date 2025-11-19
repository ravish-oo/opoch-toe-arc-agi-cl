## 🔹 WO-M6.3B – Miners for S3, S4, S8, S9

### File: `src/law_mining/mine_s3_s4_s8_s9.py`

**Goal:** For one `TaskContext` and its `RolesMapping` + `RoleStats`, mine all **always-true** instances of:

* **S3** – band/stripe templates (rows/cols in same class share pattern),
* **S4** – residue-class coloring (mod K stripes/checkerboards),
* **S8** – tiling / replication,
* **S9** – cross / plus propagation,

and return `List[SchemaInstance]` for each `mine_Sk` with **schema_params matching your existing S3/S4/S8/S9 builders**.

No best-effort, no majority votes, no defaults. A schema instance is only mined if it is exactly consistent with all training examples.

---

### 0. Imports & types

At top of `mine_s3_s4_s8_s9.py`:

```python
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance

from src.features.coords_bands import row_band_labels, col_band_labels
from src.features.neighborhoods import neighborhood_hashes
```

> 🔎 We only use numpy + existing φ operators. No custom ML, no external search libs.

You must inspect:

* `src/schemas/s3_bands.py`
* `src/schemas/s4_residue_color.py`
* `src/schemas/s8_tiling.py`
* `src/schemas/s9_cross_propagation.py`

to see exactly what `schema_params` shape each builder expects.

---

## 1. `mine_S3` – band/stripe templates

**Signature:**

```python
def mine_S3(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 1.1 Concept

S3 encodes:

> Rows (or columns) in the same “band class” share the **same color pattern** in outputs.

We mine S3 by:

* grouping rows by band class (e.g. "top", "middle", "bottom", plus maybe other φ flags),
* for each band class:

  * derive the row pattern from train outputs,
  * require that:

    * all rows in that class, in all training examples,
    * share that pattern exactly,
  * then use S3 builder to tie those rows accordingly.

### 1.2 Logic (rows; columns analogous if builder supports)

1. For each training example `ex_idx, ex`:

   ```python
   grid_out = ex.output_grid  # must not be None for training
   H_out, W_out = grid_out.shape
   row_bands = row_band_labels(H_out)  # row -> "top"/"middle"/"bottom"
   ```

2. Collect row patterns by band class:

   ```python
   # map: (band_label, maybe_extra_key) -> list of (example_idx, row_idx, row_pattern)
   band_row_patterns: Dict[Tuple[str], List[Tuple[int, int, Tuple[int,...]]]] = defaultdict(list)
   ```

   * For each row `r`:

     * `band = row_bands[r]`

     * `pattern = tuple(int(c) for c in grid_out[r, :])`

     * Append `(ex_idx, r, pattern)` to `band_row_patterns[(band,)]`.

   > If your S3 builder uses a richer notion of band (e.g. plus some φ flags like “row_has_nonzero”), you can include those in the key.

3. For each `band_key, records` in `band_row_patterns.items()`:

   * Extract all `pattern`s associated with that band across examples and rows.
   * If **all patterns are identical**:

     * Let `template = that pattern`.
     * This is a candidate S3 law: “all rows in band_key have pattern `template`”.
   * If **any pattern differs**:

     * That band_key is not S3-consistent → **skip** it.

4. Convert consistent bands into S3 params:

   * If your S3 builder expects something like:

     ```python
     params = {
         "row_band_templates": [
             {
                 "band": "top",
                 "pattern": [c0, c1, ..., c_{W-1}],
             },
             ...
         ]
     }
     ```

     then:

     * build a list of `{ "band": band, "pattern": list(template) }` for each consistent band_key,
     * set that as `schema_params`.

   * If your builder expects explicit row indices instead of band labels, change accordingly:

     * For each training example, find all rows with that band, feed them into params.

5. If at least one band yielded a consistent template, return `[SchemaInstance("S3", params)]`, otherwise `[]`.

**Constraints:**

* S3 is only mined where **all train outputs** in that band class agree on the same pattern.
* No “dominant” pattern — only exact equality.
* If your S3 builder supports columns too, you can symmetrically mine column patterns, using `col_band_labels` and column slices.

---

## 2. `mine_S4` – residue-class coloring

**Signature:**

```python
def mine_S4(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 2.1 Concept

S4 captures:

> Colors determined purely by coordinate residue class modulo K, e.g. K=2/3/4/5 stripes or checkerboards.

We mine S4 by:

* For each candidate `K` in `{2,3,4,5}`:

  * for each relevant residue `r` (e.g. `c mod K` for column-based stripes, or `r mod K` for row-based),
  * check if **output colors are constant** across all training examples for pixels with that residue,
  * if so for all residues used, create a schema instance for that pattern.

### 2.2 Logic

1. Initialize a list `instances: List[SchemaInstance] = []`.

2. For each direction in `{ "row", "col" }` (if your S4 builder supports both):

   ```python
   for direction in ["row", "col"]:
       for K in [2,3,4,5]:
           # attempt to mine a residue-class mapping for (direction, K)
   ```

3. Inside each `(direction, K)` candidate:

   * Initialize:

     ```python
     residue_to_colors: Dict[int, set[int]] = defaultdict(set)
     ```

   * For each training example `ex_idx, ex`:

     ```python
     grid_out = ex.output_grid
     H, W = grid_out.shape
     for r in range(H):
         for c in range(W):
             color = int(grid_out[r, c])
             if direction == "row":
                 residue = r % K
             else:
                 residue = c % K
             residue_to_colors[residue].add(color)
     ```

   * After processing all examples:

     * For each `residue, colors`:

       * If `len(colors) == 0` → shouldn’t happen, but harmless.
       * If `len(colors) > 1` → **conflict**; this `(direction, K)` is **invalid** → abort this K (no S4 instance here).
     * If **all** residues have exactly one color:

       * Build `residue_to_color = {residue: single_color}`.

   * If candidate is valid (no conflicts):

     * Convert into schema_params that your S4 builder expects, e.g.:

       ```python
       params = {
           "direction": direction,    # if your builder distinguishes row vs col
           "K": K,
           "residue_to_color": residue_to_color,
       }
       instances.append(SchemaInstance(family_id="S4", params=params))
       ```

4. Return `instances`.

**Constraints:**

* S4 is only mined where **all** pixels in a residue class map to the same color across **all** training outputs.
* No fallback; if any residue is inconsistent, discard that `K` (and direction combination).

---

## 3. `mine_S8` – tiling / repetition

**Signature:**

```python
def mine_S8(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 3.1 Concept

S8 encodes:

> A small base tile T is repeated (tiled) across a region with fixed stride, reconstructing the output.

We mine S8 by:

* Looking at train outputs,
* Detecting a minimal base tile that, when repeated with some stride, re-creates the output pattern exactly.

### 3.2 Logic (per training example, then consistency across examples)

Because ARC grids are small, a simple but deterministic approach is fine.

1. For each training example `ex_idx, ex`:

   ```python
   grid_out = ex.output_grid
   H, W = grid_out.shape
   ```

   * Try candidate tile sizes `(h, w)` with small divisors of `(H, W)`, e.g. all `(h,w)` such that `H % h == 0` and `W % w == 0`, and `h,w` small (≤ H, ≤ W).

2. For each candidate `(h,w)`:

   * Extract a candidate base tile:

     ```python
     base_tile = grid_out[0:h, 0:w]
     ```

   * Reconstruct a tiled grid:

     ```python
     tiled = np.tile(base_tile, (H // h, W // w))
     ```

   * If `np.array_equal(tiled, grid_out)`:

     * This `(h,w)` is a valid tiling for this example; record:

       ```python
       example_tile_info[ex_idx] = {
           "h": h,
           "w": w,
           "base_tile": base_tile.copy(),
       }
       ```

   * If no `(h,w)` passes for this example:

     * This example does not admit a simple full-grid tiling → S8 may not apply to this task.

3. Across training examples:

   * For S8 to be mined, we need:

     * the **same** `(h,w)` across all examples,
     * and the **same base_tile** (up to maybe trivial symmetries, if your builder supports that).

   * If `example_tile_info` is empty or inconsistent (different `(h,w)` or different `base_tile`) → return `[]`.

4. If consistent:

   * Build S8 `schema_params` as expected by your builder, e.g.:

     ```python
     params = {
         "tile_height": h,
         "tile_width": w,
         "base_tile": base_tile.tolist(),
     }
     return [SchemaInstance(family_id="S8", params=params)]
     ```

   * If your S8 builder supports tiling only a subregion (not full grid), you can adapt by checking region extents where tiling holds. For first version, full-grid tiling is enough.

**Constraints:**

* S8 is only mined when **all train outputs** are exact tilings of the same base tile with same stride.
* No partial matches or “mostly tiled” detection.
* If the task is not clearly tiling-like, S8 miner returns `[]`.

---

## 4. `mine_S9` – cross / plus propagation

**Signature:**

```python
def mine_S9(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 4.1 Concept

S9 encodes:

> Given plus-shaped seeds, the pattern is extended along rows/columns in specific directions.

We mine S9 by:

* Detecting plus centers (cross shapes) in train inputs and/or outputs via neighborhood hashes,
* For each seed type, inferring:

  * which directions are extended (up/down/left/right),
  * with which colors,
* Checking that this behavior is consistent across all training examples.

### 4.2 Logic

1. For each training example `ex_idx, ex`:

   ```python
   grid_in = ex.input_grid
   grid_out = ex.output_grid
   H, W = grid_in.shape
   hashes_in = neighborhood_hashes(grid_in, radius=1)
   hashes_out = neighborhood_hashes(grid_out, radius=1)
   ```

2. Detect candidate seed positions:

   * You may define a “plus pattern” code by looking at 3×3 neighborhoods:

     * center has some color,
     * up/down/left/right share that color or have particular pattern.
   * For the first version, you can:

     * treat **rarer neighborhood hashes** as “seed types”.
     * For each hash value H_in (3×3 input pattern), check if the output around that center exhibits a “plus extension” pattern.

3. For each seed type (e.g. hash value H_seed):

   * Collect all occurrences `(ex_idx, r, c)` where `hashes_in[(r,c)] == H_seed`.
   * For each such occurrence:

     * Examine `grid_out` along directions:

       * up: (r-1,c), (r-2,c), ...
       * down, left, right.
     * Determine:

       * for each direction, the maximal run of non-background / specific color,
       * whether this pattern is consistent across all occurrences in all examples.

4. Aggregate direction/color behavior:

   * For each seed type H_seed and direction d:

     * collect sequences of colors observed along that ray in grid_out,
     * if across all seeds and examples the behavior is **the same** (e.g. “paint color 3 until you hit another shape”),
     * treat that as a candidate S9 rule.

5. Translate into schema_params:

   * This heavily depends on your `build_S9_constraints` format. For example:

     ```python
     params = {
         "seed_hash": H_seed,
         "directions": {
             "up": {"color": 3, "stop_at": "border"},
             "down": {...},
             ...
         }
     }
     ```

   * The implementer must inspect `s9_cross_propagation.py` to match the exact format.

6. Only if S9 behavior is **exactly consistent** across all occurrences and all training examples do we emit a `SchemaInstance` for that seed type.

**Constraints:**

* S9 miner is allowed to be conservative:

  * if it cannot confidently detect a clean plus pattern, it returns `[]`.
* No “best guess” on plus seeds; it only emits rules that are always true on train outputs for that pattern.

---

## Thin smoke test runner

### File: `src/law_mining/test_mine_s3_s4_s8_s9_smoke.py`

**Goal:** verify that miners run without crashing and return structured `SchemaInstance` lists.

Contents:

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s3_s4_s8_s9 import mine_S3, mine_S4, mine_S8, mine_S9

def main():
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "0"  # or another simple task

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s3_instances = mine_S3(task_context, roles, role_stats)
    s4_instances = mine_S4(task_context, roles, role_stats)
    s8_instances = mine_S8(task_context, roles, role_stats)
    s9_instances = mine_S9(task_context, roles, role_stats)

    print("S3 instances:", s3_instances)
    print("S4 instances:", s4_instances)
    print("S8 instances:", s8_instances)
    print("S9 instances:", s9_instances)

if __name__ == "__main__":
    main()
```

> This is only a smoke test: we just want to see that the miners run and produce data, not that it’s correct yet.

---

## Reviewer + tester instructions

**For implementer:**

* Follow the **exact pattern**:

  * mine only always-true patterns across all training examples,
  * if any conflict is found, discard that candidate schema instance,
  * never use majority voting, thresholds, or defaults.
* For each `mine_Sk`, adapt `schema_params` to the **actual existing** builder’s expected format by inspecting `s3_bands.py`, `s4_residue_color.py`, `s8_tiling.py`, and `s9_cross_propagation.py`.
* Use only numpy, collections, and existing φ: no new “clever algorithms”.

**For reviewer/tester:**

1. **Static review:**

   * Verify that:

     * each miner checks **all training examples**,
     * any contradictory evidence makes the miner skip that schema,
     * no “most frequent” color / pattern logic is used,
     * roles and φ are only used as sources of structural facts.

2. **Smoke tests:**

   ```bash
   python -m src.law_mining.test_mine_s3_s4_s8_s9_smoke
   ```

   * Confirm:

     * script runs with no exceptions,
     * returns `SchemaInstance` lists (could be empty on some tasks).

3. **Optional deeper test:**

   * Pick a task that is clearly:

     * band/stripe (S3/S4),
     * tiling (S8),
     * or plus/cross (S9),
   * Run the miners and assemble a `TaskLawConfig` from mined schemas,
   * Call `solve_arc_task_with_diagnostics(task_id, law_config, use_training_labels=True)`,
   * Expect:

     * `diagnostics.status == "ok"` for such tasks if miners + builders are implemented correctly.

# Clarification
clarification for implementer (S9 in M6.3B)

You can give them this:

def mine_S9(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    NOTE (M6.3B):

    S9 ('cross/plus propagation') is a valid schema family, and its builder
    is implemented and used by the kernel. However, the automatic mining of
    S9 instances from training examples is intentionally NOT implemented in
    this milestone.

    Mining S9 requires a precise, invariant-based definition of plus-shape
    seeds and directional propagation, which is non-trivial to design and
    verify. To avoid guessing, heuristics, or 'best effort' behavior, S9
    mining is deferred to a dedicated later milestone (M6.3D or similar).

    For now, this function returns an empty list, meaning the law miner
    does not propose any S9 schema instances yet. Any tasks that require S9
    will remain unsolved/underconstrained until S9 mining is added properly.
    """
    return []


That’s TOE-clean: we are not pretending, and we’re explicitly stating the limit.