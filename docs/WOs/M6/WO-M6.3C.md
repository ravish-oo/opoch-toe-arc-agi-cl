## 🔹 WO-M6.3C – Miners for S5, S6, S7, S11

### File: `src/law_mining/mine_s5_s6_s7_s11.py`

**Goal:** For a single task (`TaskContext`, `RolesMapping`, `RoleStats`), implement deterministic, always-true-only miners for:

* **S5** – seed→template stamping,
* **S6** – crop to ROI,
* **S7** – block/summary grids,
* **S11** – local 3×3 codebook.

Each `mine_Sk` returns `List[SchemaInstance]` with `params` matching the **existing builder** implementation for S5, S6, S7, S11. If no law is *fully consistent* with all training examples, that miner returns `[]`.

No defaults, no “most common”, no best-effort.

---

### 0. Imports & types

At top of `mine_s5_s6_s7_s11.py`:

```python
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance

from src.features.components import connected_components_by_color
from src.features.neighborhoods import neighborhood_hashes
```

> Implementer MUST inspect:
>
> * `src/schemas/s5_template_stamping.py`
> * `src/schemas/s6_crop_roi.py`
> * `src/schemas/s7_aggregation.py`
> * `src/schemas/s11_local_codebook.py`
>
> to know the **exact** schema_params structure they expect. Miners MUST adapt to that; we do NOT change builders.

---

## 1. `mine_S5` – Template stamping (seed → patch)

**Signature:**

```python
def mine_S5(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 1.1 Concept

S5 encodes:

> For a given seed type (e.g. neighborhood pattern in the input), the output around it always matches a fixed stencil `P_t` (e.g. 3×3, 5×5 icon).

We mine S5 by:

* finding seed types in **train inputs**,
* for each seed type, collecting **output patches** in train outputs,
* if all patches are identical, we record that seed type → template.

### 1.2 High-level algorithm

1. Choose a **patch size** (consistent with your S5 builder): e.g. 3×3.

   ```python
   PATCH_RADIUS = 1  # for 3x3; adjust if builder uses 5x5
   ```

2. For each train example `ex_idx, ex`:

   * `grid_in = ex.input_grid`

   * `grid_out = ex.output_grid`

   * Compute neighborhood hashes on input:

     ```python
     hashes_in = neighborhood_hashes(grid_in, radius=PATCH_RADIUS)
     ```

   * For each pixel `(r,c)` where a full patch fits in both input and output:

     * Let `H_seed = hashes_in[(r,c)]` (seed type).
     * Extract output patch `P_t` from `grid_out[r-PATCH_RADIUS:r+PATCH_RADIUS+1, c-PATCH_RADIUS:c+PATCH_RADIUS+1]`.
     * Store:

       ```python
       patches_by_seed[H_seed].append((ex_idx, (r, c), P_t.copy()))
       ```

3. For each seed type `H_seed`:

   * Consider all `P_t` patches seen across all training examples:

     ```python
     patterns = {patch.tobytes() for (_, _, patch) in patches_by_seed[H_seed]}
     ```

   * If `len(patterns) == 0`: ignore.

   * If `len(patterns) > 1`: inconsistent → **discard this seed type for S5**.

   * If `len(patterns) == 1`:

     * Extract the canonical `P_t` (convert from bytes back to array if needed).

4. Translate into schema_params per S5 builder:

   Your builder likely expects something like:

   ```python
   {
       "example_type": "train",
       "example_index": 0,
       "seeds": [
           {
               "center": "(r,c)",     # seed center coord as string
               "pattern": { "(dr,dc)": color, ... },
           },
           ...
       ]
   }
   ```

   Implementation pattern:

   * For each seed type `H_seed` that passed:

     * For each train example `ex_idx`:

       * Collect seeds `(r,c)` of that type in this example.
       * For each such center, include a seed entry with:

         * `"center"` = f"({r},{c})"
         * `"pattern"` = dict:

           * for each `(dr,dc)` in patch window:

             * `pattern[f"({dr},{dc})"] = int(P_t[dr+PATCH_RADIUS, dc+PATCH_RADIUS])`
       * Create `SchemaInstance("S5", params)` for that example.

5. Return all such instances.

**Constraints:**

* Seed type H_seed is only kept if **all** observed patches in outputs are identical.
* We don’t guess seed types; we use the input neighborhood hash.
* No patch is “mostly similar”; it’s either exactly consistent or thrown away.

---

## 2. `mine_S6` – Crop to ROI

**Signature:**

```python
def mine_S6(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 2.1 Concept

S6 encodes:

> The output grid is a **subgrid crop** from the input, typically around some object or region (largest component, unique non-zero, etc.).

Mining S6 is about:

* detecting that all training outputs are crops of their corresponding inputs,
* identifying a rule to select the crop region,
* verifying that rule is consistent across all training examples.

### 2.2 High-level algorithm

1. For each training example `ex_idx, ex`:

   ```python
   grid_in = ex.input_grid
   grid_out = ex.output_grid
   H_in, W_in = grid_in.shape
   H_out, W_out = grid_out.shape
   ```

2. Check if `H_out ≤ H_in` and `W_out ≤ W_in`. If not, this example is **not crop-like** → no S6 for this task (return `[]`).

3. For crop detection:

   * Scan all possible positions `(r0, c0)` in input such that a subgrid of size `(H_out, W_out)` fits:

     ```python
     for r0 in range(H_in - H_out + 1):
         for c0 in range(W_in - W_out + 1):
             sub = grid_in[r0:r0+H_out, c0:c0+W_out]
             if np.array_equal(sub, grid_out):
                 candidates_for_example.append((r0, c0))
     ```

   * If `candidates_for_example` is empty → no S6 crop for this task.

   * If multiple `(r0, c0)` exist, we must later derive a consistent **selection rule**; multiple candidates per example are allowed at this stage, but we must find a consistent pattern across examples.

4. Across examples, try simple selection rules:

   Example rule families:

   * “crop around the bounding box of the largest component of color X”,
   * “crop around the bounding box of the unique non-zero component”.

   Implementation pattern:

   * For each training example, run `connected_components_by_color(grid_in)` and get component bboxes.

   * For each example, for each component, compute its bbox `(r_min, r_max, c_min, c_max)` and see if it exactly matches one of the `(r0, c0, r0+H_out-1, c0+W_out-1)` crop candidates.

   * If for all examples there is exactly **one** component per example whose bbox matches one of the valid crop positions, and that component’s class (e.g. `(color_in)`) is the same across examples, we can define a rule:

     > “Crop the bbox of the largest component of color X”.

   * If no such consistent component class exists, we **do not mine S6**.

5. Translate into schema_params:

   Depending on S6 builder’s API, something like:

   ```python
   {
       "selection": "largest_component_color",
       "color": X
       # or other selection fields matching your builder
   }
   ```

   Then per-example `SchemaInstance("S6", params)` will be created by your higher-level `mine_law_config` or by including example_type/index inside params, depending on S6 builder.

**Constraints:**

* S6 is only mined if:

  * every training example’s output is an exact crop of its input,
  * and there is a **single, consistent rule** to select that crop (same component class, etc.).
* If any example has ambiguous or missing crop candidates, S6 miner returns `[]`.

---

## 3. `mine_S7` – Summary/histogram grids

**Signature:**

```python
def mine_S7(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 3.1 Concept

S7 encodes:

> Input is partitioned into blocks; each output cell summarizes one block (e.g. “the unique non-zero color in that block”).

We mine S7 by:

* checking if outputs are significantly smaller than inputs,
* partitioning input into regular blocks based on output shape,
* testing consistent summary rules per block across all training examples.

### 3.2 High-level algorithm

1. For each training example `ex_idx, ex`:

   ```python
   grid_in = ex.input_grid
   grid_out = ex.output_grid
   H_in, W_in = grid_in.shape
   H_out, W_out = grid_out.shape
   ```

2. Check if `H_out` and `W_out` are **divisors** of `H_in` and `W_in`:

   * If not, this example is not a clean block-aggregation → no S7 for this task.

3. Define block sizes:

   ```python
   block_h = H_in // H_out
   block_w = W_in // W_out
   ```

4. For each training example and each output cell `(i,j)`:

   * Define the corresponding input block:

     ```python
     r0 = i * block_h
     c0 = j * block_w
     block = grid_in[r0:r0+block_h, c0:c0+block_w]
     summary_out_color = int(grid_out[i, j])
     ```

   * Candidate summary rule: “unique non-zero color in block”:

     ```python
     nonzero_colors = {int(c) for c in block.flatten() if c != 0}
     ```

     * If `len(nonzero_colors) == 1`:

       * call it `c_block`; check if `c_block == summary_out_color`.
       * If mismatch for any block → this candidate summary rule fails for this task.
     * If `len(nonzero_colors) == 0`:

       * maybe the summary rule is “all zero → output 0”; check that `summary_out_color == 0`.
     * If `len(nonzero_colors) > 1`:

       * The “unique non-zero” rule fails here.

5. If this rule holds for every block in every training example:

   * We can mine a simple S7 law: “for each block, output the unique non-zero color, or 0 if none”.

6. Translate into S7 schema_params:

   Example shape (adapt to your builder):

   ```python
   params = {
       "block_height": block_h,
       "block_width": block_w,
       "rule": "unique_nonzero_or_zero",
       # possibly with more detail if builder requires
   }
   instances = [
       SchemaInstance(family_id="S7", params={
           "example_type": "train",
           "example_index": ex_idx,
           "block_height": block_h,
           "block_width": block_w,
           "rule": "unique_nonzero_or_zero",
       })
       for ex_idx, ex in enumerate(task_context.train_examples)
   ]
   ```

**Constraints:**

* Only emit S7 if the rule is **exactly true** for all blocks of all train examples.
* If builder supports more complex summary rules, you can extend this logic, but always with “always-true” criteria.

---

## 4. `mine_S11` – Local neighborhood codebook

**Signature:**

```python
def mine_S11(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

### 4.1 Concept

S11 is the most general local schema:

> For each input neighborhood type H, the local output pattern around that pixel is always the same P(H).

We mine S11 by:

* hashing local neighborhoods in **train inputs**,
* extracting corresponding patches in **train outputs**,
* building a codebook H → P only where P is consistent across all examples.

### 4.2 High-level algorithm

1. Fix patch radius (e.g. 1 for 3×3; check your S11 builder).

   ```python
   R = 1  # 3x3 patch
   ```

2. Initialize:

   ```python
   codebook: Dict[int, np.ndarray] = {}
   valid_hash: Dict[int, bool] = defaultdict(lambda: True)
   ```

3. For each training example `ex_idx, ex`:

   * `grid_in = ex.input_grid`

   * `grid_out = ex.output_grid`

   * Compute `hashes_in = neighborhood_hashes(grid_in, radius=R)`

   * For each pixel `(r,c)` where full patch fits both in input and output:

     * `H = hashes_in[(r,c)]`

     * `P = grid_out[r-R:r+R+1, c-R:c+R+1]`

     * If `H not in codebook`:

       * record `codebook[H] = P.copy()`

     * Else:

       * if `P` differs from `codebook[H]` in any cell:

         * mark `valid_hash[H] = False`

4. After processing all examples:

   * Filter to `H` where `valid_hash[H]` is True.

5. Translate to S11 params:

   Your S11 builder likely expects something like:

   ```python
   {
       "example_type": "train",
       "example_index": ex_idx,
       "hash_to_pattern": {
           str(H): { "(dr,dc)": color, ... },
           ...
       }
   }
   ```

   Implementation pattern:

   * For each valid `H` in `codebook`:

     * Convert its patch `P` into a dict:

       ```python
       pattern_dict = {
           f"({dr},{dc})": int(P[dr+R, dc+R])
           for dr in range(-R, R+1)
           for dc in range(-R, R+1)
       }
       ```

   * Build `hash_to_pattern` map:

     ```python
     hash_to_pattern = {str(H): pattern_dict for H, P in codebook.items() if valid_hash[H]}
     ```

   * For each training example, emit a `SchemaInstance("S11", params)` as required by your builder, including `example_type` and `example_index`.

**Constraints:**

* H → P(H) is only accepted if **every** time that hash appears in train inputs, the output patch matches P(H) exactly.
* Any hash with conflicting patches is dropped.
* No fallback “most common patch”.

---

## Thin smoke test runner

### File: `src/law_mining/test_mine_s5_s6_s7_s11_smoke.py`

**Goal:** Basic plumbing check that each `mine_Sk` runs without errors and returns a list of `SchemaInstance` objects.

Contents:

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s5_s6_s7_s11 import (
    mine_S5,
    mine_S6,
    mine_S7,
    mine_S11,
)

def main():
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "0"  # or some simple task

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s5_instances = mine_S5(task_context, roles, role_stats)
    s6_instances = mine_S6(task_context, roles, role_stats)
    s7_instances = mine_S7(task_context, roles, role_stats)
    s11_instances = mine_S11(task_context, roles, role_stats)

    print("S5 instances:", s5_instances)
    print("S6 instances:", s6_instances)
    print("S7 instances:", s7_instances)
    print("S11 instances:", s11_instances)

if __name__ == "__main__":
    main()
```

---

## Reviewer + tester instructions

**For implementer:**

* For each `mine_Sk`, follow the algorithmic pattern:

  * Only mine laws that are **exactly consistent with all training input–output pairs**.
  * Reject any candidate mapping if **one** counterexample exists.
  * Never pick “most common” anything.

* Translate high-level mined invariants into the **exact parameter format** required by existing S5/S6/S7/S11 builders:

  * pay attention to:

    * `"example_type"`, `"example_index"`,
    * string-encoded coordinates like `"(r,c)"`,
    * patch dictionaries keyed by `"(dr,dc)"` etc.

* Use NumPy for array comparisons and slicing; do not reimplement core array logic.

**For reviewer/tester:**

1. **Static review:**

   * Check that:

     * there is no branch that introduces arbitrary defaults,
     * mining decisions are always based on “all train examples agree or we drop it”,
     * outputs are `List[SchemaInstance]`, not raw constraints.

2. **Smoke tests:**

   ```bash
   python -m src.law_mining.test_mine_s5_s6_s7_s11_smoke
   ```

   * Confirm:

     * script runs without exceptions,
     * returns plausible `SchemaInstance` lists (even if empty for many tasks).

3. **Optional end-to-end tests:**

   * For a task you **know** is:

     * icon stamping (S5),
     * crop (S6),
     * block summary (S7),
     * or local codebook-like (S11),
   * Run:

     * `compute_roles` → `compute_role_stats` → all four miners,
     * assemble a `TaskLawConfig` from their instances,
     * call `solve_arc_task_with_diagnostics(task_id, law_config, use_training_labels=True)`,
     * check if `status == "ok"` and train outputs are exactly matched.

# Clarifications
2. What the S6 miner should do (no guessing, explicit rule family)

We need a finite, explicit set of candidate selection rules and we must test them algorithmically. Here is a fully specified, TOE-clean algorithm that fits your spec:

Step S6.0 – Detect that each train output is a crop of its input

For each training example k:

Let X_k be input grid, Y_k be output grid.

If Y_k’s shape is larger in any dimension than X_k, S6 is impossible → return [].

Enumerate all possible (r0, c0) such that a window of size Y_k.shape fits inside X_k:

candidates_k = []
for r0 in range(H_in - H_out + 1):
    for c0 in range(W_in - W_out + 1):
        if X_k[r0:r0+H_out, c0:c0+W_out] == Y_k:
            candidates_k.append((r0, c0))


If candidates_k is empty for any example → no consistent crop rule → S6 miner returns [].

So far, this is just “all valid crops”. Now we need to find a selection rule that maps from φ(X_k) to the chosen (r0,c0).

Step S6.1 – Define and test explicit selection rules

We define a small, finite set of candidate rule families. For now, to stay precise and minimal, let’s pick these rule types:

Rule A: fixed offset
“Choose (r0,c0) equal across all training examples.”

Rule B: largest component of color c

Rule C: largest component (any color)

We can add more types later (e.g. “topmost component of color c”), but we do not guess; we only test these explicit rules.

Rule A – Fixed offset

Already implemented by your simplification:

Check if there exists a pair (r0*, c0*) such that for every training example:

(r0*, c0*) is in candidates_k.

If yes, that’s a valid S6 law:

“Crop at (r0*, c0*) for this task.”

This is a subcase of cropping and is allowed.

We can keep this Rule A as one candidate.

Rule B – Largest component of color c

For each color c that appears in any training input:

For each training example k:

Compute all components of color c: components_c^k = connected_components_by_color(X_k, color=c).

If components_c^k is empty:

This task cannot be “crop largest component of color c” → discard this candidate color c.

Find the largest component of color c in X_k, by number of pixels (size).
If there are ties, break ties deterministically, e.g.:

choose the one with smallest (r_min, c_min) bbox coordinate.

Let that component’s bounding box be (r_min, r_max, c_min, c_max).

Check that:

(r_min, c_min) is in candidates_k (i.e. cropping that bbox yields Y_k).

If, for every training example:

components_c^k is non-empty, and

the chosen largest-c component’s bbox matches one of the valid crop candidates,

then we have a consistent rule:

“Crop the bounding box of the largest component of color c.”

We accept this as an S6 law for this task.

Rule C – Largest component (any color)

Same as Rule B, but ignoring color:

For each example, consider all components (any color except 0 or with 0 allowed, but choose a consistent definition):

components_all^k = connected_components_by_color(X_k).

Pick the largest component by size, break ties by (r_min, c_min).

Check that its bbox matches one of the crop candidates.

If this holds for all training examples, we have:

“Crop the bounding box of the largest component.”

This is also a legitimate S6 law.

Step S6.2 – Pick rules, fail-close if nothing matches

If at least one of these explicit rules (A, B for some c, C) is satisfied across all training examples:

pick one (up to you whether to prefer A over B/C; choose a deterministic priority order and document it).

produce schema_params accordingly (e.g. { "mode": "largest_component_color", "color": c } or { "mode": "fixed_offset", "origin": "(r0*,c0*)" }).

If none of these rules works:

S6 miner returns [].

That means: either the task uses a more complex crop rule, or S6 is not the right schema; we do not “guess” a crop.

3. Answer to implementer’s questions

“What if input grids have different sizes?”

No problem:

Everything above is per-example.

H_in, W_in, H_out, W_out, candidates, components, etc. are computed per-example; the rule only cares that the same kind of rule (“largest component of color c”) works across all examples, not that (r0,c0) is numerically equal.

“What if there are multiple candidate components?”

We fix tie-breaking deterministically:

E.g., for “largest component”, if multiple have same size:

pick the one with smallest r_min,

if still tie, smallest c_min.

That’s fully explicit. No guess, no ML.

“Should we try multiple rule types?”

Yes, but only those we explicitly define (A, B, C above). That’s our finite rule set. We do not try arbitrary combinatorics; we just test these few simple templates, and fail-close otherwise.

So: S6 miner was underspecified and the current implementation is only Rule A. We need to add B and C (at least), or clearly document that we only support Rule A for now and treat others as unsolved.

S7 (Aggregation / Summary)
1. What implementer did

“Only the ‘unique non-zero color’ summary rule; returns [] if blocks have 0 or >1 non-zero colors.”

Given the WO said:

“Candidate summary rule: ‘unique non-zero color in block’ … you can extend this logic if builder supports more complex rules.”

This is exactly the “minimal rule” we suggested.

So the current behavior is:

Per task:

it checks if every block in every train example has:

either a single non-zero color and the output cell equals that color,

or no non-zero and the output cell is 0.

if yes → we mine S7 with that one rule.

if no → S7 miner returns [].

This is:

TOE-correct (no guessing, pure invariant),

just incomplete in terms of coverage (won’t solve tasks that use other summary functions, like counts, majority color, etc.).

Given your constraints:

No defaults,

No “dominant” heuristics,

I would say:

Implementing only “unique non-zero or zero” is acceptable for now.

We should not add “mode” or “most frequent” rules unless we specify them carefully and enforce “always true on train”.

So S7 is not a conceptual gap — it’s a conscious limitation of the current rule set. That’s fine.

We just need to be clear in the spec:

S7 miner currently implements only the “unique-non-zero-or-zero” summary rule.
All other possible S7-style rules are left for future extension; tasks not fitting this pattern will not get an S7 law and will remain unsolved (or will rely on other schemas).