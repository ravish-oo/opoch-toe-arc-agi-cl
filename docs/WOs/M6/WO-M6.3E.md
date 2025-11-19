## 🔹 WO-M6.3E – Law Miner for S9 (Cross / Plus Propagation)

### File: `src/law_mining/mine_s9_cross.py`

**Goal:** Implement `mine_S9` that:

* uses only `TaskContext`, `RolesMapping`, `RoleStats` and existing φ operators,
* finds **plus-shaped seeds** in **train outputs**,
* infers consistent per-direction color + arm lengths across all seeds/examples,
* emits `SchemaInstance` objects in the exact format your S9 builder expects:

```python
{
    "example_type": "train",
    "example_index": 0,
    "seeds": [{
        "center": "(2,3)",
        "up_color": 5,   "down_color": None,
        "left_color": 3, "right_color": 3,
        "max_up": 2, "max_down": 0,
        "max_left": 4, "max_right": 4
    }]
}
```

* If the pattern is not **exactly consistent on all train examples**, `mine_S9` must return `[]`.
* No fallbacks, no “closest pattern”.

---

### 1. Imports

At top of `mine_s9_cross.py`:

```python
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance
from src.features.neighborhoods import neighborhood_hashes
```

---

### 2. API

```python
def mine_S9(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine cross/plus propagation schemas (S9) for this task.

    Returns:
        A list of SchemaInstance objects, each with params:
            {
              "example_type": "train",
              "example_index": int,
              "seeds": [ { ... }, ... ]
            }

    Only emits instances when the inferred cross law is exactly consistent
    across all training examples. If no such law exists, returns [].
    """
```

---

### 3. Stage 1 – Detect plus centers in train outputs (strict 3×3 mask)

We define a **canonical plus pattern** in 3×3:

* center non-zero color `c0`,
* up/down/left/right neighbors all equal to `c0`,
* diagonals are **not** `c0` (may be 0 or other colors).

Implementation sketch:

1. For each training example `ex_idx, ex`:

   ```python
   grid_out = ex.output_grid  # shape H,W
   H, W = grid_out.shape
   seeds_per_example: Dict[int, List[Tuple[int,int]]] = defaultdict(list)

   for r in range(1, H-1):
       for c in range(1, W-1):
           center = int(grid_out[r, c])
           if center == 0:
               continue
           up    = int(grid_out[r-1, c])
           down  = int(grid_out[r+1, c])
           left  = int(grid_out[r, c-1])
           right = int(grid_out[r, c+1])
           d1    = int(grid_out[r-1, c-1])
           d2    = int(grid_out[r-1, c+1])
           d3    = int(grid_out[r+1, c-1])
           d4    = int(grid_out[r+1, c+1])

           if up == down == left == right == center and all(d != center for d in [d1,d2,d3,d4]):
               seeds_per_example[ex_idx].append((r, c))
   ```

2. If **no seeds found in any training example** → no S9 for this task → `return []`.

We **do not** use neighborhood hashes for center detection here; we use explicit 3×3 checks to keep the pattern unambiguous.

---

### 4. Stage 2 – Infer direction colors & extents (strict equality)

We restrict to a very clear rule:

> For each seed, in each direction:
>
> * if any arm exists, it is a contiguous run of cells with a **single color** `c_d`,
> * all arms of that direction in all training examples share that same `c_d` and the same length `L_d`,
> * if no seed has an arm in that direction, we treat it as “no propagation” (`color=None`, `max_* = 0`).

Algorithm:

1. Initialize accumulators:

```python
direction_colors: Dict[str, set[int]] = {
    "up": set(), "down": set(), "left": set(), "right": set()
}
direction_lengths: Dict[str, set[int]] = {
    "up": set(), "down": set(), "left": set(), "right": set()
}
```

2. For each train example `ex_idx` and seed `(r,c)` in `seeds_per_example[ex_idx]`:

For each direction `d ∈ {"up","down","left","right"}`:

* Walk from `(r,c)`:

  ```python
  dr, dc for each direction
  length = 0
  colors_seen = set()

  rr, cc = r + dr, c + dc
  while 0 <= rr < H and 0 <= cc < W:
      col = int(grid_out[rr, cc])
      if col == 0:
          break
      colors_seen.add(col)
      length += 1
      rr += dr
      cc += dc
  ```

* If `length == 0`:

  * no arm for this seed in this direction → nothing to add.

* If `length > 0`:

  * if `len(colors_seen) > 1`:

    * this seed has inconsistent colors along arm → **S9 invalid for this task** → `return []`.
  * else:

    * let `col_d = colors_seen.pop()`,
    * add `col_d` to `direction_colors[d]`,
    * add `length` to `direction_lengths[d]`.

3. After processing all seeds:

For each direction `d`:

* If `direction_colors[d]` is empty:

  * no arms in this direction anywhere → we define `color_d = None`, `max_d = 0` (no propagation).
* If `len(direction_colors[d]) == 1`:

  * let `color_d = that one color`.
* If `len(direction_colors[d]) > 1`:

  * arms in direction d use different colors across seeds/examples → **S9 invalid for this task** → `return []`.

For lengths:

* For each direction `d` where some arms exist (i.e. `direction_lengths[d]` non-empty):

  * If `len(direction_lengths[d]) == 1`:

    * `max_d = that single length`.
  * If `len(direction_lengths[d]) > 1`:

    * inconsistent extents → **S9 invalid** → `return []`.

This is very strict: S9 only applies to tasks where plus arms in each direction have a single color and length across all seeds/examples.

If that’s too restrictive for some ARC tasks, they simply won’t get S9; that’s fine — better no law than a guessed one.

---

### 5. Stage 3 – Build SchemaInstances per training example

Using the inferred `up_color`, `down_color`, `left_color`, `right_color` and `max_*`:

1. For each `ex_idx` in training:

* Get the list of seeds: `seeds_per_example[ex_idx]`.

* If empty → skip that example; no S9 constraints.

* For each `(r,c)` in `seeds_per_example[ex_idx]`:

  ```python
  seed_param = {
      "center": f"({r},{c})",
      "up_color":   up_color,     # or None
      "down_color": down_color,
      "left_color": left_color,
      "right_color": right_color,
      "max_up":     max_up,
      "max_down":   max_down,
      "max_left":   max_left,
      "max_right":  max_right,
  }
  ```

* Accumulate per-example seeds:

  ```python
  seeds_list_for_example = [seed_param, ...]
  ```

* Build params:

  ```python
  params = {
      "example_type": "train",
      "example_index": ex_idx,
      "seeds": seeds_list_for_example,
  }
  instances.append(SchemaInstance(family_id="S9", params=params))
  ```

2. Return `instances`. If no training example had seeds, we would already have returned `[]` earlier.

**Test-time behavior:**
When the system applies S9 to test inputs, it will:

* detect seeds by the same plus-center rule (3×3),
* apply the same `up_color`/`max_*` parameters,
* if a test input has no seeds → no S9 constraints for that example (and that’s fine; no default painting).

We do **not** add any code in the miner or builder that paints crosses where the pattern is not present.

---

### 6. Thin smoke test runner

**File:** `src/law_mining/test_mine_s9_cross_smoke.py`

```python
from pathlib import Path

from src.core.arc_io import load_arc_task
from src.schemas.context import build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s9_cross import mine_S9

def main():
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "0"  # replace with a known plus/cross task when you have one

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s9_instances = mine_S9(task_context, roles, role_stats)
    print("S9 instances:", s9_instances)

if __name__ == "__main__":
    main()
```

---
# Clarification 1
This is a good review – the reviewer is not confused, they’re noticing real gaps in the WO. I’ll answer per issue, in TOE mode, and be explicit about what’s spec-faithful and what needs to change.

---

## Issue 1 – Arm walking logic

> *“Should arms extend until 0/border only, or stop at any color change?”*

**Spec-faithful answer:**

For the v1 S9 miner, we should implement the **simplest, exact invariant**:

* Cross arms are contiguous runs of **the same color as the center**, and nothing else.

So the correct loop is:

```python
center = grid_out[r, c]
rr, cc = r + dr, c + dc
length = 0
while 0 <= rr < H and 0 <= cc < W:
    col = int(grid_out[rr, cc])
    if col != center:      # stop on 0 OR any different color
        break
    length += 1
    rr += dr
    cc += dc
```

And:

* `colors_seen` is not needed; we hard-require “arm color == center color”.
* If any seed ever has an arm pixel with different color, we reject S9 for this task (no law).

So:

* Arms do **not** extend over other colors; they stop as soon as color changes or at border.
* That’s strict, simple, and matches a “pure plus of same color”.

---

## Issue 2 – Test-time seed detection

> *“We detect seeds in TRAIN OUTPUTS. How do we know where seeds are in TEST INPUTS?”*

You’re right: seeds must be anchored in **inputs**, not in outputs.

**Spec-faithful pattern:**

* On training examples:

  * Detect **candidate seeds** in **inputs** (typically by color or 3×3 input neighborhood), then
  * Use training **outputs** only to **verify** that around those seed positions, we see the plus pattern and infer arms.

* From that, we derive a **seed condition**, e.g.:

  * “input color == seed_color c_seed”, or
  * “input neighborhood hash == H_seed”.

* For test examples:

  * Apply that same seed condition on the **test inputs**:

    * find all pixels in test input where this condition holds,
    * those are the `center` positions used in `"example_type": "test"` instances.

So the WO is incomplete as written: Stage 1 must be:

1. Find seed positions in **train inputs** such that their corresponding outputs have the plus pattern.
2. Encode the seed condition (e.g. specific input color) in code.
3. Use that condition to find seeds in **both** train and test inputs, and emit S9 instances for `"train"` and `"test"`.

You cannot detect seeds by output shape on test, because test outputs are unknown. That’s a real gap in the WO and needs to be corrected in the spec and code.

---

## Issue 3 – Seed detection specificity

> *“Single-pixel seeds, special colors, other 3×3 patterns – current detection might miss these. Is this acceptable or too restrictive?”*

In TOE mode:

* It is acceptable to be **conservative** in v1:

  * Only mine S9 laws for tasks where the plus in outputs is a strict 5-pixel plus with arms same color as center, and seeds are clearly identifiable by an input feature (e.g. a unique color).
  * For any tasks that use more exotic cross patterns or different seed logic, S9 miner returns `[]`. Those tasks remain unsolved by S9 and must be covered by other schemas or remain unsolved.

What is **not** acceptable is to “compensate” for this with heuristics or defaults. We must not:

* guess seeds,
* invent arms,
* or paint crosses where the pattern is not forced.

So, v1 S9 miner can be:

* “handle only classic same-color 5-pixel crosses with input seeds identifiable by a specific color/φ pattern”,

as long as:

* we clearly document that,
* and we never try to approximate other variants.

---

### Short directive you can give to reviewer/implementer for S9

1. **Arms stop at first color ≠ center** (including 0); arms must be contiguous center-color pixels only.
2. **Seeds must be detected on train inputs**, using a condition that can also be applied to test inputs; outputs are only used to verify plus shape and infer arms.
3. **Emit S9 only for simple, exact plus patterns; if any seed/example violates the pattern (different colors or lengths), `mine_S9` must return `[]` – no patching, no approximation.

# Clarification 2
This is a good set of questions from the reviewer, and they’re not nitpicks – they’re exactly the unresolved spots in the S9 WO.

You asked: **am I 100% sure how to address them, based on the spec + clarifications, in TOE mode?**
Yes, I can pin down a v1 S9 miner that is:

* deterministic,
* strictly “always-true on training or reject,”
* and explicit enough to implement without guessing.

Let me answer each blocker concretely.

---

## BLOCKER 1: Seed detection in inputs

> *How to identify seed positions in input grids? By color? By hash? By trying all positions?*

Spec + TOE-faithful v1 answer:

We define **seeds by input color**, anchored by evidence from training outputs.

Algorithm:

1. For each training example `k`:

   * Scan **all pixels** `(r,c)` in input `X_k`:

     * For each `(r,c)`, check if the corresponding output `Y_k` has a plus pattern centered at `(r,c)` (using the 3×3 shape check we discussed earlier, or a similar local criterion).
     * Record pairs `(input_color, (k, r, c))` where:

       * input color at `(r,c)` is `c_seed`,
       * output around `(r,c)` looks like a valid plus.

2. We keep only seed colors `c_seed` that satisfy:

   * In every training example:

     * **every** plus in the output is centered at an input pixel with `color == c_seed`, and
     * **no** input pixel with `color == c_seed` has a non-plus output pattern.

3. If there isn’t exactly one such `c_seed`, we reject S9 (return `[]`).

Then:

* The **seed predicate** is simply “input color == c_seed”.
* On test inputs, we use this same predicate:

  * every pixel with input color `c_seed` is treated as a seed center.

This is fully deterministic and grounded: we derive a seed color from training IO, then use that condition symmetrically on test inputs. No hashing needed for v1, though hashes could be added later.

---

## BLOCKER 2: Same-color vs multi-color arms

> *Must all arms be same color (simple plus)? Or can different directions have different colors?*

Spec-faithful v1 answer:

* **Arms can have different colors per direction** (`up_color`, `down_color`, etc.), as your builder supports.
* The only hard requirement is:

  For each direction `d ∈ {up,down,left,right}`:

  * For every seed in every training example:

    * either:

      * there is **no arm** in that direction (length 0), or
      * all cells in that direction along the arm have the **same color** `c_d`.
  * Across all seeds and training examples where an arm in direction `d` exists:

    * the color `c_d` must be the same.

So:

* Directions can have **different colors** (e.g. up=5, left=3),
* But within each direction, across all seeds and examples, there must be **exactly one** color.

This matches the builder’s `up_color`, `down_color`, etc. semantics and your original spec (“spokes get painted with specific label colors”).

We **do not** require arm colors to equal the seed color; center color is independent.

---

## BLOCKER 3: Arm length consistency

> *Must all seeds have identical arm lengths in all directions? Or just consistent per-direction across seeds?*

Spec-faithful v1 answer:

* We require **per-direction consistency** across all seeds and all training examples:

  For each direction `d`:

  * For every seed where the arm in `d` exists:

    * the arm length must be some integer `L_d`.
  * Across all seeds/examples, all observed lengths in direction `d` must be **exactly equal** to the same `L_d`.

* If:

  * any seed in direction `d` has a different length, or
  * arm detection yields contradictory lengths,

  then S9 is rejected for this task (return `[]`).

So in v1:

* arms are **constant-length** per direction per task,
* it’s not enough that lengths are “roughly similar”; they must be identical.

This is conservative but clean: we only support tasks where cross arms have uniform length per direction.

---

## BLOCKER 4: Center color handling

> *Planner uses center = grid_out[r, c] as arm color; builder says "center pixel is NOT colored by this schema". How to reconcile?*

Spec-faithful reconciliation:

* In S9, **center color is irrelevant to the law**. The schema’s job is to paint the arms, not the center.

* So:

  * We do **not** use the center color at all in computing `up_color`, `down_color`, etc.
  * Only the arm cells matter:

    * For each direction d, we detect arm cells,
    * infer `color_d` from those cells,
    * and ignore what color the center has.

* The fact that your builder “does not color center” is perfectly aligned:

  * we **never** emit any constraint on the center in S9,
  * we only emit constraints along the rays.

Therefore:

* The correct v1 miner:

  * does not rely on `center == arm color`,
  * only treats the center as a positional anchor.

---

## Summary: What changes in the S9 spec

To align the miner with the math spec and builder:

1. **Seed detection:**

   * Detect plus centers by looking at train outputs **at locations that have a specific input color c_seed**.
   * Derive c_seed by requiring that:

     * in every training example, all plus centers correspond to input color c_seed,
     * no other input c_seed positions have non-plus outputs.
   * On test, seeds are “input color == c_seed”.

2. **Arm colors:**

   * Per direction (`up`, `down`, `left`, `right`), allow a unique color per direction, but:

     * all arms in that direction must use the **same** color across all seeds/examples.

3. **Arm lengths:**

   * Each direction has a single length `L_d` (or 0 if no arms),
   * that length is the same for all seeds and all training examples.

4. **Center:**

   * We never constrain center color in S9,
   * we only constrain arms.

If any of these conditions fails on training examples, `mine_S9` returns `[]`.
For tasks that don’t fit this very clean pattern, we simply don’t mine S9 in v1.

That’s deterministic, spec-aligned, and implementable without any fallback.
