Good, this is the big one, so we need to be careful not to dump an unbuildable monster on your implementer.

Implementing **all 11 miners** in one WO would easily blow past 300 LOC, so I’m going to split **WO-M6.3** into **3 sequential sub-WOs**, each fully implementable (no stubs), grouped by similarity of schemas.

I’ll give you **high-level but precise WOs** for each group. Later, you can ask me to expand any one of them into a Claude-ready, line-by-line spec like we did for roles/role_stats.

No defaults, no best-effort; everything is “always-true on train or reject”.

---

## 🔹 WO-M6.3A – Miners for S1 (copy/tie) + S2 (component recolor) + S10 (frame/border)

**File:** `src/law_mining/mine_s1_s2_s10.py`

**Goal:** For a single task (`TaskContext`, `RolesMapping`, `RoleStats`), find all always-true instances of:

* S1 – direct pixel color ties / fixed colors per role,
* S2 – component-wise recolor maps,
* S10 – border vs interior recolor.

and output them as `SchemaInstance` objects with **coordinate-based** params matching existing builders.

### Common imports & types

At the top:

```python
from __future__ import annotations

from typing import Dict, List

from collections import defaultdict

from src.schemas.context import TaskContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance
from src.features.components import connected_components_by_color, compute_shape_signature
from src.features.object_roles import component_border_interior, component_role_bits
```

### 1. `mine_S1` – role→constant color (ties / fixes)

**Signature:**

```python
def mine_S1(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

**Logic (no wiggle room):**

1. For each `role_id`:

   * Look at `stats = role_stats[role_id]`.
   * Collect all colors seen in `stats.train_out`:

     * `colors_out = { color_out for (_,_,_,color_out) in stats.train_out }`.
   * If `colors_out` is empty:

     * Do **not** mine anything for this role (no evidence in outputs yet).
   * If `len(colors_out) > 1`:

     * **Contradiction** on this role → we do **not** mine a constant-color law for it.
   * If `len(colors_out) == 1`:

     * Let `c_out` = the single color.

2. For this role, we want to tie **all pixels with this role id** (train_in, train_out, test_in) to color `c_out`.

   But builders expect **coordinates**, not roles. So:

   * Extract all `(kind, ex_idx, r, c)` in `roles` where `roles[(kind, ex_idx, r, c)] == role_id`.
   * Build `schema_params` in the format your S1 builder expects:

     * For example, if S1 builder supports:

       * `"fixed_colors": [((kind, ex_idx, r, c), color), ...]`
       * or `"ties": [ { "pairs": [((r1,c1), (r2,c2)), ...] } ]`.
     * The miner must use the **actual S1 param format** (implementer must inspect existing `build_S1_constraints`).

3. Return a `SchemaInstance` like:

```python
SchemaInstance(
    family_id="S1",
    params={ ... schema_params ... }
)
```

* There may be:

  * one big S1 instance that includes all roles with constant colors,
  * or multiple S1 instances (e.g. per role or per color), depending on how the builder was designed. The important thing: **only add instances that are exactly consistent with training**.

### 2. `mine_S2` – component-wise recolor

**Signature:**

```python
def mine_S2(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

**Logic:**

1. For each training example:

   * Use `connected_components_by_color(input_grid)` on the **train input**.
   * For each component:

     * get:

       * `color_in`,
       * `size`,
       * `shape_signature`,
       * `pixels` list.

2. For each **component class** (e.g. defined by `(color_in, shape_signature)` or `(color_in, size)` as appropriate for your builder):

   * Look up the corresponding **output colors** for those pixels:

     * for each pixel `(r,c)` in that component:

       * output color = `train_output[r,c]`.
   * Check if **all pixels in this component class** across **all training examples** share the same output color `c_out`.

     * If any conflict → this class is **not** recolored by a simple S2 law.

3. For each class that is consistent, build a S2 mapping:

   * E.g. `mapping[(color_in, size)] = c_out`.

4. Convert this mapping into `schema_params` that your S2 builder expects, e.g.:

```python
{
    "color_in": ...,
    "size_to_color": {1: 3, 2: 2, "else": 1}
}
```

or whatever your `build_S2_constraints` expects.

5. Return a `SchemaInstance(family_id="S2", params=schema_params)` if **at least one mapping exists**.

**Constraints:**

* S2 miner must **never** introduce a mapping that is not exactly true on all train examples.
* If no class passes the consistency check, `mine_S2` returns an empty list.

### 3. `mine_S10` – border vs interior role

**Signature:**

```python
def mine_S10(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    ...
```

**Logic:**

1. For each training example:

   * Compute `component_border_interior(grid, components)` from `features.object_roles`.
   * For each component:

     * For each pixel `(r,c)`:

       * `is_border`, `is_interior` from that function.
       * Look up input color and output color at `(r,c)`.

2. For each combination:

   * `(component_class, role_type)`:

     * `component_class` could be `(color_in, shape_signature)` or `(object_id)` depending on your builder.
     * `role_type ∈ {"border", "interior"}`.

   * Collect all output colors across all train examples:

     * e.g. border color set, interior color set.

3. For each `(component_class, role_type)`:

   * If color set size is 0 → no evidence, skip.
   * If color set size > 1 → conflict, do **not** mine a law for this pair.
   * If size == 1:

     * record `border_color` or `interior_color` for that class.

4. Translate into `schema_params` expected by your S10 builder, e.g.:

```python
{
    "rules": [
        {
            "component_class": ...,
            "border_color": b,
            "interior_color": i,
        },
        ...
    ]
}
```

5. Return one or more `SchemaInstance(family_id="S10", params=...)`.

**Constraints:**

* All mined S10 rules must match **every** training example.
* No “most frequent border color”; only “always same border color”.

---

## 🔹 WO-M6.3B – Miners for S3 (bands) + S4 (residue) + S8 (tiling) + S9 (cross)

**File:** `src/law_mining/mine_s3_s4_s8_s9.py`

**Goal:** Mine all always-true band/stripe, periodic, tiling, and cross-propagation laws.

Each miner returns `List[SchemaInstance]` and must:

* use `role_stats` + φ,
* check consistency across all train examples,
* lower its findings to coordinate-based schema_params expected by S3/S4/S8/S9 builders.

At a high level:

### 1. `mine_S3` – band/stripe templates

* Use:

  * row/col bands (from φ),
  * role_stats to see if rows in same band class share color patterns in outputs.
* For each row-band class:

  * derive a color pattern (per column or per residue) from train outputs;
  * check consistency across all examples;
  * if consistent, add a schema instance tying these rows together.

### 2. `mine_S4` – residue-class coloring

* For candidate K in {2,3,4,5}:

  * for each residue `r = c(p) mod K` or `r(p) mod K`,
  * check if output color is constant across all training examples for pixels with that residue;
  * if yes for all residues used, create a schema instance with:

    * `"K": K, "residue_to_color": {...}`.

### 3. `mine_S8` – tiling / repetition

* Use φ to identify repeated tiles across train outputs (and/or inputs to outputs).
* For each candidate base tile:

  * verify that tiling pattern and stride is consistent across all training examples;
  * if yes, produce a schema_params with:

    * base tile coordinates,
    * stride `(h, w)`,
    * region to tile.

### 4. `mine_S9` – cross/plus propagation

* Use neighborhood hashes to detect plus centers in train_input / train_output.
* For each detected cross type:

  * infer which directions (up/down/left/right) are extended and with what colors;
  * check consistency across training examples;
  * encode parameters (seed positions / directions / lengths) for S9.

All of these must reject any pattern that doesn’t hold across *all* train examples.

---

## 🔹 WO-M6.3C – Miners for S5 (template stamping) + S6 (crop) + S7 (summary) + S11 (local codebook)

**File:** `src/law_mining/mine_s5_s6_s7_s11.py`

**Goal:** Mine more “structured” laws:

* S5 – seed→template stamping,
* S6 – crop to ROI,
* S7 – block/summary,
* S11 – local 3×3 codebook.

Each miner again returns `List[SchemaInstance]`.

### 1. `mine_S5` – template stamping

* For each potential seed type (e.g. rare color or specific neighborhood hash) in train inputs:

  * collect patches around that seed in train outputs (e.g. 3×3 or 5×5).
  * if for a given seed type, all patches are identical:

    * treat that as template `P_t`;
    * produce schema_params for S5 with:

      * seed condition,
      * template shape,
      * template colors.

### 2. `mine_S6` – crop to ROI

* Compare input vs output shapes for each train example.
* If outputs are strictly **subgrids** of inputs:

  * for each train example, identify the bounding box in input that corresponds to the output (by matching colors).
  * infer a rule like:

    * “crop the bbox of largest component of color X” or
    * “crop the bbox of the unique non-zero component”.
  * check that this rule (selection criterion) is consistent across all train examples.
  * encode S6 params accordingly:

    * e.g. `{ "selection": "largest_component_color", "color": 1 }`.

### 3. `mine_S7` – summary/histogram grids

* If output grids are much smaller (e.g. 2×2, 3×3) than inputs:

  * partition input into blocks (based on output shape),
  * for each block and train example:

    * compute candidate summary colors (e.g. unique non-zero color),
    * check if summary is consistent across train examples.
  * build S7 params describing:

    * block partitioning,
    * block→output cell mapping,
    * summary rule (e.g. “unique non-zero”).

### 4. `mine_S11` – local neighborhood codebook

* For each pixel in train inputs:

  * compute a neighborhood hash H (from φ).
  * look at train outputs around same location:

    * extract local pattern P(H).
* For each H:

  * if all patterns P(H) across all train examples are identical:

    * record `H -> P(H)`.
  * if any conflict: drop that H.
* Produce an S11 schema instance describing:

  * which neighborhood hashes to recognize,
  * which output patches to stamp.

Again, no “best guess”: only hashes with fully consistent P(H) across train are kept.

---

## Shared constraints across all miners

For **all** `mine_Sk`:

* **Signature:**

  ```python
  def mine_Sk(
      task_context: TaskContext,
      roles: RolesMapping,
      role_stats: Dict[int, RoleStats],
  ) -> List[SchemaInstance]:
      ...
  ```

* **Rule:**

  * A `SchemaInstance` is only emitted if its behavior is **exactly consistent** with all training IO pairs.
  * If a candidate mapping is contradicted by any training example, it is dropped.
  * No “fallback”, “most common”, or “default” behavior.

* **Coordinate translation:**

  * Miners think in roles, component classes, bands, hashes, etc.,
  * and then convert to the **coordinate/param format** your existing `build_Sk_constraints` expect.

* **No non-TOE defaults:**

  * They never fill in missing behavior “just because it seems right”.
  * Any pixel not constrained by any schema remains unconstrained; whether that yields a unique solution or not is determined later by the solver + diagnostics.

---

## Testing & review (global guidance)

For each miner module:

1. **Static review:**

   * Check that:

     * it only uses `TaskContext`, `roles`, `role_stats`, and φ primitives,
     * it does not guess or default colors/positions,
     * it rejects patterns on any contradiction.

2. **Smoke tests:**

   * For a few hand-picked ARC training tasks where you know the schema type:

     * Run `compute_roles` → `compute_role_stats` → `mine_Sk`.
     * Inspect the returned `SchemaInstance`s:

       * Do they match your intuitive law?
       * Do they produce correct train outputs when passed to the kernel?

3. **End-to-end check (later in M6.4/M6.5):**

   * For a given task_id:

     * run all miners,
     * assemble `TaskLawConfig`,
     * run `solve_arc_task_with_diagnostics(use_training_labels=True)`,
     * confirm `status == "ok"` for tasks you know are covered by S1–S11 patterns.

---

M6.3D – S1 tie miner
– Implement mine_S1 that only outputs tie constraints where training outputs force equality across roles/positions,
– uses role_stats + TaskContext,
– never fixes colors.
---

WO-M6.3E – Law Miner for S9 (Cross / Plus Propagation)

Goal: Implement a deterministic, always-true-only miner for S9 that, for each training task, discovers cross/plus propagation laws and emits SchemaInstance objects in the exact format expected by s9_cross_propagation.py:

{
  "example_type": "train",
  "example_index": 0,
  "seeds": [{
      "center": "(2,3)",             # string "r,c"
      "up_color": 5, "down_color": None,
      "left_color": 3, "right_color": 3,
      "max_up": 2, "max_down": 0,
      "max_left": 4, "max_right": 4
  }]
}


File: src/law_mining/mine_s9_cross.py

1. API
def mine_S9(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine cross/plus propagation schemas (S9) for this task.
    Return a list of SchemaInstance objects, one per (example_type, example_index)
    or per seed type, in the exact param shape expected by the S9 builder.

    Only emit instances that are provably always-true on all training examples.
    If no consistent cross law is found, return an empty list.
    """


TaskContext, Roles, RoleStats as in M6.1/M6.2.

No defaults, no “best-effort”; either schema is consistent across all train examples, or we don’t emit it.

2. Stage 1 – Candidate seed detection (plus centers) on train outputs

Idea: A “plus” is a characteristic 3×3 neighborhood shape in the output:

center pixel with some color c0,

same or related color on some subset of {up, down, left, right},

background or other color elsewhere.

High-level steps:

For each training example ex_idx, ex:

Let grid_out = ex.output_grid.

Compute 3×3 neighborhoods using neighborhood_hashes(grid_out, radius=1).

For each pixel (r,c) where a 3×3 neighborhood exists:

Extract the 3×3 patch around (r,c) as a small 3x3 array.

Check if it matches any simple plus mask template:

e.g. center non-zero,

at least two opposite neighbors share that color,

diagonal corners are background or different.

For each such match, record a candidate seed:

seeds_per_example[ex_idx].append((r, c))


Optionally, cluster seeds into “types” by their 3×3 pattern (using the hash or the raw 3×3 array), e.g.:

seed_type = neighborhood_hashes[(r, c)]
seeds_by_type[seed_type].append((ex_idx, r, c))


(This lets you mine separate laws per cross type if needed.)

3. Stage 2 – Direction/color/extent inference per seed type

For each seed_type (or just globally if you skip typing):

Initialize an accumulator:

direction_colors: Dict[str, set[int]] = {
    "up": set(), "down": set(), "left": set(), "right": set()
}
direction_lengths: Dict[str, set[int]] = {
    "up": set(), "down": set(), "left": set(), "right": set()
}


For each recorded seed (ex_idx, r, c) of this type:

Walk in each direction d ∈ {"up","down","left","right"} on grid_out:

Starting from (r,c), move one step at a time until:

you hit the grid boundary, or

you hit a pixel that is not “part of the arm” (condition depends on the spec — typically: color == center color, or matching some inferred arm color).

Record:

the color used along that direction (if any consistent non-background color exists),

the maximum extent (number of steps before stopping).

Append observed colors/lengths to direction_colors[d] and direction_lengths[d].

After scanning all seeds in all train examples:

For each direction d:

If direction_colors[d] is empty:

no clear propagation in this direction → treat as None (no propagation).

If len(direction_colors[d]) == 1:

there is a single, consistent color_d.

If len(direction_colors[d]) > 1:

inconsistent behavior → this seed type cannot be mined reliably; discard this seed type.

For lengths:

For all seeds where direction d had propagation, collect lengths in direction_lengths[d].

If lengths vary but in a consistent way, pick a well-defined rule, e.g. minimum observed length as max_d:

We must be careful here: only accept if variation is consistent with training behavior (e.g. if all arms stop exactly at same relative semantic boundary).

If lengths disagree in ways that cannot be reconciled (e.g. some arms are length 2, others 5 with no structural reason), discard this seed type for now.

This part will be fleshed out in the expanded WO, but the invariant is:

We only accept directions where both color and max extent are consistent across all seeds and training examples.

4. Stage 3 – Per-example seeds → SchemaInstance params

For each seed type that survives Stage 2:

For each training example ex_idx:

Collect seeds of this type in that example: seeds_per_example[ex_idx].

For each seed (r,c):

Build a per-seed param dict:

seed_param = {
    "center": f"({r},{c})",    # string, as builder expects
    "up_color":   inferred_up_color_or_None,
    "down_color": inferred_down_color_or_None,
    "left_color": inferred_left_color_or_None,
    "right_color": inferred_right_color_or_None,
    "max_up":     max_up_steps,
    "max_down":   max_down_steps,
    "max_left":   max_left_steps,
    "max_right":  max_right_steps,
}


Build the per-example schema_params:

params = {
    "example_type": "train",
    "example_index": ex_idx,
    "seeds": [seed_param, ...]
}


Wrap as:

instances.append(SchemaInstance(family_id="S9", params=params))


Before accepting any S9 instance, we must validate that, when passed to the S9 builder and solved together with other schemas, the resulting constraints reproduce the training outputs exactly (this is part of the global M6.4/M6.5 loop). If S9 instances produce mismatches, they must be discarded or refined; no silent keep.

5. Always-true requirement

At every stage, S9 miners must enforce:

If there is any inconsistency in direction colors or extents across training examples for a given seed type, do not emit that schema instance.

If seeds occur in some examples but not others, that’s OK if the absence is consistent with “no cross there”.

But if two different seeds of the same type behave differently in outputs (e.g. one extends up 3 steps, another up 5 without structural reason), that indicates:

either the schema fam is not the right model, or

the miner needs a more refined notion of seed type — until then, we don’t emit S9.
---

This gives you a **structured, non-monolithic M6.3**:

* **M6.3A** – S1, S2, S10 (role→color, components, frame)
* **M6.3B** – S3, S4, S8, S9 (bands, residues, tiling, crosses)
* **M6.3C** – S5, S6, S7, S11 (templates, crop, summary, local codebook)

Each of these can then be expanded into Claude-ready, code-level WOs when you’re ready, without stubs and without blowing up file size.
