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

This gives you a **structured, non-monolithic M6.3**:

* **M6.3A** – S1, S2, S10 (role→color, components, frame)
* **M6.3B** – S3, S4, S8, S9 (bands, residues, tiling, crosses)
* **M6.3C** – S5, S6, S7, S11 (templates, crop, summary, local codebook)

Each of these can then be expanded into Claude-ready, code-level WOs when you’re ready, without stubs and without blowing up file size.
