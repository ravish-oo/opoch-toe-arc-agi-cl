## Overall sequence:

> **operators → law families → schema instances → builder functions → ConstraintBuilder → solver → Pi-agent loop**

---

## 1. High-level milestones (sequential)

We don’t implement all now, just to show how operators fit.

1. **Operator / feature library (φ)** ✅ COMPLETE  

   * Grid representation, basic utilities.
   * All feature extractors: components, bands, hashes, etc.

2. **Constraint representation & ConstraintBuilder**

   * Data structures for linear constraints on y.
   * Primitive methods like `tie_pixel_colors`, `fix_pixel_color`, etc.

3. **Schema family builders (S1–S11)**

   * For each S_k, a `build_Sk_constraints(...)` that uses operators + ConstraintBuilder.

4. **Solver integration**

   * Connect to a standard LP/ILP library (`pulp`, `ortools`, or `cvxpy`).
   * Encode constraints and solve for y.

5. **Task IO + test harness**

   * Load ARC tasks, run full pipeline on training pairs, compare outputs.

6. **Pi-agent orchestration layer**

   * LLM decides which schema families + params to use, calls the pipeline, interprets results.

For now we only care about **Milestone 1**.

---

## 2. Milestone 1 – Operator / feature library (φ) ✅ COMPLETE

Goal: a clean Python module (or few small modules) that exposes a set of **feature functions** over grids, using standard libs, with minimal reinvented algorithms.

Refer to docs/WOs/M1/M1.md  for work orders and docs/WOs/M1/ to see detailed work orders 
---

## M2 – indexing + ConstraintBuilder + SchemaFamily metadata + dispatch skeleton ✅ COMPLETE

High-level Work Orders in docs/WOs/M2/M2.md 
for detailed WOs refer to docs/WOs/M2/

---

## M3 – Schema builders S1–S11
* **M3** = actually make S1–S11 *do something* using M1+M2.

### **WO-M3.0 – Define a TaskContext struct** ✅ COMPLETE

**File:** `src/schemas/context.py`
**Goal:** one coherent object we pass into every `build_Sk_constraints`, so they all see the same data.

**Scope:**

* Define a `TaskContext` dataclass with fields like:

  ```python
  @dataclass
  class TaskContext:
      grids: dict[str, Any]     # e.g. {"train_inputs": [...], "train_outputs": [...], "test_inputs": [...]}
      H: int
      W: int
      C: int
      components: list[Component]
      object_ids: dict[Pixel, int]
      sectors: dict[Pixel, dict]         # from object_roles
      border_info: dict[Pixel, dict]
      role_bits: dict[int, dict]
      row_bands: dict[int, str]
      col_bands: dict[int, str]
      row_flags: dict[int, bool]
      col_flags: dict[int, bool]
      neighborhood_hashes: dict[Pixel, int]
      # you can add fields incrementally as needed by S_k
  ```

* No logic, just an organized container built from M1 functions.

Later, the runner will precompute this once per task and pass it into each builder.

---

### **WO-M3.1 – Implement S1 + S2 builders (copy/equality + component recolor)** 

**Files:**

* `src/schemas/s1_copy_tie.py`
* `src/schemas/s2_component_recolor.py`

**Goal:** get the two most basic schemas working end-to-end.

**Scope:**

* `build_S1_constraints(context: TaskContext, params: dict, builder: ConstraintBuilder)`

  * Use features (e.g. coord/object_id) to find pixel pairs that should be tied.
  * Call `builder.tie_pixel_colors(...)` using indexing helpers.

* `build_S2_constraints(context: TaskContext, params: dict, builder: ConstraintBuilder)`

  * Use `components` + `role_bits` etc. to find which components to recolor and how.
  * For each pixel in those components, call `builder.fix_pixel_color(...)`.

**Also in this WO:**

* Update `src/schemas/dispatch.py`:

  * Import these builder functions.
  * Replace stubs for S1, S2 in `BUILDERS`.
* Update `SchemaFamily.builder_name` (in `families.py`) if needed to match the real function names.

---

### **WO-M3.2 – Implement S3 + S4 (bands/stripes + periodic residue coloring)**

**Files:**

* `src/schemas/s3_bands.py`
* `src/schemas/s4_residue_color.py`

**Goal:** handle row/col band rules & modulo-based coloring.

**Scope:**

* `build_S3_constraints(...)`:

  * Use `row_bands`, `col_bands`, row/col nonzero flags, etc.
  * Tie rows/cols with same band features using S1-style ties.
  * Possibly add constraints for periodic patterns (tie (r,j) with (r,j+K)).

* `build_S4_constraints(...)`:

  * Use coord features (`row_mod`, `col_mod`) to map residue → color.
  * For each pixel, forbid all colors ≠ `h(residue)` with `builder.forbid_pixel_color(...)`.

Wire S3, S4 into `dispatch.BUILDERS`.

---

### **WO-M3.3 – Implement S5 + S11 (template stamping & local codebook)**

**Files:**

* `src/schemas/s5_template_stamping.py`
* `src/schemas/s11_local_codebook.py`

**Goal:** handle “seed → template” and local-neighborhood codebook rules.

**Scope:**

* `build_S5_constraints(...)`:

  * Use `neighborhood_hashes` + other features to identify **seed types**.
  * From train outputs, derive a patch `P_t` per seed type.
  * For each seed in test, stamp `P_t` by fixing colors in the appropriate offsets.

* `build_S11_constraints(...)`:

  * More general local codebook:

    * For each 3×3 input hash H, learn corresponding 3×3 output pattern P(H) from train pairs.
    * For all pixels with hash H, enforce P(H) via S5-style stamping.

Wire S5, S11 into `dispatch.BUILDERS`.

---

### **WO-M3.4 – Implement S6 + S7 (cropping & summary grids)**

**Files:**

* `src/schemas/s6_crop_roi.py`
* `src/schemas/s7_aggregation.py`

**Goal:** support crop-to-ROI and block-summary style tasks.

**Scope:**

* `build_S6_constraints(...)`:

  * Use components/object roles to infer which bounding box to keep.
  * Tie output pixels to that subregion of input via S1-like ties, others fixed to background.

* `build_S7_constraints(...)`:

  * Partition input into macro-cells (e.g. using simple blocks or bands).
  * For each macro-cell, deduce a “summary color” from train outputs.
  * For each summary cell, fix its color to that summary using `fix_pixel_color(...)`.

Wire S6, S7 into `dispatch.BUILDERS`.

---

### **WO-M3.5 – Implement S8 + S9 + S10 (tiling, cross propagation, frame)**

**Files:**

* `src/schemas/s8_tiling.py`
* `src/schemas/s9_cross_propagation.py`
* `src/schemas/s10_frame_border.py`

**Goal:** finish the remaining schema types for repetitive/tiled patterns, crosses, and frame vs interior.

**Scope:**

* `build_S8_constraints(...)`:

  * Infer base tile and tiling region/stride from train outputs.
  * Tie all tile positions to the base tile’s pattern (via S1-style equality constraints).

* `build_S9_constraints(...)`:

  * Identify cross centers by neighborhood hashes.
  * For each seed, propagate along rows/cols until stopping condition, fixing colors accordingly.

* `build_S10_constraints(...)`:

  * Use `component_border_interior` features.
  * For border pixels, forbid all colors except border color; same for interior.

Wire S8, S9, S10 into `dispatch.BUILDERS`.

---

### **WO-M3.6 – Sanity test harness for schemas**

**File:** `src/runners/test_schemas_smoke.py`

**Goal:** minimal script to sanity check that each `build_Sk_constraints`:

* runs without crashing,
* adds some constraints,
* can be fed to a dummy LP solver (or a mock) for small toy grids.

**Scope:**

* Hard-code 1–2 tiny toy grids for each S_k.
* Build a minimal `TaskContext` using M1 operators.
* Call each `build_Sk_constraints`.
* Print number of constraints and maybe inspect a few.

(No need to solve real ARC tasks yet; this is just a smoke test.)

---
### How this ties together

After M1 we have:

* A **grid IO + indexing module**.
* A **feature module** that can compute:

  * coordinates and bands,
  * border flags,
  * connected components + shape signatures,
  * object_id per pixel,
  * row/col nonzero flags,
  * neighborhood hashes.

This is φ(p) in code.

* **M1** = φ + IO 
* **M2** = indexing + ConstraintBuilder + SchemaFamily metadata + dispatch skeleton

From there, we can:

* Define the **SchemaFamily registry** and **builder functions** (next milestones).
* Then hook ConstraintBuilder + solver.
* Finally, bring in the Pi-agent that uses all this.
