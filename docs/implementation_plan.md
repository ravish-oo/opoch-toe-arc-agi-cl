## Overall sequence:

> **operators → law families → schema instances → builder functions → ConstraintBuilder → solver → Pi-agent loop**

---

## 1. High-level milestones (sequential)

We don’t implement all now, just to show how operators fit.

1. **Operator / feature library (φ)** ✅ COMPLETE  

   * Grid representation, basic utilities.
   * All feature extractors: components, bands, hashes, etc.

2. **Constraint representation & ConstraintBuilder** ✅ COMPLETE 

   * Data structures for linear constraints on y.
   * Primitive methods like `tie_pixel_colors`, `fix_pixel_color`, etc.

3. **Schema family builders (S1–S11)** ✅ COMPLETE 

   * For each S_k, a `build_Sk_constraints(...)` that uses operators + ConstraintBuilder.

4. **Solver integration**

   * Connect to a standard LP/ILP library (`pulp`, `ortools`, or `cvxpy`).
   * Encode constraints and solve for y.

5. **Task IO + test harness**

   * Load ARC tasks, run full pipeline on training pairs, compare outputs.

6. **Pi-agent orchestration layer**

   * LLM decides which schema families + params to use, calls the pipeline, interprets results.

---

## Milestone 1 – Operator / feature library (φ) ✅ COMPLETE

Goal: a clean Python module (or few small modules) that exposes a set of **feature functions** over grids, using standard libs, with minimal reinvented algorithms.

Refer to docs/WOs/M1/M1.md  for work orders and docs/WOs/M1/ to see detailed work orders 
---

## M2 – indexing + ConstraintBuilder + SchemaFamily metadata + dispatch skeleton ✅ COMPLETE

High-level Work Orders in docs/WOs/M2/M2.md 
for detailed WOs refer to docs/WOs/M2/

---

## M3 – Schema builders S1–S11 ✅ COMPLETE
* **M3** = actually make S1–S11 *do something* using M1+M2.
High level work orders in docs/WOs/M3/M3.md . refer to docs/WOs/M3/ for detailed work orders

---

## M4 – Solver + Decoding + Validation
M4 is basically:
> **“Take constraints → run an LP/ILP → get y → decode to Grid(s) → check if it makes sense.”**

High-level Work Orders:
### 🔹 WO-M4.1 – LP/ILP solver wrapper ✅ COMPLETE

**File:** `src/solver/lp_solver.py`
**Goal:** turn `ConstraintBuilder.constraints` into an optimization problem, solve for y, and return the y vector.

**Scope:**

* Choose a standard library:

  * Prefer `pulp` or `ortools.linear_solver` (whichever is easier in your environment).

* Define a function like:

  ```python
  def solve_constraints(
      builder: ConstraintBuilder,
      num_pixels: int,
      num_colors: int,
      objective: str = "min_sum"
  ) -> np.ndarray:
      """
      Create binary variables y[p_idx, c], add all LinearConstraints,
      add one-hot constraints if not already included, solve the LP/ILP,
      return a flat numpy array of shape (num_pixels, num_colors) or raise if infeasible.
      """
  ```

* Features to include:

  * All constraints in `builder.constraints` as equality constraints.
  * Variables y[p,c] ∈ {0,1}.
  * One-hot constraints per pixel (you may already add these in M2; ensure they’re included exactly once).
  * Simple objective:

    * e.g. minimize sum(y) or just `0` (dummy objective) — since TU guarantees a vertex, we just need a feasible solution.

* Constraint validation:

  * If solver reports infeasible or unbounded:

    * raise a clear exception or return a special status for the Pi-agent / runner to handle.

This addresses:

* “Constraint validation”: solver will detect inconsistencies.
* Prepares for decoding in the next WO.

---

### 🔹 WO-M4.2 – Solution decoding (y → Grid(s))

**File:** `src/solver/decoding.py`
**Goal:** given the solved y (one-hot over colors), reconstruct output grid(s).

**Scope:**

* Define decoding helpers:

  ```python
  def y_to_grid(
      y: np.ndarray, 
      H: int, 
      W: int, 
      C: int
  ) -> Grid:
      """
      y: shape (H*W, C) or flat with length H*W*C.
      For each pixel p, pick color c where y[p,c] ≈ 1.
      """
  ```

* If your B(T) and indexing span **multiple outputs** (e.g. all train+test grids in one system), define a small layout struct:

  ```python
  @dataclass
  class VarLayout:
      num_examples: int
      example_shapes: list[tuple[int,int]]  # [(H1,W1), (H2,W2), ...]
      C: int
      # methods to map (example_idx, r, c, color) <-> global y index
  ```

  And a decoder:

  ```python
  def y_to_grids(y: np.ndarray, layout: VarLayout) -> list[Grid]:
      # split y into per-example grids
  ```

* Use the same indexing conventions as `src/constraints/indexing.py`.

This directly addresses your note: “solution decoding (y → Grid) belongs in M4.”

---

### 🔹 WO-M4.3 – Integrate solver into kernel runner

**File:** `src/runners/kernel.py` (augment)
**Goal:** go from **task_id + law_config** all the way to **decoded output grids**.

**Scope:**

* Extend `solve_arc_task` (or add a new function) to:

  ```python
  def solve_arc_task(
      task_id: str,
      law_config: TaskLawConfig
  ) -> dict[str, list[Grid]]:
      """
      Returns something like:
      {
        "train_outputs_pred": [...],
        "test_outputs_pred": [...]
      }
      """
  ```

* Steps inside:

  1. Load raw task (using `arc_io`).
  2. Build `TaskContext` (using `context.build_task_context_from_raw`).
  3. Create a fresh `ConstraintBuilder`.
  4. Loop over `law_config.schema_instances`, call `apply_schema_instance(...)`.
  5. Compute layout info (N, C or VarLayout).
  6. Call `solve_constraints(...)` from `lp_solver`.
  7. Decode y → output grid(s) using `decoding.y_to_grid` / `y_to_grids`.
  8. Return those grids.

* For now, you can focus on:

  * Just the **test outputs** (predictions),
  * Optionally also **predicted train outputs** for internal validation.

---

### 🔹 WO-M4.4 – Training-set validation runner (sanity / debug)

**File:** `src/runners/validate_on_training.py`
**Goal:** provide a simple script to check if a given `TaskLawConfig` actually reproduces known training outputs.

**Scope:**

* For a single `task_id`:

  * Load `arc-agi_training_challenges.json` and `arc-agi_training_solutions.json`.
  * Build `TaskContext` and call `solve_arc_task(...)` to get predicted train outputs.
  * Compare predicted vs true train outputs grid-by-grid.
  * Print:

    * exact match / mismatch,
    * maybe mismatched cells for debugging.

* This is the place where, later, a Pi-agent would learn:

  * “My law_config for this task is wrong” → refine schema/params.

This is not Pi-agent logic yet, just a **human-facing checker**.

---

So M4, at a high level, is:

1. **lp_solver.py** – encode `ConstraintBuilder` into LP/ILP and solve.
2. **decoding.py** – map solved y back to grid(s).
3. **kernel.py** – wire solver + decoding into `solve_arc_task`.
4. **validate_on_training.py** – simple validation loop on training outputs.

Once these are in place, we’ll have the full math kernel **closed**: given a task and a law_config, the system can actually produce grids and see if they match training labels. That’s the platform we need before we bring in a Pi-agent to learn laws.

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
