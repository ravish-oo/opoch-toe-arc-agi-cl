## Overall sequence:

φ operators  →  WL/q role labeller  →  law miner  →  TaskLawConfig
        →  schema builders (S1–S11)  →  ConstraintBuilder  →  solver  →  diagnostics

Step-by-step in words:

1. **Operators (φ)**

   * From raw grids for a task (train inputs/outputs, test inputs), compute all features:

     * components, object_ids, bands, residues, neighborhood hashes, roles bits, etc.
   * This is your structural “perception” of the task.

2. **WL/q role labeller**

   * Run a WL-style refinement over all pixels in all grids (train_in, train_out, test_in), using φ and local neighborhoods.
   * Output: a `role_id` for each pixel = its structural “info-geometry role” in this task.

3. **Law miner (algorithmic, no LLM)**

   * Using:

     * `TaskContext` (grids + φ),
     * roles from WL,
   * For each schema family S1–S11:

     * find parameter settings that are **always true** on all training examples,
     * create `SchemaInstance(family_id="S_k", params=...)` for those.
   * Collect all schema instances into a `TaskLawConfig`.

4. **Schema builders (S1–S11)**

   * Given `TaskLawConfig` + `TaskContext`, call each `build_Sk_constraints(...)`.
   * Each builder uses φ (and roles, if needed) to emit **linear constraints** into the `ConstraintBuilder`.

5. **ConstraintBuilder → constraint system B(T)**

   * All schema builders write rows into `ConstraintBuilder.constraints`.
   * Add one-hot constraints per pixel.

6. **Solver (LP/ILP)**

   * Feed constraints into the ILP (`solve_constraints_for_grid`).
   * Solve for y (a one-hot assignment per pixel).
   * Decode y → predicted output grids (train and test).

7. **Diagnostics**

   * Build `SolveDiagnostics`:

     * status (“ok” / “infeasible” / “mismatch” / “error”),
     * solver_status,
     * num_constraints / num_variables,
     * schema_ids_used (+ counts),
     * example summaries & train mismatches where applicable.
   * For training tasks, check if predicted train outputs == true outputs:

     * if yes → law miner succeeded for this task;
     * if no → law miner is underpowered or task is ambiguous.

* no search over programs,
* no arbitrary defaults,
* just: φ → roles → mined always-true schemas → constraints → closure.

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

4. **Solver integration** ✅ COMPLETE 

   * Connect to a standard LP/ILP library (`pulp`, `ortools`, or `cvxpy`).
   * Encode constraints and solve for y.

5. **Task IO + test harness** ✅ COMPLETE 

   * Load ARC tasks, run full pipeline on training pairs, compare outputs.

6. **WL/q Law Miner**

   * M6 = build the algorithmic law miner that sits on top of M1–M5 and acts as source of TaskLawConfig, in a way that is fully TOE-consistent (no defaults, only always-true invariants).

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

## M4 – Solver + Decoding + Validation ✅ COMPLETE
M4 is basically:
> **“Take constraints → run an LP/ILP → get y → decode to Grid(s) → check if it makes sense.”**

High-level Work Orders in docs/WOs/M4/M4.md . detailed WOs in docs/WOs/M4/

---

## M5 – diagnostics, catalog building ✅ COMPLETE
M5 is where we make the **kernel “talk”** — fail-closed, with rich, structured intermediate info that allows us to to debug and refine laws.

High-level Work Orders in docs/WOs/M5/M5.md . Detailed WOs in docs/WOs/M5/

---
 ## M6 - WL/q Law Miner

High-level M6 Work Orders – WL/q Law Miner

M6 = build the **algorithmic law miner** that sits on top of M1–M5 and replaces the Pi-agent as the source of `TaskLawConfig`, in a way that is fully TOE-consistent (no defaults, only always-true invariants).

### 🔹 WO-M6.1 – Role labeller (WL/q) over TaskContext

**Goal:** assign a structural role id to each pixel in each grid of a task (train_in, train_out, test_in), using φ + local neighborhoods, so mining operates over roles instead of raw pixels.

**File:** `src/law_mining/roles.py`

**Scope (high-level):**

* Define a `RolesMapping` type, e.g.:

  ```python
  RolesMapping = Dict[tuple[str, int, int, int], int]
  # key: (kind, example_idx, r, c)
  # kind ∈ {"train_in", "train_out", "test_in"}
  # value: role_id (0..R-1)
  ```

* Implement:

  ```python
  def compute_roles(task_context: TaskContext) -> RolesMapping:
      """
      Use φ (coords, bands, components, hashes, etc.) + WL-style refinement
      to assign stable role_ids per pixel across all grids in this task.
      """
  ```

* Behavior:

  * Construct nodes = all pixels in:

    * each training input (`train_in`),
    * each training output (`train_out`),
    * each test input (`test_in`).
  * Initialize labels from a subset of φ + kind (in/out/test).
  * Run a fixed number of WL refinement iterations (e.g. 3–5) over 4-neighbor graph in each grid.
  * Map final labels to consecutive `role_id`s.

> No defaults, no heuristics about color — just structural label refinement.

---

### 🔹 WO-M6.2 – Role statistics aggregator

**Goal:** compress raw role assignments and train IO into role-level statistics for mining.

**File:** `src/law_mining/role_stats.py`

**Scope:**

* Define:

  ```python
  @dataclass
  class RoleStats:
      train_in: List[tuple[int,int,int,int]]   # (example_idx, r, c, color_in)
      train_out: List[tuple[int,int,int,int]]  # (example_idx, r, c, color_out)
      test_in: List[tuple[int,int,int,int]]    # (example_idx, r, c, color_in_test)
  ```

* Implement:

  ```python
  def compute_role_stats(
      task_context: TaskContext,
      roles: RolesMapping
  ) -> Dict[int, RoleStats]:
      """
      For each role_id, collect:
        - all its appearances in train_in, train_out, test_in,
        - with associated colors.
      """
  ```

This is the main input to schema miners: they work at the role level, but can still consult φ when needed.

---

### 🔹 WO-M6.3 – Per-schema miners: mine_S1 … mine_S11 → SchemaInstance list

**Goal:** for each schema family S1–S11, implement a miner that:

* reads `TaskContext`, `roles`, `role_stats`, φ,
* finds parameter sets that are **always true on training**,
* and returns a list of `SchemaInstance` objects.

**Files:** new module(s) under `src/law_mining/`:

* `mine_s1_copy.py`
* `mine_s2_component_recolor.py`
* …
* or group them logically (e.g. S1–S2, S3–S4, S5/S11, etc.).

**Scope (per miner, conceptually):**

* Common signature:

  ```python
  def mine_Sk(
      task_context: TaskContext,
      roles: RolesMapping,
      role_stats: Dict[int, RoleStats],
  ) -> List[SchemaInstance]:
      ...
  ```

* Constraints:

  * A mined `SchemaInstance` must correspond to a law that is **exactly true** on all train examples.
  * If a candidate mapping (e.g. role→color, size→color, stripe pattern, tile, etc.) has any contradictions across train examples, it is **rejected**.
  * No “fallback” or “best-effort” acceptance: either always-true, or not included.

* Example patterns:

  * **S1**: for each role_id, if all train_out entries for that role have **the same color c**, create a schema instance that ties that role (or those φ conditions) to color c.
  * **S2**: for each object class, if mapping from input-color/size → output-color is consistent across all train examples, instantiate that mapping.
  * **S6/S7**: use components and output shapes to infer a unique crop/summary mapping (consistency across trains).

> This is the core “law mining engine on top of φ” from the spec — fully deterministic

---

### 🔹 WO-M6.4 – Law miner orchestrator: `mine_law_config` per task

**Goal:** combine all schema miners into a single function that produces a full `TaskLawConfig` for one ARC task.

**File:** `src/law_mining/mine_law_config.py`

**Scope:**

* Implement:

  ```python
  def mine_law_config(task_context: TaskContext) -> TaskLawConfig:
      """
      High-level miner:
        - compute roles,
        - compute role_stats,
        - call mine_S1..mine_S11,
        - assemble all SchemaInstances into a TaskLawConfig.
      """
  ```

* Steps inside:

  * `roles = compute_roles(task_context)`
  * `role_stats = compute_role_stats(task_context, roles)`
  * `schema_instances = []`
  * For each S_k:

    * `schema_instances.extend(mine_Sk(task_context, roles, role_stats))`
  * Return `TaskLawConfig(schema_instances=schema_instances)`.

* No assumptions about “coverage”:

  * if the miner produces a law_config that doesn’t solve train outputs uniquely, this will show up when we feed it to the kernel and see mismatches / infeasibility.

> This function is the non-LLM “law miner” you originally imagined: pure φ + invariants + schemas → law_config.

---

### 🔹 WO-M6.5 – Training sweep with miner: end-to-end validation

**Goal:** run the WL/q miner across all training tasks, validate via kernel, and record which tasks are solved (and which are underdetermined/failed).

**File:** `src/runners/sweep_training_with_miner.py`

**Scope:**

* For each task_id in `arc-agi_training_challenges.json`:

  * Build `TaskContext` as usual.

  * Call `law_config = mine_law_config(task_context)`.

  * Call:

    ```python
    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        challenges_path=Path("data/arc-agi_training_challenges.json")
    )
    ```

  * If `diagnostics.status == "ok"`:

    * store this `law_config` in the Catalog (via `catalog.store`).

  * Else:

    * log:

      * task_id,
      * diagnostics (mismatches, solver_status, schemas used, etc.),
      * for later inspection / refinement.

* No Pi-agent here; this is pure miner + kernel + diagnostics.

> This is the **realization** of “once S1–S11 are implemented, the rest is a small law mining engine on top of φ + a tiny LP wrapper”.

---
