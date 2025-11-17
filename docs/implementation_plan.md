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

4. **Solver integration** ✅ COMPLETE 

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

## M4 – Solver + Decoding + Validation ✅ COMPLETE
M4 is basically:
> **“Take constraints → run an LP/ILP → get y → decode to Grid(s) → check if it makes sense.”**

High-level Work Orders in docs/WOs/M4/M4.md . detailed WOs in docs/WOs/M4/

---

## M5 –  Pi-agent interface, diagnostics, catalog building
M5 is where we make the **kernel “talk”** — fail-closed, with rich, structured intermediate info that a Pi-agent can actually use to debug and refine laws.

We’re done with the math engine; now we’re building the **interface and diagnostics layer** that a Pi-agent will sit on.

High-level Work Orders
### 🔹 WO-M5.1 – Result & diagnostics struct

**Goal:** define a single, structured object that captures *everything* about a solve attempt, especially in failure.

**File:** `src/runners/results.py`

**Scope (high-level):**

* Define something like:

  ```python
  @dataclass
  class SolveDiagnostics:
      task_id: str
      law_config: TaskLawConfig
      status: Literal["ok", "infeasible", "mismatch", "error"]
      solver_status: str                 # from pulp
      num_constraints: int
      num_variables: int
      schema_ids_used: list[str]
      # optional: per-schema constraint counts

      # Only for training tasks:
      train_mismatches: list[dict]       # e.g. [{ "example_idx": 0, "diff_cells": [...] }, ...]

      # Debug:
      error_message: str | None
  ```

* This struct is what the Pi-agent will see:

  * if `status != "ok"`, it gets **explicit reasons**: infeasible, mismatch, where mismatched, etc.

---

### 🔹 WO-M5.2 – Extend kernel to return diagnostics, not just grids

**Goal:** make `solve_arc_task` produce **diagnostics + outputs** in a way that is fail-closed and Pi-agent-friendly.

**File:** `src/runners/kernel.py` (augment)

**Scope:**

* Change / wrap `solve_arc_task(task_id, law_config)` to something like:

  ```python
  def solve_arc_task_with_diagnostics(
      task_id: str,
      law_config: TaskLawConfig,
      use_training_labels: bool = False
  ) -> tuple[dict[str, list[Grid]], SolveDiagnostics]:
      """
      Returns:
        outputs: {"train": [...], "test": [...]}
        diagnostics: SolveDiagnostics
      """
  ```

* Behavior:

  * Always:

    * build TaskContext,
    * build constraints via schemas,
    * call solver,
    * decode y → grids,
    * fill `SolveDiagnostics` with:

      * status = "ok" / "infeasible" / "error".
  * If `use_training_labels=True`:

    * compare predicted vs true train outputs,
    * set status = "mismatch" if any differ,
    * populate `train_mismatches` with per-example diff info.

This is exactly the “fail-close + intermediate info” you mentioned.

---

### 🔹 WO-M5.3 – Training sweep + catalog builder script

**Goal:** one script that a Pi-agent / human can drive to **try law configs on all training tasks and build/update the Catalog**.

**File:** `src/runners/build_catalog_from_training.py`

**Scope:**

* For each `task_id` in `arc-agi_training_challenges.json`:

  * Load an existing `TaskLawConfig` (if any) from `catalog/store.py` **or** receive one from outside (Pi-agent).
  * Call `solve_arc_task_with_diagnostics(task_id, law_config, use_training_labels=True)`.
  * If `status == "ok"`:

    * mark this config as **valid** for that task,
    * write/update it in the catalog store.
  * If `status == "mismatch"` or `"infeasible"`:

    * log diagnostics to a JSON or log file for that task:

      * mismatches, solver_status, schemas used.
* This script does **no law discovery**; it just:

  * runs the kernel,
  * records successes,
  * outputs failures in a Pi-agent-readable format.

This is the main “sweep + record” entrypoint.

---

### 🔹 WO-M5.4 – Pi-agent harness / interface

**Goal:** a thin Python interface that exposes everything a Pi-agent needs via simple function calls, without forcing it to touch internals.

**File:** `src/agents/pi_interface.py`

**Scope:**

* Define functions like:

  ```python
  def load_task_summary(task_id: str) -> dict:
      """
      Returns:
        - basic info about the task,
        - maybe small grid previews,
        - counts of components/colors, etc.
      """

  def try_law_config_on_task(
      task_id: str,
      law_config: TaskLawConfig
  ) -> SolveDiagnostics:
      """
      Runs solve_arc_task_with_diagnostics(use_training_labels=True)
      but only returns diagnostics so Pi-agent can see status & mismatches.
      """

  def save_law_config(task_id: str, law_config: TaskLawConfig) -> None:
      """
      Writes to catalog via catalog.store, for configs Pi-agent believes are good.
      """
  ```

* The idea:

  * Pi-agent (you + LLM in an interactive session) only talk to `pi_interface`:

    * see task summaries,
    * propose/update law configs,
    * get rich diagnostics when it fails.

This layer is where we later plug “Pi-agent as a prompt/program” without touching the math kernel.

---

### 🔹 WO-M5.5 – Human/Pi-agent log-friendly formatting

**Goal:** make failure cases easy to read and reason about (for both humans and Pi-agent).

**File:** `src/runners/logging_utils.py`

**Scope:**

* Helpers to pretty-print `SolveDiagnostics`:

  * print which schemas used,
  * print mismatched train examples with side-by-side grids,
  * print counts (constraints, variables).
* Optionally, convert `SolveDiagnostics` into a **compact JSON** that Pi-agent can read and reason about.

This is mostly sugar, but it’s important for your “observer is observed” workflow: we want the system itself to provide introspectable state.

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
