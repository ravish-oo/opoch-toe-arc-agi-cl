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

## M2 – Constraint representation & ConstraintBuilder 

High-level Work Orders

### **WO-M2.1 – y-indexing helpers** ✅ COMPLETE

**File:** `src/constraints/indexing.py`
**Goal:** define a clean way to map `(pixel, color)` to indices in the y vector (and back).

**Scope:**

* Define functions:

  * `flatten_index(r: int, c: int, W: int) -> int`   (pixel index 0..N-1)
  * `unflatten_index(p_idx: int, W: int) -> tuple[int,int]`
  * `y_index(p_idx: int, color: int, C: int) -> int`  (0..N*C-1)
  * `y_index_to_pc(idx: int, C: int, W: int) -> (p_idx, color)`
* No solver, no constraints here — just pure indexing.
* This module will be imported by `builder.py` and schema builders.

---

### **WO-M2.2 – LinearConstraint & ConstraintBuilder core** ✅ COMPLETE

**File:** `src/constraints/builder.py`
**Goal:** core objects to collect linear equations over y.

**Scope:**

* Define:

  ```python
  @dataclass
  class LinearConstraint:
      indices: list[int]       # indices in y
      coeffs:  list[float]
      rhs:     float
  ```

  ```python
  @dataclass
  class ConstraintBuilder:
      constraints: list[LinearConstraint] = field(default_factory=list)

      def add_eq(self, indices: list[int], coeffs: list[float], rhs: float): ...
      def tie_pixel_colors(self, p_idx: int, q_idx: int, C: int): ...
      def fix_pixel_color(self, p_idx: int, color: int, C: int): ...
      def forbid_pixel_color(self, p_idx: int, color: int, C: int): ...
  ```

* Also a helper for **one-hot per pixel**:

  ```python
  def add_one_hot_constraints(builder: ConstraintBuilder, N: int, C: int): ...
  ```

* Use `src/constraints/indexing.py` for index math.

This doesn’t know anything about S1–S11 yet; it’s just the generic constraint collector.

---

### **WO-M2.3 – SchemaFamily registry (metadata only)**

**File:** `src/schemas/families.py`
**Goal:** define the **law family** objects (S1–S11) and a registry that Pi-agents/tools can inspect.

**Scope:**

* Define:

  ```python
  @dataclass
  class SchemaFamily:
      id: str                  # e.g. "S1"
      name: str                # "Direct pixel color tie"
      description: str
      parameter_spec: dict     # e.g. {"feature_predicate": "str"}
      required_features: list[str]  # e.g. ["components", "object_id"]
      builder_name: str        # e.g. "build_S1_constraints"
  ```

* Define a dict:

  ```python
  SCHEMA_FAMILIES: dict[str, SchemaFamily] = {
      "S1": SchemaFamily(...),
      "S2": SchemaFamily(...),
      ...
      "S11": SchemaFamily(...)
  }
  ```

* For M2, **builder_name** can just be strings (no actual imports yet); we’ll implement the functions in the next milestone.

This file is for the Pi-agent and system to know **what kinds of laws exist** and what parameters they require.

---

### **WO-M2.4 – Schema builder dispatch skeleton**

**File:** `src/schemas/dispatch.py`
**Goal:** a small dispatcher that, given a law family id and params, calls the right builder function stub.

**Scope:**

* Define stub signatures for builder functions:

  ```python
  # These will be implemented in M3
  def build_S1_constraints(...): ...
  def build_S2_constraints(...): ...
  # ...
  def build_S11_constraints(...): ...
  ```

* Define a mapping:

  ```python
  BUILDERS = {
      "S1": build_S1_constraints,
      "S2": build_S2_constraints,
      ...
  }
  ```

* Define a helper function:

  ```python
  def apply_schema_instance(
      family_id: str,
      params: dict,
      task_context: dict,
      builder: ConstraintBuilder
  ):
      """
      Look up the builder for family_id, call it with params and task_context.
      For now builder functions can be 'pass' or log not-implemented.
      """
  ```

* `task_context` will later carry things like:

  * grid(s),
  * features (from φ),
  * N, C, etc.

For M2 we just need **structure**; actual constraint logic per S1–S11 comes in M3.

---

That’s it for M2 at high level:

1. `indexing.py` – how y is indexed.
2. `builder.py` – how constraints are stored and basic primitives to add them.
3. `families.py` – registry of schema families S1–S11, metadata only.
4. `dispatch.py` – mapping family id → builder stub, and a generic `apply_schema_instance` entrypoint.

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

From there, we can:

* Define the **SchemaFamily registry** and **builder functions** (next milestones).
* Then hook ConstraintBuilder + solver.
* Finally, bring in the Pi-agent that uses all this.
