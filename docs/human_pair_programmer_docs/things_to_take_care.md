## In M3

### from m2 
1. This doesn’t know anything about S1–S11 yet; it’s just the generic constraint collector.

2. * For M2, **builder_name** can just be strings (no actual imports yet); we’ll implement the functions in the next milestone.
3. Schema builder - For M2 we just need **structure**; actual constraint logic per S1–S11 comes in M3.

## In M4
### from M2
1. Solution decoding (y → Grid):
    - Math spec mentions "one-hot encoding for any grid Z"
    - M2 has encoding (Grid → y indices) but not decoding (y solution → Grid)
    - Assessment: ✅ Not a gap - this naturally belongs in M4 (Solver integration) when interpreting LP solution
  
  3. Constraint validation:
    - No mention of checking constraint consistency or detecting conflicts
    - Assessment: ✅ Not a gap for M2 - LP solver will detect infeasibility in M4

# chatgpt
Totally fine, we don’t change the plan. Let’s just pin the **invariants** and **TODOs** so future-me doesn’t drift.

I’ll treat this as a checklist I must carry forward after M2.

---

## A. Things that must **stay true** (invariants)

1. **Conceptual pipeline stays exactly this:**

> Operators → law family → schema instance → builder fn → ConstraintBuilder → solver → Pi-agent checks → refine/accept.

2. **ConstraintBuilder is pure plumbing**

   * `builder.py` must **not** hard-code any S1–S11 logic.
   * It only knows how to add generic linear constraints over y (tie, fix, forbid, one-hot).

3. **Law logic lives only in schema builders**

   * All “what rule means” is in `build_Sk_constraints(...)`, not in ConstraintBuilder or solver.

4. **SchemaFamily registry is metadata only**

   * `families.py` describes families (id, params, required_features, builder_name).
   * It does **not** apply constraints by itself.

5. **Dispatch only forwards**

   * `dispatch.py` (when we do it) just maps `"S3"` → `build_S3_constraints`, etc.
   * No extra logic / heuristics there.

---

## B. What M2 will contain (and nothing more)

By end of M2:

* `src/constraints/indexing.py` ✅ y-index helpers.
* `src/constraints/builder.py` ✅ `LinearConstraint`, `ConstraintBuilder`, basic helpers, one-hot.
* `src/schemas/families.py` ✅ `SchemaFamily` + `SCHEMA_FAMILIES` (metadata; builder_name strings).
* `src/schemas/dispatch.py` ✅ mapping `family_id -> builder stub` (functions exist but may be `NotImplementedError`).

**Important:** M2 does *not* implement S1–S11 logic yet; builder stubs are placeholders only.

---

## C. What is explicitly left for **later milestones**

When you “wrap my context post M2”, I must remember:

1. **Still TODO in M3:**

   * Implement real `build_S1_constraints`, …, `build_S11_constraints` in their own files (`s1_*.py`, …).
   * Wire them into `dispatch.py` (replacing stubs).
   * Use operators from M1 + ConstraintBuilder from M2.

2. **Still TODO in M4:**

   * `solver/lp_solver.py`: encode `ConstraintBuilder.constraints` into LP/ILP with `pulp`/`ortools`/`cvxpy`.
   * Add a small `solve_constraints(...) -> Grid` API.

3. **Still TODO in M5:**

   * Pi-agent orchestration:

     * chooses law families & params,
     * calls schema builders via dispatch,
     * runs solver,
     * checks train outputs, refines, logs schemas into Catalog.

---

If you later say “context post M2”, I’ll assume:

* M1 + M2 are done,
* S1–S11 builder logic, solver, and Pi-agent loop are **not** done yet,
* and we pick up from “implement schemas with the existing plumbing”.

  