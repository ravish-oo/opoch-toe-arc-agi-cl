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

5. **Task IO + test harness** ✅ COMPLETE 

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

## M5 –  Pi-agent interface, diagnostics, catalog building ✅ COMPLETE
M5 is where we make the **kernel “talk”** — fail-closed, with rich, structured intermediate info that a Pi-agent can actually use to debug and refine laws.

We’re done with the math engine; now we’re building the **interface and diagnostics layer** that a Pi-agent will sit on.

High-level Work Orders in docs/WOs/M5/M5.md . Detailed WOs in docs/WOs/M5/

---
 ## M6 - Pi agent on langgraph

 Good, we’re finally at the “just build the thing” stage.

Given all the constraints (atomic WOs, no reinvention, just stitching), here’s how I’d slice the LangGraph implementation into **3 high-level work orders**.

No stubs, but we can keep v0 behavior for the Pi-agent node minimal so the graph runs and you can later swap in the real prompt chain.

---

## 🧩 WO-LG1 – State schema + graph skeleton ✅ COMPLETE

**Goal:** Define the shared state (`ArcPiState`), and a module that builds the LangGraph with named nodes and routing functions wired, but without heavy node logic yet.

**Files:**

* `src/agents/arc_pi_state.py`
* `src/agents/arc_pi_graph.py`

**Scope (high-level):**

1. **`arc_pi_state.py`**

   * Define `ArcPiState` as a `TypedDict` with the fields we agreed:

     * `task_id`, `max_attempts`
     * `raw_task` (dict as from `load_arc_task`)
     * `attempt`, `outcome`
     * `current_law_config`, `current_diagnostics`
     * `law_config_history`, `diagnostics_history`
     * `pi_decision`, `pi_notes`
     * `chat_history` (list of message dicts)

   * No algorithms, just types and maybe a helper to create an initial empty state.

2. **`arc_pi_graph.py`**

   * Import `ArcPiState`, `StateGraph`, `START`, `END` from LangGraph.

   * Declare nodes by name (no bodies here, just registration):

     * `"init_task_node"`
     * `"pi_agent_node"`
     * `"engine_node"`
     * `"success_node"`
     * `"failure_node"`

   * Add conditional routing functions **as pure functions** that look only at `ArcPiState` keys:

     * `route_from_init(state) -> "engine_node" | "pi_agent_node"`
     * `route_from_pi_agent(state) -> "engine_node" | "failure_node"`
     * `route_from_engine(state) -> "success_node" | "pi_agent_node" | "failure_node"`

   * Build:

     ```python
     builder = StateGraph(ArcPiState)
     # add_node(...) calls, add_edge(START→init_task_node), add_conditional_edges(...), add_edge(success/failure→END)
     graph = builder.compile()
     ```

   * Export a `get_arc_pi_graph()` function that returns the compiled `graph`.

**Reviewer/tester hint:** this WO is just types + wiring; no domain logic yet. You can unit-test that `graph` compiles and that routing functions behave correctly for a few dummy states.

---

## 🧩 WO-LG2 – Implement init/engine/success/failure nodes (kernel integration) ✅ COMPLETE

**Goal:** Implement the **non-LLM nodes**:

* `init_task_node`
* `engine_node`
* `success_node`
* `failure_node`

and wire them into the graph skeleton created in WO-LG1.

**Files:**

* `src/agents/nodes_core.py` (for node functions)
* touch `src/agents/arc_pi_graph.py` (to import and register these nodes)
* optional thin runner: `src/agents/run_single_task_graph.py`

**Scope (high-level):**

1. **`nodes_core.py`**

   Implement:

   * `init_task_node(state: ArcPiState) -> dict`:

     * Use `task_id` and `max_attempts` from state.
     * Call `load_arc_task(task_id, TRAINING_CHALLENGES_PATH)` **once**.
     * Load existing `TaskLawConfig` from `catalog.store.load_task_law_config`.
     * Initialize:

       * `raw_task`,
       * `attempt = 0`,
       * `outcome = "unresolved"`,
       * `current_law_config = existing_config or None`,
       * `current_diagnostics = None`,
       * `law_config_history = []`,
       * `diagnostics_history = []`,
       * `pi_decision = None`, `pi_notes = None`,
       * `chat_history = base_prompt_chain.messages` (the JSON you defined).

   * `engine_node(state: ArcPiState) -> dict`:

     * Require `current_law_config` is not `None`, else set a special error status and route to `failure_node`.

     * Call `solve_arc_task_with_diagnostics` from `src/runners/kernel.py`, passing:

       * `task_id`,
       * `current_law_config`,
       * `use_training_labels=True`,
       * **either** `challenges_path` constant or `raw_task` depending on how you wired the kernel.

     * Update:

       * `current_diagnostics`,
       * `diagnostics_history += [current_diagnostics]`.

   * `success_node(state: ArcPiState) -> dict`:

     * Call `save_task_law_config(task_id, current_law_config)`.
     * Set `outcome = "solved"`.

   * `failure_node(state: ArcPiState) -> dict`:

     * Inspect `pi_decision`, `attempt`, `max_attempts`, and `current_diagnostics`.

     * Decide `outcome` ∈ `{"unsolved_basis_incomplete","unsolved_attempt_limit"}`.

     * Append a JSON line with:

       * `task_id`,
       * `attempt`,
       * `outcome`,
       * `pi_decision`, `pi_notes`,
       * serialized `law_config_history`,
       * serialized `diagnostics_history` (no algorithms, just `json.dumps`).

     * Return with `{"outcome": outcome}`.

2. **Update `arc_pi_graph.py`**

   * Import node functions from `nodes_core.py`.
   * Replace dummy nodes with real ones:

     ```python
     builder.add_node("init_task_node", init_task_node)
     builder.add_node("engine_node", engine_node)
     builder.add_node("success_node", success_node)
     builder.add_node("failure_node", failure_node)
     ```

3. **Thin runner** (optional but useful): `run_single_task_graph.py`

   * Parse `--task-id` and `--max-attempts` from CLI.
   * Instantiate `graph = get_arc_pi_graph()`.
   * Call `graph.invoke({"task_id": ..., "max_attempts": N})`.
   * Print `final_state["outcome"]` and lengths of `law_config_history` / `diagnostics_history`.

**Reviewer/tester hint:** for now, `pi_agent_node` can be a simple placeholder (see WO-LG3) so the graph runs. The core test is: `engine_node` actually calls the kernel, and success/failure nodes update catalog/logs correctly.

---

## 🧩 WO-LG3 – Implement `pi_agent_node` with prompt chain + JSON parsing

**Goal:** Implement the **Pi-agent node** that uses your `base_prompt_chain.json` + `last_msg.json` + `reattempt_msg.json` + `giveup.json`, calls the LLM, and returns a valid `TaskLawConfig` or a meta decision.

**Files:**

* `src/agents/nodes_pi.py`
* touch `src/agents/arc_pi_graph.py` (register node)
* your JSON prompt files in `prompts/` (already planned)

**Scope (high-level):**

1. **`nodes_pi.py`**

   Implement `pi_agent_node(state: ArcPiState) -> dict`:

   * Decide which message template to use:

     * If `attempt == 0`:

       * use `last_msg.json` to build the first task-specific user message.
     * If `0 < attempt < max_attempts`:

       * use `reattempt_msg.json`, filling:

         * `task_id`,
         * `attempt`, `max_attempts`,
         * `last_pi_decision`,
         * `last_law_config_json`,
         * `last_diagnostics_json`.
     * If `attempt >= max_attempts`:

       * you **won’t call** `pi_agent_node` (graph will route to `failure_node`), so no special case needed here.

   * Merge:

     * `base_prompt_chain.messages` (from state),
     * plus the new task-specific message,
     * to form `messages` for LLM.

   * Call your LLM client (Claude/OpenAI) with:

     * `messages = state["chat_history"] + [new_user_msg]`,
     * system keys as needed.

   * Parse the **assistant’s JSON-only response** into:

     * `decision` string,
     * `law_config` dict or null,
     * `notes` string.

   * Update:

     * `pi_decision = decision`,
     * `pi_notes = notes`,
     * `current_law_config` = parsed TaskLawConfig (if decision == "law_config", else None),
     * `law_config_history += [current_law_config]` if applicable,
     * `chat_history` += [`assistant` message].

   * Increment `attempt` by 1.

   Return partial state dict with those updates.

   > No algorithm implementation here beyond JSON parsing and a bit of glue.

2. **Update `arc_pi_graph.py`**

   * Import `pi_agent_node` from `nodes_pi.py`.

   * Register it:

     ```python
     builder.add_node("pi_agent_node", pi_agent_node)
     ```

   * Ensure conditional edges defined in WO-LG1 point to this node.

3. **Thin test:**

   * In `run_single_task_graph.py`, use a **fake Pi-agent** at first (e.g., a version of `pi_agent_node` that always returns `decision: "give_up"` with notes), or hit a mocked LLM endpoint.
   * Once plumbing works, swap in the real LLM call.

**Reviewer/tester hint:** focus on:

* Correct construction of `messages` from `chat_history + last_msg/reattempt_msg`.
* Valid JSON parsing into `decision`, `law_config`, `notes`.
* Correct state updates: `attempt`, `current_law_config`, histories, `pi_decision`, `pi_notes`, `chat_history`.

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
