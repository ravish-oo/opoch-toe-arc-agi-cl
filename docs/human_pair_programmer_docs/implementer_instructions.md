now ur role is that of implemnter.
   u implement the expanded wo.
  pls check @docs/repo_structure.md and create only applicable folder and files for the given wo
  here is the expanded wo
  ---

  ---
  pls understnad the wo and see if it aligns with @docs/anchors/math_kernel.md and that u hv 100% clarity on how to implement this wo 
ps the code in wo is just a sketch or for illustration only. 
  =======

pls understnand this and see if this aligns with @docs/anchors/math_kernel.md and @docs/anchors/math_kernel_clarifications.md that u hv 100% clarity on how to implement this
   ps dont review/test yet. just tell if u hv 100% clarity
   ===

perfect.. so i hv cleaned up and rearragend content in implementation plan. i hv added high level work orders in it for M5. so pls read @docs/anchors/math_kernel.md
    first and then see if work orders of milesteone 5 in @docs/implementation_plan.md align perfectly with math spec and that we didnt miss something that was suposed to
    be covered









https://chatgpt.com/g/g-p-690ae6b34ba88191a521419ccd485745-toe-alive/c/691027cf-964c-8326-92da-5edc4ae09107

# Understand what we are doing
pls understand what we are doing. read @docs/anchors/maths/01_math_spec.md @docs/anchors/maths/03_math_spec_patch2.md 
@docs/anchors/maths/02_math_spec_patch1.md
  @docs/anchors/maths/04_math_spec_addendum.md @docs/anchors/engineering/computing_spec.md

# Understand ur role
Implement exactly the WO interface; no stubs, no TODOs, no extra helpers outside the spec. 
Use only the allowed primitives and frozen orders; no randomness, no floats, no heuristics.
On any unprovable case: return silent (A=all, S=0) or the specified FAIL (UNSAT, SIZE_INCOMPATIBLE, FIXED_POINT_NOT_REACHED).
Emit receipts for every public call (hashes, counts, attempts list) and ensure double-run identical hashes.
In code keep pure functions, zero side effects except receipts.


# wo prompt
here is the WO. do refer to @docs/repo_structure.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials

once u confirm above, we can start coding!

# how to debug
u may hv tried few things but see if u want to try something from here. also when u say u hypothesize, why do i need to to do guess and hope. i mean best part of programming is that u can print output at each step and study  it when out and find exactly when it breaks.. that's what debuggers formalized but old school way us to print the outputs and see. u r trained on code that probably didnt hv these prints for debug but that's how its done. 

so u must not "hypothesize" and fix. hypothesize to investigate, print outputs and settle hypothesis rather than hit and hope. that just wont work. so get back to 0th principle of coding. print and see and fix. simple as that.. 
hope this helps 


# latest
now ur role is that of an implemetner. 
u need to follow the Work order and see that it aligns with ur understanding of @docs/anchors and is in line with @docs/IMPLEMENTATION_PLAN.md  following is the work order u need to implement.
  data/ folder has training challenges of arc agi 
  do create a venv if u need any installations.
  ---

  ---
  wo above has all the details u need. pls note any code in WO is sketch/illustrative only.  LOC quota is indicative only
  pls understand and see if u hv 100% clarity on what needs to be done.
  once u confirm above, we can start coding!
─────────────────────────────────────────────
pls proceed
remember if spec is underspecified and u dont hv clarity, give a push back and DO NOT paper over such things. if the spec doesn’t cover a case, we fail loudly, we don’t improvise
Do NOT invent new interpretations or structures beyond what’s in anchors


## Patch Work Order — Progress in Harness (applies now, supports all future WOs)

**Goal:** extend `harness.py` so every WO writes (a) per-task receipts and (b) a run-level `progress.json` that aggregates exact, math-forced invariants. This lets the reviewer prove we’re implementing the calculus (Π, GLUE, FY, TU) *as we go*.

> **Anchors to read before coding:**
> `03_annex.md` A.1–A.3; `05_contracts.md` (Global, FREE vs PAID, Bit-meter, Relaxation); `04_engg_spec.md` relevant section for the current WO. All checks below are grounded in these.

### 1) Files to edit

**`src/arcsolver/harness.py`**

* Add a `--progress` flag (default on).
* Add a `collect_progress(stats: dict)` function that accumulates WO-specific counters as you process tasks.
* After the run, write `progress/progress_woXX.json` with:

  * `"wo": XX`, `"tasks_total"`, `"tasks_ok"`,
  * for **WO-1**: `bins_sum_ok`, `center_all_ok` counts,
  * for later WOs: fields listed under “Progress fields by WO” below.

**`src/arcsolver/receipts.py`**

* Add `write_run_progress(progress_dict, out_dir="progress")`.

**`scripts/run_harness.sh`**

* Accept `--upto-wo N` and always pass `--progress`.

### 2) JSON loading (PS you noted)

In `harness.py`, **glob `*.json`**, open each ARC file, then iterate **dict keys** (`train`, `test`) to extract the grids. This matches the real ARC format (dict-based files), not a flat list.

### 3) Progress fields by WO (add incrementally)

You can implement these *gradually*; the harness always writes what is available at the current WO.

**WO-1 (Bins & Predicates)**

* `bins_sum_ok`: `sum(bin_counts)==H*W` (count how many tasks satisfy).
* `bins_hash_stable`: hash invariant across two passes.
* `center_all_ok`: every training satisfies ≤0.5 distance to canvas center when predicate says `'center'`.

**Optional property-based hook (any WO)**
Add a guard to run small property tests (e.g., closure monotonicity, projector idempotence) using Hypothesis. This is ideal for algebraic laws where there’s no “ground truth” label, and is widely used for oracle-free testing. ([hypothesis.readthedocs.io][4])

### 4) Minimal code stubs to drop in (exact)

**`harness.py` additions (sketch)**

```python
# add near top-level
def _init_progress(wo:int) -> dict:
    return {"wo": wo, "tasks_total": 0, "tasks_ok": 0, "metrics": {}}

def _acc(progress: dict, key: str, ok: bool | int):
    m = progress["metrics"].setdefault(key, {"ok": 0, "total": 0, "sum": 0})
    if isinstance(ok, bool):
        m["total"] += 1
        if ok: m["ok"] += 1
    else:
        m["sum"] += int(ok)

# inside main run loop per task
progress["tasks_total"] += 1
# example WO-1:
_acc(progress, "bins_sum_ok", bins_sum_ok)
_acc(progress, "bins_hash_stable", bins_hash_stable)
_acc(progress, "center_all_ok", center_all_ok)
# when a task passes all stage checks:
progress["tasks_ok"] += 1

# at end of run:
write_run_progress(progress)
```

**`receipts.py` addition**

```python
def write_run_progress(progress: Dict[str, Any], out_dir="progress"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fn = Path(out_dir) / f"progress_wo{progress['wo']:02d}.json"
    with fn.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(progress, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
```

===
again, remember  u r role, u r the implementer
  here is the next wo. pls refer to @docs/repo_structure.md to knw what files are whr
    [Pasted text #1 +378 lines]
    ---
   work order has all the details u need. pls note any code in WO is sketch/illustrative only
  pls understand and see if u hv 100% clarity on what needs to be done.
    once u confirm above, we can start coding!

===

so we will need to run all 1120 tasks from wo 01 to 07.. but we gotta be smart about it.. i knw it will take a lot of time but once done, we must then store it
such that we can reuse the result for future WOs. here is the patch that allows us to do so. can u implement this? and u can run the script in bg and just
create a small doc that tells how to use the recipts/artifacts of this run from wo01 to 07 so that we can use it in future wos.. sounds like  a plan? here is
wo:
===
# Quick check - cache counts per stage
  find .cache/wo04 -name "*.npz" 2>/dev/null | wc -l  # WO-04 caches
  find .cache/wo05 -name "*.npz" 2>/dev/null | wc -l  # WO-05 caches
  find .cache/wo06 -name "*.npz" 2>/dev/null | wc -l  # WO-06 caches


   Dual Artifact System

  1. Receipts (in receipts/) - Audit trail
  - Contains: Metadata, hashes, stats (small JSON files)
  - Version controlled
  - WO-00 through WO-06 receipts already exist or will be created
  - Used by harness to track what's been computed

  2. Caches (in .cache/) - Materialized arrays
  - Contains: Full numpy arrays (NPZ files)
  - Git-ignored (build artifacts)
  - Currently populating for WO-04, WO-05, WO-06
  - This is what WO-07 will load from
