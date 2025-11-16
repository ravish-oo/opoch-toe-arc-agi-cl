## 2. Tiny gaps I’d fix in the plan for 1:1 alignment

These are **very small** text tweaks, not conceptual changes.

### (a) Mod / block ranges: match spec’s exact sets

**Spec (§5.1 A):**

> Mod classes: (r mod m, c mod m) for
> (m ∈ {2,…,min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W))
>
> Block sizes: (b ∈ {2,…,min(5,min(H,W))} ∪ divisors(H,W))

**Plan (WO-4.1):**

> Mod classes: all divisors of H and W (or 2..min(H,W) if simpler).
> Block coords: block sizes `b` over divisors of H,W.

This is slightly **narrower** than the spec (you’re dropping some m that aren’t divisors). It probably doesn’t matter for coverage because:

* Row/col periods are still detected in D16–D17, and
* Non-divisor mods rarely encode something that divisors + period won’t.

But if you want true 1:1:

**Change WO-4.1 lines to:**

> * Mod classes: `m ∈ {2..min(6, max(H,W))} ∪ divisors(H) ∪ divisors(W)`.
> * Block sizes: `b ∈ {2..min(5,min(H,W))} ∪ divisors(H) ∪ divisors(W)`.

Then you can **still implement efficiently** by:

* Precomputing divisors(H), divisors(W),
* Adding `{2..6}` or `{2..5}` intersected with `[2..min(H,W)]`,
* Deduplicating the set.

### (b) Anti-spurious condition: use “≠ background color”, not “non-zero”

**Spec (§5.2):**

* Defines background per training as modal color kᵢ^{bg}.
* Anti-spurious check requires **at least one training** where the T-cells have color **≠ kᵢ^{bg}**.

**Plan (WO-5.2, 5.3):**

> “non-zero” or “not trivially background”

To match spec exactly, use the **background definition**:

**Change in WO-5.2 & WO-5.3:**

Instead of:

> “non-zero (or otherwise not trivially background)”

Say:

> “at least one training where those T-cells (or T1–T2–Δ pairs) have color ≠ kᵢ^{bg} (the modal background color for that training).”

And in code:

* Compute kᵢ^{bg} = argmax_k count_i[k] for each train_out i (tie-break by smallest k).
* Anti-spurious passes only if **some** i, some occurrence has color ≠ kᵢ^{bg}.

That aligns exactly with the math text.

### (c) Minor wording for TV: stress that full-TV MILP is the spec, tree-TV is a certified fast path

Your plan already does this functionally, but if you want the story to match spec’s emphasis:

* At the top of Milestone 6, add a one-line note:

> **Spec alignment:** The conceptual ledger is full 4-neighbor TV. Tree-TV TU-LP is used as a **certified fast path**; if TU or integrality fail, we fallback to full-TV MILP, which directly minimizes the full interface cost.

This makes it obvious to any reader that you’re not quietly changing the objective.

---

For strict textual alignment, I’d update:

1. WO-4.1 to match the exact mod/block ranges (`{2..min(6,max(H,W))} ∪ divisors` and `{2..min(5,min(H,W))} ∪ divisors`).
2. WO-5.2 / 5.3 to use the **background-color anti-spurious** definition, not “non-zero”.
3. Optionally one sentence in M6 to explicitly say “tree-TV is a certified fast path; full-TV MILP is the reference objective”.

If you make those tiny edits, you’ll have a genuinely zero-gap triangle:

**TOE math spec ↔ consciousness model ↔ implementation plan.**

When you’re ready, we can design the first 2–3 concrete work orders (with function signatures) so Claude Code can start coding this **exact** spec without drifting.

====
# ATOMS
2. Mathematical / spec-level “watchpoints”

These are the only places where you can accidentally drift off-spec or break the TOE mapping.

2.1 Type key definition

Spec: type key T(p) is a specific tuple (distances, mod classes up to certain bounds, r±c, 3×3, period flags, component shape id).

Plan: “selected atoms (distances, divisors-mod classes, r±c, 3×3 hash, period class, optional component id).”

This is conceptually the same, but for code you should treat the 00_MATH_SPEC type key as canonical:

Don’t silently drop a piece (e.g., period flags or component id), or you may lose discriminative power and fail some tasks.

Don’t add unstable atoms (like something noisy or non-deterministic).

So: make sure 05_laws/atoms.py literally implements the 00_MATH_SPEC T(p) tuple, just with grid-aware ranges.

2.2 Grid-aware ranges

You switched from fixed {2..6} style ranges to divisors / min(H,W) everywhere. That matches the updated “grid-aware atom universe” idea in 00_MATH_SPEC and is fine, as long as:

Periods: only up to dimension or divisors, never arbitrary large m.

Blocks: only divisors or small b ≤ min(5, min(H,W)).

This keeps the atom space finite and within the spec.

So here: good, but stick to the bounds you’ve stated in the spec, not ad-hoc extras.

2.3 Forbids remain strictly “safe”

The plan keeps forbids optional and requires that they cannot conflict with fixes/equalities. That matches the spec. Implementation detail that must be enforced:

For each type T, color k, only emit forbid if:

k never appears at T in any train_out, and

Setting x[p,k]=0 cannot break a previously mined fix or equality.

If IIS ever appears, the plan already says: drop forbids and re-run. That’s the correct behavior.

# TU THING
2.4 TU signer + fallback policy

The spec’s guarantee depends on this exact policy:

If TU signer passes and LP is integral → accept LP solution.

Otherwise → always go to full-TV MILP and use that solution as final.

Implementation plan states exactly this. You just need to avoid any “optimizations” like:

Trying to “fix” non-integrality by rounding without MILP.

Ignoring signer failure because “it probably works”.

Stick to your written policy and the math spec remains intact.

2.5 Optional flows

You explicitly mark “connectivity flows off by default” and only on tree if ever introduced. That respects the TU + tree-TV story. I’d keep them out until you have clear tasks that need them.