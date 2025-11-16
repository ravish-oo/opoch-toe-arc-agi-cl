we now start 
## Milestone 4 — laws: atom derivations (grid‑aware)

**Goal:** implement the **frozen**, grid‑aware atom menu.

> **Grid‑aware policy:** any former finite range like `{2..6}` becomes **bounded by grid**: divisors or up to `min(H,W)` as appropriate. This preserves 100% coverage without blow‑ups on tiny boards.

now let's expand this wo. so now tell me what kinds of "mature, well-documented Python libraries for every primitive we need" can be reused for

### WO‑4.1 Coordinates & distances (A1–A7) (120–160 LOC)

* Provide arrays for `H,W; r,c; r±c; d_top,bottom,left,right; midrow/midcol`.
* Mod classes: **all divisors** of `H` and of `W` (or `2..min(H,W)` if simpler).
* Block coords: block sizes `b` over **divisors** (tiling‑valid) of `H,W`.
  **Acceptance:** shapes correct; hashes stable.

now here are the things u must take care of:
1. now thr is nothing called underspecificy in development. instead  we shud be ovespecific. but anything that u r specifying must be STRICTLY grounded in our anchor docs we created 
ls docs/anchors 
00_MATH_SPEC.md         01_STAGES.md            02_QUANTUM_MAPPING.md
2. do tell in WO which anchor docs they shud refer to before proceeding if any..
3. Plus we must be explicit about wht packges' functions and libs to use here so that claude doenst reinvent the wheel.  we dont want to implement any algo. 
4. make sure u include Brainstem run.py changes that are needed to accomodate this wo and how can reviewer use it to test on real arc agi tests . do make sure we dont make a god function. and that if we keep run.py updated, our integration effort for later WOs is removed. run.py MUST be minimal
5. We may hv receipts as part of WO with clear understanding as to how they capture and how reviewer can use it for testing/debugging. 
6. a clear instruction to reviewer that tells them how to use run.py for 2-3 arc agi tasks that tell what to expect. They shud knw how to identtify a legit implementation gap or unsatisifable task if any in WO may be using receipts. this instruction MUST ensure that math and implementation match 100% 
7. we do not get trapped in optimization fixes that enforces simplified implemtnatinns unless we are really looking at huge compute times (ps we will do on CPU)
8. STRICTLY no short cuts or hacks or simplified suggestion. if spec is underspecified and u dont hv clarity, give a push back and DO NOT paper over such things. if the spec doesn’t cover a case, we fail loudly, we don’t improvise
9. Do NOT invent new interpretations or structures beyond what’s in anchors

so can u create an expanded WO accommodating all of what i said above with a small checklist at the end of WO to show how u took care of each of these?
