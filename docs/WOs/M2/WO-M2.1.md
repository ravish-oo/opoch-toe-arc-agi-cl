## WO-M2.1 – `y`-indexing helpers

**File:** `src/constraints/indexing.py`
**Goal:** provide a *single, canonical* way to map between:

* pixel coordinates `(r, c)` ↔ flat pixel index `p_idx` (0..N-1),
* pixel/color `(p_idx, color)` ↔ flat y-index `y_idx` (0..N*C-1).

This must be **pure indexing**, no solver, no constraints.

### 1. Conventions (important, no ambiguity)

We fix the following conventions to match the math spec:

* Grid has shape `(H, W)`, with:

  * rows `r` in `0..H-1`,
  * cols `c` in `0..W-1`.
* Pixel ordering:

  * flat pixel index `p_idx` runs from `0` to `N-1` where `N = H * W`.
  * We use **row-major order**: all columns of row 0, then all columns of row 1, etc.
  * Formula: `p_idx = r * W + c`.
* Color index `color` is in `0..C-1`.
* y-vector indexing:

  * y is a vector of length `N * C`.
  * For pixel `p_idx` and color `color`, we use:

    * `y_idx = p_idx * C + color`.

These conventions must be used everywhere in constraints & solver; no deviations.

### 2. Functions to implement

In `src/constraints/indexing.py`, implement:

```python
from typing import Tuple

def flatten_index(r: int, c: int, W: int) -> int:
    """
    Convert row/col coordinates to a flat pixel index (0 .. N-1).

    Args:
        r: row index, 0 <= r < H
        c: col index, 0 <= c < W
        W: grid width

    Returns:
        p_idx: flat pixel index, p_idx = r * W + c
    """
    ...

def unflatten_index(p_idx: int, W: int) -> Tuple[int, int]:
    """
    Convert a flat pixel index back to (row, col).

    Args:
        p_idx: flat pixel index, 0 <= p_idx < N
        W: grid width

    Returns:
        (r, c): row and column indices, consistent with flatten_index.
    """
    ...

def y_index(p_idx: int, color: int, C: int) -> int:
    """
    Convert a (pixel index, color index) pair to a y-vector index (0 .. N*C-1).

    Args:
        p_idx: flat pixel index, 0 <= p_idx < N
        color: color index, 0 <= color < C
        C: number of colors

    Returns:
        y_idx: index in the y vector, y_idx = p_idx * C + color
    """
    ...

def y_index_to_pc(y_idx: int, C: int, W: int) -> Tuple[int, int]:
    """
    Convert a y-vector index back to (pixel index, color index).

    Args:
        y_idx: index in y, 0 <= y_idx < N*C
        C: number of colors
        W: grid width (needed so that later we can recover (r,c) from p_idx)

    Returns:
        (p_idx, color) pair, such that:
            y_idx == p_idx * C + color
        NOTE: actual (r,c) can be recovered by calling unflatten_index(p_idx, W).
    """
    ...
```

**Requirements / constraints:**

* Use only **standard Python** (no need for numpy here).
* Implement exactly the formulas above; do **not** invent alternative layouts.
* Add type hints and docstrings as shown.
* No global state, no classes needed.

### 3. Thin runner for quick local testing

In the same file, add a small `if __name__ == "__main__":` block to sanity-check the roundtrips. Example behavior:

```python
if __name__ == "__main__":
    H, W, C = 3, 4, 5
    # Check pixel index roundtrip
    for r in range(H):
        for c in range(W):
            p = flatten_index(r, c, W)
            rr, cc = unflatten_index(p, W)
            assert (rr, cc) == (r, c), f"Roundtrip (r,c)->p->(r,c) failed at {(r,c)}"

    # Check y-index roundtrip
    N = H * W
    for p_idx in range(N):
        for color in range(C):
            y_idx = y_index(p_idx, color, C)
            p_back, color_back = y_index_to_pc(y_idx, C, W)
            assert p_back == p_idx, "p_idx mismatch"
            assert color_back == color, "color mismatch"

    print("indexing.py sanity checks passed.")
```

This is a simple runner that:

* doesn’t depend on ARC data at all,
* gives immediate feedback if indexing is wired incorrectly.

### 4. Reviewer/tester instructions

For the **reviewer+tester** (a separate Claude Code instance or you):

1. **Code review:**

   * Check that:

     * `flatten_index` uses `r * W + c`.
     * `unflatten_index` uses integer division and modulo: `r = p_idx // W`, `c = p_idx % W`.
     * `y_index` uses `p_idx * C + color`.
     * `y_index_to_pc` does the true inverse: `p_idx = y_idx // C`, `color = y_idx % C`.
   * Confirm docstrings match these formulas exactly.
   * Confirm no extra logic or dependencies were added.

2. **Run the thin runner:**

   * `python -m src.constraints.indexing` (adjust path as needed).
   * Ensure it prints `indexing.py sanity checks passed.` and raises no assertion errors.

3. **(Optional) Integration smoke test with ARC grid:**

   * Use `arc_io.py` (from M1) to load one sample task.
   * Take a grid from `train[0]["input"]`.
   * For a few `(r,c)` values, manually check:

     * `p_idx = flatten_index(r,c,W)`
     * `(rr,cc) = unflatten_index(p_idx,W)`
       → verify `(rr,cc)` matches `(r,c)`.
   * This is just a sanity check that the H,W from actual ARC data flow in correctly.

No ARC-specific math enters here; this is purely index geometry consistent with the spec’s (\Omega = {1..N}) and y ∈ {0,1}^{NC}, just implemented in 0-based Python form.

---
