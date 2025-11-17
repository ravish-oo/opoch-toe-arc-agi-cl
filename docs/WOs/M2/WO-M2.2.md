## WO-M2.2 – `LinearConstraint` & `ConstraintBuilder` core

**File:** `src/constraints/builder.py`
**Goal:** define the core data structures to **collect linear equations over y**, and a few primitive helpers to add constraints. No solver logic here.

This module will be used later by schema builders and the LP solver.

---

### 1. Imports & dependencies

* Use standard libs only:

  * `from dataclasses import dataclass, field`
  * `from typing import List`
* Import y-indexing helpers from M2.1:

```python
from src.constraints.indexing import y_index
```

(No other non-standard libs.)

---

### 2. `LinearConstraint` dataclass

Define:

```python
@dataclass
class LinearConstraint:
    """
    Represents a single linear equality constraint over the y vector:

        sum_i coeffs[i] * y[indices[i]] = rhs

    where y is the flattened (N*C)-dimensional one-hot vector of pixel colors.
    """
    indices: List[int]    # indices into y (0 .. N*C-1)
    coeffs:  List[float]  # same length as indices
    rhs:     float         # right-hand side
```

Constraints:

* `indices` and `coeffs` **must have the same length**.
* This module will not enforce that, but we’ll keep it consistent in our code.

---

### 3. `ConstraintBuilder` dataclass

Define:

```python
@dataclass
class ConstraintBuilder:
    """
    Collects linear equality constraints over the y vector.
    """
    constraints: List[LinearConstraint] = field(default_factory=list)

    def add_eq(self, indices: List[int], coeffs: List[float], rhs: float) -> None:
        """
        Add a generic linear equality constraint:

            sum_i coeffs[i] * y[indices[i]] = rhs
        """
        ...
```

**Implementation details for `add_eq`:**

* Assert `len(indices) == len(coeffs)`.
* Append a new `LinearConstraint(indices=indices, coeffs=coeffs, rhs=rhs)` to `self.constraints`.

---

### 4. Helper methods on `ConstraintBuilder`

All these helpers **must use** `y_index` from `src.constraints.indexing`.

Assume:

* `p_idx` is a flat pixel index `0 <= p_idx < N`.
* `color` in `0 <= color < C`.

#### 4.1 `tie_pixel_colors`

```python
def tie_pixel_colors(self, p_idx: int, q_idx: int, C: int) -> None:
    """
    Enforce that pixels p and q have the same color:

        for all c in 0..C-1:
            y[p,c] - y[q,c] = 0
    """
    ...
```

Implementation:

* For each `c` in `0..C-1`:

  * `i_p = y_index(p_idx, c, C)`
  * `i_q = y_index(q_idx, c, C)`
  * Call `self.add_eq(indices=[i_p, i_q], coeffs=[1.0, -1.0], rhs=0.0)`.

#### 4.2 `fix_pixel_color`

```python
def fix_pixel_color(self, p_idx: int, color: int, C: int) -> None:
    """
    Enforce that pixel p has exactly the given color:

        y[p,color] = 1
        and (later, via one-hot constraints) y[p,c!=color] = 0

    NOTE: This method only enforces y[p,color] = 1.
          Zeroing out other colors is done by global one-hot + other schema logic.
    """
    ...
```

Important: to avoid over-constraining, in M2 **we only set y[p,color] = 1 here**, and rely on one-hot constraints (and other schema constraints) to prevent other colors.

Implementation:

* Compute `i = y_index(p_idx, color, C)`.
* Call `self.add_eq(indices=[i], coeffs=[1.0], rhs=1.0)`.

*(We won’t set other colors to 0 here; that’s safer and more flexible.)*

#### 4.3 `forbid_pixel_color`

```python
def forbid_pixel_color(self, p_idx: int, color: int, C: int) -> None:
    """
    Enforce that pixel p does NOT have the given color:

        y[p,color] = 0
    """
    ...
```

Implementation:

* `i = y_index(p_idx, color, C)`.
* `self.add_eq(indices=[i], coeffs=[1.0], rhs=0.0)`.

---

### 5. One-hot per pixel helper

Outside the class, define:

```python
def add_one_hot_constraints(builder: ConstraintBuilder, N: int, C: int) -> None:
    """
    For each pixel index p in 0..N-1, enforce:

        sum_{c=0..C-1} y[p,c] = 1

    This ensures every pixel has exactly one color.
    """
    ...
```

Implementation:

* For `p_idx` in `0..N-1`:

  * `indices = [y_index(p_idx, c, C) for c in range(C)]`
  * `coeffs = [1.0] * C`
  * `rhs = 1.0`
  * `builder.add_eq(indices, coeffs, rhs)`

No other logic here.

---

### 6. Thin runner for local sanity checks

Add at bottom of `builder.py`:

```python
if __name__ == "__main__":
    from src.constraints.indexing import y_index

    # Simple sanity checks with tiny N, C
    N, C = 2, 3  # two pixels, three colors
    b = ConstraintBuilder()

    # Check one-hot constraints
    add_one_hot_constraints(b, N, C)
    assert len(b.constraints) == N, "Expected one constraint per pixel"

    # Check tie_pixel_colors produces C constraints
    b2 = ConstraintBuilder()
    b2.tie_pixel_colors(p_idx=0, q_idx=1, C=C)
    assert len(b2.constraints) == C, "Expected C tie constraints"

    # Check fix_pixel_color produces one constraint
    b3 = ConstraintBuilder()
    b3.fix_pixel_color(p_idx=0, color=2, C=C)
    assert len(b3.constraints) == 1
    lc = b3.constraints[0]
    assert lc.rhs == 1.0
    assert len(lc.indices) == 1 and len(lc.coeffs) == 1

    # Check forbid_pixel_color produces one constraint with rhs 0
    b4 = ConstraintBuilder()
    b4.forbid_pixel_color(p_idx=0, color=1, C=C)
    lc2 = b4.constraints[0]
    assert lc2.rhs == 0.0

    print("builder.py sanity checks passed.")
```

This runner doesn’t solve anything, it just ensures the methods produce the expected number/shape of constraints.

---

### 7. Reviewer/tester instructions

For the reviewer/tester:

1. **Code correctness:**

   * Verify imports: only `dataclasses`, `typing`, and `y_index`.
   * Check `add_eq` asserts `len(indices) == len(coeffs)` and appends a `LinearConstraint`.
   * Confirm:

     * `tie_pixel_colors` loops over `c in range(C)` and calls `add_eq` with `[1, -1]` coeffs.
     * `fix_pixel_color` sets exactly `y[p,color] = 1` (no extra zeroing).
     * `forbid_pixel_color` sets `y[p,color] = 0`.
   * Check `add_one_hot_constraints` creates exactly `N` constraints, each summing `C` entries of y to 1.

2. **Run sanity runner:**

   * `python -m src.constraints.builder`
   * Confirm it prints: `builder.py sanity checks passed.` and no assertion errors.

3. **Optional integration test:**

   * Import `ConstraintBuilder` and `add_one_hot_constraints` in a separate test script.
   * For a tiny grid (e.g. H=1, W=2, C=3), compute N=2, build constraints, and check a few manually:

     * indices match expected `y_index(p,c)` values.

No schemas S1–S11 should be referenced here; this file is *pure generic constraint plumbing*.

---
