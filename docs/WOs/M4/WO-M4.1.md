## WO-M4.1 – LP/ILP solver wrapper (with thin runner)

### ✅ Libraries to use (no reinvention)

* **Optimization:** `pulp` (Python linear programming library)

  * `from pulp import LpProblem, LpMinimize, LpStatusOptimal, LpVariable, LpBinary, value`
* **Array handling:** `numpy` as `np`

Absolutely **no custom simplex / solver**. Only stitch these together.

---

## A. `src/solver/lp_solver.py`

### A.1 Purpose

Implement a **single-grid** solver:

* Takes:

  * a `ConstraintBuilder` (from M2),
  * `num_pixels` (N for this grid),
  * `num_colors` (C),
* Creates binary variables y[p,c] ∈ {0,1},
* Adds:

  * all constraints from the builder,
  * one-hot constraints per pixel,
* Solves an ILP with `pulp`,
* Returns a `(num_pixels, num_colors)` numpy array of 0/1.

We treat **one grid per call**. Multi-grid global LP is out-of-scope for this WO.

### A.2 Types and imports

At the top of `lp_solver.py`:

```python
from dataclasses import dataclass
from typing import List

import numpy as np
import pulp

from src.constraints.builder import ConstraintBuilder, LinearConstraint
from src.constraints.indexing import y_index_to_pc  # or equivalent
```

If your `indexing` module has slightly different names, adjust accordingly.

### A.3 Custom exception for infeasibility

Define:

```python
class InfeasibleModelError(Exception):
    """Raised when the ILP model is infeasible or not optimal."""
    pass
```

### A.4 Main API function

Define exactly this function:

```python
def solve_constraints_for_grid(
    builder: ConstraintBuilder,
    num_pixels: int,
    num_colors: int,
    objective: str = "min_sum"
) -> np.ndarray:
    """
    Build and solve an ILP for a single grid.

    Args:
        builder: ConstraintBuilder with collected LinearConstraint objects.
        num_pixels: number of pixels in this grid (H * W).
        num_colors: number of colors in the palette (C).
        objective: currently supports:
            - "min_sum": minimize sum of all y[p,c]
            - "none":    zero objective (feasibility only)

    Returns:
        y: numpy array of shape (num_pixels, num_colors), with entries 0 or 1.

    Raises:
        InfeasibleModelError: if the model is infeasible or no optimal solution is found.
    """
```

### A.5 Implementation details (step-by-step)

Inside `solve_constraints_for_grid`:

1. **Create model**

   ```python
   prob = pulp.LpProblem("arc_ilp", pulp.LpMinimize)
   ```

2. **Create binary variables y[p][c]**

   * Use a 2D structure:

     ```python
     y = [
         [pulp.LpVariable(f"y_{p}_{c}", lowBound=0, upBound=1, cat=LpBinary)
          for c in range(num_colors)]
         for p in range(num_pixels)
     ]
     ```

3. **Add constraints from `builder.constraints`**

   You must map each flat `y_index` back to `(p_idx, color)`.

   Assuming `y_index_to_pc(idx, num_colors, width_or_dummy)` returns `(p_idx, c)`; if your `indexing` API is slightly different, adapt accordingly.

   For each `LinearConstraint lc` in `builder.constraints`:

   ```python
   for lc in builder.constraints:
       assert len(lc.indices) == len(lc.coeffs)
       expr = 0
       for idx, coeff in zip(lc.indices, lc.coeffs):
           p_idx, color = y_index_to_pc(idx, num_colors)  # adapt signature
           expr += coeff * y[p_idx][color]
       prob += (expr == lc.rhs)
   ```

   **Important:** equality constraints only.

4. **Add one-hot constraints per pixel**

   Even if `builder` might have added some, **v0 always enforces them here**:

   ```python
   for p in range(num_pixels):
       prob += (sum(y[p][c] for c in range(num_colors)) == 1)
   ```

5. **Objective**

   Two modes:

   ```python
   if objective == "min_sum":
       prob += sum(y[p][c] for p in range(num_pixels) for c in range(num_colors))
   elif objective == "none":
       prob += 0
   else:
       raise ValueError(f"Unknown objective: {objective}")
   ```

6. **Solve using pulp’s CBC solver**

   ```python
   status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
   if pulp.LpStatus[status] != "Optimal":
       raise InfeasibleModelError(f"Solver status: {pulp.LpStatus[status]}")
   ```

7. **Extract solution into numpy array**

   ```python
   y_sol = np.zeros((num_pixels, num_colors), dtype=int)
   for p in range(num_pixels):
       for c in range(num_colors):
           val = pulp.value(y[p][c])
           # Guard against None / float noise
           y_sol[p, c] = 1 if val is not None and val > 0.5 else 0
   ```

8. **Optional sanity check**

   * Each pixel should be exactly one-hot:

     ```python
     assert np.all(y_sol.sum(axis=1) == 1), "One-hot constraint violated in solution"
     ```

   * You can keep this assertion; if it ever fails, it’s a bug.

9. **Return**

   ```python
   return y_sol
   ```

That’s the whole file: exception + this function.

---

## B. `src/runners/test_lp_solver.py` – thin runner / smoke test

This is for the reviewer/tester to verify the ILP wrapper in isolation, without touching real ARC yet.

### B.1 Purpose

* Build a tiny, artificial constraint system using `ConstraintBuilder`.
* Solve it with `solve_constraints_for_grid`.
* Check the solution is as expected.

### B.2 Imports

```python
import numpy as np

from src.constraints.builder import ConstraintBuilder
from src.constraints.indexing import y_index
from src.solver.lp_solver import solve_constraints_for_grid
```

Adjust `y_index` import to your actual function name.

### B.3 Test scenario: 2 pixels, 3 colors

We’ll make a super simple test:

* 2 pixels: p0, p1
* 3 colors: c0, c1, c2
* Constraints:

  * Pixel 0 must be color 1
  * Pixel 1 must have the same color as pixel 0

### B.4 Test code

In `test_lp_solver.py`:

```python
def build_simple_test_constraints() -> tuple[ConstraintBuilder, int, int]:
    num_pixels = 2
    num_colors = 3
    builder = ConstraintBuilder()

    # p0 = color 1
    p0_idx = 0
    color1 = 1
    # y_{p0,1} = 1, y_{p0,c≠1} = 0
    # We'll use add_eq with single terms or direct fix_pixel_color if you prefer.

    from src.constraints.builder import LinearConstraint

    # y_{p0,1} = 1
    idx_p0_c1 = y_index(p0_idx, color1, num_colors)
    builder.add_eq(indices=[idx_p0_c1], coeffs=[1.0], rhs=1.0)

    # y_{p0,0} = 0, y_{p0,2} = 0  (optional, one-hot will enforce this)
    idx_p0_c0 = y_index(p0_idx, 0, num_colors)
    idx_p0_c2 = y_index(p0_idx, 2, num_colors)
    builder.add_eq(indices=[idx_p0_c0], coeffs=[1.0], rhs=0.0)
    builder.add_eq(indices=[idx_p0_c2], coeffs=[1.0], rhs=0.0)

    # p1 has same color as p0:
    # For each color c: y_{p1,c} - y_{p0,c} = 0
    p1_idx = 1
    for c in range(num_colors):
        idx_p1_c = y_index(p1_idx, c, num_colors)
        idx_p0_c = y_index(p0_idx, c, num_colors)
        builder.add_eq(
            indices=[idx_p1_c, idx_p0_c],
            coeffs=[1.0, -1.0],
            rhs=0.0
        )

    return builder, num_pixels, num_colors


def test_simple_ilp():
    builder, num_pixels, num_colors = build_simple_test_constraints()
    y_sol = solve_constraints_for_grid(builder, num_pixels, num_colors, objective="min_sum")

    assert y_sol.shape == (num_pixels, num_colors)
    # pixel 0 must be [0,1,0]
    assert np.array_equal(y_sol[0], np.array([0,1,0]))
    # pixel 1 must be same as pixel 0
    assert np.array_equal(y_sol[1], np.array([0,1,0]))
    print("test_simple_ilp: OK")


if __name__ == "__main__":
    test_simple_ilp()
```

### B.5 How reviewers/testers should validate

* From repo root, run:

  ```bash
  python -m src.runners.test_lp_solver
  ```

* Expected:

  * No exceptions,
  * Console output: `test_simple_ilp: OK`

* If an exception is raised (InfeasibleModelError, assertion, etc.), they know:

  * either constraint building is wrong,
  * or the solver wrapper has a bug.

Once this passes, we know:

* `ConstraintBuilder` → `solve_constraints_for_grid` pipeline is working.
* We’re ready to plug it into `kernel.py` and later decoding.

---