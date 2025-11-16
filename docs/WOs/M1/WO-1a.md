## 🔹 Work Order 1a – Core grid types, indexing, and pretty-print

**Files:**

* `src/core/grid_types.py`

**Goal:**
Define a standard `Grid` representation and a few core utilities (index mapping + printing) that everything else will reuse. No ARC JSON here yet.

**Libraries to use (and only these):**

* `numpy` (as `np`)
* `typing` (`TypeAlias`, `Tuple`)
* `dataclasses` (for future structs if needed, but keep minimal here)

**Tasks for implementer (Claude Code):**

1. **Define Grid type alias**

   ```python
   import numpy as np
   from typing import TypeAlias, Tuple

   Grid: TypeAlias = np.ndarray  # shape: (H, W), dtype: int
   Pixel: TypeAlias = Tuple[int, int]  # (row, col)
   ```

   * Add docstring: Grid must always be `dtype=int` and shape `(H, W)`.

2. **Pixel index helpers (row-major)**

   Implement exactly:

   ```python
   def pixel_index(r: int, c: int, width: int) -> int:
       """
       Map (row r, col c) to a flat index idx in [0, H*W-1], row-major.
       idx = r * width + c
       """
       ...

   def index_to_pixel(idx: int, width: int) -> Tuple[int, int]:
       """
       Inverse of pixel_index: given flat idx and width, return (row, col).
       r = idx // width, c = idx % width
       """
       ...
   ```

   * Enforce integer arithmetic, no tricks.
   * Raise a `ValueError` if `idx < 0`.

3. **Pretty-print function for debugging**

   Implement:

   ```python
   def print_grid(grid: Grid) -> None:
       """
       Print a small ASCII representation of the grid:
       each row on its own line, space-separated integers.
       """
       ...
   ```

   * Requirements:

     * Assert `grid.ndim == 2`.
     * Convert values to ints and print them with spaces, e.g.:

       ```
       0 0 1
       2 3 3
       0 0 0
       ```

4. **Tiny self-test block**

   At the bottom of `grid_types.py`:

   ```python
   if __name__ == "__main__":
       import numpy as np

       grid = np.array([[0, 1], [2, 3]], dtype=int)
       print("Grid:")
       print_grid(grid)

       width = grid.shape[1]
       for r in range(grid.shape[0]):
           for c in range(width):
               idx = pixel_index(r, c, width)
               r2, c2 = index_to_pixel(idx, width)
               assert (r, c) == (r2, c2), f"Roundtrip failed at {(r,c)}"
       print("Index roundtrip test passed.")
   ```

   * This ensures index logic is correct.

**Reviewer/tester instructions (human or test agent):**

* Run:

  ```bash
  python -m src.core.grid_types
  ```

* Check:

  * The printed grid matches:

    ```
    Grid:
    0 1
    2 3
    Index roundtrip test passed.
    ```

  * No stack traces.

  * `Grid` is used consistently as `np.ndarray` with `dtype=int`.

No ARC data involved yet; this is pure, math-spec-compliant grid representation.

---

