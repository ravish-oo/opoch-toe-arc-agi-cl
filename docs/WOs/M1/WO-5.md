## 🔹 Work Order 5 – Line features & neighborhood hashes

**File:**
`src/features/neighborhoods.py`

**Goal:**
Implement per-row / per-column nonzero flags and per-pixel 3×3 (or general (2r+1)×(2r+1)) neighborhood hashes, as per the math spec (line features + local pattern hashes).

**Libraries to use (and only these):**

* `numpy` as `np`
* `typing` (`Dict`, `Tuple`)
* `from src.core.grid_types import Grid, Pixel`

No custom algorithms beyond basic loops / numpy slicing.

---

### 1. Module imports

At top of `src/features/neighborhoods.py`:

```python
from typing import Dict, Tuple
import numpy as np

from src.core.grid_types import Grid, Pixel
```

---

### 2. `row_nonzero_flags(grid: Grid)`

**Signature:**

```python
def row_nonzero_flags(grid: Grid) -> Dict[int, bool]:
    """
    For each row r in [0, H-1], return True if any cell in that row != 0.

    Returns:
      dict mapping row index r -> bool
    """
    ...
```

**Requirements:**

* Assume `grid.ndim == 2`.
* Use **numpy** operations, no manual nested loops needed:

  * You can use `(grid != 0).any(axis=1)` to get a 1D boolean array of length H.
* Convert to a dict `{r: bool_value}` for all rows `0..H-1`.

---

### 3. `col_nonzero_flags(grid: Grid)`

**Signature:**

```python
def col_nonzero_flags(grid: Grid) -> Dict[int, bool]:
    """
    For each column c in [0, W-1], return True if any cell in that column != 0.

    Returns:
      dict mapping column index c -> bool
    """
    ...
```

**Requirements:**

* Use numpy, similar to rows:

  * `(grid != 0).any(axis=0)` gives a 1D boolean array of length W.
* Convert to dict `{c: bool_value}`.

---

### 4. `neighborhood_hashes(grid: Grid, radius: int = 1)`

**Signature:**

```python
def neighborhood_hashes(grid: Grid, radius: int = 1) -> Dict[Pixel, int]:
    """
    For each pixel (r,c) in the grid, compute a 'hash' of its local neighborhood.

    Neighborhood definition:
      - Neighborhood size = (2*radius + 1) x (2*radius + 1)
      - Centered at (r, c)
      - If (r+dr, c+dc) is out of bounds, use sentinel value -1 at that position.

    Procedure:
      - Build a small 2D patch for each pixel with these rules.
      - Flatten the patch row-major into a 1D list of ints.
      - Convert to a tuple and pass it to Python's built-in hash(...) to get an int.

    Returns:
      dict mapping (r,c) -> hash_value (int)
    """
    ...
```

**Precise behavior:**

* Assume `grid.ndim == 2`, `grid.dtype` is integer-like.
* Let `H, W = grid.shape`.
* For each pixel `(r,c)` with `0 <= r < H`, `0 <= c < W`:

  * Build a patch of shape `(K, K)` where `K = 2*radius + 1`.
  * For `dr` in `[-radius..+radius]`, `dc` in `[-radius..+radius]`:

    * Compute `rr = r + dr`, `cc = c + dc`.
    * If `0 <= rr < H` and `0 <= cc < W`:

      * Use `grid[rr, cc]`.
    * Else:

      * Use sentinel value `-1`.
  * Flatten patch row-major to a list of ints.
  * Make `patch_tuple = tuple(flat_list)`.
  * Compute `h = hash(patch_tuple)` (**Python built-in hash** is fine; we only require equality within a single run).
  * Set `result[(r,c)] = h`.

**No wiggle room:**

* Sentinel **must** be `-1`.
* Flatten order is row-major (iterate rows, then columns).
* Use `hash(tuple)` exactly as described.

---

### 5. Self-test in `__main__`

At bottom of `neighborhoods.py`:

```python
if __name__ == "__main__":
    # Small test grid with repeated local patterns
    grid = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    row_flags = row_nonzero_flags(grid)
    col_flags = col_nonzero_flags(grid)
    print("\nRow nonzero flags:", row_flags)
    print("Col nonzero flags:", col_flags)

    hashes = neighborhood_hashes(grid, radius=1)
    print("\nNeighborhood hashes (radius=1) for a few pixels:")

    # Choose two interior pixels with same 3x3 pattern and verify hashes equal
    p1 = (0, 1)  # top-left '1' in the 2x2 block
    p2 = (1, 2)  # bottom-right '1' in the same 2x2 block
    print(f"  hash{p1} =", hashes[p1])
    print(f"  hash{p2} =", hashes[p2])
    print("  hashes equal? ->", hashes[p1] == hashes[p2])
```

You can pick other positions if you prefer; the key is that some **visually identical** local pattern yields the same hash.

---

### 6. Reviewer/tester instructions

1. **Run the module directly:**

   ```bash
   python -m src.features.neighborhoods
   ```

2. **Check output:**

   * It should print the 4x4 grid as:

     ```text
     Grid:
     0 1 1 0
     0 1 1 0
     2 2 0 0
     2 2 0 0
     ```

   * It should print reasonable `row_nonzero_flags` and `col_nonzero_flags`:

     * e.g. each row that has any non-zero value is `True`.

   * It should print hashes for the two chosen pixels and confirm equality:

     ```text
       hash(0, 1) = <int>
       hash(1, 2) = <int>
       hashes equal? -> True
     ```

3. **Optional integration check with ARC data:**

   * Load one training grid via `load_arc_training_challenges` from `src/core/arc_io`.
   * Pick a medium-sized grid (e.g. from the first training task).
   * Run:

     * `row_nonzero_flags(grid)`
     * `col_nonzero_flags(grid)`
     * `neighborhood_hashes(grid)`
   * Confirm:

     * row/col flags align with visual non-zero patterns,
     * `len(hashes) == H*W`,
     * no errors.

---