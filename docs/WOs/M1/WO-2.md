## 🔹 Work Order 2 – Coordinate, bands, and border features

**File:**
`src/features/coords_bands.py`

**Goal:**
Implement simple coordinate-based φ features exactly as in the math spec:

* row / col,
* row/col residues mod {2,3,4,5},
* row/col bands (top/middle/bottom, left/middle/right),
* global border mask.

This module is small enough to be a single unit (<200 LOC).

**Libraries to use (and only these):**

* `numpy` as `np`
* `typing` (`Dict`, `Tuple`)
* `from src.core.grid_types import Grid, Pixel`

No custom algorithms beyond basic loops / numpy ops.

---

### 1. Module structure & imports

At top of `src/features/coords_bands.py`:

```python
from typing import Dict, Tuple
import numpy as np

from src.core.grid_types import Grid, Pixel
```

---

### 2. `coord_features(grid: Grid)`

**Signature:**

```python
def coord_features(grid: Grid) -> Dict[Pixel, Dict]:
    """
    For each pixel (r,c) in the grid, return a nested dict of coordinate features:
      - "row": int
      - "col": int
      - "row_mod": {2: r % 2, 3: r % 3, 4: r % 4, 5: r % 5}
      - "col_mod": {2: c % 2, 3: c % 3, 4: c % 4, 5: c % 5}

    Keys are (r,c) tuples. Grid indices are 0-based.
    """
    ...
```

**Requirements:**

* Assume `grid.ndim == 2`.
* H, W = `grid.shape`.
* Return type: `Dict[Tuple[int,int], Dict]`.
* Use 0-based indexing for row/col and for mod results.
* Implementation can be simple nested loops; no need to over-vectorize.

---

### 3. `row_band_labels(H: int)`

**Signature:**

```python
def row_band_labels(H: int) -> Dict[int, str]:
    """
    Assign each row index r in [0, H-1] to a band: 'top', 'middle', or 'bottom'.

    Convention (fixed, deterministic):

      - If H == 1:
          row 0 -> 'middle'
      - If H == 2:
          row 0 -> 'top'
          row 1 -> 'bottom'
      - If H >= 3:
          row 0         -> 'top'
          row H-1       -> 'bottom'
          all rows 1..H-2 -> 'middle'
    """
    ...
```

**Requirements:**

* MUST follow the convention above exactly (no thirds splitting).
* Return a dict: `r -> "top"/"middle"/"bottom"` for every `r` in `0..H-1`.

---

### 4. `col_band_labels(W: int)`

**Signature:**

```python
def col_band_labels(W: int) -> Dict[int, str]:
    """
    Assign each column index c in [0, W-1] to a band: 'left', 'middle', or 'right'.

    Convention (fixed, deterministic):

      - If W == 1:
          col 0 -> 'middle'
      - If W == 2:
          col 0 -> 'left'
          col 1 -> 'right'
      - If W >= 3:
          col 0         -> 'left'
          col W-1       -> 'right'
          all cols 1..W-2 -> 'middle'
    """
    ...
```

**Requirements:**

* MUST follow this convention exactly.
* Return a dict: `c -> "left"/"middle"/"right"` for every `c` in `0..W-1`.

---

### 5. `border_mask(grid: Grid)`

**Signature:**

```python
def border_mask(grid: Grid) -> np.ndarray:
    """
    Return a boolean mask of shape (H, W) where True indicates
    that the pixel is on the outer border of the grid:

      - r == 0 or r == H-1 or c == 0 or c == W-1
    """
    ...
```

**Requirements:**

* `mask.dtype` must be `bool`.
* `mask.shape` == `grid.shape`.
* Implement with straightforward numpy logic (`np.zeros(shape, dtype=bool)` and set True where needed, or via broadcasting).

---

### 6. Small self-test in `__main__`

At bottom of the file:

```python
if __name__ == "__main__":
    # Simple smoke test on a toy 4x5 grid
    import numpy as np

    grid = np.arange(20, dtype=int).reshape(4, 5)
    print("Grid:")
    from src.core.grid_types import print_grid
    print_grid(grid)

    H, W = grid.shape

    cf = coord_features(grid)
    print("\nFeatures for a few pixels:")
    for (r, c) in [(0,0), (0,W-1), (H-1,0), (H-1,W-1)]:
        print(f"  (r={r}, c={c}):", cf[(r,c)])

    row_bands = row_band_labels(H)
    col_bands = col_band_labels(W)
    print("\nRow bands:", row_bands)
    print("Col bands:", col_bands)

    bm = border_mask(grid)
    print("\nBorder mask (True on border pixels):")
    print(bm.astype(int))  # print as 0/1 for clarity
```

---

### Reviewer/tester instructions

1. **Run module directly:**

   ```bash
   python -m src.features.coords_bands
   ```

2. **Check output:**

   * It should print a 4x5 grid (0..19) via `print_grid`.
   * It should print coord features for the four corners:

     * rows and cols (0-based),
     * row_mod / col_mod with keys {2,3,4,5}.
   * It should print row bands mapping like:

     * `0: 'top', 1: 'middle', 2: 'middle', 3: 'bottom'` for H=4.
   * It should print col bands mapping like:

     * `0: 'left', 1: 'middle', 2: 'middle', 3: 'middle', 4: 'right'` for W=5.
   * Border mask printed as 0/1 should have:

     * 1s on all edges (first/last row, first/last col),
     * 0s in strictly interior pixels.

3. **(Optional integration check):**

   * In a REPL or a small script, load one ARC training task using `load_arc_training_challenges` from `arc_io`, take its first train grid, and run:

     * `coord_features(grid)`,
     * `row_band_labels(grid.shape[0])`,
     * `col_band_labels(grid.shape[1])`,
     * `border_mask(grid)`.
   * Confirm nothing crashes and shapes are correct.

---