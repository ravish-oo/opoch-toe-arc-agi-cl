## 🔹 Work Order 3 – Connected components per color

**File:**
`src/features/components.py`

**Goal:**
Implement per-color connected components and basic stats (size, bbox), exactly as in the math spec 1.2.4, using **4-connectivity**.

**Libraries to use:**

* `numpy` as `np`
* `dataclasses.dataclass`
* `typing` (`List`, `Tuple`)
* Prefer **`scipy.ndimage.label`** for component labeling; only if SciPy is truly not available, fall back to a simple BFS/DFS.

**Imports (fixed):**

At top of `components.py`:

```python
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.core.grid_types import Grid, Pixel
```

Try SciPy:

```python
try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None
```

---

### 1. Define `Component` dataclass

Exactly:

```python
@dataclass
class Component:
    id: int
    color: int
    pixels: List[Pixel]
    size: int
    bbox: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
```

**Conventions:**

* `id`: unique integer for each component across all colors in the grid (e.g. 0,1,2,...).
* `color`: the color value (int in [0..C-1]).
* `pixels`: list of `(row, col)` tuples (0-based).
* `size`: number of pixels in this component (len(pixels)).
* `bbox`: `(r_min, r_max, c_min, c_max)`, all 0-based, inclusive.

---

### 2. Implement `connected_components_by_color`

Signature:

```python
def connected_components_by_color(grid: Grid) -> List[Component]:
    """
    Compute 4-connected components for each distinct color in the grid.

    For each color value v in the grid:
      - Create a binary mask (grid == v).
      - Run 4-connected component labeling on that mask.
      - For each non-empty component:
          - collect all pixels (r,c),
          - compute size and bbox,
          - create a Component object.

    Components across all colors are returned in a flat list,
    with monotonically increasing ids (0,1,2,...).
    """
    ...
```

**Detailed requirements:**

1. **Grid assumptions:**

   * `grid.ndim == 2`
   * `grid.dtype` is integer-like (as defined in `Grid`).

2. **Distinct colors:**

   * Use `np.unique(grid)` to get all color values (ints).
   * Iterate colors in ascending order.

3. **Labeling with SciPy (preferred path):**

   * If `ndi is not None`:

     * Create a mask: `mask = (grid == color)` (bool array).

     * Use 4-connectivity structure:

       ```python
       structure = np.array([[0,1,0],
                             [1,1,1],
                             [0,1,0]], dtype=int)
       ```

     * Call: `labels, num = ndi.label(mask, structure=structure)`

     * For `label_id` in `1..num`:

       * `coords = np.argwhere(labels == label_id)` → array of shape (k,2)

       * If `coords.size == 0`, skip (should not happen, but safe).

       * Extract pixels:

         ```python
         pixels = [(int(r), int(c)) for r, c in coords]
         ```

       * Compute:

         ```python
         rows = coords[:,0]
         cols = coords[:,1]
         r_min, r_max = int(rows.min()), int(rows.max())
         c_min, c_max = int(cols.min()), int(cols.max())
         size = len(pixels)
         ```

       * Create `Component(id=component_id, color=color, pixels=pixels, size=size, bbox=(r_min, r_max, c_min, c_max))`

       * Increment `component_id` each time.

4. **Fallback without SciPy:**

   * If `ndi is None`, implement a **simple 4-connected BFS**:

     * For each color:

       * Mask = (grid == color).
       * Maintain a visited boolean mask same shape.
       * For each pixel `(r,c)` where mask is True and not visited:

         * BFS on 4 neighbors (up/down/left/right) constrained to grid bounds and `mask==True`.
         * Collect pixels in a list.
         * Compute bbox and size exactly as above.
         * Create a `Component`.
   * Keep BFS implementation straightforward and short; no cleverness.

5. **Return value:**

   * Flat `List[Component]`, ordered by:

     * color in ascending order,
     * then label index in ascending order.
   * Component ids must be 0,1,2,... in that order.

---

### 3. Tiny self-test in `__main__`

At bottom of `components.py`:

```python
if __name__ == "__main__":
    # Simple test grid with 2 colors and a few blobs
    grid = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [0, 0, 2, 1],
        [3, 3, 3, 1],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    comps = connected_components_by_color(grid)
    print("\nComponents found:")
    for comp in comps:
        print(f"  id={comp.id}, color={comp.color}, size={comp.size}, bbox={comp.bbox}")
```

No extra tests, no stubs, just this.

---

### 4. Reviewer/tester instructions

1. **Run module directly:**

   ```bash
   python -m src.features.components
   ```

2. **Check output:**

   * It prints the test grid:

     ```text
     Grid:
     0 0 1 1
     0 2 2 1
     0 0 2 1
     3 3 3 1
     ```

   * Then a listing of components, e.g. (exact order may vary, but should be consistent with our convention):

     ```text
     Components found:
       id=0, color=0, size=5, bbox=(0, 2, 0, 1)
       id=1, color=1, size=4, bbox=(0, 3, 2, 3)
       id=2, color=2, size=3, bbox=(1, 2, 1, 2)
       id=3, color=3, size=3, bbox=(3, 3, 0, 2)
     ```

     (Sizes/bboxes should match the blobs visually.)

3. **SciPy path:**

   * If SciPy is installed:

     * Confirm it is actually using `ndi.label` (you can temporarily add a `print("Using scipy.ndimage.label")` to verify, then remove it).
   * If SciPy is not installed:

     * Ensure fallback BFS still finds components with correct sizes and bboxes.

4. **Optional integration check:**

   * Load a real ARC training grid with `load_arc_training_challenges` from `arc_io`,
   * call `connected_components_by_color` on one train grid,
   * verify it returns a non-empty list of components and that all returned pixels correspond to colors in the grid.

---
