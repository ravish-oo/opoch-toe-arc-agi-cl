## 🔹 Work Order 6 – Component-relative roles & sectors

**File:**
`src/features/object_roles.py`

**Goal:**
Implement higher-level features derived from components, as per the math spec:

* per-pixel quadrant/sector within each component’s bounding box,
* per-pixel interior vs border within each component,
* per-component role bits: `is_small`, `is_big`, `is_unique_shape`.

**Libraries to use (and only these):**

* `numpy` as `np`
* `typing` (`Dict`, `Tuple`, `List`)
* `from src.core.grid_types import Grid, Pixel`
* `from src.features.components import Component, connected_components_by_color, compute_shape_signature`

No external libs, no fancy algorithms — pure Python + numpy.

---

### 0. Module imports

At top of `src/features/object_roles.py`:

```python
from typing import Dict, Tuple, List
import numpy as np

from src.core.grid_types import Grid, Pixel
from src.features.components import Component, connected_components_by_color, compute_shape_signature
```

---

### 6.1 `component_sectors(components: List[Component])`

**Signature:**

```python
def component_sectors(
    components: List[Component]
) -> Dict[Pixel, Dict[str, str]]:
    """
    For each pixel (r,c) in each component, assign sector labels
    relative to that component's bounding box.

    Returns:
      a dict mapping (r,c) -> {
        "vert_sector": "top" | "center" | "bottom",
        "horiz_sector": "left" | "center" | "right"
      }
    """
    ...
```

**Exact behavior:**

* For each `Component comp` with `bbox = (r_min, r_max, c_min, c_max)`:

  * Let `h = r_max - r_min + 1` (component height).
  * Let `w = c_max - c_min + 1` (component width).

* **Vertical sector (`vert_sector`):**

  * If `h == 1`:

    * All pixels → `"center"`.
  * If `h == 2`:

    * row `r_min` → `"top"`,
    * row `r_max` → `"bottom"`.
  * If `h >= 3`:

    * row `r_min` → `"top"`,
    * row `r_max` → `"bottom"`,
    * all rows between → `"center"`.

* **Horizontal sector (`horiz_sector`):**

  * If `w == 1`:

    * All pixels → `"center"`.
  * If `w == 2`:

    * col `c_min` → `"left"`,
    * col `c_max` → `"right"`.
  * If `w >= 3`:

    * col `c_min` → `"left"`,
    * col `c_max` → `"right"`,
    * all cols between → `"center"`.

* For each `(r,c) in comp.pixels`:

  * Determine its `vert_sector` and `horiz_sector` using the above logic.
  * Store:

    ```python
    result[(r, c)] = {
        "vert_sector": vert_sector,
        "horiz_sector": horiz_sector,
    }
    ```

**No wiggle room:** use the bounding box of each component, not the whole grid; follow the above rules exactly.

---

### 6.2 `component_border_interior(grid: Grid, components: List[Component])`

**Signature:**

```python
def component_border_interior(
    grid: Grid,
    components: List[Component]
) -> Dict[Pixel, Dict[str, bool]]:
    """
    For each pixel (r,c) in each component, mark whether it is
    'border' or 'interior' with respect to that component.

    A pixel is 'interior' if all 4-connected neighbors of same
    color are also in the component. Otherwise it's 'border'.

    Returns:
      (r,c) -> {
        "is_border": bool,
        "is_interior": bool
      }
    """
    ...
```

**Exact behavior:**

* Build a lookup `(r,c) -> comp_id` for all pixels in all components:

  ```python
  pixel_to_comp_id: Dict[Pixel, int] = {}
  for comp in components:
      for (r, c) in comp.pixels:
          pixel_to_comp_id[(r, c)] = comp.id
  ```

* Get `H, W = grid.shape`.

* For each `comp` in `components`, for each `(r,c)` in `comp.pixels`:

  * Initialize `is_border = False`.

  * For each 4-connected neighbor:

    ```python
    neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    ```

    For each `(rr, cc)` in neighbors:

    * If `(rr, cc)` is out of grid bounds → this pixel is **border**.
    * Else if `grid[rr, cc] != comp.color` → border.
    * Else if `pixel_to_comp_id.get((rr, cc)) != comp.id` → border.
    * If any of these conditions hold, mark `is_border = True` and break.

  * After checking neighbors:

    * `is_interior = not is_border`.
    * Store:

      ```python
      result[(r, c)] = {
          "is_border": is_border,
          "is_interior": is_interior,
      }
      ```

* This ensures:

  * Each pixel in components is either border or interior, never both.

This is the per-component version of interior/border needed for S10.

---

### 6.3 `component_role_bits(components: List[Component])`

**Signature:**

```python
def component_role_bits(
    components: List[Component]
) -> Dict[int, Dict[str, bool]]:
    """
    Assign simple 'role bits' to each component.id:

      - is_small: size in lowest third of sizes
      - is_big:   size in highest third of sizes
      - is_unique_shape: this (color, shape_signature) occurs only once

    Returns:
      comp_id -> {
        "is_small": bool,
        "is_big": bool,
        "is_unique_shape": bool
      }
    """
    ...
```

**Exact behavior:**

1. **Compute size thresholds:**

   * Let `sizes = sorted([comp.size for comp in components])`.

   * If `components` is empty, return `{}`.

   * Compute:

     ```python
     n = len(sizes)
     small_idx = n // 3          # integer division
     big_idx = (2 * n) // 3
     small_cutoff = sizes[small_idx]   # lowest third
     big_cutoff = sizes[big_idx]       # highest third
     ```

   * This is a simple rank-based cutoff; don’t overcomplicate.

2. **Compute shape uniqueness per (color, shape_signature):**

   * Ensure `shape_signature` is computed:

     ```python
     for comp in components:
         if comp.shape_signature is None:
             comp.shape_signature = compute_shape_signature(comp)
     ```

   * Count occurrences:

     ```python
     from collections import Counter
     key_counts = Counter((comp.color, comp.shape_signature) for comp in components)
     ```

3. **Assign role bits per component.id:**

   * For each `comp`:

     ```python
     size = comp.size
     key = (comp.color, comp.shape_signature)
     is_small = size <= small_cutoff
     is_big = size >= big_cutoff
     is_unique_shape = key_counts[key] == 1
     roles[comp.id] = {
         "is_small": is_small,
         "is_big": is_big,
         "is_unique_shape": is_unique_shape,
     }
     ```

* Return `roles: Dict[int, Dict[str, bool]]`.

This gives you the role flags the math spec calls “role bits”.

---

### 6.4 Self-test / thin runner in `__main__`

At the bottom of `object_roles.py`:

```python
if __name__ == "__main__":
    # Simple grid with two components of the same shape and one different
    grid = np.array([
        [0, 1, 1, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 2],
        [3, 3, 0, 0, 0],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    # Get components first
    comps = connected_components_by_color(grid)
    print("\nComponents:")
    for comp in comps:
        print(f"  id={comp.id}, color={comp.color}, size={comp.size}, bbox={comp.bbox}")

    # Sectors
    sectors = component_sectors(comps)
    print("\nSectors for a few pixels:")
    sample_pixels = list(sectors.keys())[:5]
    for p in sample_pixels:
        print(f"  {p}: {sectors[p]}")

    # Border vs interior
    border_info = component_border_interior(grid, comps)
    print("\nBorder/interior flags for a few pixels:")
    for p in sample_pixels:
        print(f"  {p}: {border_info[p]}")

    # Role bits
    roles = component_role_bits(comps)
    print("\nComponent role bits:")
    for comp in comps:
        print(f"  comp.id={comp.id}: {roles[comp.id]}")
```

This doesn’t need to be exhaustive; just enough to smoke-test behavior.

---

### Reviewer/tester instructions

1. **Run directly:**

   ```bash
   python -m src.features.object_roles
   ```

2. **Check output qualitatively:**

   * Printed grid should show 3–4 blobs of colors 1,2,3.
   * `Components:` section:

     * IDs, colors, sizes, and bboxes should match the visual blobs.
   * `Sectors for a few pixels:`:

     * For pixels in the same component, sectors should be consistent with their position in that component’s bbox (e.g., top-left part → `"top"/"left"`, etc.).
   * `Border/interior flags:`:

     * Pixels at the outer edges of a blob should be `"is_border": True`.
     * Pixels fully surrounded by same-color neighbors (if any) should be `"is_interior": True`.

3. **Role bits sanity:**

   * `component_role_bits`:

     * You should see some components flagged as `is_small=True`, some as `is_big=True`, according to relative sizes.
     * Components whose `(color, shape_signature)` pattern appears only once should have `is_unique_shape=True`.

4. **Optional ARC integration:**

   * Load a real training grid with `load_arc_training_challenges`.
   * Run:

     * `components = connected_components_by_color(grid)`
     * `sectors = component_sectors(components)`
     * `border_info = component_border_interior(grid, components)`
     * `roles = component_role_bits(components)`
   * Confirm that:

     * No errors occur,
     * `sectors`, `border_info`, and `roles` all cover the relevant pixels / component IDs.

---
