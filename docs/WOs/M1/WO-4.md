## 🔹 Work Order 4 – Object shape signatures and object_id map

**File (existing):**
`src/features/components.py`

We are **extending** this file:

* augmenting `Component` with `shape_signature`,
* adding two functions: `compute_shape_signature` and `assign_object_ids`,
* adding a small `__main__` test to validate.

**Libraries to use (and only these):**

* `numpy` as `np`
* `dataclasses.dataclass`, `dataclasses.field`
* `typing` (`List`, `Tuple`, `Dict`, `Any`)

No extra imports beyond what you already have unless needed.

---

### 1. Extend `Component` dataclass with `shape_signature`

In `src/features/components.py`, modify the existing `Component` definition to:

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Any

from src.core.grid_types import Grid, Pixel
import numpy as np
# ... plus scipy fallback stuff you already have ...

@dataclass
class Component:
    id: int
    color: int
    pixels: List[Pixel]
    size: int
    bbox: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
    shape_signature: Any = field(default=None)
```

**Notes:**

* `shape_signature` will be a tuple of relative coordinates, but we keep it typed as `Any` for flexibility.
* When constructing Components in `connected_components_by_color`, you can initially set `shape_signature=None`; it will be filled by `compute_shape_signature` later.

---

### 2. Implement `compute_shape_signature(comp: Component)`

Add **below** `connected_components_by_color` (or near it):

```python
def compute_shape_signature(comp: Component) -> Tuple[Tuple[int, int], ...]:
    """
    Compute a translation-invariant shape signature for a component.

    Steps:
      - Use comp.bbox = (r_min, r_max, c_min, c_max)
      - For each pixel (r,c) in comp.pixels:
          dr = r - r_min
          dc = c - c_min
      - Collect all (dr, dc) pairs
      - Sort them in lexicographic order
      - Return them as a tuple of (dr, dc) pairs

    This makes shapes equal up to translation (same pattern, different location).
    """
    (r_min, r_max, c_min, c_max) = comp.bbox
    rel_coords: List[Tuple[int, int]] = []
    for (r, c) in comp.pixels:
        dr = r - r_min
        dc = c - c_min
        rel_coords.append((dr, dc))

    rel_coords_sorted = sorted(rel_coords)
    signature: Tuple[Tuple[int, int], ...] = tuple(rel_coords_sorted)
    return signature
```

**No wiggle room:**

* Use `comp.bbox` values exactly.
* 0-based coordinates.
* Sort before returning.

You may choose to set `comp.shape_signature = compute_shape_signature(comp)` either inside `connected_components_by_color` (recommended) or just before using it in `assign_object_ids`.

---

### 3. Implement `assign_object_ids(components: List[Component])`

Add:

```python
def assign_object_ids(components: List[Component]) -> Dict[Pixel, int]:
    """
    Given a list of Components, compute shape_signature for each,
    group components with the same (color, shape_signature),
    and assign an object_id (0,1,2,...) per group.

    Returns:
      mapping from (r,c) -> object_id for all pixels in all components.
    """
    # First, ensure all components have shape_signature computed
    for comp in components:
        if comp.shape_signature is None:
            comp.shape_signature = compute_shape_signature(comp)

    # Group by (color, shape_signature)
    groups: Dict[Tuple[int, Tuple[Tuple[int,int], ...]], List[Component]] = {}
    for comp in components:
        key = (comp.color, comp.shape_signature)
        groups.setdefault(key, []).append(comp)

    # Assign object_ids
    pixel_to_object_id: Dict[Pixel, int] = {}
    current_object_id = 0
    for key, comps_in_group in groups.items():
        # all comps in this group share the same object_id
        for comp in comps_in_group:
            for (r, c) in comp.pixels:
                pixel_to_object_id[(r, c)] = current_object_id
        current_object_id += 1

    return pixel_to_object_id
```

**Conventions:**

* **Grouping key** is `(color, shape_signature)` — so two shapes in different colors are not the same object class.
* Object IDs are **0-based**, incrementing per group.
* Every pixel belonging to some component must appear in the returned dict with exactly one object_id.

---

### 4. Tiny self-test in `__main__`

At the bottom of `components.py` (extend existing test or replace it; keep it simple):

```python
if __name__ == "__main__":
    # Simple grid with two identical shapes in different places
    grid = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
    ], dtype=int)

    from src.core.grid_types import print_grid
    print("Grid:")
    print_grid(grid)

    comps = connected_components_by_color(grid)
    print("\nComponents:")
    for comp in comps:
        comp.shape_signature = compute_shape_signature(comp)
        print(f"  id={comp.id}, color={comp.color}, size={comp.size}, bbox={comp.bbox}, shape_sig={comp.shape_signature}")

    pixel_to_obj = assign_object_ids(comps)
    print("\nPixel to object_id:")
    # print object_id map in grid form for clarity
    H, W = grid.shape
    obj_grid = -np.ones_like(grid)
    for (r, c), oid in pixel_to_obj.items():
        obj_grid[r, c] = oid
    print(obj_grid)
```

**What this test should demonstrate:**

* The two 2×2 blocks of 1’s and 2’s are both squares; each color will be its own object type due to `(color, shape_sig)` key.
* You should see object IDs per color/shape group laid out in `obj_grid`.

---

### 5. Reviewer/tester instructions

1. **Run:**

   ```bash
   python -m src.features.components
   ```

2. **Check output:**

   * Printed grid should be:

     ```text
     Grid:
     0 1 1 0 0
     0 1 1 0 0
     0 0 0 2 2
     0 0 0 2 2
     ```

   * Components list should show:

     * At least one component for color 1 (the 2×2 block),
     * At least one component for color 2 (the 2×2 block),
     * shape_signature for the 1-block and 2-block should have the same pattern of (dr, dc) pairs, but grouped under different colors.

   * The `obj_grid` printed should show groups with distinct IDs per `(color, shape)`:

     For example, something like (exact IDs may differ, but pattern should be consistent):

     ```text
     [[-1  0  0 -1 -1]
      [-1  0  0 -1 -1]
      [-1 -1 -1  1  1]
      [-1 -1 -1  1  1]]
     ```

     Where:

     * object_id 0 assigned to the color 1 square,
     * object_id 1 assigned to the color 2 square,
     * -1 on background / color 0 pixels.

3. **Optional integration with a real ARC grid:**

   * Pick a small training grid via `load_arc_training_challenges`,
   * run `connected_components_by_color`,
   * then `assign_object_ids`,
   * inspect a few `(r,c)` points:

     * identical shapes in different locations with same color should share object_id,
     * obviously different shapes should get different ids.

---
