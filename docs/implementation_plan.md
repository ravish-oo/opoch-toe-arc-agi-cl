## Overall sequence:

> **operators → law families → schema instances → builder functions → ConstraintBuilder → solver → Pi-agent loop**

---

## 1. High-level milestones (sequential)

We don’t implement all now, just to show how operators fit.

1. **Operator / feature library (φ)**

   * Grid representation, basic utilities.
   * All feature extractors: components, bands, hashes, etc.

2. **Constraint representation & ConstraintBuilder**

   * Data structures for linear constraints on y.
   * Primitive methods like `tie_pixel_colors`, `fix_pixel_color`, etc.

3. **Schema family builders (S1–S11)**

   * For each S_k, a `build_Sk_constraints(...)` that uses operators + ConstraintBuilder.

4. **Solver integration**

   * Connect to a standard LP/ILP library (`pulp`, `ortools`, or `cvxpy`).
   * Encode constraints and solve for y.

5. **Task IO + test harness**

   * Load ARC tasks, run full pipeline on training pairs, compare outputs.

6. **Pi-agent orchestration layer**

   * LLM decides which schema families + params to use, calls the pipeline, interprets results.

For now we only care about **Milestone 1**.

---

## 2. Milestone 1 – Operator / feature library (φ)

Goal: a clean Python module (or few small modules) that exposes a set of **feature functions** over grids, using standard libs, with minimal reinvented algorithms.

Below are **work orders** you can hand to Claude Code as separate tasks.

---

### Work Order 1 – Core grid data structures & IO ✅ COMPLETE

**Goal:** basic `Grid` representation and helpers, plus simple ARC JSON loading.

**Use:** `numpy`, `pathlib`, `json`.

**Tasks:**

* Define `Grid` as `np.ndarray` of shape `(H, W)`, `dtype=int`.
* Implement:

  * `load_arc_task(path) -> dict` that reads the ARC JSON and returns a `{ "train": [...], "test": [...] }` structure with `Grid`s.
  * `print_grid(grid: Grid)` for debugging (small ASCII view).
* Implement pixel index helpers:

  * `pixel_index(r, c, W) -> int` and `index_to_pixel(idx, W) -> (r, c)`.
* Keep this under ~150 lines, with a few tiny tests (e.g. roundtrip index↔pixel).

---

### Work Order 2 – Coordinate, bands, and border features ✅ COMPLETE

**Goal:** implement simple coordinate-based φ features.

**Use:** `numpy`.

**Tasks:**

* Given a `Grid`, implement functions:

  ```python
  def coord_features(grid: Grid) -> dict[tuple[int,int], dict]:
      # returns per-pixel dict with:
      #  "row", "col",
      #  "row_mod": {2,3,4,5},
      #  "col_mod": {2,3,4,5}
  ```

  ```python
  def row_band_labels(H: int) -> dict[int, str]:
      # row -> "top"/"middle"/"bottom"
  ```

  ```python
  def col_band_labels(W: int) -> dict[int, str]:
      # col -> "left"/"middle"/"right"
  ```

  ```python
  def border_mask(grid: Grid) -> np.ndarray[bool]:
      # True for pixels on the outer border of the grid
  ```

* Add a small `if __name__ == "__main__":` test that constructs a toy grid and prints these features for a few pixels.

---

### Work Order 3 – Connected components per color ✅ COMPLETE

**Goal:** get connected components and basic stats (size, bbox).

**Use:** `numpy` + `scipy.ndimage` (`label`)
(if SciPy is not available, use a simple BFS/DFS, but explicitly tell Claude to **prefer scipy.ndimage.label**).

**Tasks:**

* Define a `Component` dataclass:

  ```python
  @dataclass
  class Component:
      id: int
      color: int
      pixels: list[tuple[int,int]]
      size: int
      bbox: tuple[int,int,int,int]  # (r_min, r_max, c_min, c_max)
  ```

* Implement:

  ```python
  def connected_components_by_color(grid: Grid) -> list[Component]:
      # for each distinct color in grid:
      #   create a binary mask (grid == color)
      #   run scipy.ndimage.label(mask) to get connected labels
      #   for each label, collect pixels, size, bbox
      # return flat list of Component objects
  ```

* For now use 4-connectivity (up/down/left/right).

* Add a tiny test: small grid with 2–3 blobs, print component sizes and bboxes.

---

### Work Order 4 – Object shape signatures and object_id map ✅ COMPLETE

**Goal:** group components into “objects” up to translation (shape equivalence).

**Use:** `numpy`, pure Python.

**Tasks:**

* Extend `Component` to include `shape_signature: Any`.

* Implement:

  ```python
  def compute_shape_signature(comp: Component) -> tuple:
      # normalize component pixels by:
      #   - take all (r,c), subtract (r_min, c_min)
      #   - sort the relative coords
      #   - return as a tuple of (dr, dc) pairs
  ```

* Implement:

  ```python
  def assign_object_ids(components: list[Component]) -> dict[tuple[int,int], int]:
      """
      Given components, compute shape_signature for each,
      group components with same (color, shape_signature),
      assign an object_id (0,1,2,...) per group.
      Return a dict: (r,c) -> object_id.
      """
  ```

* Add a quick test: two separate identical shapes in different places should get the same object_id.

---

### Work Order 5 – Line features & neighborhood hashes

**Goal:** get per-row/col flags and 3×3 local pattern hashes.

**Use:** `numpy`.

**Tasks:**

* Implement:

  ```python
  def row_nonzero_flags(grid: Grid) -> dict[int, bool]:
      # row -> True if any cell != 0
  ```

  ```python
  def col_nonzero_flags(grid: Grid) -> dict[int, bool]:
      # col -> True if any cell != 0
  ```

* Implement 3×3 neighborhood hash:

  ```python
  def neighborhood_hashes(grid: Grid, radius: int = 1) -> dict[tuple[int,int], int]:
      """
      For each pixel (r,c), extract a (2*radius+1)x(2*radius+1) window,
      pad with a sentinel (e.g. -1) at borders,
      flatten to a tuple and hash (e.g. via Python's hash() or a stable tuple).
      Return dict[(r,c)] = hash_value.
      """
  ```

* Add a couple of simple tests:

  * Compare hashes for identical local patterns in different positions.

---

## Work Order 6 – Component-relative roles & sectors

**File:** `src/features/object_roles.py`
**Goal:** add higher-level features derived from components:

* per-pixel quadrant/sector within a component’s bounding box,
* per-pixel interior vs border *within* each component,
* per-component “role bits” like is_small / is_big / is_unique_shape.

**Dependencies:**

* `numpy`
* `from src.core.grid_types import Grid, Component`
* `from src.features.components import connected_components_by_color, compute_shape_signature`
  (or whatever you named them)

### 6.1 Quadrant / sector features (per-pixel, per-component)

Implement:

```python
def component_sectors(
    components: list[Component]
) -> dict[tuple[int, int], dict[str, str]]:
    """
    For each pixel (r,c) in each component, assign sector labels
    relative to that component's bounding box.

    Returns:
      a dict mapping (r,c) -> {
        "vert_sector": "top" | "center" | "bottom",
        "horiz_sector": "left" | "center" | "right"
      }
    """
```

Suggested logic:

* For each `comp`:

  * get `r_min, r_max, c_min, c_max` from `comp.bbox`.
  * compute vertical thirds:

    * if r in [r_min, mid_top] → "top"
    * if r in (mid_top, mid_bottom) → "center"
    * if r in [mid_bottom, r_max] → "bottom"
  * similar for columns: "left" / "center" / "right".
* Assign for each `(r,c)` in `comp.pixels`.

Keep it simple; exact split logic is not critical, just stable.

### 6.2 Interior vs border per-component

Implement:

```python
def component_border_interior(
    grid: Grid,
    components: list[Component]
) -> dict[tuple[int,int], dict[str, bool]]:
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
```

Suggested logic:

* Build a fast lookup: `(r,c) -> comp.id` for all pixels in components.
* For each pixel `(r,c)` in a component:

  * check 4 neighbors (up/down/left/right) inside grid bounds:

    * if any neighbor is either:

      * not same color, or
      * not in the same component,
        → mark `is_border=True`.
  * otherwise `is_interior=True`.
* Ensure `is_border` and `is_interior` are mutually exclusive.

This is what S10 (border vs interior laws) will use.

### 6.3 Object role bits (is_big / is_small / is_unique_pattern)

Implement:

```python
def component_role_bits(
    components: list[Component]
) -> dict[int, dict[str, bool]]:
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
```

Suggested logic:

* Collect `sizes = [comp.size for comp in components]`.
* Compute thresholds, e.g.:

  * small_cutoff = 33rd percentile,
  * big_cutoff = 66th percentile (or use sorted ranks).
* Mark:

  * `is_small` if `size <= small_cutoff`,
  * `is_big` if `size >= big_cutoff`.
* For uniqueness:

  * For each `(color, shape_signature)` pair, count how many components have it.
  * `is_unique_shape=True` if count == 1 for that `(color, shape_signature)`.

Return a dict keyed by `comp.id`.

---

### Usage in the rest of the code

* `connected_components_by_color(grid)` gives you `components`.
* This new module gives you:

  * `sectors = component_sectors(components)` → per-pixel sectors inside object bbox,
  * `border_info = component_border_interior(grid, components)` → interior vs border per pixel,
  * `role_bits = component_role_bits(components)` → role flags per component.

Schemas S2/S6/S7/S10 can then refer to these features when deciding *which* components or pixels to constrain.

---

### How this ties together

Once these work orders are done, you will have:

* A **grid IO + indexing module**.
* A **feature module** that can compute:

  * coordinates and bands,
  * border flags,
  * connected components + shape signatures,
  * object_id per pixel,
  * row/col nonzero flags,
  * neighborhood hashes.

This is φ(p) in code.

From there, we can:

* Define the **SchemaFamily registry** and **builder functions** (next milestones).
* Then hook ConstraintBuilder + solver.
* Finally, bring in the Pi-agent that uses all this.
