## WO-M4.2 – Solution decoding (y → Grid(s))

### Libraries & dependencies (no reinvention)

* Use **only**:

  ```python
  import numpy as np
  ```

* And imports from your own modules:

  ```python
  from src.core.grid_types import Grid
  from src.constraints.indexing import unflatten_index  # if needed
  ```

No custom algorithms; just reshape, argmax, and simple indexing math.

---

## A. `src/solver/decoding.py`

### A.1 Design choice (single-grid first)

Given M4.1 `solve_constraints_for_grid` returns **one** grid’s y as shape `(num_pixels, num_colors)`, v0 decoding will be:

* one helper for a **single grid**,
* an optional helper that accepts **flat y** for convenience,
* we *do not* implement multi-grid `VarLayout` in this WO (we already decided to start per-grid).

So we keep it minimal, and consistent with:

* **math spec**: one-hot y → grid,
* **M4.1**: per-grid ILP.

### A.2 API functions

Implement these two functions:

```python
def y_to_grid(
    y: np.ndarray,
    H: int,
    W: int,
    C: int
) -> Grid:
    """
    Decode a solved y vector into a grid.

    Args:
        y: numpy array either:
           - shape (H*W, C), or
           - flat of length H*W*C.
        H: output grid height.
        W: output grid width.
        C: number of colors.

    Returns:
        grid: numpy array of shape (H, W) with integer color values in [0, C-1].
    """
```

```python
def y_flat_to_grid(
    y_flat: np.ndarray,
    H: int,
    W: int,
    C: int
) -> Grid:
    """
    Convenience wrapper for flat y of length H*W*C.

    Args:
        y_flat: 1D numpy array of length H*W*C.
        H, W, C: as above.

    Returns:
        grid: shape (H, W), same as y_to_grid.
    """
```

### A.3 Implementation details

#### `y_to_grid`

Inside `y_to_grid`:

1. **Validate shape**

   ```python
   if y.ndim == 1:
       # Expect length H*W*C
       if y.size != H * W * C:
           raise ValueError("Flat y length does not match H*W*C")
       y2 = y.reshape(H * W, C)
   elif y.ndim == 2:
       if y.shape != (H * W, C):
           raise ValueError(f"y shape {y.shape} does not match (H*W, C) = ({H*W}, {C})")
       y2 = y
   else:
       raise ValueError(f"y must be 1D or 2D, got ndim={y.ndim}")
   ```

2. **Argmax per pixel**

   * y is one-hot, but solver might return small numeric noise.
   * We use `argmax` per row:

     ```python
     # shape (H*W,), entries in [0..C-1]
     color_indices = np.argmax(y2, axis=1)
     ```

3. **Reshape to grid**

   ```python
   grid = color_indices.reshape(H, W).astype(int)
   ```

4. **Optional sanity check**

   * you may check that each row is one-hot-ish:

     ```python
     # row sums should be close to 1; not enforced strictly here
     # but could be added as assertion if desired.
     ```

5. **Return**

   ```python
   return grid
   ```

#### `y_flat_to_grid`

Just a thin wrapper:

```python
def y_flat_to_grid(y_flat: np.ndarray, H: int, W: int, C: int) -> Grid:
    return y_to_grid(y_flat, H, W, C)
```

No duplication.

---

## B. `src/runners/test_decoding.py` – thin runner / tests

### B.1 Purpose

* Verify that:

  * `y_to_grid` decodes known one-hot arrays correctly,
  * It works with both 2D and flat 1D representations,
  * It respects H, W, C dimensions.

No ARC integration yet; just pure decoding sanity.

### B.2 Imports

```python
import numpy as np

from src.core.grid_types import Grid
from src.solver.decoding import y_to_grid, y_flat_to_grid
```

### B.3 Test 1 – Simple 2×2 grid, 3 colors

Define:

* H = 2, W = 2, C = 3
* Four pixels, choose colors [0,1,2,1] e.g.:

```python
def test_y_to_grid_2x2():
    H, W, C = 2, 2, 3
    num_pixels = H * W

    # Pixel colors (flattened by pixel):
    # p0 -> color 0
    # p1 -> color 1
    # p2 -> color 2
    # p3 -> color 1
    y = np.zeros((num_pixels, C), dtype=int)
    y[0, 0] = 1
    y[1, 1] = 1
    y[2, 2] = 1
    y[3, 1] = 1

    grid = y_to_grid(y, H, W, C)
    expected = np.array([[0, 1],
                         [2, 1]], dtype=int)
    assert grid.shape == (H, W)
    assert np.array_equal(grid, expected), f"Got {grid}, expected {expected}"
    print("test_y_to_grid_2x2: OK")
```

### B.4 Test 2 – Flat y (1D)

```python
def test_y_flat_to_grid_2x2():
    H, W, C = 2, 2, 3
    num_pixels = H * W

    y2 = np.zeros((num_pixels, C), dtype=int)
    y2[0, 2] = 1
    y2[1, 0] = 1
    y2[2, 1] = 1
    y2[3, 2] = 1

    y_flat = y2.reshape(-1)
    grid = y_flat_to_grid(y_flat, H, W, C)
    expected = np.array([[2, 0],
                         [1, 2]], dtype=int)
    assert grid.shape == (H, W)
    assert np.array_equal(grid, expected), f"Got {grid}, expected {expected}"
    print("test_y_flat_to_grid_2x2: OK")
```

### B.5 Main runner

```python
if __name__ == "__main__":
    test_y_to_grid_2x2()
    test_y_flat_to_grid_2x2()
    print("All decoding tests passed.")
```

### B.6 Reviewer/tester instructions

From repo root:

```bash
python -m src.runners.test_decoding
```

Expected:

* Output:

  * `test_y_to_grid_2x2: OK`
  * `test_y_flat_to_grid_2x2: OK`
  * `All decoding tests passed.`
* No exceptions or assertion errors.

If anything fails:

* It’s likely a mismatch between the shapes and the reshape logic.
* The reviewer should check:

  * That H, W, C match what they passed in,
  * That the argmax logic is correct.

---
