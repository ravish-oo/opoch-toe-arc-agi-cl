now we expand next. first assess the atomicity of it ie. is it around 300 loc or not. if not we break it into sequential WOs so that we dont bring in stubs.

### Work Order 2 – Coordinate, bands, and border features

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
repeating same instrcutions so that u dont miss
1. in this we dont want underspecificity so that claude gets some wiggle room. so be specific and dont leave a wiggle room 
2. we explicitly want to use mature and standard python libs so that claude doenst reinvent the wheel or implements any algo. we must resuse what is out thr and just stitch it. that is the smart move 
3. give clear reviwer+tester instructions so that they can knw how to test and make sure things are working as expected. may be involve real arc agi tasks if applicable? 
4. include instruction to build thin runner in parallel if applicable to this WO 
5. make sure u stick to the math spec my friend shared 

pls operate as a pi agent..