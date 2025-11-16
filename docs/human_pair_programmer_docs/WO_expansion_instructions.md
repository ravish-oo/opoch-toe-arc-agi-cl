now we expand next. first assess the atomicity of it ie. is it around 300 loc or not. if not we break it into sequential WOs so that we dont bring in stubs.

### Work Order 3 – Connected components per color

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
repeating same instrcutions so that u dont miss
1. in this we dont want underspecificity so that claude gets some wiggle room. so be specific and dont leave a wiggle room 
2. we explicitly want to use mature and standard python libs so that claude doenst reinvent the wheel or implements any algo. we must resuse what is out thr and just stitch it. that is the smart move 
3. give clear reviwer+tester instructions so that they can knw how to test and make sure things are working as expected. may be involve real arc agi tasks if applicable? 
4. include instruction to build thin runner in parallel if applicable to this WO 
5. make sure u stick to the math spec my friend shared 

pls operate as a pi agent..