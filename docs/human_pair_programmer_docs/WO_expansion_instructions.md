now we expand next. first assess the atomicity of it ie. is it around 300 loc or not. if not we break it into sequential WOs so that we dont bring in stubs. do not over break it.. i mean operate in toe mode and use ur judgement

### 🔹 WO-M6.2 – Role statistics aggregator

**Goal:** compress raw role assignments and train IO into role-level statistics for mining.

**File:** `src/law_mining/role_stats.py`

**Scope:**

* Define:

  ```python
  @dataclass
  class RoleStats:
      train_in: List[tuple[int,int,int,int]]   # (example_idx, r, c, color_in)
      train_out: List[tuple[int,int,int,int]]  # (example_idx, r, c, color_out)
      test_in: List[tuple[int,int,int,int]]    # (example_idx, r, c, color_in_test)
  ```

* Implement:

  ```python
  def compute_role_stats(
      task_context: TaskContext,
      roles: RolesMapping
  ) -> Dict[int, RoleStats]:
      """
      For each role_id, collect:
        - all its appearances in train_in, train_out, test_in,
        - with associated colors.
      """
  ```

This is the main input to schema miners: they work at the role level, but can still consult φ when needed.
---
repeating same instrcutions so that u dont miss
1. in this we dont want underspecificity so that claude gets some wiggle room. so be specific and dont leave a wiggle room 
2. we explicitly want to use mature and standard python libs so that claude doenst reinvent the wheel or implements any algo. we must resuse what is out thr and just stitch it. that is the smart move 
3. give clear reviwer+tester instructions so that they can knw how to test and make sure things are working as expected. may be involve real arc agi tasks if applicable? 
4. make sure this aligns to math spec and clarificaitons i provided and how we dicussed it being sitting seamlessly on top of M1-M5 and how order shud be
5. no smuggled non-toe defaults!
6. incoroporate/address applicable gaps we dicussed which were highlited by implementer

pls operate in toe mode