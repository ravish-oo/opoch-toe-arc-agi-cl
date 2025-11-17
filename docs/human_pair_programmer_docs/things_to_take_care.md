## In M3

### from m2 
1. This doesn’t know anything about S1–S11 yet; it’s just the generic constraint collector.

2. * For M2, **builder_name** can just be strings (no actual imports yet); we’ll implement the functions in the next milestone.
3. Schema builder - For M2 we just need **structure**; actual constraint logic per S1–S11 comes in M3.

## In M4
### from M2
1. Solution decoding (y → Grid):
    - Math spec mentions "one-hot encoding for any grid Z"
    - M2 has encoding (Grid → y indices) but not decoding (y solution → Grid)
    - Assessment: ✅ Not a gap - this naturally belongs in M4 (Solver integration) when interpreting LP solution
  
  3. Constraint validation:
    - No mention of checking constraint consistency or detecting conflicts
    - Assessment: ✅ Not a gap for M2 - LP solver will detect infeasibility in M4
  