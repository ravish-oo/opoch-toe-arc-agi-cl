## WO-M2.4 – Schema builder dispatch skeleton

**File:** `src/schemas/dispatch.py`
**Goal:** provide a **single entrypoint** that, given a `family_id` and `params`, calls the appropriate `build_Sk_constraints(...)` function. In M2 these builders are *stubs* (raise `NotImplementedError`); real logic comes in M3.

No math logic here, just clean wiring.

---

### 1. Imports & dependencies

Use only standard libs + our existing ConstraintBuilder:

```python
from typing import Callable, Dict, Any

from src.constraints.builder import ConstraintBuilder
```

No numpy, no solver, no feature modules.

---

### 2. Standard builder function signature

We fix the builder function signature **once** so M3 can implement them consistently.

Each schema builder `build_Sk_constraints` must have:

```python
def build_Sk_constraints(
    task_context: Dict[str, Any],
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema Sk to the given ConstraintBuilder.

    Args:
        task_context: dict with task-specific data, e.g.:
            {
                "input_grids": List[Grid],
                "output_grids": List[Grid] or None for test,
                "features": Dict[str, Any],  # precomputed φ, components, etc.
                "N": int,                    # number of pixels
                "C": int                     # palette size
            }
        schema_params: parameters for this schema instance (from SchemaFamily.parameter_spec).
        builder: the ConstraintBuilder to add constraints into.
    """
    ...
```

For **M2**, each `build_Sk_constraints` will just raise `NotImplementedError`. No logic, no features.

We have 11 such functions: `build_S1_constraints` … `build_S11_constraints`.

---

### 3. Define stub builder functions

In `dispatch.py`, define:

```python
def build_S1_constraints(
    task_context: Dict[str, Any],
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    raise NotImplementedError("build_S1_constraints is not implemented yet (M3).")


def build_S2_constraints(
    task_context: Dict[str, Any],
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    raise NotImplementedError("build_S2_constraints is not implemented yet (M3).")

# ...

def build_S11_constraints(
    task_context: Dict[str, Any],
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    raise NotImplementedError("build_S11_constraints is not implemented yet (M3).")
```

Make sure all 11 functions exist and share the exact same argument names and types.

---

### 4. `BUILDERS` mapping

Define a global dict mapping family_id → builder function:

```python
BUILDERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any], ConstraintBuilder], None]] = {
    "S1": build_S1_constraints,
    "S2": build_S2_constraints,
    "S3": build_S3_constraints,
    "S4": build_S4_constraints,
    "S5": build_S5_constraints,
    "S6": build_S6_constraints,
    "S7": build_S7_constraints,
    "S8": build_S8_constraints,
    "S9": build_S9_constraints,
    "S10": build_S10_constraints,
    "S11": build_S11_constraints,
}
```

No other keys, no extra logic in this dict.

Later, `SchemaFamily.builder_name` in `families.py` will match these names, but for M2 we do not import `families` here to avoid circular dependencies.

---

### 5. `apply_schema_instance` helper

Define:

```python
def apply_schema_instance(
    family_id: str,
    schema_params: Dict[str, Any],
    task_context: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Look up the builder function for the given family_id and apply it.

    Args:
        family_id: e.g. "S1", "S2", ..., "S11".
        schema_params: parameters for this schema instance.
        task_context: task-specific context (grids, features, N, C, etc.).
        builder: ConstraintBuilder to accumulate constraints.

    Raises:
        KeyError: if no builder is registered for the given family_id.
        NotImplementedError: if the builder function is still a stub (M2).
    """
    if family_id not in BUILDERS:
        raise KeyError(f"No builder registered for schema family '{family_id}'")
    builder_fn = BUILDERS[family_id]
    builder_fn(task_context, schema_params, builder)
```

No extra branching or heuristics.

---

### 6. Thin runner for sanity

Add at the bottom of `dispatch.py`:

```python
if __name__ == "__main__":
    from pprint import pprint
    from src.constraints.builder import ConstraintBuilder

    print("Available schema builders:")
    pprint(list(BUILDERS.keys()))
    assert set(BUILDERS.keys()) == {f"S{i}" for i in range(1, 12)}, \
        "Expected builder keys S1..S11"

    # Test that apply_schema_instance dispatches correctly and raises NotImplementedError
    dummy_context: Dict[str, Any] = {}
    dummy_params: Dict[str, Any] = {}
    cb = ConstraintBuilder()

    try:
        apply_schema_instance("S1", dummy_params, dummy_context, cb)
    except NotImplementedError as e:
        print("Caught expected NotImplementedError for S1:", e)
    else:
        raise AssertionError("Expected NotImplementedError for S1 builder stub")

    # Test that unknown family_id raises KeyError
    try:
        apply_schema_instance("S99", dummy_params, dummy_context, cb)
    except KeyError as e:
        print("Caught expected KeyError for unknown family:", e)
    else:
        raise AssertionError("Expected KeyError for unknown family_id")

    print("dispatch.py sanity checks passed.")
```

This verifies:

* `BUILDERS` has exactly S1..S11.
* `apply_schema_instance("S1", ...)` calls the stub and raises `NotImplementedError`.
* `apply_schema_instance("S99", ...)` raises `KeyError`.

No ARC data involved yet.

---

### 7. Reviewer/tester instructions

For the reviewer/tester:

1. **Code review:**

   * Check imports: only `typing`, `ConstraintBuilder` from `src.constraints.builder`.
   * Confirm all 11 functions `build_S1_constraints` .. `build_S11_constraints` exist with the same signature.
   * Verify each stub raises `NotImplementedError` with a clear message.
   * Confirm `BUILDERS` contains exactly `"S1"`..`"S11"` mapped to the correct functions.
   * Confirm `apply_schema_instance`:

     * raises `KeyError` for unknown `family_id`,
     * otherwise forwards to the correct builder function.

2. **Run thin runner:**

   * `python -m src.schemas.dispatch`
   * Expect output listing S1..S11, a caught `NotImplementedError` for S1, a caught `KeyError` for S99, and `dispatch.py sanity checks passed.` with no assertions.

3. **Future consistency (no action now, just awareness):**

   * Later, when M3 implements real `build_Sk_constraints`, they must **reuse this exact signature** and be wired into `BUILDERS` without changing `apply_schema_instance`.
