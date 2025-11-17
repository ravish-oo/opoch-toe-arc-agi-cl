"""
Schema builder dispatch layer.

This module provides a single entrypoint for applying schema constraint builders.
In M2, all builders are stubs (raise NotImplementedError).
In M3, these will be replaced with actual constraint logic.

All schema builders share a standard signature:
    def build_Sk_constraints(task_context, schema_params, builder) -> None

Where:
  - task_context: TaskContext with grids, features, N, C, etc.
  - schema_params: parameters for this schema instance
  - builder: ConstraintBuilder to accumulate constraints
"""

from typing import Callable, Dict, Any

from src.constraints.builder import ConstraintBuilder
from src.schemas.context import TaskContext

# Import actual schema builders (M3.1+)
from src.schemas.s1_copy_tie import build_S1_constraints
from src.schemas.s2_component_recolor import build_S2_constraints


# =============================================================================
# S1 and S2 are implemented in separate modules (M3.1)
# S3-S11 remain as stubs below (to be implemented in M3.2+)
# =============================================================================

def build_S3_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S3 (Band / stripe laws).

    In M3, this will enforce shared patterns for row/column classes.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S3_constraints is not implemented yet (M3).")


def build_S4_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S4 (Periodicity / residue-class coloring).

    In M3, this will assign colors based on coordinate residues mod K.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S4_constraints is not implemented yet (M3).")


def build_S5_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S5 (Template stamping).

    In M3, this will stamp template patches around seed pixels.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S5_constraints is not implemented yet (M3).")


def build_S6_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S6 (Cropping to ROI / dominant object).

    In M3, this will constrain output to be a crop of selected bbox.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S6_constraints is not implemented yet (M3).")


def build_S7_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S7 (Aggregation / summary grid).

    In M3, this will summarize macro-cells into smaller output grid.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S7_constraints is not implemented yet (M3).")


def build_S8_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S8 (Tiling / replication).

    In M3, this will replicate a base tile pattern to fill region.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S8_constraints is not implemented yet (M3).")


def build_S9_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S9 (Cross / plus propagation).

    In M3, this will propagate spokes from cross-shaped seeds.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S9_constraints is not implemented yet (M3).")


def build_S10_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S10 (Frame / border vs interior).

    In M3, this will assign different colors to border vs interior pixels.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S10_constraints is not implemented yet (M3).")


def build_S11_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add constraints for schema S11 (Local neighborhood codebook).

    In M3, this will apply learned codebook mappings from hash to patch.

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters for this schema instance
        builder: ConstraintBuilder to add constraints to

    Raises:
        NotImplementedError: M2 stub, implementation in M3
    """
    raise NotImplementedError("build_S11_constraints is not implemented yet (M3).")


# =============================================================================
# Builder registry
# =============================================================================

BUILDERS: Dict[str, Callable[[TaskContext, Dict[str, Any], ConstraintBuilder], None]] = {
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


# =============================================================================
# Dispatch entrypoint
# =============================================================================

def apply_schema_instance(
    family_id: str,
    schema_params: Dict[str, Any],
    task_context: TaskContext,
    builder: ConstraintBuilder
) -> None:
    """
    Look up the builder function for the given family_id and apply it.

    This is the main entrypoint for schema constraint building.
    Given a schema family ID and its parameters, this dispatches to the
    appropriate builder function which emits constraints into the builder.

    Args:
        family_id: Schema family identifier (e.g. "S1", "S2", ..., "S11")
        schema_params: Parameters for this schema instance
        task_context: TaskContext with all φ features and grids
        builder: ConstraintBuilder to accumulate constraints

    Raises:
        KeyError: If no builder is registered for the given family_id
        NotImplementedError: If the builder function is still a stub (M2)

    Example:
        >>> builder = ConstraintBuilder()
        >>> context = {"N": 12, "C": 10, "features": {...}}
        >>> params = {"K": 3, "residue_to_color": {0: 1, 1: 2, 2: 0}}
        >>> apply_schema_instance("S4", params, context, builder)
        # In M3, this will add S4 constraints to builder
    """
    if family_id not in BUILDERS:
        raise KeyError(f"No builder registered for schema family '{family_id}'")

    builder_fn = BUILDERS[family_id]
    builder_fn(task_context, schema_params, builder)


if __name__ == "__main__":
    from pprint import pprint

    print("Schema builder dispatch self-test")
    print("=" * 70)

    print("\n1. Available schema builders:")
    print("-" * 70)
    pprint(sorted(BUILDERS.keys()))

    expected_keys = {f"S{i}" for i in range(1, 12)}
    assert set(BUILDERS.keys()) == expected_keys, \
        f"Expected builder keys S1..S11, got {set(BUILDERS.keys())}"
    print(f"  ✓ All 11 builders registered (S1-S11)")

    print("\n2. Testing dispatch with stub builders:")
    print("-" * 70)

    # Test that apply_schema_instance dispatches correctly
    # S1/S2 are implemented (M3.1), S3-S11 are stubs
    # Test S3 (stub) raises NotImplementedError
    import numpy as np
    from src.schemas.context import build_example_context

    dummy_grid = np.array([[0, 1], [2, 3]], dtype=int)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    dummy_context = TaskContext(train_examples=[dummy_ex], test_examples=[], C=4)
    dummy_params: Dict[str, Any] = {}
    cb = ConstraintBuilder()

    try:
        apply_schema_instance("S3", dummy_params, dummy_context, cb)
        raise AssertionError("Expected NotImplementedError for S3 builder stub")
    except NotImplementedError as e:
        print(f"  ✓ Caught expected NotImplementedError for S3 (stub):")
        print(f"    {e}")

    print("\n3. Testing unknown family_id:")
    print("-" * 70)

    # Test that unknown family_id raises KeyError
    try:
        apply_schema_instance("S99", dummy_params, dummy_context, cb)
        raise AssertionError("Expected KeyError for unknown family_id")
    except KeyError as e:
        print(f"  ✓ Caught expected KeyError for unknown family:")
        print(f"    {e}")

    print("\n4. Verifying builder function signatures:")
    print("-" * 70)

    # Check that all builder functions have the same signature
    import inspect

    signatures = {}
    for fid, builder_fn in BUILDERS.items():
        sig = inspect.signature(builder_fn)
        signatures[fid] = list(sig.parameters.keys())

    # All should have: task_context, schema_params, builder
    expected_params = ['task_context', 'schema_params', 'builder']
    for fid, params in signatures.items():
        assert params == expected_params, \
            f"{fid} has wrong signature: {params} != {expected_params}"

    print(f"  ✓ All builders have standard signature: {expected_params}")

    print("\n" + "=" * 70)
    print("✓ dispatch.py sanity checks passed.")
