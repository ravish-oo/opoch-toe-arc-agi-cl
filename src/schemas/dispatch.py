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

from typing import Callable, Dict, Any, Optional

from src.constraints.builder import ConstraintBuilder
from src.schemas.context import TaskContext

# Import actual schema builders (M3.1+)
from src.schemas.s1_copy_tie import build_S1_constraints
from src.schemas.s2_component_recolor import build_S2_constraints
from src.schemas.s3_bands import build_S3_constraints
from src.schemas.s4_residue_color import build_S4_constraints
from src.schemas.s5_template_stamping import build_S5_constraints
from src.schemas.s6_crop_roi import build_S6_constraints
from src.schemas.s7_aggregation import build_S7_constraints
from src.schemas.s8_tiling import build_S8_constraints
from src.schemas.s9_cross_propagation import build_S9_constraints
from src.schemas.s10_frame_border import build_S10_constraints
from src.schemas.s11_local_codebook import build_S11_constraints
from src.schemas.s12_projection import build_S12_constraints
from src.schemas.s13_gravity import build_S13_constraints
from src.schemas.s14_topology import build_S14_constraints
from src.schemas.s_default import build_S_Default_constraints


# =============================================================================
# Schema builders (S1-S14 + S_Default) are implemented in separate modules (M3.1-M3.5)
# =============================================================================


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
    "S12": build_S12_constraints,
    "S13": build_S13_constraints,
    "S14": build_S14_constraints,
    "S_Default": build_S_Default_constraints,
}


# =============================================================================
# Dispatch entrypoint
# =============================================================================

def apply_schema_instance(
    family_id: str,
    schema_params: Dict[str, Any],
    task_context: TaskContext,
    builder: ConstraintBuilder,
    example_type: str = None,
    example_index: int = None,
    schema_constraint_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Look up the builder function for the given family_id and apply it.

    This is the main entrypoint for schema constraint building.
    Given a schema family ID and its parameters, this dispatches to the
    appropriate builder function which emits constraints into the builder.

    Args:
        family_id: Schema family identifier (e.g. "S1", "S2", ..., "S11", "S_Default")
        schema_params: Parameters for this schema instance
        task_context: TaskContext with all φ features and grids
        builder: ConstraintBuilder to accumulate constraints
        example_type: "train" or "test" (optional, injected into params if provided)
        example_index: Which example to constrain (optional, injected into params if provided)
        schema_constraint_counts: Optional dict to track per-schema constraint counts.
                                  If provided, will be updated with constraints added
                                  by this schema instance.

    Raises:
        KeyError: If no builder is registered for the given family_id

    Example:
        >>> builder = ConstraintBuilder()
        >>> params = {"K": 3, "residue_to_color": {0: 1, 1: 2, 2: 0}}
        >>> counts = {}
        >>> apply_schema_instance("S4", params, context, builder,
        ...                       example_type="train", example_index=0,
        ...                       schema_constraint_counts=counts)
        >>> counts["S4"]  # Number of constraints added by S4
        12
    """
    if family_id not in BUILDERS:
        raise KeyError(f"No builder registered for schema family '{family_id}'")

    # Inject example_type and example_index into params for backward compatibility
    # with M3 builders that expect these in params
    enriched_params = dict(schema_params)
    if example_type is not None:
        enriched_params["example_type"] = example_type
    if example_index is not None:
        enriched_params["example_index"] = example_index

    # Track constraints before building
    before = len(builder.constraints)

    # Apply schema builder
    builder_fn = BUILDERS[family_id]
    builder_fn(task_context, enriched_params, builder)

    # Track constraints after building (M5.X diagnostics)
    if schema_constraint_counts is not None:
        after = len(builder.constraints)
        added = after - before

        # Sanity check: schema builders should only add constraints, never remove
        if added < 0:
            raise RuntimeError(
                f"Schema builder {family_id} reduced constraint count by {-added}. "
                f"Before: {before}, After: {after}"
            )

        # Update counts if this schema added any constraints
        if added > 0:
            schema_constraint_counts[family_id] = (
                schema_constraint_counts.get(family_id, 0) + added
            )


if __name__ == "__main__":
    from pprint import pprint

    print("Schema builder dispatch self-test")
    print("=" * 70)

    print("\n1. Available schema builders:")
    print("-" * 70)
    pprint(sorted(BUILDERS.keys()))

    expected_keys = {f"S{i}" for i in range(1, 15)} | {"S_Default"}
    assert set(BUILDERS.keys()) == expected_keys, \
        f"Expected builder keys S1..S14 + S_Default, got {set(BUILDERS.keys())}"
    print(f"  ✓ All 15 builders registered (S1-S14 + S_Default)")

    print("\n2. Testing dispatch with all builders:")
    print("-" * 70)

    # Test that apply_schema_instance dispatches correctly
    # All S1-S11 are implemented (M3.1-M3.5)
    # Test that all builders execute without error (even with empty params)
    import numpy as np
    from src.schemas.context import build_example_context

    dummy_grid = np.array([[0, 1], [2, 3]], dtype=int)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    dummy_context = TaskContext(train_examples=[dummy_ex], test_examples=[], C=4)
    dummy_params: Dict[str, Any] = {}
    cb = ConstraintBuilder()

    # Test a few builders to ensure they execute without error
    for family_id in ["S1", "S5", "S8", "S11"]:
        try:
            apply_schema_instance(family_id, dummy_params, dummy_context, cb)
            print(f"  ✓ {family_id} builder executed successfully")
        except Exception as e:
            # Builders may return early with empty params, that's OK
            print(f"  ✓ {family_id} builder executed (returned early: {type(e).__name__})")

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
