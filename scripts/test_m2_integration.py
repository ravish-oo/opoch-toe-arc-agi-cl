#!/usr/bin/env python3
"""
M2 Integration Test - All Components Working Together

Tests the integration of all M2 work orders:
- WO-M2.1: y-indexing (indexing.py)
- WO-M2.2: ConstraintBuilder (builder.py)
- WO-M2.3: SchemaFamily registry (families.py)
- WO-M2.4: Schema builder dispatch (dispatch.py)

Verifies the complete infrastructure is ready for M3.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.constraints.indexing import (
    flatten_index,
    unflatten_index,
    y_index,
    y_index_to_pc
)
from src.constraints.builder import (
    ConstraintBuilder,
    add_one_hot_constraints
)
from src.schemas.families import SCHEMA_FAMILIES
from src.schemas.dispatch import BUILDERS, apply_schema_instance


def test_m2_1_and_m2_2_integration():
    """Test indexing works with ConstraintBuilder"""
    print("Testing M2.1 (indexing) + M2.2 (builder) integration...")

    # Small grid: H=2, W=3, C=5
    H, W, C = 2, 3, 5
    N = H * W

    # Create builder
    builder = ConstraintBuilder()

    # Add one-hot constraints using indexing
    add_one_hot_constraints(builder, N, C)

    # Verify constraints were created correctly
    assert len(builder.constraints) == N

    # Check first constraint uses correct indices
    lc = builder.constraints[0]  # First pixel (p_idx=0)
    expected_indices = [y_index(0, c, C) for c in range(C)]
    assert lc.indices == expected_indices

    # Verify indices can be decoded back
    for idx in lc.indices:
        p_idx, color = y_index_to_pc(idx, C, W)
        assert p_idx == 0  # First pixel
        assert 0 <= color < C

    print("  ✓ M2.1 indexing works with M2.2 ConstraintBuilder")


def test_m2_3_and_m2_4_integration():
    """Test SCHEMA_FAMILIES matches BUILDERS"""
    print("Testing M2.3 (families) + M2.4 (dispatch) integration...")

    # Every family should have a builder
    for fid, family in SCHEMA_FAMILIES.items():
        # Check builder exists
        assert fid in BUILDERS, \
            f"Family {fid} has no builder in BUILDERS"

        # Check builder name matches
        builder_fn = BUILDERS[fid]
        assert builder_fn.__name__ == family.builder_name, \
            f"Family {fid} builder name mismatch"

    # Every builder should have a family
    for fid in BUILDERS.keys():
        assert fid in SCHEMA_FAMILIES, \
            f"Builder {fid} has no family in SCHEMA_FAMILIES"

    print(f"  ✓ All {len(SCHEMA_FAMILIES)} families match {len(BUILDERS)} builders")


def test_full_pipeline_structure():
    """
    Test the full M2 pipeline structure (without M3 logic).

    Simulates what M3 will do:
    1. Create task_context with features (M1)
    2. Select a schema family (M2.3)
    3. Dispatch to builder (M2.4)
    4. Builder would add constraints (M3 - stubbed)
    """
    print("Testing full M2 pipeline structure...")

    # Create proper TaskContext for S1-S4
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    dummy_grid = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    task_context = TaskContext(train_examples=[dummy_ex], test_examples=[], C=10)
    N = 6  # 2x3 grid

    # Create constraint builder
    builder = ConstraintBuilder()

    # Add one-hot constraints (these work in M2)
    add_one_hot_constraints(builder, N, task_context.C)
    initial_constraint_count = len(builder.constraints)

    # Select a schema family that's still a stub (S5, since S1-S4 are implemented in M3.1-M3.2)
    family = SCHEMA_FAMILIES["S5"]

    # Prepare schema parameters (would come from law mining in M3)
    schema_params = {
        "feature_predicate": "test"  # Dummy parameter
    }

    # Dispatch to builder (S5 should raise NotImplementedError, S1-S4 are implemented)
    try:
        apply_schema_instance(
            family_id=family.id,
            schema_params=schema_params,
            task_context=task_context,
            builder=builder
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        # This is expected for S5-S11 (still stubs)
        assert "M3" in str(e)

    # Builder should still have only one-hot constraints (stub didn't add any)
    assert len(builder.constraints) == initial_constraint_count

    print("  ✓ Full pipeline structure ready for M3 (S1-S4 implemented, S5-S11 stubs)")


def test_all_schemas_dispatchable():
    """Test that all 11 schemas can be dispatched (S1-S4 implemented, S5-S11 stubs)"""
    print("Testing all 11 schemas are dispatchable...")

    # Need proper TaskContext for S1-S4
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    dummy_grid = np.array([[0, 1], [2, 3]], dtype=int)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    task_context = TaskContext(train_examples=[dummy_ex], test_examples=[], C=4)
    schema_params = {}

    for fid, family in SCHEMA_FAMILIES.items():
        builder = ConstraintBuilder()

        if fid in ["S1", "S2", "S3", "S4"]:
            # S1-S4 are implemented - should NOT raise NotImplementedError
            # They may or may not add constraints depending on params (empty params = no constraints)
            try:
                apply_schema_instance(fid, schema_params, task_context, builder)
                # Success - builder implemented
            except NotImplementedError:
                raise AssertionError(f"{fid} should be implemented (M3.1-M3.2)")
        else:
            # S5-S11 should still raise NotImplementedError
            try:
                apply_schema_instance(fid, schema_params, task_context, builder)
                raise AssertionError(f"{fid} should raise NotImplementedError (stub)")
            except NotImplementedError:
                # Expected for stubs
                pass

    print(f"  ✓ All {len(SCHEMA_FAMILIES)} schemas dispatchable (S1-S4 implemented, S5-S11 stubs)")


def test_constraint_accumulation():
    """
    Test that constraints can be accumulated from multiple sources.

    This simulates what M3 will do: add one-hot constraints, then add
    schema-specific constraints from multiple schemas.
    """
    print("Testing constraint accumulation...")

    N, C = 6, 5
    builder = ConstraintBuilder()

    # Add one-hot constraints
    add_one_hot_constraints(builder, N, C)
    assert len(builder.constraints) == N

    # Add tie constraints
    builder.tie_pixel_colors(0, 1, C)
    assert len(builder.constraints) == N + C

    # Add fix constraint
    builder.fix_pixel_color(2, 3, C)
    assert len(builder.constraints) == N + C + 1

    # Add forbid constraint
    builder.forbid_pixel_color(3, 4, C)
    assert len(builder.constraints) == N + C + 2

    # All constraints should be LinearConstraint objects
    from src.constraints.builder import LinearConstraint
    assert all(isinstance(lc, LinearConstraint) for lc in builder.constraints)

    print(f"  ✓ Accumulated {len(builder.constraints)} constraints from multiple sources")


def test_ready_for_m3():
    """
    Verify that M2 infrastructure is complete and ready for M3.

    M3 will need to:
    1. Implement actual builder functions
    2. Mine laws from training data
    3. Use M1 features (φ)
    4. Emit constraints via builder
    5. Maintain standard signature
    """
    print("Testing M2 readiness for M3...")

    # Check all components exist
    assert callable(flatten_index), "M2.1: indexing functions"
    assert ConstraintBuilder is not None, "M2.2: ConstraintBuilder class"
    assert len(SCHEMA_FAMILIES) == 11, "M2.3: 11 schema families"
    assert len(BUILDERS) == 11, "M2.4: 11 builder stubs"

    # Check standard signature is defined
    import inspect
    sig = inspect.signature(BUILDERS["S1"])
    params = list(sig.parameters.keys())
    assert params == ['task_context', 'schema_params', 'builder'], \
        "Standard signature defined"

    # Check dispatcher works
    assert callable(apply_schema_instance), "M2.4: dispatcher function"

    # S1-S4 are implemented (M3.1-M3.2), S5-S11 are stubs
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    dummy_grid = np.array([[0, 1], [2, 3]], dtype=int)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    task_context = TaskContext(train_examples=[dummy_ex], test_examples=[], C=4)

    for fid in BUILDERS.keys():
        builder = ConstraintBuilder()
        if fid in ["S1", "S2", "S3", "S4"]:
            # Implemented in M3.1-M3.2 - should NOT raise
            try:
                apply_schema_instance(fid, {}, task_context, builder)
                # Success
            except NotImplementedError:
                raise AssertionError(f"{fid} should be implemented (M3.1-M3.2)")
        else:
            # S5-S11 still stubs
            try:
                apply_schema_instance(fid, {}, task_context, builder)
                raise AssertionError(f"{fid} should be stub")
            except NotImplementedError:
                pass  # Good - stub

    print("  ✓ M2 infrastructure complete, M3.1-M3.2 implemented (S1-S4), S5-S11 ready")


def test_no_circular_imports():
    """Test that M2 modules don't have circular import issues"""
    print("Testing no circular imports...")

    # This script already imports all M2 modules, so if we got here,
    # there are no circular import issues

    # Verify we can import all at once
    try:
        from src.constraints.indexing import y_index
        from src.constraints.builder import ConstraintBuilder
        from src.schemas.families import SCHEMA_FAMILIES
        from src.schemas.dispatch import apply_schema_instance
    except ImportError as e:
        raise AssertionError(f"Circular import detected: {e}")

    print("  ✓ No circular imports between M2 modules")


def test_m2_module_organization():
    """Test that M2 modules are organized correctly"""
    print("Testing M2 module organization...")

    # Check all M2 files exist
    m2_files = [
        "src/constraints/indexing.py",
        "src/constraints/builder.py",
        "src/schemas/families.py",
        "src/schemas/dispatch.py",
    ]

    for filepath in m2_files:
        full_path = project_root / filepath
        assert full_path.exists(), f"Missing M2 file: {filepath}"

    print(f"  ✓ All {len(m2_files)} M2 module files present")


def main():
    print("=" * 70)
    print("M2 Integration Test - All Components")
    print("=" * 70)
    print()

    try:
        # Cross-component integration
        test_m2_1_and_m2_2_integration()
        test_m2_3_and_m2_4_integration()

        # Full pipeline
        test_full_pipeline_structure()
        test_all_schemas_dispatchable()
        test_constraint_accumulation()

        # Readiness checks
        test_ready_for_m3()
        test_no_circular_imports()
        test_m2_module_organization()

        print()
        print("=" * 70)
        print("✅ M2 INTEGRATION - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("M2 Milestone Complete:")
        print("  ✓ WO-M2.1: y-indexing helpers")
        print("  ✓ WO-M2.2: LinearConstraint & ConstraintBuilder")
        print("  ✓ WO-M2.3: SchemaFamily registry (S1-S11)")
        print("  ✓ WO-M2.4: Schema builder dispatch skeleton")
        print()
        print("Infrastructure ready for M3 (schema implementations)")
        print("=" * 70)
        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
