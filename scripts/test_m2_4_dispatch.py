#!/usr/bin/env python3
"""
Comprehensive test for WO-M2.4 dispatch.py

Tests:
1. All 11 builder stubs exist
2. Signature consistency across all builders
3. All stubs raise NotImplementedError
4. BUILDERS registry completeness
5. apply_schema_instance dispatcher
6. Error handling
7. Connection to M2.3 families.py
"""

import sys
import inspect
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schemas.dispatch import (
    BUILDERS,
    apply_schema_instance,
    build_S1_constraints,
    build_S2_constraints,
    build_S3_constraints,
    build_S4_constraints,
    build_S5_constraints,
    build_S6_constraints,
    build_S7_constraints,
    build_S8_constraints,
    build_S9_constraints,
    build_S10_constraints,
    build_S11_constraints,
)
from src.constraints.builder import ConstraintBuilder


def test_builder_count():
    """Test that exactly 11 builder stubs exist"""
    print("Testing builder stub count...")

    # All builder functions should be in BUILDERS
    assert len(BUILDERS) == 11, \
        f"Expected exactly 11 builders, got {len(BUILDERS)}"

    print(f"  ✓ Exactly 11 builder stubs")


def test_builder_registry_keys():
    """Test that BUILDERS has keys S1 through S11"""
    print("Testing BUILDERS registry keys...")

    expected_keys = {f"S{i}" for i in range(1, 12)}
    actual_keys = set(BUILDERS.keys())

    assert actual_keys == expected_keys, \
        f"Expected keys {expected_keys}, got {actual_keys}"

    # Check they're strings
    assert all(isinstance(k, str) for k in BUILDERS.keys()), \
        "All keys should be strings"

    print(f"  ✓ BUILDERS has keys S1 through S11")


def test_signature_consistency():
    """
    CRITICAL TEST: All 11 builders must have IDENTICAL signatures.

    This is the highest priority check - any deviation breaks M3 compatibility.
    """
    print("Testing signature consistency...")

    # Expected signature
    expected_params = ['task_context', 'schema_params', 'builder']

    # Collect all signatures
    signatures = {}
    for fid, builder_fn in BUILDERS.items():
        sig = inspect.signature(builder_fn)
        param_names = list(sig.parameters.keys())
        signatures[fid] = param_names

        # Check this builder has correct signature
        assert param_names == expected_params, \
            f"{fid} has wrong signature: {param_names} != {expected_params}"

    # Verify all are identical
    unique_signatures = set(tuple(sig) for sig in signatures.values())
    assert len(unique_signatures) == 1, \
        f"All builders must have same signature, found variations: {unique_signatures}"

    print(f"  ✓ All 11 builders have identical signature: {expected_params}")


def test_signature_parameter_types():
    """Test that all builders have correct type annotations"""
    print("Testing parameter type annotations...")

    from typing import get_type_hints

    # Expected type hints
    # Note: get_type_hints returns the actual types, not string annotations
    for fid, builder_fn in BUILDERS.items():
        sig = inspect.signature(builder_fn)

        # Check parameter annotations
        params = sig.parameters

        # task_context should be Dict[str, Any]
        assert 'task_context' in params
        # Can't easily check Dict[str, Any] at runtime, but we can check it's annotated
        assert params['task_context'].annotation != inspect.Parameter.empty, \
            f"{fid}: task_context should have type annotation"

        # schema_params should be Dict[str, Any]
        assert 'schema_params' in params
        assert params['schema_params'].annotation != inspect.Parameter.empty, \
            f"{fid}: schema_params should have type annotation"

        # builder should be ConstraintBuilder
        assert 'builder' in params
        assert params['builder'].annotation != inspect.Parameter.empty, \
            f"{fid}: builder should have type annotation"

        # Return type should be None
        assert sig.return_annotation is None or sig.return_annotation == type(None), \
            f"{fid}: return type should be None"

    print("  ✓ All parameter types correctly annotated")


def test_all_stubs_raise_not_implemented():
    """
    CRITICAL TEST: All stubs must raise NotImplementedError.

    No placeholder logic, no 'pass', only NotImplementedError.
    """
    print("Testing all stubs raise NotImplementedError...")

    dummy_context = {}
    dummy_params = {}
    builder = ConstraintBuilder()

    for fid, builder_fn in BUILDERS.items():
        try:
            builder_fn(dummy_context, dummy_params, builder)
            raise AssertionError(
                f"{fid} should raise NotImplementedError, but didn't raise anything"
            )
        except NotImplementedError as e:
            # Check the error message mentions M3
            error_msg = str(e)
            assert "M3" in error_msg, \
                f"{fid}: NotImplementedError message should mention M3, got: {error_msg}"
            assert fid.replace("S", "build_S") in error_msg or fid in error_msg, \
                f"{fid}: Error message should identify the function"

    print(f"  ✓ All 11 stubs raise NotImplementedError with M3 message")


def test_no_logic_in_stubs():
    """Test that stubs don't modify the builder before raising"""
    print("Testing stubs don't execute logic...")

    dummy_context = {}
    dummy_params = {}

    for fid, builder_fn in BUILDERS.items():
        builder = ConstraintBuilder()

        # Builder should start empty
        assert len(builder.constraints) == 0

        try:
            builder_fn(dummy_context, dummy_params, builder)
        except NotImplementedError:
            pass

        # Builder should still be empty (no constraints added before exception)
        assert len(builder.constraints) == 0, \
            f"{fid} added constraints before raising NotImplementedError"

    print("  ✓ No logic executed in stubs (all raise immediately)")


def test_apply_schema_instance_dispatch():
    """Test that apply_schema_instance correctly dispatches"""
    print("Testing apply_schema_instance dispatch...")

    dummy_context = {}
    dummy_params = {}
    builder = ConstraintBuilder()

    # Test dispatching to each builder
    for fid in BUILDERS.keys():
        try:
            apply_schema_instance(fid, dummy_params, dummy_context, builder)
            raise AssertionError(f"Expected NotImplementedError for {fid}")
        except NotImplementedError as e:
            # Success - stub was called
            assert fid.replace("S", "build_S") in str(e) or fid in str(e)

    print("  ✓ apply_schema_instance correctly dispatches to all builders")


def test_apply_schema_instance_unknown_family():
    """Test that apply_schema_instance raises KeyError for unknown family"""
    print("Testing apply_schema_instance with unknown family_id...")

    dummy_context = {}
    dummy_params = {}
    builder = ConstraintBuilder()

    # Test various invalid family IDs
    invalid_ids = ["S0", "S12", "S99", "X1", "schema1", ""]

    for invalid_id in invalid_ids:
        try:
            apply_schema_instance(invalid_id, dummy_params, dummy_context, builder)
            raise AssertionError(f"Expected KeyError for invalid family_id '{invalid_id}'")
        except KeyError as e:
            # Check error message mentions the invalid ID
            assert invalid_id in str(e) or "registered" in str(e).lower()

    print(f"  ✓ apply_schema_instance raises KeyError for {len(invalid_ids)} invalid IDs")


def test_apply_schema_instance_parameter_order():
    """
    Test that apply_schema_instance forwards parameters in correct order.

    The dispatcher should pass: (task_context, schema_params, builder)
    """
    print("Testing apply_schema_instance parameter order...")

    # We can't test this directly without modifying stubs, but we can
    # verify the call signature of apply_schema_instance matches expectations

    sig = inspect.signature(apply_schema_instance)
    params = list(sig.parameters.keys())

    # apply_schema_instance should have: family_id, schema_params, task_context, builder
    expected = ['family_id', 'schema_params', 'task_context', 'builder']
    assert params == expected, \
        f"apply_schema_instance signature: {params} != {expected}"

    print(f"  ✓ apply_schema_instance has correct parameter order")


def test_builder_names_match_convention():
    """Test that builder function names match the expected convention"""
    print("Testing builder function name convention...")

    for fid, builder_fn in BUILDERS.items():
        expected_name = f"build_{fid}_constraints"
        actual_name = builder_fn.__name__

        assert actual_name == expected_name, \
            f"Builder for {fid} has wrong name: {actual_name} != {expected_name}"

    print("  ✓ All builder names follow 'build_S{k}_constraints' convention")


def test_connection_to_families():
    """
    Test connection to M2.3 families.py.

    Verify that builder names in BUILDERS match the builder_name field
    in SCHEMA_FAMILIES from M2.3.
    """
    print("Testing connection to M2.3 families.py...")

    from src.schemas.families import SCHEMA_FAMILIES

    # Check each family has a matching builder
    for fid, family in SCHEMA_FAMILIES.items():
        # Family should specify a builder_name
        builder_name = family.builder_name

        # This builder should exist in BUILDERS
        assert fid in BUILDERS, \
            f"Family {fid} specifies builder but no builder in BUILDERS"

        # The builder function name should match
        builder_fn = BUILDERS[fid]
        assert builder_fn.__name__ == builder_name, \
            f"Family {fid} expects builder '{builder_name}', " \
            f"but BUILDERS[{fid}] is '{builder_fn.__name__}'"

    # Reverse check: all builders have corresponding families
    for fid in BUILDERS.keys():
        assert fid in SCHEMA_FAMILIES, \
            f"Builder {fid} exists but no corresponding family in SCHEMA_FAMILIES"

    print(f"  ✓ All {len(BUILDERS)} builders match M2.3 families")


def test_docstrings_present():
    """Test that all builder stubs have docstrings"""
    print("Testing builder docstrings...")

    for fid, builder_fn in BUILDERS.items():
        doc = builder_fn.__doc__

        assert doc is not None, f"{fid} has no docstring"
        assert len(doc) > 20, f"{fid} docstring too short"

        # Check mentions M3
        assert "M3" in doc, f"{fid} docstring should mention M3"

        # Check mentions Args
        assert "Args:" in doc, f"{fid} docstring should have Args section"

    print("  ✓ All builders have comprehensive docstrings")


def test_no_feature_imports():
    """Test that dispatch.py doesn't import feature modules"""
    print("Testing no feature imports...")

    # Check the source file doesn't import features
    dispatch_file = Path(project_root) / "src" / "schemas" / "dispatch.py"
    content = dispatch_file.read_text()

    # Should NOT import from src.features
    forbidden_imports = [
        "from src.features",
        "import src.features",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in content, \
            f"dispatch.py should not import features (found '{forbidden}')"

    # Should NOT import from src.schemas.families (circular dependency)
    assert "from src.schemas.families" not in content, \
        "dispatch.py should not import families (would be circular in M3)"

    print("  ✓ No feature imports (pure dispatch)")


def test_builders_are_functions():
    """Test that all BUILDERS values are callable"""
    print("Testing BUILDERS values are callable...")

    for fid, builder_fn in BUILDERS.items():
        assert callable(builder_fn), f"{fid} builder is not callable"
        assert inspect.isfunction(builder_fn), f"{fid} builder is not a function"

    print("  ✓ All BUILDERS values are functions")


def test_individual_builder_imports():
    """Test that all individual builder functions can be imported"""
    print("Testing individual builder imports...")

    # All these should be importable from dispatch module
    all_builders = [
        build_S1_constraints,
        build_S2_constraints,
        build_S3_constraints,
        build_S4_constraints,
        build_S5_constraints,
        build_S6_constraints,
        build_S7_constraints,
        build_S8_constraints,
        build_S9_constraints,
        build_S10_constraints,
        build_S11_constraints,
    ]

    assert len(all_builders) == 11, "Should import all 11 builders"

    # Each should be a function
    for builder_fn in all_builders:
        assert callable(builder_fn)

    print("  ✓ All 11 builders individually importable")


def main():
    print("=" * 70)
    print("WO-M2.4 Comprehensive Test - dispatch.py")
    print("=" * 70)
    print()

    try:
        # Basic structure tests
        test_builder_count()
        test_builder_registry_keys()
        test_builders_are_functions()
        test_individual_builder_imports()

        # CRITICAL: Signature consistency
        test_signature_consistency()
        test_signature_parameter_types()
        test_builder_names_match_convention()

        # CRITICAL: Stub behavior
        test_all_stubs_raise_not_implemented()
        test_no_logic_in_stubs()

        # Dispatcher tests
        test_apply_schema_instance_dispatch()
        test_apply_schema_instance_unknown_family()
        test_apply_schema_instance_parameter_order()

        # Connection to M2.3
        test_connection_to_families()

        # Documentation and imports
        test_docstrings_present()
        test_no_feature_imports()

        print()
        print("=" * 70)
        print("✅ ALL TESTS PASSED")
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
