"""
Comprehensive review test for WO-M6.3A: Schema miners for S1, S2, S10.

This test verifies:
  1. S1: Stub implementation (correct signature, docstring, returns [])
  2. S2: Full implementation with always-true mining
  3. S10: Full implementation with always-true mining

Review priorities (HIGHEST to lowest):
  1. No TODOs, stubs, or simplified implementations (except S1 intentional stub)
  2. S2 and S10 check ALL training examples (not just first)
  3. Reject conflicts: if len(colors) > 1: skip
  4. Skip unobserved: if len(colors) == 0: skip
  5. Params format matches existing builders exactly
"""

import inspect
from pathlib import Path
from typing import Dict, List, Set

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s1_s2_s10 import mine_S1, mine_S2, mine_S10
from src.catalog.types import SchemaInstance


def test_s1_stub_implementation():
    """
    Test S1 miner is intentional stub with correct signature and docstring.

    Requirements:
      - Correct signature: (TaskContext, RolesMapping, Dict[int, RoleStats])
      - Explanatory docstring (not just "TODO")
      - Returns empty list
      - No raise NotImplementedError
    """
    print("\n" + "=" * 70)
    print("TEST 1: S1 Stub Implementation")
    print("=" * 70)

    # Check signature
    sig = inspect.signature(mine_S1)
    params = list(sig.parameters.keys())
    assert params == ["task_context", "roles", "role_stats"], \
        f"S1 signature wrong: {params}"
    print("  ✓ Signature correct")

    # Check docstring exists and is explanatory
    docstring = inspect.getdoc(mine_S1)
    assert docstring is not None, "S1 has no docstring"
    assert len(docstring) > 50, "S1 docstring too short"
    assert "S1" in docstring, "S1 docstring doesn't mention S1"
    assert "not implemented" in docstring.lower() or "empty" in docstring.lower(), \
        "S1 docstring doesn't explain why it's a stub"
    print("  ✓ Docstring explanatory")

    # Check returns empty list (not raises NotImplementedError)
    from src.schemas.context import TaskContext, build_example_context
    import numpy as np

    # Create minimal dummy context
    dummy_grid = np.zeros((3, 3), dtype=np.int8)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    dummy_task = TaskContext(
        train_examples=[dummy_ex],
        test_examples=[],
        C=1
    )
    dummy_roles = {}
    dummy_stats = {}

    result = mine_S1(dummy_task, dummy_roles, dummy_stats)
    assert result == [], f"S1 should return [], got {result}"
    print("  ✓ Returns empty list")

    print("\n✓ S1 stub implementation correct")


def test_s2_always_true_mining():
    """
    Test S2 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - Verifies component uniformity (all pixels same output color)
      - Aggregates across examples
      - Rejects conflicts (len(color_set) > 1)
      - Params format matches build_S2_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 2: S2 Always-True Mining")
    print("=" * 70)

    # Check source code for TODOs
    source = inspect.getsource(mine_S2)
    assert "TODO" not in source, "S2 contains TODO"
    assert "NotImplementedError" not in source, "S2 raises NotImplementedError"
    assert "simplified" not in source.lower(), "S2 mentions 'simplified'"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S2 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check verifies component uniformity
    assert "output_colors_in_comp" in source or "uniform" in source.lower(), \
        "S2 doesn't verify component uniformity"
    assert "len(" in source and "!= 1" in source, \
        "S2 doesn't check for single output color"
    print("  ✓ Verifies component uniformity")

    # Check aggregates across examples
    assert "class_to_colors" in source or "defaultdict" in source, \
        "S2 doesn't aggregate across examples"
    print("  ✓ Aggregates across examples")

    # Check rejects conflicts
    assert "len(color_set) == 1" in source or "len(color_set) != 1" in source, \
        "S2 doesn't check for conflict rejection"
    print("  ✓ Rejects conflicts")

    # Test on real task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s2_instances = mine_S2(task_context, roles, role_stats)
    print(f"  S2 instances mined: {len(s2_instances)}")

    # Verify params format
    if s2_instances:
        inst = s2_instances[0]
        assert inst.family_id == "S2", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "input_color" in inst.params, "Missing input_color"
        assert "size_to_color" in inst.params, "Missing size_to_color"

        # Verify size_to_color uses string keys (not int)
        size_to_color = inst.params["size_to_color"]
        assert isinstance(size_to_color, dict), "size_to_color not a dict"
        if size_to_color:
            first_key = next(iter(size_to_color.keys()))
            assert isinstance(first_key, str), \
                f"size_to_color keys should be strings, got {type(first_key)}"
        print("  ✓ Params format matches builder")

    print("\n✓ S2 always-true mining correct")


def test_s10_always_true_mining():
    """
    Test S10 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - Uses existing component_border_interior φ operator
      - Aggregates border and interior colors separately
      - Requires BOTH consistent (len(border_colors) != 1 or len(interior_colors) != 1)
      - Params format matches build_S10_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 3: S10 Always-True Mining")
    print("=" * 70)

    # Check source code for TODOs
    source = inspect.getsource(mine_S10)
    assert "TODO" not in source, "S10 contains TODO"
    assert "NotImplementedError" not in source, "S10 raises NotImplementedError"
    assert "simplified" not in source.lower(), "S10 mentions 'simplified'"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S10 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check uses component_border_interior
    assert "component_border_interior" in source, \
        "S10 doesn't use component_border_interior φ operator"
    print("  ✓ Uses component_border_interior φ operator")

    # Check aggregates border and interior separately
    assert "border_colors" in source and "interior_colors" in source, \
        "S10 doesn't aggregate border and interior separately"
    print("  ✓ Aggregates border and interior colors")

    # Check requires BOTH consistent
    assert ("len(border_colors) != 1 or len(interior_colors) != 1" in source or
            "len(border_colors) == 1 and len(interior_colors) == 1" in source), \
        "S10 doesn't check BOTH border and interior consistency"
    print("  ✓ Requires BOTH border and interior consistent")

    # Test on real task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s10_instances = mine_S10(task_context, roles, role_stats)
    print(f"  S10 instances mined: {len(s10_instances)}")

    # Verify params format
    if s10_instances:
        inst = s10_instances[0]
        assert inst.family_id == "S10", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "border_color" in inst.params, "Missing border_color"
        assert "interior_color" in inst.params, "Missing interior_color"

        # Verify colors are ints
        assert isinstance(inst.params["border_color"], int), \
            f"border_color should be int, got {type(inst.params['border_color'])}"
        assert isinstance(inst.params["interior_color"], int), \
            f"interior_color should be int, got {type(inst.params['interior_color'])}"
        print("  ✓ Params format matches builder")

    print("\n✓ S10 always-true mining correct")


def test_multi_task_coverage():
    """
    Test miners on multiple tasks to ensure robustness.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Task Coverage")
    print("=" * 70)

    test_task_ids = ["00576224", "007bbfb7", "00d62c1b"]
    challenges_path = Path("data/arc-agi_training_challenges.json")

    for task_id in test_task_ids:
        print(f"\n  Task: {task_id}")

        raw_task = load_arc_task(task_id, challenges_path)
        task_context = build_task_context_from_raw(raw_task)
        roles = compute_roles(task_context)
        role_stats = compute_role_stats(task_context, roles)

        # Mine all schemas
        s1_instances = mine_S1(task_context, roles, role_stats)
        s2_instances = mine_S2(task_context, roles, role_stats)
        s10_instances = mine_S10(task_context, roles, role_stats)

        # Verify S1 always empty
        assert len(s1_instances) == 0, f"S1 should be empty, got {len(s1_instances)}"

        # Verify S2 instances (if any) have correct structure
        for inst in s2_instances:
            assert inst.family_id == "S2"
            assert all(k in inst.params for k in ["example_type", "example_index",
                                                   "input_color", "size_to_color"])

        # Verify S10 instances (if any) have correct structure
        for inst in s10_instances:
            assert inst.family_id == "S10"
            assert all(k in inst.params for k in ["example_type", "example_index",
                                                   "border_color", "interior_color"])

        print(f"    S1: {len(s1_instances)} (expected 0)")
        print(f"    S2: {len(s2_instances)}")
        print(f"    S10: {len(s10_instances)}")
        print(f"    ✓ Task passed")

    print("\n✓ Multi-task coverage correct")


def main():
    """Run all comprehensive review tests."""
    print("=" * 70)
    print("WO-M6.3A COMPREHENSIVE REVIEW TEST")
    print("Schema Miners for S1, S2, S10")
    print("=" * 70)

    try:
        test_s1_stub_implementation()
        test_s2_always_true_mining()
        test_s10_always_true_mining()
        test_multi_task_coverage()

        print("\n" + "=" * 70)
        print("✓ ALL REVIEW TESTS PASSED")
        print("=" * 70)
        print("\nWO-M6.3A implementation is CORRECT and ready for integration.")

    except AssertionError as e:
        print(f"\n✗ REVIEW TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
