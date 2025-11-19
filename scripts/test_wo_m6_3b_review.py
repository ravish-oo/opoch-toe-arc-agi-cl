"""
Comprehensive review test for WO-M6.3B: Schema miners for S3, S4, S8, S9.

This test verifies:
  1. S3: Band/stripe patterns (rows in same band tied together)
  2. S4: Residue-class coloring (mod K patterns)
  3. S8: Tiling/replication (repeated base tile)
  4. S9: Stub implementation (deferred like S1)

Review priorities (HIGHEST to lowest):
  1. No TODOs, stubs, simplified implementations (except S9 intentional stub)
  2. S3, S4, S8 check ALL training examples (not just first)
  3. Reject conflicts: if len(colors) > 1 or patterns differ: skip
  4. Skip unobserved: if len(colors) == 0: skip
  5. Params format matches existing builders exactly
"""

import inspect
from pathlib import Path
from typing import Dict, List, Set

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s3_s4_s8_s9 import mine_S3, mine_S4, mine_S8, mine_S9
from src.catalog.types import SchemaInstance


def test_s9_stub_implementation():
    """
    Test S9 miner is intentional stub (like S1 in M6.3A).

    Requirements:
      - Correct signature
      - Explanatory docstring
      - Returns empty list
      - No raise NotImplementedError
    """
    print("\n" + "=" * 70)
    print("TEST 1: S9 Stub Implementation")
    print("=" * 70)

    # Check signature
    sig = inspect.signature(mine_S9)
    params = list(sig.parameters.keys())
    assert params == ["task_context", "roles", "role_stats"], \
        f"S9 signature wrong: {params}"
    print("  ✓ Signature correct")

    # Check docstring
    docstring = inspect.getdoc(mine_S9)
    assert docstring is not None, "S9 has no docstring"
    assert len(docstring) > 50, "S9 docstring too short"
    assert "S9" in docstring, "S9 docstring doesn't mention S9"
    assert ("not implemented" in docstring.lower() or
            "deferred" in docstring.lower() or
            "empty" in docstring.lower()), \
        "S9 docstring doesn't explain why it's a stub"
    print("  ✓ Docstring explanatory")

    # Check returns empty list
    from src.schemas.context import TaskContext, build_example_context
    import numpy as np

    dummy_grid = np.zeros((3, 3), dtype=np.int8)
    dummy_ex = build_example_context(dummy_grid, dummy_grid)
    dummy_task = TaskContext(
        train_examples=[dummy_ex],
        test_examples=[],
        C=1
    )
    dummy_roles = {}
    dummy_stats = {}

    result = mine_S9(dummy_task, dummy_roles, dummy_stats)
    assert result == [], f"S9 should return [], got {result}"
    print("  ✓ Returns empty list")

    print("\n✓ S9 stub implementation correct")


def test_s3_always_true_mining():
    """
    Test S3 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - Checks all rows in same band have identical patterns
      - Intersects valid bands across examples
      - Params format matches build_S3_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 2: S3 Always-True Mining")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S3)
    assert "TODO" not in source, "S3 contains TODO"
    assert "NotImplementedError" not in source, "S3 raises NotImplementedError"
    assert "simplified" not in source.lower(), "S3 mentions 'simplified'"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S3 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check pattern consistency check
    assert "patterns" in source.lower() and "set(patterns)" in source, \
        "S3 doesn't check pattern consistency"
    print("  ✓ Checks pattern consistency")

    # Check intersects across examples
    assert "intersection" in source.lower() or "intersect" in source.lower(), \
        "S3 doesn't intersect valid bands across examples"
    print("  ✓ Intersects valid bands across examples")

    # Test on real task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s3_instances = mine_S3(task_context, roles, role_stats)
    print(f"  S3 instances mined: {len(s3_instances)}")

    # Verify params format
    for inst in s3_instances:
        assert inst.family_id == "S3", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "row_classes" in inst.params, "Missing row_classes"
        assert isinstance(inst.params["row_classes"], list), "row_classes not a list"
        if inst.params["row_classes"]:
            assert isinstance(inst.params["row_classes"][0], list), \
                "row_classes should be list of lists"
    print("  ✓ Params format matches builder")

    print("\n✓ S3 always-true mining correct")


def test_s4_always_true_mining():
    """
    Test S4 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - Checks all pixels with same residue map to same color
      - Rejects conflicts (len(colors) > 1)
      - Params format matches build_S4_constraints (axis, string keys)
    """
    print("\n" + "=" * 70)
    print("TEST 3: S4 Always-True Mining")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S4)
    assert "TODO" not in source, "S4 contains TODO"
    assert "NotImplementedError" not in source, "S4 raises NotImplementedError"
    assert "simplified" not in source.lower(), "S4 mentions 'simplified'"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S4 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check residue aggregation
    assert "residue_to_colors" in source or "residue" in source.lower(), \
        "S4 doesn't aggregate by residue"
    print("  ✓ Aggregates by residue")

    # Check conflict rejection
    assert "len(colors) != 1" in source or "len(colors) == 1" in source, \
        "S4 doesn't check for conflicts"
    print("  ✓ Rejects conflicts")

    # Test on real task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s4_instances = mine_S4(task_context, roles, role_stats)
    print(f"  S4 instances mined: {len(s4_instances)}")

    # Verify params format
    for inst in s4_instances:
        assert inst.family_id == "S4", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "axis" in inst.params, "Missing axis"
        assert inst.params["axis"] in ["row", "col"], \
            f"axis should be 'row' or 'col', got {inst.params['axis']}"
        assert "K" in inst.params, "Missing K"
        assert "residue_to_color" in inst.params, "Missing residue_to_color"
        # Verify string keys
        assert all(isinstance(k, str) for k in inst.params["residue_to_color"].keys()), \
            "residue_to_color keys should be strings"
    print("  ✓ Params format matches builder")

    print("\n✓ S4 always-true mining correct")


def test_s8_always_true_mining():
    """
    Test S8 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - Finds tilings for each example
      - Intersects to find common tiling
      - Returns [] if no consistent tiling
      - Params format matches build_S8_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 4: S8 Always-True Mining")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S8)
    assert "TODO" not in source, "S8 contains TODO"
    assert "NotImplementedError" not in source, "S8 raises NotImplementedError"
    assert "simplified" not in source.lower(), "S8 mentions 'simplified'"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S8 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check tiling logic
    assert "np.tile" in source or "tile" in source.lower(), \
        "S8 doesn't use tiling"
    assert "np.array_equal" in source, \
        "S8 doesn't check exact tiling match"
    print("  ✓ Uses tiling with exact match check")

    # Check consistency across examples
    assert "common" in source.lower() or "intersect" in source.lower(), \
        "S8 doesn't check for common tiling across examples"
    print("  ✓ Checks for common tiling")

    # Test on real task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s8_instances = mine_S8(task_context, roles, role_stats)
    print(f"  S8 instances mined: {len(s8_instances)}")

    # Verify params format
    for inst in s8_instances:
        assert inst.family_id == "S8", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "tile_height" in inst.params, "Missing tile_height"
        assert "tile_width" in inst.params, "Missing tile_width"
        assert "tile_pattern" in inst.params, "Missing tile_pattern"
        assert "region_origin" in inst.params, "Missing region_origin"
        assert "region_height" in inst.params, "Missing region_height"
        assert "region_width" in inst.params, "Missing region_width"
        # Verify string keys
        assert all(isinstance(k, str) for k in inst.params["tile_pattern"].keys()), \
            "tile_pattern keys should be strings"
        # Verify region_origin is string
        assert isinstance(inst.params["region_origin"], str), \
            "region_origin should be a string"
    print("  ✓ Params format matches builder")

    print("\n✓ S8 always-true mining correct")


def test_multi_task_coverage():
    """
    Test miners on multiple tasks to ensure robustness.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Task Coverage")
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
        s3_instances = mine_S3(task_context, roles, role_stats)
        s4_instances = mine_S4(task_context, roles, role_stats)
        s8_instances = mine_S8(task_context, roles, role_stats)
        s9_instances = mine_S9(task_context, roles, role_stats)

        # Verify S9 always empty
        assert len(s9_instances) == 0, f"S9 should be empty, got {len(s9_instances)}"

        # Verify S3 instances (if any) have correct structure
        for inst in s3_instances:
            assert inst.family_id == "S3"
            assert all(k in inst.params for k in ["example_type", "example_index", "row_classes"])

        # Verify S4 instances (if any) have correct structure
        for inst in s4_instances:
            assert inst.family_id == "S4"
            assert all(k in inst.params for k in ["example_type", "example_index", "axis", "K", "residue_to_color"])

        # Verify S8 instances (if any) have correct structure
        for inst in s8_instances:
            assert inst.family_id == "S8"
            assert all(k in inst.params for k in ["example_type", "example_index",
                                                   "tile_height", "tile_width", "tile_pattern",
                                                   "region_origin", "region_height", "region_width"])

        print(f"    S3: {len(s3_instances)}")
        print(f"    S4: {len(s4_instances)}")
        print(f"    S8: {len(s8_instances)}")
        print(f"    S9: {len(s9_instances)} (expected 0)")
        print(f"    ✓ Task passed")

    print("\n✓ Multi-task coverage correct")


def main():
    """Run all comprehensive review tests."""
    print("=" * 70)
    print("WO-M6.3B COMPREHENSIVE REVIEW TEST")
    print("Schema Miners for S3, S4, S8, S9")
    print("=" * 70)

    try:
        test_s9_stub_implementation()
        test_s3_always_true_mining()
        test_s4_always_true_mining()
        test_s8_always_true_mining()
        test_multi_task_coverage()

        print("\n" + "=" * 70)
        print("✓ ALL REVIEW TESTS PASSED")
        print("=" * 70)
        print("\nWO-M6.3B implementation is CORRECT and ready for integration.")

    except AssertionError as e:
        print(f"\n✗ REVIEW TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
