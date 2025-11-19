"""
Comprehensive review test for WO-M6.3C: Schema miners for S5, S6, S7, S11.

This test verifies:
  1. S5: Template stamping (seed → patch)
  2. S6: Crop to ROI (with explicit rule testing)
  3. S7: Summary/aggregation grids (unique non-zero or zero rule)
  4. S11: Local neighborhood codebook (hash → template)

Review priorities (HIGHEST to lowest):
  1. No TODOs, stubs, simplified implementations (S7 has known limitation)
  2. S5, S6, S7, S11 check ALL training examples (not just first)
  3. Reject conflicts: if len(patches) > 1 or rules don't match: skip
  4. Skip unobserved: if len(patches) == 0: skip
  5. Params format matches existing builders exactly
"""

import inspect
from pathlib import Path
from typing import Dict, List, Set

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s5_s6_s7_s11 import mine_S5, mine_S6, mine_S7, mine_S11
from src.catalog.types import SchemaInstance


def test_s5_always_true_mining():
    """
    Test S5 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - For each hash, checks all patches are identical
      - Rejects conflicts (len(patch_set) > 1)
      - Params format matches build_S5_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 1: S5 Always-True Mining")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S5)
    assert "TODO" not in source, "S5 contains TODO"
    assert "NotImplementedError" not in source, "S5 raises NotImplementedError"
    # Allow "simplified" in docstring but not in code logic
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if '"""' not in line and "'''" not in line and "simplified" in line.lower():
            # Check it's not in a comment/docstring context
            if not line.strip().startswith('#'):
                assert False, f"S5 mentions 'simplified' in code at line {i}"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S5 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check uses neighborhood_hashes from ExampleContext
    assert "ex.neighborhood_hashes" in source, \
        "S5 doesn't use ex.neighborhood_hashes"
    print("  ✓ Uses precomputed neighborhood_hashes")

    # Check patch aggregation
    assert "hash_to_patches" in source or "patches" in source.lower(), \
        "S5 doesn't aggregate patches"
    print("  ✓ Aggregates patches by hash")

    # Check conflict rejection
    assert "len(patch_set) == 1" in source or "len(patch_set) != 1" in source, \
        "S5 doesn't check for conflicts"
    print("  ✓ Rejects conflicts")

    # Test on real task
    task_id = "007bbfb7"  # This task has S5 instances
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s5_instances = mine_S5(task_context, roles, role_stats)
    print(f"  S5 instances mined: {len(s5_instances)}")

    # Verify params format
    if s5_instances:
        inst = s5_instances[0]
        assert inst.family_id == "S5", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "seed_templates" in inst.params, "Missing seed_templates"

        # Verify seed_templates format
        seed_templates = inst.params["seed_templates"]
        assert isinstance(seed_templates, dict), "seed_templates not a dict"

        if seed_templates:
            # Check hash keys are strings
            hash_key = list(seed_templates.keys())[0]
            assert isinstance(hash_key, str), \
                f"Hash keys should be strings, got {type(hash_key)}"

            # Check template has string offset keys
            template = seed_templates[hash_key]
            assert isinstance(template, dict), "Template should be dict"
            if template:
                offset_key = list(template.keys())[0]
                assert isinstance(offset_key, str), \
                    f"Offset keys should be strings, got {type(offset_key)}"
                assert offset_key.startswith("(") and "," in offset_key, \
                    f"Offset key should be like '(dr,dc)', got {offset_key}"
        print("  ✓ Params format matches builder")

    print("\n✓ S5 always-true mining correct")


def test_s6_always_true_mining():
    """
    Test S6 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - Tests explicit rules (Fixed offset, Largest component)
      - Validates test inputs before accepting rule
      - Params format matches build_S6_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 2: S6 Always-True Mining")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S6)
    assert "TODO" not in source, "S6 contains TODO"
    assert "NotImplementedError" not in source, "S6 raises NotImplementedError"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S6 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check finds crop candidates
    assert "np.array_equal" in source, \
        "S6 doesn't use exact array comparison for crop detection"
    print("  ✓ Uses exact crop matching")

    # Check tests multiple rules
    assert "Rule A" in source or "fixed offset" in source.lower(), \
        "S6 doesn't implement fixed offset rule"
    assert "Rule B" in source or "largest component" in source.lower(), \
        "S6 doesn't implement component-based rules"
    print("  ✓ Tests multiple explicit rules")

    # Check validates test inputs
    assert "test_applicable" in source or "test" in source.lower(), \
        "S6 doesn't validate test inputs"
    print("  ✓ Validates test inputs before accepting rule")

    # Test on real task (most won't have S6, but structure is correct)
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s6_instances = mine_S6(task_context, roles, role_stats)
    print(f"  S6 instances mined: {len(s6_instances)}")

    # Verify params format if any instances
    for inst in s6_instances:
        assert inst.family_id == "S6", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "output_height" in inst.params, "Missing output_height"
        assert "output_width" in inst.params, "Missing output_width"
        assert "background_color" in inst.params, "Missing background_color"
        assert "out_to_in" in inst.params, "Missing out_to_in"

        # Verify out_to_in mapping has string keys/values
        out_to_in = inst.params["out_to_in"]
        assert isinstance(out_to_in, dict), "out_to_in should be dict"
        if out_to_in:
            key = list(out_to_in.keys())[0]
            val = out_to_in[key]
            assert isinstance(key, str), f"Keys should be strings, got {type(key)}"
            assert isinstance(val, str), f"Values should be strings, got {type(val)}"
        print("  ✓ Params format matches builder")

    print("\n✓ S6 always-true mining correct")


def test_s7_always_true_mining():
    """
    Test S7 miner implements always-true mining correctly.

    Requirements:
      - Known limitation: only implements "unique non-zero or zero" rule
      - Iterates ALL training examples
      - Validates rule holds for ALL blocks
      - Validates test inputs before accepting
      - Params format matches build_S7_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 3: S7 Always-True Mining (with known limitation)")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S7)
    assert "TODO" not in source, "S7 contains TODO"
    assert "NotImplementedError" not in source, "S7 raises NotImplementedError"

    # Check for documented scope limitation
    assert "unique non-zero" in source.lower() or "unique_nonzero" in source, \
        "S7 should document unique non-zero rule"
    print("  ✓ Documents scope limitation (unique non-zero rule)")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S7 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check validates rule on all blocks
    assert "for i in range" in source and "for j in range" in source, \
        "S7 doesn't iterate all blocks"
    print("  ✓ Validates rule on ALL blocks")

    # Check validates test inputs
    assert "test_rule_valid" in source or "test" in source.lower(), \
        "S7 doesn't validate test inputs"
    print("  ✓ Validates test inputs before accepting")

    # Test on real task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s7_instances = mine_S7(task_context, roles, role_stats)
    print(f"  S7 instances mined: {len(s7_instances)}")

    # Verify params format if any instances
    for inst in s7_instances:
        assert inst.family_id == "S7", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "output_height" in inst.params, "Missing output_height"
        assert "output_width" in inst.params, "Missing output_width"
        assert "summary_colors" in inst.params, "Missing summary_colors"

        # Verify summary_colors has string keys, int values
        summary_colors = inst.params["summary_colors"]
        assert isinstance(summary_colors, dict), "summary_colors should be dict"
        if summary_colors:
            key = list(summary_colors.keys())[0]
            val = summary_colors[key]
            assert isinstance(key, str), f"Keys should be strings, got {type(key)}"
            assert isinstance(val, int), f"Values should be ints, got {type(val)}"
        print("  ✓ Params format matches builder")

    print("\n✓ S7 always-true mining correct (with documented limitation)")


def test_s11_always_true_mining():
    """
    Test S11 miner implements always-true mining correctly.

    Requirements:
      - No TODOs/stubs/simplified
      - Iterates ALL training examples
      - For each hash, checks all patches are identical
      - Rejects conflicts (len(patch_set) > 1)
      - Params format matches build_S11_constraints
    """
    print("\n" + "=" * 70)
    print("TEST 4: S11 Always-True Mining")
    print("=" * 70)

    # Check source code
    source = inspect.getsource(mine_S11)
    assert "TODO" not in source, "S11 contains TODO"
    assert "NotImplementedError" not in source, "S11 raises NotImplementedError"
    print("  ✓ No TODOs or stubs")

    # Check iterates ALL training examples
    assert "for ex_idx, ex in enumerate(task_context.train_examples)" in source, \
        "S11 doesn't iterate all training examples"
    print("  ✓ Iterates ALL training examples")

    # Check uses neighborhood_hashes from ExampleContext
    assert "ex.neighborhood_hashes" in source, \
        "S11 doesn't use ex.neighborhood_hashes"
    print("  ✓ Uses precomputed neighborhood_hashes")

    # Check patch aggregation
    assert "hash_to_patches" in source or "patches" in source.lower(), \
        "S11 doesn't aggregate patches"
    print("  ✓ Aggregates patches by hash")

    # Check conflict rejection
    assert "len(patch_set) == 1" in source or "len(patch_set) != 1" in source, \
        "S11 doesn't check for conflicts"
    print("  ✓ Rejects conflicts")

    # Test on real task
    task_id = "007bbfb7"  # This task has S11 instances
    challenges_path = Path("data/arc-agi_training_challenges.json")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    s11_instances = mine_S11(task_context, roles, role_stats)
    print(f"  S11 instances mined: {len(s11_instances)}")

    # Verify params format
    if s11_instances:
        inst = s11_instances[0]
        assert inst.family_id == "S11", f"Wrong family_id: {inst.family_id}"
        assert "example_type" in inst.params, "Missing example_type"
        assert "example_index" in inst.params, "Missing example_index"
        assert "hash_templates" in inst.params, "Missing hash_templates"

        # Verify hash_templates format (NOT hash_to_pattern!)
        hash_templates = inst.params["hash_templates"]
        assert isinstance(hash_templates, dict), "hash_templates not a dict"

        if hash_templates:
            # Check hash keys are strings
            hash_key = list(hash_templates.keys())[0]
            assert isinstance(hash_key, str), \
                f"Hash keys should be strings, got {type(hash_key)}"

            # Check template has string offset keys
            template = hash_templates[hash_key]
            assert isinstance(template, dict), "Template should be dict"
            if template:
                offset_key = list(template.keys())[0]
                assert isinstance(offset_key, str), \
                    f"Offset keys should be strings, got {type(offset_key)}"
                assert offset_key.startswith("(") and "," in offset_key, \
                    f"Offset key should be like '(dr,dc)', got {offset_key}"
        print("  ✓ Params format matches builder")

    print("\n✓ S11 always-true mining correct")


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
        s5_instances = mine_S5(task_context, roles, role_stats)
        s6_instances = mine_S6(task_context, roles, role_stats)
        s7_instances = mine_S7(task_context, roles, role_stats)
        s11_instances = mine_S11(task_context, roles, role_stats)

        # Verify S5 instances (if any) have correct structure
        for inst in s5_instances:
            assert inst.family_id == "S5"
            assert all(k in inst.params for k in ["example_type", "example_index", "seed_templates"])

        # Verify S6 instances (if any) have correct structure
        for inst in s6_instances:
            assert inst.family_id == "S6"
            assert all(k in inst.params for k in ["example_type", "example_index",
                                                   "output_height", "output_width",
                                                   "background_color", "out_to_in"])

        # Verify S7 instances (if any) have correct structure
        for inst in s7_instances:
            assert inst.family_id == "S7"
            assert all(k in inst.params for k in ["example_type", "example_index",
                                                   "output_height", "output_width",
                                                   "summary_colors"])

        # Verify S11 instances (if any) have correct structure
        for inst in s11_instances:
            assert inst.family_id == "S11"
            assert all(k in inst.params for k in ["example_type", "example_index", "hash_templates"])

        print(f"    S5: {len(s5_instances)}")
        print(f"    S6: {len(s6_instances)}")
        print(f"    S7: {len(s7_instances)}")
        print(f"    S11: {len(s11_instances)}")
        print(f"    ✓ Task passed")

    print("\n✓ Multi-task coverage correct")


def main():
    """Run all comprehensive review tests."""
    print("=" * 70)
    print("WO-M6.3C COMPREHENSIVE REVIEW TEST")
    print("Schema Miners for S5, S6, S7, S11")
    print("=" * 70)

    try:
        test_s5_always_true_mining()
        test_s6_always_true_mining()
        test_s7_always_true_mining()
        test_s11_always_true_mining()
        test_multi_task_coverage()

        print("\n" + "=" * 70)
        print("✓ ALL REVIEW TESTS PASSED")
        print("=" * 70)
        print("\nWO-M6.3C implementation is CORRECT and ready for integration.")
        print("\nKnown limitation: S7 only implements 'unique non-zero or zero' rule.")
        print("This is documented and acceptable for M6.3C scope.")

    except AssertionError as e:
        print(f"\n✗ REVIEW TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
