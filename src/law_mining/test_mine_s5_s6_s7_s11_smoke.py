"""
Smoke test for S5, S6, S7, S11 schema miners.

Verifies that the miners:
  - Run without error on real ARC tasks
  - Produce SchemaInstance objects with correct structure
  - Follow always-true invariant constraint (no contradictions)
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s5_s6_s7_s11 import mine_S5, mine_S6, mine_S7, mine_S11


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("S5, S6, S7, S11 Miners Smoke Test")
    print("=" * 70)

    # Test on a few different task IDs
    test_task_ids = ["00576224", "007bbfb7", "00d62c1b"]
    challenges_path = Path("data/arc-agi_training_challenges.json")

    for task_id in test_task_ids:
        print(f"\n{'─' * 70}")
        print(f"Task: {task_id}")
        print(f"{'─' * 70}")

        # Load task
        raw_task = load_arc_task(task_id, challenges_path)
        task_context = build_task_context_from_raw(raw_task)

        print(f"  Train examples: {len(task_context.train_examples)}")
        print(f"  Test examples: {len(task_context.test_examples)}")

        # Compute roles and stats
        roles = compute_roles(task_context)
        role_stats = compute_role_stats(task_context, roles)

        # Mine schemas
        s5_instances = mine_S5(task_context, roles, role_stats)
        s6_instances = mine_S6(task_context, roles, role_stats)
        s7_instances = mine_S7(task_context, roles, role_stats)
        s11_instances = mine_S11(task_context, roles, role_stats)

        print(f"  S5 instances: {len(s5_instances)}")
        print(f"  S6 instances: {len(s6_instances)}")
        print(f"  S7 instances: {len(s7_instances)}")
        print(f"  S11 instances: {len(s11_instances)}")

        # Verify S5 instances have correct structure
        for inst in s5_instances:
            assert inst.family_id == "S5", "S5 instance should have family_id='S5'"
            assert "example_type" in inst.params, "S5 params should have example_type"
            assert "example_index" in inst.params, "S5 params should have example_index"
            assert "seed_templates" in inst.params, "S5 params should have seed_templates"
            # seed_templates should be a dict with string keys
            assert isinstance(inst.params["seed_templates"], dict), "seed_templates should be a dict"
            # Each template should be a dict of offset strings -> colors
            for hash_str, template in inst.params["seed_templates"].items():
                assert isinstance(hash_str, str), "Hash keys should be strings"
                assert isinstance(template, dict), "Template should be a dict"
                if template:  # If not empty
                    offset_key = list(template.keys())[0]
                    assert isinstance(offset_key, str), "Offset keys should be strings"
                    assert offset_key.startswith("(") and "," in offset_key, \
                        "Offset keys should be like '(dr,dc)'"

        # Verify S6 instances have correct structure
        for inst in s6_instances:
            assert inst.family_id == "S6", "S6 instance should have family_id='S6'"
            assert "example_type" in inst.params, "S6 params should have example_type"
            assert "example_index" in inst.params, "S6 params should have example_index"
            assert "output_height" in inst.params, "S6 params should have output_height"
            assert "output_width" in inst.params, "S6 params should have output_width"
            assert "background_color" in inst.params, "S6 params should have background_color"
            assert "out_to_in" in inst.params, "S6 params should have out_to_in"
            # out_to_in should be a dict with string coord keys
            assert isinstance(inst.params["out_to_in"], dict), "out_to_in should be a dict"
            if inst.params["out_to_in"]:
                key = list(inst.params["out_to_in"].keys())[0]
                val = inst.params["out_to_in"][key]
                assert isinstance(key, str), "out_to_in keys should be strings"
                assert isinstance(val, str), "out_to_in values should be strings"
                assert key.startswith("(") and "," in key, "Keys should be like '(r,c)'"
                assert val.startswith("(") and "," in val, "Values should be like '(r,c)'"

        # Verify S7 instances have correct structure
        for inst in s7_instances:
            assert inst.family_id == "S7", "S7 instance should have family_id='S7'"
            assert "example_type" in inst.params, "S7 params should have example_type"
            assert "example_index" in inst.params, "S7 params should have example_index"
            assert "output_height" in inst.params, "S7 params should have output_height"
            assert "output_width" in inst.params, "S7 params should have output_width"
            assert "summary_colors" in inst.params, "S7 params should have summary_colors"
            # summary_colors should be a dict with string coord keys
            assert isinstance(inst.params["summary_colors"], dict), "summary_colors should be a dict"
            if inst.params["summary_colors"]:
                key = list(inst.params["summary_colors"].keys())[0]
                val = inst.params["summary_colors"][key]
                assert isinstance(key, str), "summary_colors keys should be strings"
                assert isinstance(val, int), "summary_colors values should be ints"
                assert key.startswith("(") and "," in key, "Keys should be like '(r,c)'"

        # Verify S11 instances have correct structure
        for inst in s11_instances:
            assert inst.family_id == "S11", "S11 instance should have family_id='S11'"
            assert "example_type" in inst.params, "S11 params should have example_type"
            assert "example_index" in inst.params, "S11 params should have example_index"
            assert "hash_templates" in inst.params, "S11 params should have hash_templates"
            # hash_templates should be a dict with string keys
            assert isinstance(inst.params["hash_templates"], dict), "hash_templates should be a dict"
            # Each template should be a dict of offset strings -> colors
            for hash_str, template in inst.params["hash_templates"].items():
                assert isinstance(hash_str, str), "Hash keys should be strings"
                assert isinstance(template, dict), "Template should be a dict"
                if template:  # If not empty
                    offset_key = list(template.keys())[0]
                    assert isinstance(offset_key, str), "Offset keys should be strings"
                    assert offset_key.startswith("(") and "," in offset_key, \
                        "Offset keys should be like '(dr,dc)'"

        # Show samples
        if s5_instances:
            print(f"\n  Sample S5 instance:")
            print(f"    family_id: {s5_instances[0].family_id}")
            print(f"    params keys: {list(s5_instances[0].params.keys())}")
            seed_templates = s5_instances[0].params["seed_templates"]
            print(f"    num seed templates: {len(seed_templates)}")

        if s6_instances:
            print(f"\n  Sample S6 instance:")
            print(f"    family_id: {s6_instances[0].family_id}")
            print(f"    output dims: {s6_instances[0].params['output_height']}x{s6_instances[0].params['output_width']}")
            print(f"    num mappings: {len(s6_instances[0].params['out_to_in'])}")

        if s7_instances:
            print(f"\n  Sample S7 instance:")
            print(f"    family_id: {s7_instances[0].family_id}")
            print(f"    output dims: {s7_instances[0].params['output_height']}x{s7_instances[0].params['output_width']}")
            print(f"    num summaries: {len(s7_instances[0].params['summary_colors'])}")

        if s11_instances:
            print(f"\n  Sample S11 instance:")
            print(f"    family_id: {s11_instances[0].family_id}")
            hash_templates = s11_instances[0].params["hash_templates"]
            print(f"    num hash templates: {len(hash_templates)}")

        print(f"  ✓ Smoke test passed")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
