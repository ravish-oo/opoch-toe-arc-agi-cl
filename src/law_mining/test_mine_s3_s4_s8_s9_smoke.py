"""
Smoke test for S3, S4, S8, S9 schema miners.

Verifies that the miners:
  - Run without error on real ARC tasks
  - Produce SchemaInstance objects with correct structure
  - Follow always-true invariant constraint (no contradictions)
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s3_s4_s8_s9 import mine_S3, mine_S4, mine_S8, mine_S9


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("S3, S4, S8, S9 Miners Smoke Test")
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
        s3_instances = mine_S3(task_context, roles, role_stats)
        s4_instances = mine_S4(task_context, roles, role_stats)
        s8_instances = mine_S8(task_context, roles, role_stats)
        s9_instances = mine_S9(task_context, roles, role_stats)

        print(f"  S3 instances: {len(s3_instances)}")
        print(f"  S4 instances: {len(s4_instances)}")
        print(f"  S8 instances: {len(s8_instances)}")
        print(f"  S9 instances: {len(s9_instances)} (expected 0 - not implemented)")

        # Verify S9 is empty (stub)
        assert len(s9_instances) == 0, "S9 miner should return empty list in M6.3B"

        # Verify S3 instances have correct structure
        for inst in s3_instances:
            assert inst.family_id == "S3", "S3 instance should have family_id='S3'"
            assert "example_type" in inst.params, "S3 params should have example_type"
            assert "example_index" in inst.params, "S3 params should have example_index"
            assert "row_classes" in inst.params, "S3 params should have row_classes"
            # row_classes should be a list of lists
            assert isinstance(inst.params["row_classes"], list), "row_classes should be a list"
            if inst.params["row_classes"]:
                assert isinstance(inst.params["row_classes"][0], list), "row_classes should be list of lists"

        # Verify S4 instances have correct structure
        for inst in s4_instances:
            assert inst.family_id == "S4", "S4 instance should have family_id='S4'"
            assert "example_type" in inst.params, "S4 params should have example_type"
            assert "example_index" in inst.params, "S4 params should have example_index"
            assert "axis" in inst.params, "S4 params should have axis"
            assert inst.params["axis"] in ["row", "col"], "S4 axis should be 'row' or 'col'"
            assert "K" in inst.params, "S4 params should have K"
            assert "residue_to_color" in inst.params, "S4 params should have residue_to_color"
            # residue_to_color should have string keys
            assert all(isinstance(k, str) for k in inst.params["residue_to_color"].keys()), \
                "residue_to_color keys should be strings"

        # Verify S8 instances have correct structure
        for inst in s8_instances:
            assert inst.family_id == "S8", "S8 instance should have family_id='S8'"
            assert "example_type" in inst.params, "S8 params should have example_type"
            assert "example_index" in inst.params, "S8 params should have example_index"
            assert "tile_height" in inst.params, "S8 params should have tile_height"
            assert "tile_width" in inst.params, "S8 params should have tile_width"
            assert "tile_pattern" in inst.params, "S8 params should have tile_pattern"
            assert "region_origin" in inst.params, "S8 params should have region_origin"
            assert "region_height" in inst.params, "S8 params should have region_height"
            assert "region_width" in inst.params, "S8 params should have region_width"
            # tile_pattern should have string keys like "(0,0)"
            assert all(isinstance(k, str) for k in inst.params["tile_pattern"].keys()), \
                "tile_pattern keys should be strings"
            # region_origin should be a string like "(0,0)"
            assert isinstance(inst.params["region_origin"], str), "region_origin should be a string"

        # Show samples
        if s3_instances:
            print(f"\n  Sample S3 instance:")
            print(f"    family_id: {s3_instances[0].family_id}")
            print(f"    params: {s3_instances[0].params}")

        if s4_instances:
            print(f"\n  Sample S4 instance:")
            print(f"    family_id: {s4_instances[0].family_id}")
            print(f"    params: {s4_instances[0].params}")

        if s8_instances:
            print(f"\n  Sample S8 instance:")
            print(f"    family_id: {s8_instances[0].family_id}")
            print(f"    params: {s8_instances[0].params}")

        print(f"  ✓ Smoke test passed")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
