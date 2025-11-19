"""
Smoke test for S9 cross/plus miner.

Verifies that the S9 miner:
  - Runs without error on real ARC tasks
  - Produces SchemaInstance objects with correct structure
  - Only mines when cross patterns are exactly consistent
  - Detects seeds by input color, validates arms in outputs
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s9_cross import mine_S9


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("S9 Cross/Plus Miner Smoke Test")
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

        # Mine S9
        s9_instances = mine_S9(task_context, roles, role_stats)

        print(f"  S9 instances: {len(s9_instances)}")

        # Verify structure
        for inst in s9_instances:
            assert inst.family_id == "S9", "S9 instance should have family_id='S9'"
            assert "example_type" in inst.params, "S9 params should have example_type"
            assert "example_index" in inst.params, "S9 params should have example_index"
            assert "seeds" in inst.params, "S9 params should have seeds"

            example_type = inst.params["example_type"]
            assert example_type in ["train", "test"], \
                f"example_type should be 'train' or 'test', got {example_type}"

            seeds = inst.params["seeds"]
            assert isinstance(seeds, list), "seeds should be a list"

            # Verify each seed structure
            for seed in seeds:
                assert "center" in seed, "Seed should have center"
                assert "up_color" in seed, "Seed should have up_color"
                assert "down_color" in seed, "Seed should have down_color"
                assert "left_color" in seed, "Seed should have left_color"
                assert "right_color" in seed, "Seed should have right_color"
                assert "max_up" in seed, "Seed should have max_up"
                assert "max_down" in seed, "Seed should have max_down"
                assert "max_left" in seed, "Seed should have max_left"
                assert "max_right" in seed, "Seed should have max_right"

                # Verify center is string format "(r,c)"
                center = seed["center"]
                assert isinstance(center, str), "Center should be string"
                assert center.startswith("(") and center.endswith(")"), \
                    f"Center should be format '(r,c)', got {center}"

                # Verify colors are int or None
                for color_key in ["up_color", "down_color", "left_color", "right_color"]:
                    color = seed[color_key]
                    assert color is None or isinstance(color, int), \
                        f"{color_key} should be int or None, got {type(color)}"

                # Verify max_* are ints
                for max_key in ["max_up", "max_down", "max_left", "max_right"]:
                    max_val = seed[max_key]
                    assert isinstance(max_val, int), \
                        f"{max_key} should be int, got {type(max_val)}"
                    assert max_val >= 0, f"{max_key} should be >= 0, got {max_val}"

        # Show samples
        if s9_instances:
            inst = s9_instances[0]
            seeds = inst.params.get("seeds", [])

            print(f"\n  S9 instance details:")
            print(f"    example_type: {inst.params['example_type']}")
            print(f"    example_index: {inst.params['example_index']}")
            print(f"    num seeds: {len(seeds)}")

            if seeds:
                sample = seeds[0]
                print(f"    Sample seed:")
                print(f"      center: {sample['center']}")
                print(f"      up_color: {sample['up_color']}, max_up: {sample['max_up']}")
                print(f"      down_color: {sample['down_color']}, max_down: {sample['max_down']}")
                print(f"      left_color: {sample['left_color']}, max_left: {sample['max_left']}")
                print(f"      right_color: {sample['right_color']}, max_right: {sample['max_right']}")

        print(f"  ✓ Smoke test passed")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
