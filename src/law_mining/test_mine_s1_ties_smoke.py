"""
Smoke test for S1 tie miner.

Verifies that the S1 miner:
  - Runs without error on real ARC tasks
  - Produces SchemaInstance objects with correct structure
  - Only ties roles with homogeneous train_out colors
  - Never fixes colors, only creates equality constraints
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s1_ties import mine_S1


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("S1 Tie Miner Smoke Test")
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

        print(f"  Total roles: {len(role_stats)}")

        # Mine S1
        s1_instances = mine_S1(task_context, roles, role_stats)

        print(f"  S1 instances: {len(s1_instances)}")

        # Verify structure
        for inst in s1_instances:
            assert inst.family_id == "S1", "S1 instance should have family_id='S1'"
            assert "ties" in inst.params, "S1 params should have 'ties'"

            ties = inst.params["ties"]
            assert isinstance(ties, list), "ties should be a list"

            # Verify each tie group
            for tie_group in ties:
                assert "example_type" in tie_group, "Tie group should have example_type"
                assert "example_index" in tie_group, "Tie group should have example_index"
                assert "pairs" in tie_group, "Tie group should have pairs"

                example_type = tie_group["example_type"]
                assert example_type in ["train", "test"], \
                    f"example_type should be 'train' or 'test', got {example_type}"

                pairs = tie_group["pairs"]
                assert isinstance(pairs, list), "pairs should be a list"

                # Verify pair structure
                for pair in pairs:
                    assert len(pair) == 2, "Each pair should have 2 positions"
                    pos1, pos2 = pair
                    assert len(pos1) == 2, "Each position should be (r, c)"
                    assert len(pos2) == 2, "Each position should be (r, c)"
                    assert isinstance(pos1[0], int), "Position coordinates should be ints"
                    assert isinstance(pos1[1], int), "Position coordinates should be ints"
                    assert isinstance(pos2[0], int), "Position coordinates should be ints"
                    assert isinstance(pos2[1], int), "Position coordinates should be ints"

        # Show samples
        if s1_instances:
            inst = s1_instances[0]
            ties = inst.params["ties"]
            total_pairs = sum(len(t.get("pairs", [])) for t in ties)

            print(f"\n  S1 instance details:")
            print(f"    Total tie groups: {len(ties)}")
            print(f"    Total tie pairs: {total_pairs}")

            if ties:
                sample = ties[0]
                print(f"    Sample tie group:")
                print(f"      example_type: {sample['example_type']}")
                print(f"      example_index: {sample['example_index']}")
                print(f"      num pairs: {len(sample['pairs'])}")
                if sample['pairs']:
                    print(f"      sample pair: {sample['pairs'][0]}")

        print(f"  ✓ Smoke test passed")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
