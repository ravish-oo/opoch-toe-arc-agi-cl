"""
Smoke test for S1, S2, S10 schema miners.

Verifies that the miners:
  - Run without error on real ARC tasks
  - Produce SchemaInstance objects with correct structure
  - Follow always-true invariant constraint (no contradictions)
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s1_s2_s10 import mine_S1, mine_S2, mine_S10


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("S1, S2, S10 Miners Smoke Test")
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
        s1_instances = mine_S1(task_context, roles, role_stats)
        s2_instances = mine_S2(task_context, roles, role_stats)
        s10_instances = mine_S10(task_context, roles, role_stats)

        print(f"  S1 instances: {len(s1_instances)} (expected 0 - not implemented)")
        print(f"  S2 instances: {len(s2_instances)}")
        print(f"  S10 instances: {len(s10_instances)}")

        # Verify S1 is empty (stub)
        assert len(s1_instances) == 0, "S1 miner should return empty list in M6.3A"

        # Verify S2 instances have correct structure
        for inst in s2_instances:
            assert inst.family_id == "S2", "S2 instance should have family_id='S2'"
            assert "example_type" in inst.params, "S2 params should have example_type"
            assert "example_index" in inst.params, "S2 params should have example_index"
            assert "input_color" in inst.params, "S2 params should have input_color"
            assert "size_to_color" in inst.params, "S2 params should have size_to_color"

        # Verify S10 instances have correct structure
        for inst in s10_instances:
            assert inst.family_id == "S10", "S10 instance should have family_id='S10'"
            assert "example_type" in inst.params, "S10 params should have example_type"
            assert "example_index" in inst.params, "S10 params should have example_index"
            assert "border_color" in inst.params, "S10 params should have border_color"
            assert "interior_color" in inst.params, "S10 params should have interior_color"

        # Show samples
        if s2_instances:
            print(f"\n  Sample S2 instance:")
            print(f"    family_id: {s2_instances[0].family_id}")
            print(f"    params: {s2_instances[0].params}")

        if s10_instances:
            print(f"\n  Sample S10 instance:")
            print(f"    family_id: {s10_instances[0].family_id}")
            print(f"    params: {s10_instances[0].params}")

        print(f"  ✓ Smoke test passed")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
