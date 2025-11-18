"""
Smoke test for role statistics aggregator.

Verifies that compute_role_stats:
  - Runs without error on real ARC tasks
  - Produces role stats for all roles
  - Correctly aggregates train_in, train_out, test_in appearances
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("Role Statistics Aggregator Smoke Test")
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

        # Compute roles
        roles = compute_roles(task_context)
        num_roles = len(set(roles.values()))
        print(f"  Distinct roles: {num_roles}")

        # Compute role stats
        role_stats = compute_role_stats(task_context, roles)
        print(f"  Role stats computed: {len(role_stats)}")

        # Verify all roles have stats
        role_ids_from_roles = set(roles.values())
        role_ids_from_stats = set(role_stats.keys())
        assert role_ids_from_roles == role_ids_from_stats, \
            f"Mismatch: {len(role_ids_from_roles)} roles but {len(role_ids_from_stats)} stats"

        # Count total appearances
        total_train_in = sum(len(stats.train_in) for stats in role_stats.values())
        total_train_out = sum(len(stats.train_out) for stats in role_stats.values())
        total_test_in = sum(len(stats.test_in) for stats in role_stats.values())

        print(f"  Total appearances:")
        print(f"    train_in:  {total_train_in}")
        print(f"    train_out: {total_train_out}")
        print(f"    test_in:   {total_test_in}")

        # Sanity checks
        assert total_train_in > 0, "Should have train_in appearances"
        assert total_test_in > 0, "Should have test_in appearances"
        # For training tasks with outputs, should have train_out
        if all(ex.output_grid is not None for ex in task_context.train_examples):
            assert total_train_out > 0, "Should have train_out appearances for tasks with outputs"

        print(f"  ✓ Smoke test passed")

        # Show a few sample role stats
        print(f"\n  Sample role stats:")
        for i, (role_id, stats) in enumerate(list(role_stats.items())[:3]):
            print(f"    Role {role_id}: train_in={len(stats.train_in)}, "
                  f"train_out={len(stats.train_out)}, test_in={len(stats.test_in)}")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
