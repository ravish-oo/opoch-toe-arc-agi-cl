"""
Smoke test for role labeller.

Verifies that compute_roles:
  - Runs without error on real ARC tasks
  - Produces a reasonable number of roles
  - Is deterministic (same input -> same output)
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("Role Labeller Smoke Test")
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
        num_pixels = len(roles)
        num_roles = len(set(roles.values()))

        print(f"  Total pixels: {num_pixels}")
        print(f"  Distinct roles: {num_roles}")
        print(f"  Compression ratio: {num_roles}/{num_pixels} = {num_roles/num_pixels:.2%}")

        # Sanity checks
        assert num_roles > 0, "Should have at least one role"
        assert num_roles <= num_pixels, "Can't have more roles than pixels"

        # Test determinism
        roles2 = compute_roles(task_context)
        assert roles == roles2, f"compute_roles not deterministic on {task_id}"

        print(f"  ✓ Smoke test passed")

        # Show a few sample role assignments
        print(f"\n  Sample role assignments:")
        for i, (key, role_id) in enumerate(list(roles.items())[:5]):
            kind, ex_idx, r, c = key
            print(f"    {kind:12s} ex={ex_idx} ({r:2d},{c:2d}) -> role_id={role_id}")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
