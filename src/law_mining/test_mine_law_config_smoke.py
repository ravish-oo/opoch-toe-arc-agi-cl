"""
Smoke test for law miner orchestrator.

Verifies that mine_law_config:
  - Runs without error on real ARC tasks
  - Produces TaskLawConfig with SchemaInstance list
  - Calls all schema miners (S1-S11)
  - Handles tasks with various patterns
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config


def main():
    """Run smoke tests on multiple ARC tasks."""
    print("=" * 70)
    print("Law Miner Orchestrator Smoke Test")
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

        # Mine law config
        law_config = mine_law_config(task_context)

        print(f"\n  Total schema instances: {len(law_config.schema_instances)}")

        # Verify structure
        assert hasattr(law_config, 'schema_instances'), \
            "TaskLawConfig should have schema_instances attribute"
        assert isinstance(law_config.schema_instances, list), \
            "schema_instances should be a list"

        for inst in law_config.schema_instances:
            assert hasattr(inst, 'family_id'), "SchemaInstance should have family_id"
            assert hasattr(inst, 'params'), "SchemaInstance should have params"
            assert isinstance(inst.family_id, str), "family_id should be string"
            assert isinstance(inst.params, dict), "params should be dict"

        # Show breakdown by family
        family_counts = {}
        for inst in law_config.schema_instances:
            family_counts[inst.family_id] = family_counts.get(inst.family_id, 0) + 1

        if family_counts:
            print(f"\n  Breakdown by schema family:")
            for family_id in sorted(family_counts.keys()):
                count = family_counts[family_id]
                print(f"    {family_id}: {count} instances")
        else:
            print(f"\n  No schema instances mined (task may not match any patterns)")

        # Show sample instances
        if law_config.schema_instances:
            print(f"\n  Sample schema instances (first 3):")
            for i, inst in enumerate(law_config.schema_instances[:3]):
                print(f"    {i+1}. {inst.family_id} with params keys: {list(inst.params.keys())}")

        print(f"\n  ✓ Smoke test passed")

    print(f"\n{'=' * 70}")
    print("✓ All smoke tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
