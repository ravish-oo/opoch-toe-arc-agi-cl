"""
Debug error tasks - check test example application.
"""

import sys
from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance


def debug_test_example(task_id: str):
    """Debug test example schema application."""
    print(f"Debugging test example for task: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")

    # Load and mine
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(task_context)

    print(f"Mined {len(law_config.schema_instances)} schema instances")
    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    # Try to apply schemas to first TEST example
    print("\nApplying schemas to TEST example 0...")
    builder = ConstraintBuilder()

    for i, schema_inst in enumerate(law_config.schema_instances):
        print(f"  [{i+1}/{len(law_config.schema_instances)}] Applying {schema_inst.family_id}...")

        apply_schema_instance(
            family_id=schema_inst.family_id,
            schema_params=schema_inst.params,
            task_context=task_context,
            builder=builder,
            example_type="test",  # ← TEST example
            example_index=0,
            schema_constraint_counts={},
        )
        print(f"      ✓ Applied (constraints: {len(builder.constraints)})")

    print(f"\nTotal constraints: {len(builder.constraints)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.debug_error_test <task_id>")
        sys.exit(1)

    task_id = sys.argv[1]
    debug_test_example(task_id)
