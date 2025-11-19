"""
Debug error tasks with uncaught exceptions to see full traceback.
"""

import sys
from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance


def debug_error_task(task_id: str):
    """Debug error task without catching exceptions."""
    print(f"Debugging error task: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")

    # Load and mine (this works)
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(task_context)

    print(f"Mined {len(law_config.schema_instances)} schema instances")

    # Try to apply schemas to first train example
    print("\nApplying schemas to train example 0...")
    builder = ConstraintBuilder()

    for i, schema_inst in enumerate(law_config.schema_instances):
        print(f"  [{i+1}/{len(law_config.schema_instances)}] Applying {schema_inst.family_id}...")

        # This is where the error likely occurs
        apply_schema_instance(
            family_id=schema_inst.family_id,
            schema_params=schema_inst.params,
            task_context=task_context,
            builder=builder,
            example_type="train",
            example_index=0,
            schema_constraint_counts={},
        )
        print(f"      ✓ Applied (constraints: {len(builder.constraints)})")

    print(f"\nTotal constraints: {len(builder.constraints)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.debug_error_uncaught <task_id>")
        sys.exit(1)

    task_id = sys.argv[1]
    debug_error_task(task_id)
