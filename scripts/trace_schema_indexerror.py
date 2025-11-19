"""
Debug script to find EXACT location of IndexError in schema application.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance

task_id = "00d62c1b"
challenges_path = Path("data/arc-agi_training_challenges.json")

print(f"Tracing IndexError for task: {task_id}")
print("=" * 70)

# Load and mine
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)
law_config = mine_law_config(task_context)

print(f"Mined {len(law_config.schema_instances)} schema instances")

# Try to apply each schema to train example 0
print(f"\nApplying schemas to train example 0...")

builder = ConstraintBuilder()

for i, schema_inst in enumerate(law_config.schema_instances):
    print(f"\n[{i+1}/{len(law_config.schema_instances)}] Applying {schema_inst.family_id}...")
    print(f"  Params keys: {list(schema_inst.params.keys())}")

    try:
        apply_schema_instance(
            family_id=schema_inst.family_id,
            schema_params=schema_inst.params,
            task_context=task_context,
            builder=builder,
            example_type="train",
            example_index=0,
            schema_constraint_counts={},
        )
        print(f"  ✓ Applied (constraints: {len(builder.constraints)})")
    except Exception as e:
        print(f"  ✗ EXCEPTION: {type(e).__name__}: {e}")
        print(f"\n  Full traceback:")
        import traceback
        traceback.print_exc()
        print(f"\n  Schema that crashed:")
        print(f"    Family: {schema_inst.family_id}")
        print(f"    Params: {schema_inst.params}")
        break
