"""
Core kernel runner for ARC-AGI constraint solver.

This module provides the main entrypoint for solving ARC tasks using
the constraint-based approach:
  1. Load task data
  2. Build TaskContext with all φ features
  3. Apply schema instances to generate constraints
  4. (Future) Solve LP to get output grids

For M3.2, this implements steps 1-3 only. Solver integration comes later.
"""

from pathlib import Path
from typing import Optional

from src.core.arc_io import load_arc_training_challenges
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance
from src.catalog.types import TaskLawConfig


def solve_arc_task(
    task_id: str,
    law_config: TaskLawConfig,
    challenges_path: Optional[Path] = None
) -> ConstraintBuilder:
    """
    Core kernel entrypoint: build constraints for a task using law config.

    This is the main pipeline that:
      1. Loads task data (train + test examples)
      2. Builds TaskContext (all φ features from M1)
      3. Applies all schema instances from law_config
      4. Returns ConstraintBuilder with all constraints

    Future: will integrate LP solver to return predicted grids.

    Args:
        task_id: ARC task identifier
        law_config: TaskLawConfig with schema instances to apply
        challenges_path: Optional path to challenges file (defaults to standard location)

    Returns:
        ConstraintBuilder with all constraints generated

    Example:
        >>> from src.catalog.types import SchemaInstance, TaskLawConfig
        >>> config = TaskLawConfig(schema_instances=[
        ...     SchemaInstance("S1", {"ties": [...]})
        ... ])
        >>> builder = solve_arc_task("00576224", config)
        >>> len(builder.constraints)
        10
    """
    # 1. Load task
    if challenges_path is None:
        challenges_path = Path("data/arc-agi_training_challenges.json")

    task_data = load_arc_task(task_id, challenges_path)

    # 2. Build TaskContext
    ctx = build_task_context_from_raw(task_data)

    # 3. Create ConstraintBuilder
    builder = ConstraintBuilder()

    # 4. Apply all schema instances
    for schema_instance in law_config.schema_instances:
        apply_schema_instance(
            family_id=schema_instance.family_id,
            schema_params=schema_instance.params,
            task_context=ctx,
            builder=builder,
        )

    # 5. Return builder (solver integration will come later)
    return builder


if __name__ == "__main__":
    # Self-test with toy law config
    from src.catalog.types import SchemaInstance, TaskLawConfig

    print("Testing kernel runner with toy law config...")
    print("=" * 70)

    # Create a simple law config: S1 with one tie
    task_id = "00576224"  # Use a real task

    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={
                "ties": [{
                    "example_type": "train",
                    "example_index": 0,
                    "pairs": [((0, 0), (0, 1))]  # Tie top-left to top-right
                }]
            }
        )
    ])

    print(f"Task ID: {task_id}")
    print(f"Law config: {len(law_config.schema_instances)} schema instances")
    for inst in law_config.schema_instances:
        print(f"  - {inst.family_id}")

    # Run kernel
    builder = solve_arc_task(task_id, law_config)

    print(f"\nConstraints generated: {len(builder.constraints)}")
    assert len(builder.constraints) > 0, "Should have generated constraints"

    # Inspect first constraint
    if builder.constraints:
        c0 = builder.constraints[0]
        print(f"\nSample constraint (first):")
        print(f"  indices: {c0.indices}")
        print(f"  coeffs: {c0.coeffs}")
        print(f"  rhs: {c0.rhs}")

    print("\n" + "=" * 70)
    print("✓ Kernel runner self-test passed.")

    # Test with multi-schema config
    print("\n" + "=" * 70)
    print("Testing kernel with multi-schema config (S1 + S4)...")
    print("=" * 70)

    multi_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={
                "ties": [{
                    "example_type": "train",
                    "example_index": 0,
                    "pairs": [((0, 0), (0, 1))]
                }]
            }
        ),
        SchemaInstance(
            family_id="S4",
            params={
                "example_type": "train",
                "example_index": 0,
                "axis": "col",
                "K": 2,
                "residue_to_color": {"0": 1, "1": 3}
            }
        )
    ])

    print(f"Multi-schema config: {len(multi_config.schema_instances)} instances")
    for inst in multi_config.schema_instances:
        print(f"  - {inst.family_id}")

    builder2 = solve_arc_task(task_id, multi_config)
    print(f"\nConstraints generated: {len(builder2.constraints)}")
    assert len(builder2.constraints) > 0, "Should have generated constraints"

    print("\n" + "=" * 70)
    print("✓ Multi-schema kernel test passed.")
