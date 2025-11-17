"""
Catalog types for schema instances and task law configurations.

This module defines the data structures used to represent:
- SchemaInstance: A single schema (S1-S11) with its parameters
- TaskLawConfig: A collection of schema instances for a task

These are used by the kernel runner and Pi-agent to configure
constraint generation for specific tasks.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class SchemaInstance:
    """
    Represents a single schema instance with its parameters.

    A schema instance is a specific instantiation of a schema family
    (e.g., S1, S2, ..., S11) with concrete parameter values.

    Attributes:
        family_id: Schema family identifier ("S1", "S2", ..., "S11")
        params: Schema-specific parameters (format varies by family)

    Example:
        >>> # S1 instance: tie pixels (0,0) and (0,1) in train example 0
        >>> s1 = SchemaInstance(
        ...     family_id="S1",
        ...     params={
        ...         "ties": [{
        ...             "example_type": "train",
        ...             "example_index": 0,
        ...             "pairs": [((0, 0), (0, 1))]
        ...         }]
        ...     }
        ... )
        >>> s1.family_id
        'S1'
    """
    family_id: str          # "S1", "S2", etc.
    params: Dict[str, Any]  # schema-specific params


@dataclass
class TaskLawConfig:
    """
    Represents a complete law configuration for a task.

    A task law configuration is a collection of schema instances that
    together define all constraints for solving a task.

    Attributes:
        schema_instances: List of SchemaInstance objects to apply

    Example:
        >>> # Config with S1 tie and S2 recolor
        >>> config = TaskLawConfig(schema_instances=[
        ...     SchemaInstance("S1", {"ties": [...]}),
        ...     SchemaInstance("S2", {"input_color": 1, ...})
        ... ])
        >>> len(config.schema_instances)
        2
    """
    schema_instances: List[SchemaInstance]


if __name__ == "__main__":
    # Self-test
    print("Testing catalog types...")
    print("=" * 70)

    # Create a schema instance
    s1_instance = SchemaInstance(
        family_id="S1",
        params={
            "ties": [{
                "example_type": "train",
                "example_index": 0,
                "pairs": [((0, 0), (0, 1))]
            }]
        }
    )

    print(f"SchemaInstance created: family_id={s1_instance.family_id}")
    print(f"  params keys: {list(s1_instance.params.keys())}")

    # Create a task law config
    config = TaskLawConfig(schema_instances=[s1_instance])
    print(f"\nTaskLawConfig created: {len(config.schema_instances)} instances")

    # Create multi-schema config
    s2_instance = SchemaInstance(
        family_id="S2",
        params={
            "example_type": "train",
            "example_index": 0,
            "input_color": 1,
            "size_to_color": {"1": 3}
        }
    )

    multi_config = TaskLawConfig(schema_instances=[s1_instance, s2_instance])
    print(f"Multi-schema config: {len(multi_config.schema_instances)} instances")
    for inst in multi_config.schema_instances:
        print(f"  - {inst.family_id}")

    print("\n" + "=" * 70)
    print("✓ Catalog types self-test passed.")
