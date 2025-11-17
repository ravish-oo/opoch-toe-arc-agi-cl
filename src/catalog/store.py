"""
Catalog storage for task law configurations.

This module provides functions to load and save TaskLawConfig objects
to/from a persistent catalog on disk. Each task has its own JSON file
storing the validated law configuration.

Storage structure:
    catalog/tasks/{task_id}.json

Each JSON file contains a serialized TaskLawConfig with schema instances
and their parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.catalog.types import TaskLawConfig, SchemaInstance


# Default catalog directory
DEFAULT_CATALOG_DIR = Path("catalog/tasks")


def load_task_law_config(
    task_id: str,
    catalog_dir: Path = DEFAULT_CATALOG_DIR
) -> Optional[TaskLawConfig]:
    """
    Load a TaskLawConfig for a given task from the catalog.

    Args:
        task_id: ARC task identifier
        catalog_dir: Directory containing task config files

    Returns:
        TaskLawConfig if file exists and is valid, None otherwise

    Example:
        >>> config = load_task_law_config("00576224")
        >>> if config:
        ...     print(f"Found {len(config.schema_instances)} schemas")
    """
    config_path = catalog_dir / f"{task_id}.json"

    # Return None if file doesn't exist
    if not config_path.exists():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct TaskLawConfig from JSON
        schema_instances = []
        for si_data in data.get("schema_instances", []):
            schema_instances.append(
                SchemaInstance(
                    family_id=si_data["family_id"],
                    params=si_data["params"]
                )
            )

        return TaskLawConfig(schema_instances=schema_instances)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # If file is corrupted or has wrong format, return None
        # (Could log warning here in future)
        return None


def save_task_law_config(
    task_id: str,
    config: TaskLawConfig,
    catalog_dir: Path = DEFAULT_CATALOG_DIR
) -> None:
    """
    Save a TaskLawConfig for a given task to the catalog.

    Creates the catalog directory if it doesn't exist.
    Overwrites any existing config for this task.

    Args:
        task_id: ARC task identifier
        config: TaskLawConfig to save
        catalog_dir: Directory to store task config files

    Example:
        >>> from src.catalog.types import SchemaInstance, TaskLawConfig
        >>> config = TaskLawConfig(schema_instances=[
        ...     SchemaInstance("S1", {"ties": [...]})
        ... ])
        >>> save_task_law_config("00576224", config)
    """
    # Create directory if needed
    catalog_dir.mkdir(parents=True, exist_ok=True)

    config_path = catalog_dir / f"{task_id}.json"

    # Serialize TaskLawConfig to JSON
    data = {
        "task_id": task_id,
        "schema_instances": [
            {
                "family_id": si.family_id,
                "params": si.params
            }
            for si in config.schema_instances
        ]
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Self-test: save and load a config
    print("Testing catalog store...")
    print("=" * 70)

    from src.catalog.types import SchemaInstance, TaskLawConfig

    # Create test config
    test_task_id = "test_task_00576224"
    test_config = TaskLawConfig(
        schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={
                    "ties": [{
                        "pairs": [((0, 0), (0, 1))]
                    }]
                }
            )
        ]
    )

    print(f"Test task ID: {test_task_id}")
    print(f"Original config: {len(test_config.schema_instances)} schema(s)")

    # Save
    save_task_law_config(test_task_id, test_config)
    print(f"✓ Saved to catalog/tasks/{test_task_id}.json")

    # Load
    loaded_config = load_task_law_config(test_task_id)
    assert loaded_config is not None, "Failed to load config"
    assert len(loaded_config.schema_instances) == 1
    assert loaded_config.schema_instances[0].family_id == "S1"
    print(f"✓ Loaded config: {len(loaded_config.schema_instances)} schema(s)")

    # Test non-existent task
    missing_config = load_task_law_config("nonexistent_task_999")
    assert missing_config is None, "Should return None for missing config"
    print("✓ Returns None for non-existent task")

    # Cleanup test file
    test_path = Path(f"catalog/tasks/{test_task_id}.json")
    if test_path.exists():
        test_path.unlink()
        print(f"✓ Cleaned up test file")

    print("\n" + "=" * 70)
    print("✓ Catalog store self-test passed")
    print("=" * 70)
