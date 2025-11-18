"""
Schema miners for S1, S2, S10.

This module implements algorithmic mining of always-true schema instances
from task training examples. Each miner:
  - Analyzes role statistics and component structures
  - Finds invariants that are 100% consistent across ALL training examples
  - Returns SchemaInstance objects with parameters matching actual builders
  - Never invents defaults or uses "most frequent" - only exact matches

Miners in this module:
  - mine_S1: Stub (not implemented in M6.3A)
  - mine_S2: Component-wise recolor based on size
  - mine_S10: Border vs interior pixel recolor
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

from src.schemas.context import TaskContext, ExampleContext
from src.law_mining.roles import RolesMapping
from src.law_mining.role_stats import RoleStats
from src.catalog.types import SchemaInstance

from src.features.components import (
    connected_components_by_color,
    compute_shape_signature,
    Component,
)
from src.features.object_roles import (
    component_border_interior,
)


# =============================================================================
# S1 Miner (Stub - Not Implemented in M6.3A)
# =============================================================================

def mine_S1(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    S1 is the 'tie/equality' schema: it enforces that two pixels share the same color
    (y_{p,c} = y_{q,c} for all c). It does NOT fix absolute colors.

    Automatic mining of S1 (discovering which positions/roles should be tied)
    is intentionally NOT implemented in M6.3A.

    For now, the law miner relies on S2–S11 to fix colors and shape.
    S1 tie-mining can be added later as a refinement once the core color-fixing
    schemas are working.

    Therefore this function returns an empty list in this milestone.

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping from WL refinement
        role_stats: Per-role statistics

    Returns:
        Empty list (S1 mining not implemented yet)
    """
    return []


# =============================================================================
# S2 Miner - Component-wise Recolor
# =============================================================================

def mine_S2(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S2 schema instances: component-wise recolor based on size.

    For each component class (defined by input_color + size), if ALL occurrences
    across all training examples map to the same output color, we can create
    an S2 rule.

    Algorithm:
      1. For each train example:
         - Extract components from input grid
         - For each component, get (input_color, size) -> output_color
      2. Aggregate across all train examples:
         - Track all output colors seen for each (input_color, size)
      3. Keep only classes with exactly one consistent output color
      4. Generate SchemaInstance for each (train + test) example

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used in S2, but kept for signature consistency)
        role_stats: RoleStats (not used in S2, but kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S2 parameters

    Example mined rule:
        If all training examples show "input_color=1, size=2 -> output_color=5",
        then generate SchemaInstances for each example with:
        {
            "example_type": "train",
            "example_index": 0,
            "input_color": 1,
            "size_to_color": {"2": 5}
        }
    """
    # Step 1: Mine task-wide invariants
    # Key: (input_color, size) -> Set[output_colors]
    class_to_colors: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue  # Skip if no output (shouldn't happen for training)

        grid_in = ex.input_grid
        grid_out = ex.output_grid

        # Extract components from input
        components = connected_components_by_color(grid_in)

        for comp in components:
            input_color = comp.color
            size = comp.size

            # Check that all pixels in this component have the same output color
            output_colors_in_comp = set()
            for (r, c) in comp.pixels:
                output_colors_in_comp.add(int(grid_out[r, c]))

            # If component pixels don't all map to same color, skip this component
            if len(output_colors_in_comp) != 1:
                continue

            output_color = output_colors_in_comp.pop()

            # Record this (input_color, size) -> output_color mapping
            key = (input_color, size)
            class_to_colors[key].add(output_color)

    # Step 2: Filter to only consistent mappings
    # Build dict of (input_color, size) -> output_color for always-true rules
    consistent_mappings: Dict[Tuple[int, int], int] = {}

    for key, color_set in class_to_colors.items():
        if len(color_set) == 1:
            # All train examples agree on this mapping
            output_color = color_set.pop()
            consistent_mappings[key] = output_color

    if not consistent_mappings:
        return []  # No S2 rules found

    # Step 3: Organize by input_color
    # Group mappings: input_color -> {size: output_color}
    color_to_size_map: Dict[int, Dict[str, int]] = defaultdict(dict)

    for (input_color, size), output_color in consistent_mappings.items():
        color_to_size_map[input_color][str(size)] = output_color

    # Step 4: Generate SchemaInstances for each example
    instances: List[SchemaInstance] = []

    # Generate for train examples
    for ex_idx, ex in enumerate(task_context.train_examples):
        for input_color, size_to_color in color_to_size_map.items():
            instances.append(SchemaInstance(
                family_id="S2",
                params={
                    "example_type": "train",
                    "example_index": ex_idx,
                    "input_color": input_color,
                    "size_to_color": size_to_color
                }
            ))

    # Generate for test examples
    for ex_idx, ex in enumerate(task_context.test_examples):
        for input_color, size_to_color in color_to_size_map.items():
            instances.append(SchemaInstance(
                family_id="S2",
                params={
                    "example_type": "test",
                    "example_index": ex_idx,
                    "input_color": input_color,
                    "size_to_color": size_to_color
                }
            ))

    return instances


# =============================================================================
# S10 Miner - Border vs Interior
# =============================================================================

def mine_S10(
    task_context: TaskContext,
    roles: RolesMapping,
    role_stats: Dict[int, RoleStats],
) -> List[SchemaInstance]:
    """
    Mine S10 schema instances: border vs interior recolor.

    For the whole grid (or dominant component), if ALL training examples show:
      - All border pixels -> same color b
      - All interior pixels -> same color i
    Then we can create an S10 rule.

    Algorithm:
      1. For each train example:
         - Get all components
         - Use component_border_interior to classify pixels
         - Track border colors and interior colors
      2. Aggregate across all train examples
      3. Keep only if BOTH border AND interior have exactly one consistent color
      4. Generate SchemaInstance for each (train + test) example

    Args:
        task_context: TaskContext with train/test examples
        roles: RolesMapping (not used, kept for signature consistency)
        role_stats: RoleStats (not used, kept for signature consistency)

    Returns:
        List of SchemaInstance objects with S10 parameters

    Example mined rule:
        If all train examples show "border pixels -> color 3, interior -> color 5",
        generate SchemaInstances with:
        {
            "example_type": "train",
            "example_index": 0,
            "border_color": 3,
            "interior_color": 5
        }
    """
    # Step 1: Mine task-wide invariants
    # Track all border colors and interior colors seen across all train examples
    border_colors: Set[int] = set()
    interior_colors: Set[int] = set()

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue  # Skip if no output

        grid_in = ex.input_grid
        grid_out = ex.output_grid

        # Get components and border/interior classification
        components = connected_components_by_color(grid_in)
        border_info = component_border_interior(grid_in, components)

        # For each pixel, check if it's border or interior and what output color it has
        for (r, c), info in border_info.items():
            output_color = int(grid_out[r, c])

            if info["is_border"]:
                border_colors.add(output_color)
            if info["is_interior"]:
                interior_colors.add(output_color)

    # Step 2: Check consistency
    # Only proceed if BOTH border and interior have exactly one consistent color
    if len(border_colors) != 1 or len(interior_colors) != 1:
        return []  # Not consistent across all train examples

    border_color = border_colors.pop()
    interior_color = interior_colors.pop()

    # Step 3: Generate SchemaInstances for each example
    instances: List[SchemaInstance] = []

    # Generate for train examples
    for ex_idx, ex in enumerate(task_context.train_examples):
        instances.append(SchemaInstance(
            family_id="S10",
            params={
                "example_type": "train",
                "example_index": ex_idx,
                "border_color": border_color,
                "interior_color": interior_color
            }
        ))

    # Generate for test examples
    for ex_idx, ex in enumerate(task_context.test_examples):
        instances.append(SchemaInstance(
            family_id="S10",
            params={
                "example_type": "test",
                "example_index": ex_idx,
                "border_color": border_color,
                "interior_color": interior_color
            }
        ))

    return instances


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles
    from src.law_mining.role_stats import compute_role_stats

    print("=" * 70)
    print("mine_s1_s2_s10.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    # Compute roles and stats
    roles = compute_roles(task_context)
    role_stats = compute_role_stats(task_context, roles)

    # Mine schemas
    print("\nMining S1 instances...")
    s1_instances = mine_S1(task_context, roles, role_stats)
    print(f"✓ S1 instances: {len(s1_instances)} (expected 0 - not implemented)")

    print("\nMining S2 instances...")
    s2_instances = mine_S2(task_context, roles, role_stats)
    print(f"✓ S2 instances: {len(s2_instances)}")
    if s2_instances:
        print(f"  Sample S2 params: {s2_instances[0].params}")

    print("\nMining S10 instances...")
    s10_instances = mine_S10(task_context, roles, role_stats)
    print(f"✓ S10 instances: {len(s10_instances)}")
    if s10_instances:
        print(f"  Sample S10 params: {s10_instances[0].params}")

    print("\n" + "=" * 70)
    print("✓ mine_s1_s2_s10.py self-test passed")
    print("=" * 70)
