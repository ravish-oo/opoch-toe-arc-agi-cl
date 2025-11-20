"""
Diagnose infeasible tasks: Find conflicting schema constraints.

For infeasible tasks, the ILP solver can't find a solution because
constraints are contradictory. This script identifies:
1. Which pixel in test output has conflicting constraints
2. Which schemas are fighting (e.g., S5 says color 2, S11 says color 3)
3. Why they conflict (spurious co-occurrence in training data)

Usage:
    python scripts/diagnose_infeasible.py <task_id>
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance


def diagnose_infeasible_conflicts(task_id: str):
    """
    Find conflicting constraints in an infeasible task.

    Args:
        task_id: ARC task ID that has status=infeasible
    """
    print("=" * 70)
    print(f"INFEASIBLE CONFLICT DIAGNOSIS: {task_id}")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")

    # 1. Load and mine
    print("\n[1] Loading task and mining schemas...")
    raw_task = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(ctx)

    num_schemas = len(law_config.schema_instances)
    print(f"    ✓ Mined {num_schemas} schema instances")

    # Count by family
    schema_counts = defaultdict(int)
    for si in law_config.schema_instances:
        schema_counts[si.family_id] += 1

    print(f"    Schema breakdown:")
    for family_id, count in sorted(schema_counts.items()):
        print(f"      {family_id}: {count}")

    # 2. Analyze ALL examples (train + test)
    print(f"\n[2] Analyzing all example constraints...")

    all_examples = []
    for i, ex in enumerate(ctx.train_examples):
        all_examples.append(("train", i, ex))
    for i, ex in enumerate(ctx.test_examples):
        all_examples.append(("test", i, ex))

    for example_type, ex_idx, ex in all_examples:
        print(f"\n{'=' * 70}")
        print(f"{example_type.upper()} EXAMPLE {ex_idx}")
        print(f"{'=' * 70}")
        print(f"Input: {ex.input_H}×{ex.input_W}")
        print(f"Output: {ex.output_H}×{ex.output_W}")

        # Skip if output dimensions unknown
        if ex.output_H is None or ex.output_W is None:
            print("  (Skipping - output dimensions unknown)")
            continue

        # Track constraints per pixel per schema
        # pixel_constraints[(r, c)][schema_family] = [(constraint_type, color), ...]
        pixel_constraints = defaultdict(lambda: defaultdict(list))

        # Apply each schema and track what it constrains
        for si in law_config.schema_instances:
            builder = ConstraintBuilder()

            # Apply schema to this example
            apply_schema_instance(
                family_id=si.family_id,
                schema_params=si.params,
                task_context=ctx,
                builder=builder,
                example_type=example_type,
                example_index=ex_idx
            )

            # Parse constraints to find pixel fixings
            for constraint in builder.constraints:
                # Constraint format: LinearConstraint(indices, coeffs, rhs)
                # For fix_pixel_color(p_idx, color, C):
                #   Constraint: y[p_idx * C + color] == 1
                #   indices: [p_idx * C + color]
                #   coeffs: [1.0]
                #   rhs: 1.0

                # Check if this is a pixel fixing constraint (single var == 1)
                if constraint.rhs == 1.0 and len(constraint.indices) == 1 and constraint.coeffs[0] == 1.0:
                    var_idx = constraint.indices[0]

                    # Decode var_idx to (pixel_idx, color)
                    # var_idx = p_idx * C + color
                    C = ctx.C
                    p_idx = var_idx // C
                    color = var_idx % C

                    # Decode pixel index to (r, c)
                    W = ex.output_W
                    r = p_idx // W
                    c = p_idx % W

                    # Record this constraint
                    pixel_constraints[(r, c)][si.family_id].append(("fix", color))

        # 3. Find conflicting pixels
        print(f"\n[3] Checking for conflicts...")

        conflicts_found = 0

        for (r, c), schema_constraints in sorted(pixel_constraints.items()):
            # Get all colors fixed by all schemas for this pixel
            colors_by_schema = {}
            for schema_family, constraints in schema_constraints.items():
                colors = set(color for constraint_type, color in constraints if constraint_type == "fix")
                if colors:
                    colors_by_schema[schema_family] = colors

            # Check for conflicts
            if len(colors_by_schema) > 1:
                # Multiple schemas constraining same pixel
                # Check if they agree
                all_colors = set()
                for colors in colors_by_schema.values():
                    all_colors.update(colors)

                if len(all_colors) > 1:
                    # CONFLICT FOUND!
                    conflicts_found += 1

                    print(f"\n{'─' * 70}")
                    print(f"🔴 CONFLICT #{conflicts_found}: Pixel ({r}, {c})")
                    print(f"{'─' * 70}")

                    print(f"\nMultiple schemas constraining this pixel to DIFFERENT colors:")
                    for schema_family, colors in sorted(colors_by_schema.items()):
                        color_str = ", ".join(str(c) for c in sorted(colors))
                        print(f"  {schema_family:12s}: color {color_str}")

                    # Show which schemas are fighting
                    schema_list = sorted(colors_by_schema.keys())
                    if len(schema_list) == 2:
                        s1, s2 = schema_list
                        c1 = sorted(colors_by_schema[s1])[0]
                        c2 = sorted(colors_by_schema[s2])[0]
                        print(f"\n  → {s1} says: Must be color {c1}")
                        print(f"  → {s2} says: Must be color {c2}")
                        print(f"  → IMPOSSIBLE: Pixel can't be both!")
                    else:
                        print(f"\n  → {len(schema_list)} schemas fighting over this pixel")

                    # Only show first 3 conflicts
                    if conflicts_found >= 3:
                        break

        if conflicts_found == 0:
            print("\n  ✓ No conflicts found in test example")
            print("  (Infeasibility may be due to other constraint types)")
        else:
            print(f"\n{'=' * 70}")
            print(f"SUMMARY: {conflicts_found} conflicting pixel(s) found")
            print(f"{'=' * 70}")

    print(f"\n{'=' * 70}")
    print(f"END OF DIAGNOSIS")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/diagnose_infeasible.py <task_id>")
        print("Example: python scripts/diagnose_infeasible.py 025d127b")
        sys.exit(1)

    task_id = sys.argv[1]
    diagnose_infeasible_conflicts(task_id)
