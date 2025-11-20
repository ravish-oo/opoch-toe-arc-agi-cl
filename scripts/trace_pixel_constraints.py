"""
Trace which schemas constrain specific pixels.

For mismatch_train tasks, identifies which schemas generated preferences
for mismatched pixels to understand "The Void" vs "The Lie".
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance


def trace_pixel_constraints(task_id: str, example_type: str, example_idx: int,
                            target_pixels: List[Tuple[int, int]]):
    """
    Trace which schemas generated preferences for specific pixels.

    Args:
        task_id: ARC task ID
        example_type: "train" or "test"
        example_idx: Example index
        target_pixels: List of (r, c) pixel coordinates to trace
    """
    print("=" * 70)
    print(f"PIXEL CONSTRAINT TRACE: {task_id}")
    print(f"Example: {example_type}[{example_idx}]")
    print("=" * 70)

    challenges_path = Path("data/arc-agi_training_challenges.json")

    # 1. Load and mine
    print("\n[1] Loading task and mining schemas...")
    raw_task = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(raw_task)
    law_config = mine_law_config(ctx)

    num_schemas = len(law_config.schema_instances)
    print(f"    ✓ Mined {num_schemas} schema instances")

    # Get example
    if example_type == "train":
        ex = ctx.train_examples[example_idx]
    else:
        ex = ctx.test_examples[example_idx]

    W = ex.output_W
    H = ex.output_H

    # 2. Trace each target pixel
    for r, c in target_pixels:
        print(f"\n{'=' * 70}")
        print(f"PIXEL ({r}, {c})")
        print(f"{'=' * 70}")

        p_idx = r * W + c

        # Track preferences from each schema
        pixel_preferences = []  # List of (schema_family, schema_idx, color, weight)

        # Apply each schema and check if it constrains this pixel
        for schema_idx, si in enumerate(law_config.schema_instances):
            builder = ConstraintBuilder()

            # Apply schema to this example
            apply_schema_instance(
                family_id=si.family_id,
                schema_params=si.params,
                task_context=ctx,
                builder=builder,
                example_type=example_type,
                example_index=example_idx
            )

            # Check preferences for this pixel
            for pref_p_idx, color, weight in builder.preferences:
                if pref_p_idx == p_idx:
                    pixel_preferences.append((si.family_id, schema_idx, color, weight))

        # Display results
        if not pixel_preferences:
            print("⚫ THE VOID: No schemas constrained this pixel")
            print("   → S_Default was the only constraint (if any)")
            print("   → This pixel fell through all S1-S11 patterns")
        else:
            print(f"📊 {len(pixel_preferences)} preference(s) found:")
            print()

            # Group by schema family
            by_family = defaultdict(list)
            for family, idx, color, weight in pixel_preferences:
                by_family[family].append((idx, color, weight))

            # Sort by weight (highest first)
            families_sorted = sorted(by_family.items(),
                                    key=lambda x: max(w for _, _, w in x[1]),
                                    reverse=True)

            for family, prefs in families_sorted:
                max_weight = max(w for _, _, w in prefs)
                print(f"  {family:12s} (weight={max_weight:>5.1f}):")
                for idx, color, weight in prefs:
                    print(f"    Instance {idx}: prefers color {color} (weight={weight})")

            # Calculate expected color (lowest cost)
            color_costs = defaultdict(float)
            C = ctx.C

            for color_candidate in range(C):
                cost = 0
                for _, _, pref_color, weight in pixel_preferences:
                    if color_candidate != pref_color:
                        cost += weight
                color_costs[color_candidate] = cost

            # Find minimum cost color
            expected_color = min(color_costs.items(), key=lambda x: x[1])[0]
            expected_cost = color_costs[expected_color]

            print()
            print(f"  💡 Expected (lowest cost): color {expected_color} (cost={expected_cost:.1f})")

            # Show top 3 alternatives
            print(f"  📋 Cost breakdown (top 3):")
            for color, cost in sorted(color_costs.items(), key=lambda x: x[1])[:3]:
                print(f"     Color {color}: cost={cost:.1f}")

    print(f"\n{'=' * 70}")
    print(f"END OF TRACE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python scripts/trace_pixel_constraints.py <task_id> <train|test> <ex_idx> <r,c> [<r,c> ...]")
        print("Example: python scripts/trace_pixel_constraints.py 0b148d64 train 0 5,8 6,7")
        sys.exit(1)

    task_id = sys.argv[1]
    example_type = sys.argv[2]
    example_idx = int(sys.argv[3])

    target_pixels = []
    for pixel_str in sys.argv[4:]:
        r, c = map(int, pixel_str.split(','))
        target_pixels.append((r, c))

    trace_pixel_constraints(task_id, example_type, example_idx, target_pixels)
