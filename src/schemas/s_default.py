"""
S_Default schema builder: Law of Inertia & Colored Vacuum.

This implements the S_Default schema for unconstrained pixels:
    "For pixels not covered by S1-S11, apply default behavior based on role:
     - Fixed-color roles: fix to consistent color (e.g., fixed_4, fixed_0)
     - Inert roles: copy input color
     - Expansion zones (no role): apply global vacuum rule (e.g., vacuum_7)"

S_Default prevents the ILP solver from assigning arbitrary colors to
background/unconstrained pixels. Handles both geometry-preserving and
geometry-changing tasks. Supports ANY consistent color, not just 0.
"""

from typing import Dict, Any

from src.schemas.context import TaskContext
from src.constraints.builder import ConstraintBuilder


def build_S_Default_constraints(
    task_context: TaskContext,
    schema_params: Dict[str, Any],
    builder: ConstraintBuilder
) -> None:
    """
    Add S_Default constraints: apply default behavior to pixels based on roles.

    S_Default uses role statistics to determine default behavior:
      - "fixed_0": Fix pixel to color 0 (vacuum/background)
      - "copy_input": Fix pixel to input color (inertia)

    Param schema:
        {
          "example_type": "train" | "test",
          "example_index": int,
          "rules": {
            role_id: "fixed_0" | "copy_input",
            ...
          },
          "roles_mapping": {
            "(kind, ex_idx, r, c)": role_id,
            ...
          }
        }

    Where:
        - rules maps role_id to behavior ("fixed_0" or "copy_input")
        - roles_mapping maps pixel positions to role_ids
        - For train examples: lookup role via ("train_out", ex_idx, r, c)
        - For test examples: lookup role via ("test_in", ex_idx, r, c)

    Args:
        task_context: TaskContext with all φ features and grids
        schema_params: Parameters specifying default rules and role mapping
        builder: ConstraintBuilder to add constraints to

    Example:
        >>> # Fix role 5 to color 0, copy role 7 from input
        >>> params = {
        ...     "example_type": "train",
        ...     "example_index": 0,
        ...     "rules": {5: "fixed_0", 7: "copy_input"},
        ...     "roles_mapping": {
        ...         "('train_out', 0, 0, 0)": 5,
        ...         "('train_out', 0, 1, 1)": 7
        ...     }
        ... }
        >>> build_S_Default_constraints(ctx, params, builder)
    """
    # 1. Select example
    example_type = schema_params.get("example_type", "train")
    example_index = schema_params.get("example_index", 0)

    if example_type == "train":
        if example_index >= len(task_context.train_examples):
            return  # Invalid index
        ex = task_context.train_examples[example_index]
    else:  # "test"
        if example_index >= len(task_context.test_examples):
            return  # Invalid index
        ex = task_context.test_examples[example_index]

    # 2. Get grid dimensions
    # S_Default applies to OUTPUT grid
    H = ex.output_H
    W = ex.output_W
    if H is None or W is None:
        return  # No output grid to constrain
    C = task_context.C

    # 3. Extract rules and roles_mapping
    rules = schema_params.get("rules", {})
    roles_mapping_raw = schema_params.get("roles_mapping", {})

    if not rules:
        return  # No rules to apply

    # Convert roles_mapping keys to tuples
    # Handle both tuple keys (in-memory) and string keys (from JSON)
    roles_mapping: Dict[tuple, int] = {}
    for key, role_id in roles_mapping_raw.items():
        try:
            if isinstance(key, tuple):
                # Already a tuple (in-memory SchemaInstance)
                key_tuple = key
            else:
                # String key from JSON serialization: "('train_out', 0, 1, 2)"
                import ast
                key_tuple = ast.literal_eval(key)
            roles_mapping[key_tuple] = int(role_id)
        except (ValueError, SyntaxError, TypeError):
            continue  # Skip malformed keys

    # 4. Determine lookup kind based on example type
    if example_type == "train":
        lookup_kind = "train_out"
    else:  # "test"
        # Use test_in role as proxy for test output (geometry-preserving assumption)
        lookup_kind = "test_in"

    # 5. Apply constraints for each pixel in output grid
    for r in range(H):
        for c in range(W):
            # Lookup role_id for this pixel
            role_key = (lookup_kind, example_index, r, c)
            role_id = roles_mapping.get(role_key)

            # Determine rule to apply
            rule = None
            if role_id is not None:
                # Case A: Inertia (we have a role from input mapping)
                rule = rules.get(role_id)
            else:
                # Case B: Expansion Zone (no role - pixel outside input bounds)
                # Check for Global Vacuum Rule (special rule ID -1)
                rule = rules.get(-1)

            if rule is None:
                continue  # No rule for this pixel (active role, S1-S11 handles it)

            # Compute flat pixel index
            p_idx = r * W + c

            # Apply constraint based on rule
            # Use SOFT PREFERENCES instead of HARD CONSTRAINTS
            # This allows S1-S11 (matter laws) to override S_Default (inertia)
            if rule == "copy_input":
                # Prefer input color (inertia)
                # For test examples, use test input grid
                # For train examples, use train input grid
                input_grid = ex.input_grid

                # Validate coordinates within input grid
                if r < ex.input_H and c < ex.input_W:
                    c_in = int(input_grid[r, c])
                    # Validate color is in palette
                    if 0 <= c_in < C:
                        builder.prefer_pixel_color(p_idx, c_in, weight=1.0)
                # else: pixel is outside input grid (expansion zone)
                # Can't copy from non-existent input, skip preference

            elif rule.startswith("fixed_") or rule.startswith("vacuum_"):
                # Parse dynamic color: "fixed_4", "vacuum_7", etc.
                try:
                    target_color = int(rule.split("_")[1])
                    # Validate color is in palette
                    if 0 <= target_color < C:
                        builder.prefer_pixel_color(p_idx, target_color, weight=1.0)
                except (ValueError, IndexError):
                    pass  # Malformed rule, skip


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S_Default builder with toy example...")
    print("=" * 70)

    # Create a 3x3 input grid
    input_grid = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()  # Geometry-preserving

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    print("Test 1: Fixed-0 rule")
    print("-" * 70)

    # Create a simple roles_mapping (all pixels have role 1)
    roles_mapping = {}
    for r in range(3):
        for c in range(3):
            roles_mapping[f"('train_out', 0, {r}, {c})"] = 1

    params1 = {
        "example_type": "train",
        "example_index": 0,
        "rules": {1: "fixed_0"},  # Role 1 → fix to 0
        "roles_mapping": roles_mapping
    }

    builder1 = ConstraintBuilder()
    build_S_Default_constraints(ctx, params1, builder1)

    # Should have 9 preferences (3x3 grid, all prefer color 0)
    expected1 = 9
    print(f"  Expected: {expected1} preferences (3x3 grid, all prefer color 0)")
    print(f"  Actual: {len(builder1.preferences)}")
    assert len(builder1.preferences) == expected1, \
        f"Expected {expected1} preferences, got {len(builder1.preferences)}"

    print("\nTest 2: Copy-input rule")
    print("-" * 70)

    params2 = {
        "example_type": "train",
        "example_index": 0,
        "rules": {1: "copy_input"},  # Role 1 → copy from input
        "roles_mapping": roles_mapping
    }

    builder2 = ConstraintBuilder()
    build_S_Default_constraints(ctx, params2, builder2)

    # Should have 9 preferences (3x3 grid, all copy from input)
    expected2 = 9
    print(f"  Expected: {expected2} preferences (3x3 grid, all copy from input)")
    print(f"  Actual: {len(builder2.preferences)}")
    assert len(builder2.preferences) == expected2, \
        f"Expected {expected2} preferences, got {len(builder2.preferences)}"

    print("\nTest 3: Mixed rules (different roles)")
    print("-" * 70)

    # Create roles_mapping with two roles
    # Border pixels (role 1) → fixed_0
    # Center pixel (role 2) → copy_input
    roles_mapping_mixed = {}
    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                # Center pixel
                roles_mapping_mixed[f"('train_out', 0, {r}, {c})"] = 2
            else:
                # Border pixels
                roles_mapping_mixed[f"('train_out', 0, {r}, {c})"] = 1

    params3 = {
        "example_type": "train",
        "example_index": 0,
        "rules": {
            1: "fixed_0",      # Border → 0
            2: "copy_input"    # Center → copy
        },
        "roles_mapping": roles_mapping_mixed
    }

    builder3 = ConstraintBuilder()
    build_S_Default_constraints(ctx, params3, builder3)

    # Should have 9 preferences (8 border + 1 center)
    expected3 = 9
    print(f"  Expected: {expected3} preferences (8 border prefer 0, 1 center copy)")
    print(f"  Actual: {len(builder3.preferences)}")
    assert len(builder3.preferences) == expected3, \
        f"Expected {expected3} preferences, got {len(builder3.preferences)}"

    print("\nTest 4: No rules (active roles)")
    print("-" * 70)

    params4 = {
        "example_type": "train",
        "example_index": 0,
        "rules": {},  # No rules
        "roles_mapping": roles_mapping
    }

    builder4 = ConstraintBuilder()
    build_S_Default_constraints(ctx, params4, builder4)

    expected4 = 0
    print(f"  Expected: {expected4} preferences (no rules)")
    print(f"  Actual: {len(builder4.preferences)}")
    assert len(builder4.preferences) == expected4, \
        f"Expected {expected4} preferences, got {len(builder4.preferences)}"

    print("\n" + "=" * 70)
    print("✓ S_Default builder self-test passed.")
