"""
End-to-end integration test: TaskContext → dispatch → ConstraintBuilder.

This demonstrates the complete M3.0 flow:
  1. Load ARC task
  2. Build TaskContext with all φ features
  3. Pass TaskContext to schema builder dispatch
  4. Verify builder receives correct context structure

This validates the handoff from M1 (features) → M2 (constraints) → M3.0 (context).
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.schemas.dispatch import apply_schema_instance
from src.constraints.builder import ConstraintBuilder


def test_context_dispatch_integration():
    """Test that TaskContext can be passed to dispatch layer."""
    print("End-to-end integration: TaskContext → dispatch → ConstraintBuilder")
    print("=" * 70)

    # 1. Load task and build context
    print("\n1. Building TaskContext...")
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "00576224"

    task_data = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(task_data)

    print(f"   ✓ TaskContext built for task {task_id}")
    print(f"     - {len(ctx.train_examples)} train examples")
    print(f"     - {len(ctx.test_examples)} test examples")
    print(f"     - Palette size C={ctx.C}")

    # 2. Create ConstraintBuilder
    print("\n2. Creating ConstraintBuilder...")
    builder = ConstraintBuilder()
    print(f"   ✓ ConstraintBuilder created")

    # 3. Try to call dispatch with TaskContext
    # S1/S2 are implemented, S3-S11 are stubs
    # Test S3 (stub) raises NotImplementedError
    print("\n3. Testing dispatch with TaskContext...")
    schema_params = {"dummy": "params"}

    try:
        apply_schema_instance("S3", schema_params, ctx, builder)
        print("   ✗ ERROR: Expected NotImplementedError")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        print(f"   ✓ Caught expected NotImplementedError from S3 (stub):")
        print(f"     {e}")

    # 4. Verify TaskContext structure is accessible
    print("\n4. Verifying TaskContext is accessible in builder context...")

    # Simulate what a real schema builder would do:
    # Access train examples
    for i, ex in enumerate(ctx.train_examples):
        assert ex.input_grid is not None, f"Train example {i} missing input"
        assert ex.output_grid is not None, f"Train example {i} missing output"
        assert len(ex.components) > 0, f"Train example {i} has no components"

    # Access test examples
    for i, ex in enumerate(ctx.test_examples):
        assert ex.input_grid is not None, f"Test example {i} missing input"
        assert ex.output_grid is None, f"Test example {i} should have no output"

    # Access palette size
    assert ctx.C > 0, "Palette size should be > 0"

    print(f"   ✓ TaskContext structure is fully accessible")
    print(f"     - Train examples accessible: {len(ctx.train_examples)}")
    print(f"     - Test examples accessible: {len(ctx.test_examples)}")
    print(f"     - Palette C accessible: {ctx.C}")

    # 5. Verify builder signature matches
    print("\n5. Verifying builder signature compatibility...")
    from src.schemas.dispatch import BUILDERS
    import inspect

    for family_id, builder_fn in BUILDERS.items():
        sig = inspect.signature(builder_fn)
        params = list(sig.parameters.keys())
        assert params == ['task_context', 'schema_params', 'builder'], \
            f"{family_id} has wrong signature: {params}"

    print(f"   ✓ All {len(BUILDERS)} builders have compatible signatures")

    print("\n" + "=" * 70)
    print("✓ End-to-end integration test passed!")
    print("=" * 70)


def test_s1_builder():
    """Test S1 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S1 builder (Direct pixel color tie)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a simple 2x2 grid
    input_grid = np.array([[1, 2], [3, 4]], dtype=int)
    output_grid = np.array([[1, 1], [3, 3]], dtype=int)

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Create params: tie (0,0) to (0,1) and (1,0) to (1,1)
    params = {
        "ties": [{
            "example_type": "train",
            "example_index": 0,
            "pairs": [
                ((0, 0), (0, 1)),  # tie top row
                ((1, 0), (1, 1))   # tie bottom row
            ]
        }]
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S1", params, ctx, builder)

    # Verify constraints were added
    # 2 pairs × 5 colors = 10 constraints
    expected = 2 * ctx.C
    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S1 added {len(builder.constraints)} tie constraints")
    print(f"    (2 pixel pairs × {ctx.C} colors = {expected})")


def test_s2_builder():
    """Test S2 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S2 builder (Component-wise recolor)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a 3x3 grid with components of different sizes
    input_grid = np.array([
        [0, 1, 0],
        [1, 2, 1],
        [0, 0, 0]
    ], dtype=int)

    output_grid = np.array([
        [0, 3, 0],
        [4, 2, 4],
        [0, 0, 0]
    ], dtype=int)

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    print(f"  Components found: {len(ex.components)}")
    color_1_comps = [c for c in ex.components if c.color == 1]
    print(f"  Components with color 1: {len(color_1_comps)}")
    for comp in color_1_comps:
        print(f"    Component {comp.id}: size={comp.size}, pixels={comp.pixels}")

    # Create params: recolor components of color 1 based on size
    params = {
        "example_type": "train",
        "example_index": 0,
        "input_color": 1,
        "size_to_color": {
            "1": 3,   # size 1 → color 3
            "2": 4,   # size 2 → color 4
            "else": 0
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S2", params, ctx, builder)

    # Count total pixels in color-1 components
    total_pixels = sum(c.size for c in color_1_comps)
    assert len(builder.constraints) == total_pixels, \
        f"Expected {total_pixels} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S2 added {len(builder.constraints)} fix constraints")
    print(f"    (one per pixel in color-1 components)")


if __name__ == "__main__":
    test_context_dispatch_integration()
    test_s1_builder()
    test_s2_builder()

    print("\n" + "=" * 70)
    print("✓ All integration tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print("  - TaskContext successfully built from ARC task")
    print("  - TaskContext contains all φ features from M1")
    print("  - dispatch.apply_schema_instance accepts TaskContext")
    print("  - All builder functions have compatible signatures")
    print("  - S1 builder works (Direct pixel color tie)")
    print("  - S2 builder works (Component-wise recolor)")
    print("  - Ready for M3.2+ schema implementations!")
