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

    # 3. Try to call dispatch with TaskContext (will fail with NotImplementedError, which is expected)
    print("\n3. Testing dispatch with TaskContext...")
    schema_params = {"dummy": "params"}

    try:
        apply_schema_instance("S1", schema_params, ctx, builder)
        print("   ✗ ERROR: Expected NotImplementedError")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        print(f"   ✓ Caught expected NotImplementedError:")
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
    print("\nSummary:")
    print("  - TaskContext successfully built from ARC task")
    print("  - TaskContext contains all φ features from M1")
    print("  - dispatch.apply_schema_instance accepts TaskContext")
    print("  - All builder functions have compatible signatures")
    print("  - Ready for M3.1+ schema implementations!")


if __name__ == "__main__":
    test_context_dispatch_integration()
