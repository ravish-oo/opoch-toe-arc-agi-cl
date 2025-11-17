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
    # S1-S7 and S11 are implemented, S8-S10 are stubs
    # Test S8 (stub) raises NotImplementedError
    print("\n3. Testing dispatch with TaskContext...")
    schema_params = {"dummy": "params"}

    try:
        apply_schema_instance("S8", schema_params, ctx, builder)
        print("   ✗ ERROR: Expected NotImplementedError")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        print(f"   ✓ Caught expected NotImplementedError from S8 (stub):")
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


def test_s3_builder():
    """Test S3 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S3 builder (Band / stripe laws)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a 4x4 grid
    input_grid = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 2, 3, 4],  # Same as row 0
        [9, 8, 7, 6]
    ], dtype=int)

    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Create params: tie rows 0 and 2
    params = {
        "example_type": "train",
        "example_index": 0,
        "row_classes": [[0, 2]],  # Rows 0 and 2 share pattern
        "col_classes": [],
        "col_period_K": None,
        "row_period_K": None
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S3", params, ctx, builder)

    # 4 columns × 10 colors = 40 constraints
    expected = 4 * ctx.C
    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S3 added {len(builder.constraints)} tie constraints")
    print(f"    (tie rows 0,2 across 4 columns × {ctx.C} colors)")


def test_s4_builder():
    """Test S4 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S4 builder (Residue-class coloring)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a 4x4 grid
    input_grid = np.array([
        [0, 1, 0, 1],
        [2, 3, 2, 3],
        [0, 1, 0, 1],
        [2, 3, 2, 3]
    ], dtype=int)

    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=5)

    # Create params: even columns → color 1, odd columns → color 3
    params = {
        "example_type": "train",
        "example_index": 0,
        "axis": "col",
        "K": 2,
        "residue_to_color": {
            "0": 1,  # even columns → color 1
            "1": 3   # odd columns → color 3
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S4", params, ctx, builder)

    # 4x4 grid = 16 pixels, one fix per pixel
    expected = 16
    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S4 added {len(builder.constraints)} fix constraints")
    print(f"    (one per pixel in 4x4 grid)")


def test_s5_builder():
    """Test S5 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S5 builder (Template stamping)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a 5x5 grid with a seed pixel
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Get hash at (1,1)
    if (1, 1) in ex.neighborhood_hashes:
        seed_hash = ex.neighborhood_hashes[(1, 1)]
    else:
        print("  ⚠ Warning: No hash at (1,1), skipping S5 test")
        return

    # Create params: stamp 2x2 square around seed
    params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(seed_hash): {
                "(0,0)": 5,
                "(0,1)": 5,
                "(1,0)": 5,
                "(1,1)": 5
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S5", params, ctx, builder)

    # Should have 4 constraints (one 2x2 template)
    expected = 4
    assert len(builder.constraints) >= expected, \
        f"Expected at least {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S5 added {len(builder.constraints)} template constraints")
    print(f"    (stamped 2x2 template at seed pixel)")


def test_s11_builder():
    """Test S11 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S11 builder (Local neighborhood codebook)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a 3x3 grid
    input_grid = np.array([
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ], dtype=int)

    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Get a hash from the grid
    if ex.neighborhood_hashes:
        sample_hash = list(ex.neighborhood_hashes.values())[0]
        pixels_with_hash = sum(1 for h in ex.neighborhood_hashes.values() if h == sample_hash)
    else:
        print("  ⚠ Warning: No hashes found, skipping S11 test")
        return

    # Create params: rewrite center pixel for this hash
    params = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(sample_hash): {
                "(0,0)": 7
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S11", params, ctx, builder)

    # Should have 1 constraint per pixel with this hash
    expected = pixels_with_hash
    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S11 added {len(builder.constraints)} codebook constraints")
    print(f"    (applied template to {pixels_with_hash} matching pixels)")


def test_s6_builder():
    """Test S6 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S6 builder (Cropping to ROI)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a 4x4 input grid
    input_grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    output_grid = None  # S6 doesn't need output for context building

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Crop central 2x2 square
    params = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "background_color": 0,
        "out_to_in": {
            "(0,0)": "(1,1)",  # output (0,0) <- input (1,1) = 1
            "(0,1)": "(1,2)",  # output (0,1) <- input (1,2) = 2
            "(1,0)": "(2,1)",  # output (1,0) <- input (2,1) = 3
            "(1,1)": "(2,2)"   # output (1,1) <- input (2,2) = 4
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S6", params, ctx, builder)

    # Should have 4 constraints (2x2 output)
    expected = 4
    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S6 added {len(builder.constraints)} crop constraints")
    print(f"    (cropped 2x2 from 4x4 input)")


def test_s7_builder():
    """Test S7 builder with toy example."""
    print("\n" + "=" * 70)
    print("Testing S7 builder (Aggregation / summary grid)")
    print("=" * 70)

    import numpy as np
    from src.schemas.context import build_example_context, TaskContext
    from src.constraints.builder import ConstraintBuilder

    # Create a larger input grid (doesn't matter for S7, only summary matters)
    input_grid = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ], dtype=int)

    output_grid = None

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Create 2x2 summary grid
    params = {
        "example_type": "train",
        "example_index": 0,
        "output_height": 2,
        "output_width": 2,
        "summary_colors": {
            "(0,0)": 1,  # Top-left block -> color 1
            "(0,1)": 2,  # Top-right block -> color 2
            "(1,0)": 3,  # Bottom-left block -> color 3
            "(1,1)": 4   # Bottom-right block -> color 4
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S7", params, ctx, builder)

    # Should have 4 constraints (2x2 summary)
    expected = 4
    assert len(builder.constraints) == expected, \
        f"Expected {expected} constraints, got {len(builder.constraints)}"

    print(f"  ✓ S7 added {len(builder.constraints)} summary constraints")
    print(f"    (2x2 summary grid)")


if __name__ == "__main__":
    test_context_dispatch_integration()
    test_s1_builder()
    test_s2_builder()
    test_s3_builder()
    test_s4_builder()
    test_s5_builder()
    test_s6_builder()
    test_s7_builder()
    test_s11_builder()

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
    print("  - S3 builder works (Band / stripe laws)")
    print("  - S4 builder works (Residue-class coloring)")
    print("  - S5 builder works (Template stamping)")
    print("  - S6 builder works (Cropping to ROI)")
    print("  - S7 builder works (Aggregation / summary grid)")
    print("  - S11 builder works (Local neighborhood codebook)")
    print("  - Ready for M3.5+ schema implementations!")
