"""
Smoke test for kernel runner with full solver integration (M4.3).

This test verifies that the complete math kernel pipeline works:
  1. Load task data
  2. Build TaskContext with all φ features
  3. Apply schema instances to generate constraints
  4. Solve LP/ILP per example
  5. Decode y → grids

No accuracy validation; just ensures the pipeline runs without crashing.
"""

from pathlib import Path

from src.catalog.types import SchemaInstance, TaskLawConfig
from src.runners.kernel import solve_arc_task, TaskSolveError


def make_dummy_law_config() -> TaskLawConfig:
    """
    Create a minimal law configuration for smoke testing.

    Uses S1 (copy tie) schema with a single tie constraint.
    This is the simplest schema that produces valid constraints.

    Returns:
        TaskLawConfig with one S1 schema instance
    """
    return TaskLawConfig(
        schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={
                    "ties": [{
                        "pairs": [((0, 0), (0, 1))]  # Tie top-left to top-right
                    }]
                }
            )
        ]
    )


def test_kernel_smoke():
    """
    Smoke test: verify that kernel can load, constrain, solve, and decode.

    This does NOT validate correctness of the output grids.
    It only checks that the pipeline runs without errors and returns
    the expected structure.
    """
    print("\n" + "=" * 70)
    print("KERNEL SMOKE TEST")
    print("=" * 70)

    # Use a simple task from training set
    task_id = "00576224"
    law_config = make_dummy_law_config()

    print(f"\nTask ID: {task_id}")
    print(f"Law config: {len(law_config.schema_instances)} schema instance(s)")
    print(f"  - {law_config.schema_instances[0].family_id}")

    # Attempt to solve
    try:
        result = solve_arc_task(task_id, law_config)

        # Validate result structure
        assert isinstance(result, dict), \
            f"Expected dict result, got {type(result)}"

        assert "train_outputs_pred" in result, \
            "Result missing 'train_outputs_pred' key"

        assert "test_outputs_pred" in result, \
            "Result missing 'test_outputs_pred' key"

        assert isinstance(result["train_outputs_pred"], list), \
            "train_outputs_pred should be a list"

        assert isinstance(result["test_outputs_pred"], list), \
            "test_outputs_pred should be a list"

        # Display results
        print(f"\n✓ Kernel pipeline completed successfully!")
        print(f"  Train outputs predicted: {len(result['train_outputs_pred'])}")
        print(f"  Test outputs predicted: {len(result['test_outputs_pred'])}")

        # Show first train output shape if available
        if result["train_outputs_pred"]:
            grid0 = result["train_outputs_pred"][0]
            print(f"\n  First train output shape: {grid0.shape}")
            print(f"  First train output (preview):")
            print(f"    {grid0}")

        print("\n✓ test_kernel_smoke: PASSED")

    except TaskSolveError as e:
        print(f"\n✗ Task solve failed: {e}")
        print(f"  Task: {e.task_id}")
        print(f"  Example: {e.example_type}[{e.example_index}]")
        print(f"  Reason: {e.original_error}")
        raise

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise

    print("=" * 70)


if __name__ == "__main__":
    test_kernel_smoke()

    print("\n" + "=" * 70)
    print("✓ ALL KERNEL SMOKE TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - Task loading: ✓")
    print("  - TaskContext building: ✓")
    print("  - Schema constraint application: ✓")
    print("  - ILP solving per example: ✓")
    print("  - Grid decoding: ✓")
    print("\nKernel runner is ready for integration with Pi-agent")
    print()
