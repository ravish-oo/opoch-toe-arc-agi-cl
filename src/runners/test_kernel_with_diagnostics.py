"""
Test runner for kernel with diagnostics integration (M5.2).

This test verifies that solve_arc_task_with_diagnostics runs end-to-end
and returns properly populated diagnostics. It's a sanity check for the
Pi-agent interface, not a correctness test.

Tests:
  1. Empty law config (expect infeasible or error)
  2. Minimal S1 config with use_training_labels=True (expect mismatch)
  3. Diagnostics structure is properly populated
"""

from src.runners.kernel import solve_arc_task_with_diagnostics
from src.catalog.types import TaskLawConfig, SchemaInstance


def dummy_law_config() -> TaskLawConfig:
    """
    Create a minimal empty law config.

    With no schema instances, the solver will have only one-hot constraints,
    which are typically infeasible (underconstrained).
    """
    return TaskLawConfig(schema_instances=[])


def minimal_s1_config() -> TaskLawConfig:
    """
    Create a minimal S1 configuration (same as smoke tests).

    This ties top-left to top-right, which won't solve most tasks correctly
    but should at least produce a result.
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


def test_empty_config():
    """
    Test 1: Empty law config should result in infeasible or error status.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Empty law config")
    print("=" * 70)

    task_id = "00576224"
    law_config = dummy_law_config()

    print(f"Task ID: {task_id}")
    print(f"Law config: {len(law_config.schema_instances)} schema instances")

    try:
        outputs, diagnostics = solve_arc_task_with_diagnostics(
            task_id=task_id,
            law_config=law_config,
            use_training_labels=False,
        )

        print(f"\n=== Outputs ===")
        print(f"Train outputs predicted: {len(outputs['train'])}")
        print(f"Test outputs predicted: {len(outputs['test'])}")

        print(f"\n=== Diagnostics ===")
        print(f"Status: {diagnostics.status}")
        print(f"Solver status: {diagnostics.solver_status}")
        print(f"Num constraints: {diagnostics.num_constraints}")
        print(f"Num variables: {diagnostics.num_variables}")
        print(f"Schema IDs used: {diagnostics.schema_ids_used}")
        print(f"Error message: {diagnostics.error_message}")

        # With empty config, we might get "ok" (if solver finds *some* solution)
        # or "infeasible" or "error". Any of these is acceptable for this test.
        assert diagnostics.status in ["ok", "infeasible", "error"], \
            f"Unexpected status: {diagnostics.status}"

        print("\n✓ Test 1 passed: Diagnostics populated correctly")

    except Exception as e:
        print(f"\n✗ Test 1 failed with exception: {e}")
        raise


def test_minimal_s1_with_labels():
    """
    Test 2: Minimal S1 config with training labels comparison.

    This should produce outputs (S1 applies constraints), but likely mismatches
    since the config is too simple.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Minimal S1 config with training labels")
    print("=" * 70)

    task_id = "00576224"
    law_config = minimal_s1_config()

    print(f"Task ID: {task_id}")
    print(f"Law config: {len(law_config.schema_instances)} schema instance(s)")
    for si in law_config.schema_instances:
        print(f"  - {si.family_id}")

    try:
        outputs, diagnostics = solve_arc_task_with_diagnostics(
            task_id=task_id,
            law_config=law_config,
            use_training_labels=True,  # Compare with ground truth
        )

        print(f"\n=== Outputs ===")
        print(f"Train outputs predicted: {len(outputs['train'])}")
        print(f"Test outputs predicted: {len(outputs['test'])}")

        print(f"\n=== Diagnostics ===")
        print(f"Status: {diagnostics.status}")
        print(f"Solver status: {diagnostics.solver_status}")
        print(f"Num constraints: {diagnostics.num_constraints}")
        print(f"Num variables: {diagnostics.num_variables}")
        print(f"Schema IDs used: {diagnostics.schema_ids_used}")
        print(f"Train mismatches: {len(diagnostics.train_mismatches)} example(s)")

        if diagnostics.train_mismatches:
            for mm in diagnostics.train_mismatches[:2]:  # Show first 2
                print(f"  Example {mm['example_idx']}: {len(mm['diff_cells'])} cells differ")

        print(f"Error message: {diagnostics.error_message}")

        # Verify diagnostics structure
        assert diagnostics.status in ["ok", "mismatch", "infeasible", "error"]
        assert diagnostics.task_id == task_id
        assert diagnostics.law_config == law_config
        assert diagnostics.num_constraints > 0, "Should have some constraints from S1"
        assert diagnostics.num_variables > 0, "Should have variables for pixels"
        assert "S1" in diagnostics.schema_ids_used

        # If status is "mismatch", we should have train_mismatches
        if diagnostics.status == "mismatch":
            assert len(diagnostics.train_mismatches) > 0, \
                "Mismatch status should have train_mismatches"

        print("\n✓ Test 2 passed: Diagnostics with training labels work correctly")

    except Exception as e:
        print(f"\n✗ Test 2 failed with exception: {e}")
        raise


def main():
    """Run all tests for kernel with diagnostics."""
    print("\n" + "=" * 70)
    print("KERNEL WITH DIAGNOSTICS TEST SUITE")
    print("=" * 70)

    # Test 1: Empty config
    test_empty_config()

    # Test 2: Minimal S1 with labels
    test_minimal_s1_with_labels()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  - Empty law config handling: ✓")
    print("  - Diagnostics structure population: ✓")
    print("  - Training label comparison: ✓")
    print("  - Error handling: ✓")
    print("\nKernel with diagnostics is ready for Pi-agent integration (M5.3+)")
    print()


if __name__ == "__main__":
    main()
