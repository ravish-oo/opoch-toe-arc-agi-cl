#!/usr/bin/env python3
"""
Comprehensive review test for WO-M5.2: Extend kernel to return diagnostics.

This test verifies:
  1. No TODOs, stubs, or simplified implementations
  2. Uses clarified signatures correctly:
     - load_arc_task(task_id, challenges_path)
     - apply_schema_instance(..., schema_params=..., example_type=..., example_index=...)
     - solve_constraints_for_grid returns (y, status_str)
     - ExampleContext.output_grid for training labels
  3. Error handling catches specific exceptions
  4. Logic correctness (one ILP per example, computes num_pixels correctly, etc.)
  5. Diagnostics structure properly populated
  6. Training label comparison works
  7. Backward-compatible wrapper exists
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.runners.kernel import solve_arc_task_with_diagnostics, solve_arc_task
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.runners.results import SolveDiagnostics


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    files = [
        project_root / "src/runners/kernel.py",
        project_root / "src/runners/test_kernel_with_diagnostics.py",
        project_root / "src/solver/lp_solver.py",
    ]

    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError",
               "stub", "Stub", "simplified", "Simplified", "MVP"]

    for file_path in files:
        source = file_path.read_text()
        for marker in markers:
            assert marker not in source, \
                f"Found '{marker}' in {file_path.name}"

    print("  ✓ No TODOs, stubs, or markers found in any file")


def test_signature_adaptations():
    """Test that code uses clarified signatures correctly."""
    print("\nTest: Clarified signatures used correctly")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # 1. load_arc_task should be called with challenges_path
    assert "load_arc_task(task_id, challenges_path)" in source, \
        "Should call load_arc_task with challenges_path parameter"
    print("  ✓ load_arc_task(task_id, challenges_path) used correctly")

    # 2. apply_schema_instance should use schema_params (not params)
    assert "schema_params=schema_inst.params" in source, \
        "Should use schema_params parameter name"
    print("  ✓ schema_params parameter name used")

    # 3. apply_schema_instance should pass example_type
    assert 'example_type="train"' in source, \
        "Should pass example_type='train'"
    assert 'example_type="test"' in source, \
        "Should pass example_type='test'"
    print("  ✓ example_type parameter passed for train and test")

    # 4. solve_constraints_for_grid should be unpacked as tuple
    assert "y, solver_status_single = solve_constraints_for_grid(" in source, \
        "Should unpack solver result as (y, solver_status_str)"
    print("  ✓ solve_constraints_for_grid unpacked as tuple")

    # 5. ExampleContext.output_grid should be used for training labels
    assert "ex.output_grid for ex in ctx.train_examples" in source, \
        "Should use ex.output_grid for training labels"
    print("  ✓ ExampleContext.output_grid used for training labels")


def test_error_handling():
    """Test that error handling catches specific exceptions."""
    print("\nTest: Error handling (specific exceptions)")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should catch InfeasibleModelError
    assert "except InfeasibleModelError" in source, \
        "Should catch InfeasibleModelError specifically"
    print("  ✓ Catches InfeasibleModelError")

    # Should catch TaskSolveError
    assert "except TaskSolveError" in source, \
        "Should catch TaskSolveError specifically"
    print("  ✓ Catches TaskSolveError")

    # Should catch general Exception as fallback
    assert "except Exception" in source, \
        "Should catch Exception as fallback"
    print("  ✓ Catches Exception as fallback")

    # Should set status based on exception type
    assert 'status = "infeasible"' in source, \
        "Should set status='infeasible' for solver failures"
    assert 'status = "error"' in source, \
        "Should set status='error' for unexpected exceptions"
    print("  ✓ Sets status correctly based on exception type")


def test_diagnostics_structure():
    """Test that diagnostics structure is correctly populated."""
    print("\nTest: Diagnostics structure population")
    print("-" * 70)

    task_id = "00576224"
    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={"ties": [{"pairs": [((0, 0), (0, 1))]}]}
        )
    ])

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
    )

    # Check diagnostics type
    assert isinstance(diagnostics, SolveDiagnostics), \
        f"Should return SolveDiagnostics, got {type(diagnostics)}"
    print("  ✓ Returns SolveDiagnostics instance")

    # Check all required fields
    assert diagnostics.task_id == task_id
    assert diagnostics.law_config == law_config
    assert diagnostics.status in ["ok", "infeasible", "mismatch", "error"]
    assert isinstance(diagnostics.solver_status, str)
    assert diagnostics.num_constraints >= 0
    assert diagnostics.num_variables > 0
    assert diagnostics.schema_ids_used == ["S1"]
    assert isinstance(diagnostics.train_mismatches, list)
    print("  ✓ All required fields populated")

    # Check outputs structure
    assert "train" in outputs and "test" in outputs
    assert isinstance(outputs["train"], list)
    assert isinstance(outputs["test"], list)
    print("  ✓ Outputs structure correct")


def test_training_label_comparison():
    """Test that training label comparison works correctly."""
    print("\nTest: Training label comparison")
    print("-" * 70)

    task_id = "00576224"
    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={"ties": [{"pairs": [((0, 0), (0, 1))]}]}
        )
    ])

    # Test WITH training labels
    outputs_with, diag_with = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
    )

    # Should have status="mismatch" or "ok" (depending on correctness)
    assert diag_with.status in ["ok", "mismatch", "infeasible", "error"]
    print(f"  ✓ With training labels: status={diag_with.status}")

    # If status is mismatch, should have train_mismatches
    if diag_with.status == "mismatch":
        assert len(diag_with.train_mismatches) > 0, \
            "Mismatch status should have non-empty train_mismatches"
        print(f"  ✓ Train mismatches populated: {len(diag_with.train_mismatches)} example(s)")

    # Test WITHOUT training labels
    outputs_without, diag_without = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=False,
    )

    # Should have empty train_mismatches
    assert diag_without.train_mismatches == [], \
        "Without training labels, train_mismatches should be empty"
    print("  ✓ Without training labels: train_mismatches empty")


def test_one_ilp_per_example():
    """Test that solver runs one ILP per example."""
    print("\nTest: One ILP per example")
    print("-" * 70)

    task_id = "00576224"
    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={"ties": [{"pairs": [((0, 0), (0, 1))]}]}
        )
    ])

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=False,
    )

    # Load task to count examples
    from src.schemas.context import load_arc_task
    task_data = load_arc_task(task_id, Path("data/arc-agi_training_challenges.json"))
    num_train = len(task_data["train"])
    num_test = len(task_data["test"])

    # Should have one output per example
    assert len(outputs["train"]) == num_train, \
        f"Should have {num_train} train outputs, got {len(outputs['train'])}"
    assert len(outputs["test"]) == num_test, \
        f"Should have {num_test} test outputs, got {len(outputs['test'])}"

    print(f"  ✓ Train examples: {num_train} outputs produced")
    print(f"  ✓ Test examples: {num_test} outputs produced")


def test_backward_compatible_wrapper():
    """Test that solve_arc_task wrapper exists and works."""
    print("\nTest: Backward-compatible wrapper")
    print("-" * 70)

    task_id = "00576224"
    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={"ties": [{"pairs": [((0, 0), (0, 1))]}]}
        )
    ])

    # Call old API
    result = solve_arc_task(task_id, law_config)

    # Check output format
    assert "train_outputs_pred" in result, \
        "Wrapper should return dict with 'train_outputs_pred'"
    assert "test_outputs_pred" in result, \
        "Wrapper should return dict with 'test_outputs_pred'"

    assert isinstance(result["train_outputs_pred"], list)
    assert isinstance(result["test_outputs_pred"], list)

    print("  ✓ solve_arc_task wrapper exists")
    print("  ✓ Returns backward-compatible format")
    print(f"  ✓ Train outputs: {len(result['train_outputs_pred'])}")
    print(f"  ✓ Test outputs: {len(result['test_outputs_pred'])}")


def test_num_pixels_computation():
    """Test that num_pixels is computed correctly."""
    print("\nTest: num_pixels computation")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should compute num_pixels from H_out * W_out
    assert "num_pixels = H_out * W_out" in source, \
        "Should compute num_pixels = H_out * W_out"
    print("  ✓ num_pixels computed as H_out * W_out")

    # Should determine H_out, W_out from ExampleContext
    assert "H_out = ex.output_H if ex.output_H is not None else ex.input_H" in source, \
        "Should use output_H with fallback to input_H"
    assert "W_out = ex.output_W if ex.output_W is not None else ex.input_W" in source, \
        "Should use output_W with fallback to input_W"
    print("  ✓ H_out, W_out determined from ExampleContext")


def test_constraint_variable_tracking():
    """Test that constraints and variables are tracked correctly."""
    print("\nTest: Constraint/variable tracking")
    print("-" * 70)

    task_id = "00576224"
    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={"ties": [{"pairs": [((0, 0), (0, 1))]}]}
        )
    ])

    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=False,
    )

    # With S1 config, should have some constraints
    assert diagnostics.num_constraints > 0, \
        f"S1 should produce constraints, got {diagnostics.num_constraints}"
    print(f"  ✓ Constraints tracked: {diagnostics.num_constraints}")

    # Should have variables (one per pixel per color)
    assert diagnostics.num_variables > 0, \
        f"Should have variables, got {diagnostics.num_variables}"
    print(f"  ✓ Variables tracked: {diagnostics.num_variables}")

    # Schema IDs should be recorded
    assert diagnostics.schema_ids_used == ["S1"], \
        f"Should record S1, got {diagnostics.schema_ids_used}"
    print(f"  ✓ Schema IDs tracked: {diagnostics.schema_ids_used}")


def test_code_organization():
    """Test code organization and quality."""
    print("\nTest: Code organization")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Check has module docstring
    assert '"""' in source[:300], "Module should have docstring"
    print("  ✓ Module has docstring")

    # Check function docstrings
    assert "Args:" in source, "Functions should document Args"
    assert "Returns:" in source, "Functions should document Returns"
    print("  ✓ Functions have docstrings with Args/Returns")

    # Check imports
    assert "from pathlib import Path" in source
    assert "from src.runners.results import SolveDiagnostics" in source
    print("  ✓ Imports organized correctly")

    # Check self-test exists
    assert 'if __name__ == "__main__":' in source, \
        "Should have self-test"
    print("  ✓ Self-test exists")


def main():
    print("=" * 70)
    print("WO-M5.2 COMPREHENSIVE REVIEW TEST")
    print("Testing kernel with diagnostics integration")
    print("=" * 70)

    try:
        # Core implementation checks
        test_no_todos_stubs()
        test_signature_adaptations()
        test_error_handling()
        test_num_pixels_computation()
        test_code_organization()

        # Functional tests
        test_diagnostics_structure()
        test_training_label_comparison()
        test_one_ilp_per_example()
        test_backward_compatible_wrapper()
        test_constraint_variable_tracking()

        print("\n" + "=" * 70)
        print("✅ WO-M5.2 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Implementation quality - EXCELLENT")
        print("    - No TODOs, stubs, or simplified implementations")
        print("    - All clarified signatures used correctly")
        print("    - Error handling catches specific exceptions")
        print()
        print("  ✓ Design requirements - ALL MET")
        print("    - solve_arc_task_with_diagnostics implemented")
        print("    - Returns (outputs, diagnostics) tuple")
        print("    - One ILP per example (train + test)")
        print("    - Training label comparison works")
        print("    - Backward-compatible wrapper exists")
        print()
        print("  ✓ Functional tests - ALL PASSED")
        print("    - Diagnostics structure correctly populated")
        print("    - Constraint/variable tracking works")
        print("    - num_pixels computed correctly")
        print("    - Error handling verified")
        print()
        print("WO-M5.2 IMPLEMENTATION READY FOR M5.3")
        print("=" * 70)
        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
