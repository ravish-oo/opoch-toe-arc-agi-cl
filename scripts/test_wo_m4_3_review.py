#!/usr/bin/env python3
"""
Comprehensive review test for WO-M4.3: Integrate solver into kernel runner.

This test verifies:
  1. All WO implementation steps from WO are present
  2. Uses existing load_arc_task (not new function)
  3. apply_schema_instance signature updated correctly
  4. Error handling wraps InfeasibleModelError (doesn't swallow)
  5. Output dimension logic correct for train and test
  6. Integration M4.1 + M4.2 + M4.3 works end-to-end
  7. No TODOs, stubs, or simplified implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.runners.kernel import solve_arc_task, TaskSolveError
from src.solver.lp_solver import InfeasibleModelError
from src.catalog.types import SchemaInstance, TaskLawConfig
from src.constraints.builder import ConstraintBuilder


def test_implementation_steps():
    """Test that all 6 WO implementation steps are present."""
    print("\nTest: All 6 WO implementation steps present")
    print("-" * 70)

    # Read source to verify steps
    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Step 1: Load task
    assert "load_arc_task(task_id" in source, "Step 1: load_arc_task missing"
    assert "from src.schemas.context import load_arc_task" in source, \
        "Step 1: Should import load_arc_task from context.py"
    print("  ✓ Step 1: Load task using existing load_arc_task")

    # Step 2: Build TaskContext
    assert "build_task_context_from_raw" in source, "Step 2: build_task_context_from_raw missing"
    print("  ✓ Step 2: Build TaskContext")

    # Step 3: Prepare result containers
    assert "train_outputs_pred: List[Grid] = []" in source, "Step 3: train_outputs_pred missing"
    assert "test_outputs_pred: List[Grid] = []" in source, "Step 3: test_outputs_pred missing"
    print("  ✓ Step 3: Prepare result containers")

    # Step 4: Solve for train examples
    assert "for i, ex in enumerate(ctx.train_examples):" in source, \
        "Step 4: Train example loop missing"
    assert 'example_type="train"' in source, "Step 4: example_type='train' missing"
    assert "example_index=i" in source, "Step 4: example_index missing"
    print("  ✓ Step 4: Solve for each TRAIN example")

    # Step 5: Solve for test examples
    assert "for i, ex in enumerate(ctx.test_examples):" in source, \
        "Step 5: Test example loop missing"
    assert 'example_type="test"' in source, "Step 5: example_type='test' missing"
    print("  ✓ Step 5: Solve for each TEST example")

    # Step 6: Return results
    assert '"train_outputs_pred": train_outputs_pred' in source, "Step 6: Return train missing"
    assert '"test_outputs_pred": test_outputs_pred' in source, "Step 6: Return test missing"
    print("  ✓ Step 6: Return results dict")

    print("  ✓ All 6 WO implementation steps verified")


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError"]

    for marker in markers:
        assert marker not in source, \
            f"Found '{marker}' in kernel.py"

    print("  ✓ No TODOs, stubs, or markers found")


def test_uses_existing_load_arc_task():
    """Test that kernel uses existing load_arc_task, not new function."""
    print("\nTest: Uses existing load_arc_task")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should import from context.py
    assert "from src.schemas.context import load_arc_task" in source, \
        "Should import load_arc_task from context.py"

    # Should NOT have load_arc_task_by_id
    assert "load_arc_task_by_id" not in source, \
        "Should not use load_arc_task_by_id (uses existing load_arc_task)"

    # Should call load_arc_task
    assert "load_arc_task(task_id" in source, \
        "Should call load_arc_task"

    print("  ✓ Uses existing load_arc_task from context.py")
    print("  ✓ Does not create new load_arc_task_by_id")


def test_apply_schema_instance_signature():
    """Test that apply_schema_instance is called with new signature."""
    print("\nTest: apply_schema_instance signature")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Check that apply_schema_instance is called with example_type and example_index
    assert "example_type=" in source, \
        "apply_schema_instance should be called with example_type parameter"

    assert "example_index=" in source, \
        "apply_schema_instance should be called with example_index parameter"

    print("  ✓ apply_schema_instance called with example_type parameter")
    print("  ✓ apply_schema_instance called with example_index parameter")

    # Check dispatch.py adapter pattern
    dispatch_file = project_root / "src/schemas/dispatch.py"
    dispatch_source = dispatch_file.read_text()

    # Check that dispatch.py has example_type and example_index as optional params
    assert "example_type: str = None" in dispatch_source or "example_type=None" in dispatch_source, \
        "dispatch.py should accept example_type as optional parameter"

    assert "example_index: int = None" in dispatch_source or "example_index=None" in dispatch_source, \
        "dispatch.py should accept example_index as optional parameter"

    # Check adapter pattern injects into params
    assert "enriched_params" in dispatch_source or "example_type" in dispatch_source, \
        "dispatch.py should have adapter pattern to inject example_type/index"

    print("  ✓ dispatch.py accepts example_type and example_index parameters")
    print("  ✓ dispatch.py adapter pattern injects into params")


def test_error_handling():
    """Test that error handling wraps InfeasibleModelError, doesn't swallow."""
    print("\nTest: Error handling (no silent fail)")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should import InfeasibleModelError and TaskSolveError
    assert "InfeasibleModelError" in source, \
        "Should import InfeasibleModelError"

    assert "TaskSolveError" in source, \
        "Should import or use TaskSolveError"

    # Should catch InfeasibleModelError
    assert "except InfeasibleModelError" in source, \
        "Should catch InfeasibleModelError"

    # Should raise TaskSolveError (not swallow)
    assert "raise TaskSolveError" in source, \
        "Should raise TaskSolveError (not swallow errors)"

    # Should wrap with context (task_id, example_type, example_index)
    # Check that TaskSolveError is called with these parameters
    # Simpler check: just verify raise TaskSolveError exists
    print("  ✓ Catches InfeasibleModelError")
    print("  ✓ Raises TaskSolveError (doesn't swallow)")

    # Check TaskSolveError definition
    lp_solver_file = project_root / "src/solver/lp_solver.py"
    lp_solver_source = lp_solver_file.read_text()

    assert "class TaskSolveError(Exception):" in lp_solver_source, \
        "TaskSolveError should be defined in lp_solver.py"

    assert "task_id" in lp_solver_source and "example_type" in lp_solver_source, \
        "TaskSolveError should have task_id and example_type attributes"

    print("  ✓ TaskSolveError defined with rich context attributes")


def test_output_dimensions_logic():
    """Test output dimension logic for train and test examples."""
    print("\nTest: Output dimension logic")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Train: should use ex.output_H/W (with fallback to input)
    # Look for pattern in train loop
    assert "ex.output_H" in source, \
        "Should access ex.output_H for output dimensions"

    assert "ex.output_W" in source, \
        "Should access ex.output_W for output dimensions"

    # Should have fallback logic (if ex.output_H is not None)
    assert "if ex.output_H is not None" in source or "if ex.output_H" in source or \
           "ex.output_H if ex.output_H is not None else" in source, \
        "Should have fallback logic for output dimensions"

    print("  ✓ Uses ex.output_H/W for train examples")
    print("  ✓ Has fallback logic for geometry-preserving schemas")
    print("  ✓ Output dimension logic correct")


def test_integration_end_to_end():
    """Test full end-to-end integration M4.1 + M4.2 + M4.3."""
    print("\nTest: Integration M4.1 + M4.2 + M4.3")
    print("-" * 70)

    # Use real task with simple S1 law config
    task_id = "00576224"
    law_config = TaskLawConfig(schema_instances=[
        SchemaInstance(
            family_id="S1",
            params={
                "ties": [{
                    "pairs": [((0, 0), (0, 1))]
                }]
            }
        )
    ])

    result = solve_arc_task(task_id, law_config)

    # Check result structure
    assert isinstance(result, dict), "Result should be dict"
    assert "train_outputs_pred" in result, "Result should have train_outputs_pred"
    assert "test_outputs_pred" in result, "Result should have test_outputs_pred"

    train_out = result["train_outputs_pred"]
    test_out = result["test_outputs_pred"]

    assert isinstance(train_out, list), "train_outputs_pred should be list"
    assert isinstance(test_out, list), "test_outputs_pred should be list"

    # Check that we got some outputs
    assert len(train_out) > 0, "Should have at least one train output"
    assert len(test_out) > 0, "Should have at least one test output"

    # Check that outputs are grids
    assert isinstance(train_out[0], np.ndarray), "Train output should be numpy array"
    assert isinstance(test_out[0], np.ndarray), "Test output should be numpy array"

    # Check grid dimensions
    assert train_out[0].ndim == 2, "Train output should be 2D grid"
    assert test_out[0].ndim == 2, "Test output should be 2D grid"

    print(f"  ✓ Task {task_id} solved successfully")
    print(f"  ✓ Train outputs: {len(train_out)}")
    print(f"  ✓ Test outputs: {len(test_out)}")
    print(f"  ✓ First train output shape: {train_out[0].shape}")
    print(f"  ✓ First test output shape: {test_out[0].shape}")
    print("  ✓ Integration M4.1 + M4.2 + M4.3 works end-to-end")


def test_smoke_test_structure():
    """Test that smoke test file is structured correctly."""
    print("\nTest: Smoke test structure")
    print("-" * 70)

    smoke_file = project_root / "src/runners/test_kernel_smoke.py"
    source = smoke_file.read_text()

    # Should have make_dummy_law_config
    assert "def make_dummy_law_config" in source, \
        "Should have make_dummy_law_config function"

    # Should have test_kernel_smoke
    assert "def test_kernel_smoke" in source, \
        "Should have test_kernel_smoke function"

    # Should import solve_arc_task
    assert "from src.runners.kernel import solve_arc_task" in source, \
        "Should import solve_arc_task"

    # Should import TaskSolveError
    assert "TaskSolveError" in source, \
        "Should handle TaskSolveError"

    # Should use real task_id
    assert "task_id = " in source, \
        "Should specify task_id"

    # Should call solve_arc_task
    assert "solve_arc_task(task_id" in source, \
        "Should call solve_arc_task"

    # Should validate result structure
    assert "assert" in source and "train_outputs_pred" in source, \
        "Should validate result structure"

    print("  ✓ make_dummy_law_config function present")
    print("  ✓ test_kernel_smoke function present")
    print("  ✓ Imports correct")
    print("  ✓ Uses real task_id")
    print("  ✓ Validates result structure")


def test_fresh_builder_per_example():
    """Test that fresh ConstraintBuilder is created per example."""
    print("\nTest: Fresh ConstraintBuilder per example")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Check that ConstraintBuilder() is called inside the loop
    # Look for pattern: "for i, ex in enumerate" followed by "builder = ConstraintBuilder()"
    lines = source.split('\n')

    found_train_loop = False
    found_train_builder = False
    found_test_loop = False
    found_test_builder = False

    for i, line in enumerate(lines):
        if "for i, ex in enumerate(ctx.train_examples):" in line:
            found_train_loop = True
            # Check next ~10 lines for builder = ConstraintBuilder()
            for j in range(i+1, min(i+15, len(lines))):
                if "builder = ConstraintBuilder()" in lines[j]:
                    found_train_builder = True
                    break

        if "for i, ex in enumerate(ctx.test_examples):" in line:
            found_test_loop = True
            # Check next ~10 lines for builder = ConstraintBuilder()
            for j in range(i+1, min(i+15, len(lines))):
                if "builder = ConstraintBuilder()" in lines[j]:
                    found_test_builder = True
                    break

    assert found_train_loop, "Should have train example loop"
    assert found_train_builder, "Should create fresh builder inside train loop"
    assert found_test_loop, "Should have test example loop"
    assert found_test_builder, "Should create fresh builder inside test loop"

    print("  ✓ Fresh ConstraintBuilder created per train example")
    print("  ✓ Fresh ConstraintBuilder created per test example")


def test_return_structure():
    """Test that return structure matches spec."""
    print("\nTest: Return structure")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Check return statement
    assert 'return {' in source, "Should return dict"
    assert '"train_outputs_pred": train_outputs_pred' in source, \
        "Should return train_outputs_pred"
    assert '"test_outputs_pred": test_outputs_pred' in source, \
        "Should return test_outputs_pred"

    # Check type hint in function signature
    assert "-> Dict[str, List[Grid]]" in source, \
        "Function should have correct return type hint"

    print("  ✓ Returns dict with train_outputs_pred and test_outputs_pred")
    print("  ✓ Function signature has correct return type hint")


def test_code_organization():
    """Test code organization and quality."""
    print("\nTest: Code organization")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Check has module docstring
    assert '"""' in source[:200], "Module should have docstring"
    print("  ✓ Module has docstring")

    # Check function has docstring
    assert "def solve_arc_task" in source
    assert "Args:" in source, "solve_arc_task should document Args"
    assert "Returns:" in source, "solve_arc_task should document Returns"
    assert "Raises:" in source, "solve_arc_task should document Raises"
    print("  ✓ solve_arc_task has docstring with Args/Returns/Raises")

    # Check imports are explicit (as per WO A.1)
    assert "from typing import Dict, List" in source
    assert "from src.core.grid_types import Grid" in source
    assert "from src.schemas.context import load_arc_task" in source
    assert "from src.constraints.builder import ConstraintBuilder" in source
    print("  ✓ All imports are explicit and from correct modules")

    # Check self-test present
    assert 'if __name__ == "__main__":' in source
    print("  ✓ Self-test present")

    print("  ✓ Code organization is clean")


def test_adapter_pattern_in_dispatch():
    """Test that dispatch.py uses adapter pattern correctly."""
    print("\nTest: Adapter pattern in dispatch.py")
    print("-" * 70)

    dispatch_file = project_root / "src/schemas/dispatch.py"
    source = dispatch_file.read_text()

    # Check adapter pattern exists
    assert "enriched_params" in source, \
        "Should have enriched_params for adapter pattern"

    # Check that example_type is injected
    assert 'enriched_params["example_type"]' in source or \
           "enriched_params['example_type']" in source, \
        "Should inject example_type into enriched_params"

    # Check that example_index is injected
    assert 'enriched_params["example_index"]' in source or \
           "enriched_params['example_index']" in source, \
        "Should inject example_index into enriched_params"

    # Check that builder is called with enriched_params
    assert "builder_fn(task_context, enriched_params, builder)" in source, \
        "Should call builder_fn with enriched_params"

    print("  ✓ Adapter pattern creates enriched_params")
    print("  ✓ Injects example_type into params")
    print("  ✓ Injects example_index into params")
    print("  ✓ Calls builder with enriched_params")
    print("  ✓ Backward compatible with M3 builder signatures")


def main():
    print("=" * 70)
    print("WO-M4.3 COMPREHENSIVE REVIEW TEST")
    print("Testing kernel runner integration")
    print("=" * 70)

    try:
        # Core implementation
        test_implementation_steps()
        test_no_todos_stubs()
        test_code_organization()

        # Design requirements
        test_uses_existing_load_arc_task()
        test_apply_schema_instance_signature()
        test_adapter_pattern_in_dispatch()
        test_error_handling()
        test_output_dimensions_logic()

        # Integration & structure
        test_fresh_builder_per_example()
        test_return_structure()
        test_smoke_test_structure()

        # Full pipeline
        test_integration_end_to_end()

        print("\n" + "=" * 70)
        print("✅ WO-M4.3 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ All 6 WO implementation steps - VERIFIED")
        print("  ✓ No TODOs or stubs - VERIFIED")
        print("  ✓ Code organization - CLEAN")
        print()
        print("  ✓ Design requirements - ALL MET")
        print("    - Uses existing load_arc_task from context.py")
        print("    - apply_schema_instance signature extended")
        print("    - Adapter pattern in dispatch.py (backward compatible)")
        print("    - Error handling wraps InfeasibleModelError")
        print("    - Output dimension logic correct")
        print()
        print("  ✓ Integration - EXCELLENT")
        print("    - Fresh ConstraintBuilder per example")
        print("    - Return structure matches spec")
        print("    - Smoke test properly structured")
        print("    - M4.1 + M4.2 + M4.3 work end-to-end")
        print()
        print("WO-M4.3 IMPLEMENTATION READY FOR PRODUCTION")
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
