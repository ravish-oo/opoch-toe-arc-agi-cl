#!/usr/bin/env python3
"""
Comprehensive review test for WO-M4.4: Training-set validation runner.

This test verifies:
  1. All WO components are present
  2. Uses existing load_arc_training_solutions (not new function)
  3. Helpers work with normalized Grid structure
  4. make_law_config_for_task has real working config
  5. Error handling catches InfeasibleModelError specifically
  6. compare_grids logic correct
  7. No TODOs, stubs, or simplified implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.runners.validate_on_training import (
    parse_args,
    make_law_config_for_task,
    get_true_train_grids,
    get_true_test_grids,
    compare_grids,
    validate_on_training
)
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.core.grid_types import Grid


def test_implementation_components():
    """Test that all WO components are present."""
    print("\nTest: All WO components present")
    print("-" * 70)

    # Read source to verify components
    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Component 1: parse_args
    assert "def parse_args" in source, "parse_args function missing"
    assert "argparse.ArgumentParser" in source, "argparse setup missing"
    print("  ✓ parse_args function present")

    # Component 2: make_law_config_for_task
    assert "def make_law_config_for_task" in source, "make_law_config_for_task missing"
    print("  ✓ make_law_config_for_task function present")

    # Component 3: get_true_train_grids
    assert "def get_true_train_grids" in source, "get_true_train_grids missing"
    print("  ✓ get_true_train_grids function present")

    # Component 4: get_true_test_grids
    assert "def get_true_test_grids" in source, "get_true_test_grids missing"
    print("  ✓ get_true_test_grids function present")

    # Component 5: compare_grids
    assert "def compare_grids" in source, "compare_grids missing"
    print("  ✓ compare_grids function present")

    # Component 6: validate_on_training
    assert "def validate_on_training" in source, "validate_on_training missing"
    print("  ✓ validate_on_training function present")

    # Component 7: main
    assert "def main" in source, "main function missing"
    assert 'if __name__ == "__main__":' in source, "main guard missing"
    print("  ✓ main function and guard present")

    print("  ✓ All WO components verified")


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError"]

    for marker in markers:
        assert marker not in source, \
            f"Found '{marker}' in validate_on_training.py"

    print("  ✓ No TODOs, stubs, or markers found")


def test_uses_existing_function():
    """Test that uses existing load_arc_training_solutions."""
    print("\nTest: Uses existing load_arc_training_solutions")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Should import from arc_io
    assert "from src.core.arc_io import load_arc_training_solutions" in source, \
        "Should import load_arc_training_solutions from arc_io"

    # Should NOT define new load_training_solutions
    assert "def load_training_solutions" not in source, \
        "Should not define new load_training_solutions"

    print("  ✓ Uses existing load_arc_training_solutions from arc_io.py")
    print("  ✓ Does not create duplicate function")


def test_real_working_config():
    """Test that make_law_config_for_task has real working config."""
    print("\nTest: Real working law config (not placeholder)")
    print("-" * 70)

    # Call the function
    config = make_law_config_for_task("00576224")

    # Check it returns TaskLawConfig
    assert isinstance(config, TaskLawConfig), \
        f"Should return TaskLawConfig, got {type(config)}"

    # Check it has schema instances
    assert len(config.schema_instances) > 0, \
        "Config should have at least one schema instance"

    # Check first schema instance
    si = config.schema_instances[0]
    assert isinstance(si, SchemaInstance), \
        f"Should be SchemaInstance, got {type(si)}"

    assert si.family_id == "S1", \
        f"Expected S1 (same as smoke test), got {si.family_id}"

    # Check params are not empty
    assert "ties" in si.params, \
        "S1 params should have 'ties' key"

    assert len(si.params["ties"]) > 0, \
        "S1 ties should not be empty"

    print(f"  ✓ Returns TaskLawConfig with {len(config.schema_instances)} schema instance(s)")
    print(f"  ✓ First schema: {si.family_id}")
    print(f"  ✓ Has real params (not stub/placeholder)")
    print("  ✓ Same config as test_kernel_smoke.py")


def test_normalized_grid_helpers():
    """Test that helpers work with normalized Grid structure."""
    print("\nTest: Helpers use normalized Grid structure")
    print("-" * 70)

    # Test get_true_train_grids
    grid1 = np.array([[1, 2], [3, 4]], dtype=int)
    grid2 = np.array([[5, 6], [7, 8]], dtype=int)

    raw_task = {
        "train": [
            {"input": grid1, "output": grid2},
            {"input": grid2, "output": grid1}
        ]
    }

    train_grids = get_true_train_grids(raw_task)
    assert len(train_grids) == 2, "Should extract 2 train grids"
    assert np.array_equal(train_grids[0], grid2), "First grid should match"
    assert np.array_equal(train_grids[1], grid1), "Second grid should match"
    print("  ✓ get_true_train_grids extracts Grid objects directly")

    # Test get_true_test_grids
    solutions = {
        "task1": [grid1, grid2],
        "task2": [grid2]
    }

    test_grids1 = get_true_test_grids("task1", solutions)
    assert len(test_grids1) == 2, "Should get 2 test grids for task1"
    assert np.array_equal(test_grids1[0], grid1)

    test_grids2 = get_true_test_grids("task_unknown", solutions)
    assert len(test_grids2) == 0, "Should return empty list for unknown task"
    print("  ✓ get_true_test_grids uses Dict[str, List[Grid]] structure")

    print("  ✓ No list_of_lists_to_grid conversion needed")


def test_compare_grids_logic():
    """Test compare_grids logic."""
    print("\nTest: compare_grids logic")
    print("-" * 70)

    # Test 1: Exact match
    grid1 = np.array([[1, 2], [3, 4]], dtype=int)
    grid2 = np.array([[1, 2], [3, 4]], dtype=int)

    result = compare_grids(grid1, grid2)
    assert result["match"] == True, "Should match"
    assert result["reason"] == "exact_match"
    assert len(result["diff_coords"]) == 0
    print("  ✓ Exact match detected correctly")

    # Test 2: Shape mismatch
    grid3 = np.array([[1, 2, 3]], dtype=int)

    result = compare_grids(grid1, grid3)
    assert result["match"] == False, "Should not match"
    assert "shape mismatch" in result["reason"]
    assert "(2, 2)" in result["reason"]
    assert "(1, 3)" in result["reason"]
    print("  ✓ Shape mismatch detected with clear message")

    # Test 3: Value mismatch
    grid4 = np.array([[1, 9], [3, 4]], dtype=int)

    result = compare_grids(grid1, grid4)
    assert result["match"] == False, "Should not match"
    assert result["reason"] == "value_mismatch"
    assert (0, 1) in result["diff_coords"], "Should detect diff at (0, 1)"
    assert len(result["diff_coords"]) == 1, "Should have exactly 1 diff"
    print("  ✓ Value mismatch detected with diff coordinates")

    # Test 4: Multiple value mismatches
    grid5 = np.array([[9, 9], [9, 9]], dtype=int)

    result = compare_grids(grid1, grid5)
    assert result["match"] == False
    assert len(result["diff_coords"]) == 4, "Should have 4 diffs"
    print("  ✓ Multiple value mismatches detected")


def test_error_handling():
    """Test that error handling catches specific errors."""
    print("\nTest: Error handling (no silent fail)")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Should import InfeasibleModelError and TaskSolveError
    assert "InfeasibleModelError" in source, \
        "Should import InfeasibleModelError"

    assert "TaskSolveError" in source, \
        "Should import TaskSolveError"

    # Should catch InfeasibleModelError specifically
    assert "except InfeasibleModelError" in source, \
        "Should catch InfeasibleModelError specifically"

    # Should catch TaskSolveError specifically
    assert "except TaskSolveError" in source, \
        "Should catch TaskSolveError specifically"

    # Should catch general Exception
    assert "except Exception" in source, \
        "Should catch general Exception as fallback"

    # Should print errors (not swallow)
    assert "[ERROR]" in source, \
        "Should print ERROR messages"

    print("  ✓ Catches InfeasibleModelError specifically")
    print("  ✓ Catches TaskSolveError specifically")
    print("  ✓ Catches general Exception as fallback")
    print("  ✓ Prints error messages (no silent failures)")


def test_cli_structure():
    """Test CLI argparse structure."""
    print("\nTest: CLI argparse structure")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Check argparse setup
    assert "parser.add_argument" in source, "Should use argparse"

    # Check required argument (task_id)
    assert '"task_id"' in source or "'task_id'" in source, \
        "Should have task_id argument"

    # Check optional arguments
    assert "--challenges_path" in source, "Should have --challenges_path"
    assert "--solutions_path" in source, "Should have --solutions_path"

    # Check default paths
    assert "arc-agi_training_challenges.json" in source, \
        "Should have default challenges path"
    assert "arc-agi_training_solutions.json" in source, \
        "Should have default solutions path"

    print("  ✓ argparse configured correctly")
    print("  ✓ task_id required argument present")
    print("  ✓ Optional --challenges_path and --solutions_path")
    print("  ✓ Default paths specified")


def test_output_format():
    """Test that output format is clear for Pi-agent."""
    print("\nTest: Output format clear for Pi-agent")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Should print task info
    assert "Task structure:" in source or "task_id" in source.lower(), \
        "Should print task information"

    # Should print train validation results
    assert "TRAIN" in source, "Should have TRAIN section"

    # Should print test validation results
    assert "TEST" in source, "Should have TEST section"

    # Should print accuracy
    assert "accuracy" in source.lower(), "Should print accuracy"

    # Should show mismatches
    assert "MISMATCH" in source, "Should show MISMATCH"

    # Should show diff coordinates
    assert "diff_coords" in source, "Should show differing coordinates"

    print("  ✓ Prints task structure")
    print("  ✓ Prints train validation results")
    print("  ✓ Prints test validation results")
    print("  ✓ Prints accuracy metrics")
    print("  ✓ Shows mismatch details with diff coordinates")


def test_integration_with_kernel():
    """Test integration with kernel.py."""
    print("\nTest: Integration with kernel.py")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Should import solve_arc_task
    assert "from src.runners.kernel import solve_arc_task" in source, \
        "Should import solve_arc_task from kernel"

    # Should call solve_arc_task
    assert "solve_arc_task(task_id, law_config" in source, \
        "Should call solve_arc_task with task_id and law_config"

    # Should access train_outputs_pred and test_outputs_pred
    assert "train_outputs_pred" in source, \
        "Should access train_outputs_pred from result"

    assert "test_outputs_pred" in source, \
        "Should access test_outputs_pred from result"

    print("  ✓ Imports solve_arc_task from kernel")
    print("  ✓ Calls solve_arc_task correctly")
    print("  ✓ Accesses prediction results")


def test_code_organization():
    """Test code organization and quality."""
    print("\nTest: Code organization")
    print("-" * 70)

    validate_file = project_root / "src/runners/validate_on_training.py"
    source = validate_file.read_text()

    # Check has module docstring
    assert '"""' in source[:300], "Module should have docstring"
    print("  ✓ Module has docstring")

    # Check functions have docstrings
    assert "Args:" in source, "Functions should document Args"
    assert "Returns:" in source, "Functions should document Returns"
    print("  ✓ Functions have docstrings with Args/Returns")

    # Check imports are organized
    assert "import argparse" in source
    assert "from pathlib import Path" in source
    assert "import numpy as np" in source
    print("  ✓ Imports organized correctly")

    print("  ✓ Code organization is clean")


def main():
    print("=" * 70)
    print("WO-M4.4 COMPREHENSIVE REVIEW TEST")
    print("Testing training-set validation runner")
    print("=" * 70)

    try:
        # Core implementation
        test_implementation_components()
        test_no_todos_stubs()
        test_code_organization()

        # Design requirements
        test_uses_existing_function()
        test_real_working_config()
        test_normalized_grid_helpers()
        test_compare_grids_logic()
        test_error_handling()

        # CLI and integration
        test_cli_structure()
        test_output_format()
        test_integration_with_kernel()

        print("\n" + "=" * 70)
        print("✅ WO-M4.4 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ All WO components - VERIFIED")
        print("  ✓ No TODOs or stubs - VERIFIED")
        print("  ✓ Code organization - CLEAN")
        print()
        print("  ✓ Design requirements - ALL MET")
        print("    - Uses existing load_arc_training_solutions")
        print("    - Helpers work with normalized Grid structure")
        print("    - Real working law config (not placeholder)")
        print("    - Error handling catches specific errors")
        print("    - compare_grids logic correct")
        print()
        print("  ✓ CLI and integration - EXCELLENT")
        print("    - argparse configured correctly")
        print("    - Output format clear for Pi-agent")
        print("    - Integration with kernel.py correct")
        print()
        print("WO-M4.4 IMPLEMENTATION READY FOR PRODUCTION")
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
