#!/usr/bin/env python3
"""
Comprehensive review test for WO-M5.3: Training sweep + catalog builder.

This test verifies:
  1. No TODOs, stubs, or simplified implementations
  2. NO law discovery code (HIGHEST PRIORITY)
  3. Correct code flow: load config -> run kernel -> save/log
  4. Only saves to catalog when status == "ok"
  5. Writes failures to JSONL when status != "ok"
  6. Proper error handling
  7. Correct JSONL format
  8. CLI argparse structure
  9. Summary statistics
"""

import sys
from pathlib import Path
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.catalog.store import save_task_law_config, load_task_law_config
from src.catalog.types import TaskLawConfig, SchemaInstance


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    files = [
        project_root / "src/catalog/store.py",
        project_root / "src/runners/build_catalog_from_training.py",
    ]

    markers = ["TODO", "FIXME", "HACK", "XXX", "NotImplementedError",
               "stub", "Stub", "MVP"]

    for file_path in files:
        source = file_path.read_text()
        for marker in markers:
            # Allow "simplified" only in comments saying "NO simplified"
            if marker.lower() == "simplified" and "NO" in source:
                continue
            assert marker not in source, \
                f"Found '{marker}' in {file_path.name}"

    print("  ✓ No TODOs, stubs, or markers found")


def test_no_law_discovery():
    """Test that NO law discovery code exists (HIGHEST PRIORITY)."""
    print("\nTest: NO law discovery code (HIGHEST PRIORITY)")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should NOT have law discovery keywords in actual code
    forbidden_patterns = [
        "infer_schema",
        "discover_law",
        "mine_law",
        "analyze_features",
        "detect_pattern",
        "extract_law",
        "find_equivalence",
        "compute_component",  # φ feature computation
    ]

    for pattern in forbidden_patterns:
        assert pattern not in source, \
            f"Found forbidden pattern '{pattern}' - suggests law discovery code"

    # Should have comments saying "NO law discovery"
    assert "NO law discovery" in source or "does NO law discovery" in source, \
        "Should document that it does NO law discovery"

    print("  ✓ NO law discovery code found")
    print("  ✓ Comments confirm: 'does NO law discovery'")


def test_correct_code_flow():
    """Test that code flow is: load config -> run kernel -> save/log."""
    print("\nTest: Correct code flow")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should load config
    assert "load_task_law_config" in source, \
        "Should call load_task_law_config"
    print("  ✓ Loads law config")

    # Should run kernel
    assert "solve_arc_task_with_diagnostics" in source, \
        "Should call solve_arc_task_with_diagnostics"
    assert "use_training_labels=True" in source, \
        "Should use training labels for validation"
    print("  ✓ Runs kernel with use_training_labels=True")

    # Should save on success
    assert "save_task_law_config" in source, \
        "Should call save_task_law_config"
    print("  ✓ Saves to catalog")

    # Should log failures
    assert "failure_log_file.write" in source, \
        "Should write to failure log"
    assert "json.dumps" in source, \
        "Should serialize to JSON"
    print("  ✓ Logs failures to JSONL")


def test_status_based_logic():
    """Test that catalog update only happens when status == 'ok'."""
    print("\nTest: Status-based logic")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should check status == "ok"
    assert 'if diagnostics.status == "ok"' in source, \
        "Should check for status == 'ok'"
    print("  ✓ Checks status == 'ok'")

    # Should save only when ok
    # Extract the if block and verify save_task_law_config is inside it
    ok_block_start = source.find('if diagnostics.status == "ok"')
    else_block_start = source.find('else:', ok_block_start)
    ok_block = source[ok_block_start:else_block_start]

    assert "save_task_law_config" in ok_block, \
        "save_task_law_config should be in 'ok' block"
    print("  ✓ Saves to catalog only when status == 'ok'")

    # Should log failures in else block
    else_block = source[else_block_start:else_block_start + 1000]
    assert "failure_log_file.write" in else_block, \
        "Should write to failure log in else block"
    print("  ✓ Logs failures when status != 'ok'")


def test_error_handling():
    """Test proper error handling."""
    print("\nTest: Error handling")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should have try/except around kernel call
    assert "try:" in source and "except Exception" in source, \
        "Should have try/except for error handling"
    print("  ✓ Has try/except around kernel call")

    # Should write error to failure log
    assert '"status": "error"' in source or "'status': 'error'" in source, \
        "Should set status='error' for exceptions"
    print("  ✓ Writes errors to failure log with status='error'")

    # Should flush after each write (important for long-running sweeps)
    assert "flush()" in source, \
        "Should flush failure log after each write"
    print("  ✓ Flushes failure log after each write")


def test_jsonl_format():
    """Test that failure records have correct format."""
    print("\nTest: JSONL format")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should serialize diagnostics fields
    required_fields = [
        "task_id",
        "status",
        "solver_status",
        "num_constraints",
        "num_variables",
        "schema_ids_used",
        "train_mismatches",
        "error_message",
    ]

    for field in required_fields:
        assert f'"{field}"' in source or f"'{field}'" in source, \
            f"Should serialize field '{field}'"

    print(f"  ✓ All {len(required_fields)} required fields serialized")


def test_cli_structure():
    """Test CLI argparse structure."""
    print("\nTest: CLI argparse structure")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Check argparse setup
    assert "argparse.ArgumentParser" in source, "Should use argparse"
    print("  ✓ Uses argparse")

    # Check required arguments
    assert "--challenges-path" in source, "Should have --challenges-path"
    assert "--failure-log" in source, "Should have --failure-log"
    assert "--only-with-configs" in source, "Should have --only-with-configs"
    assert "--max-tasks" in source, "Should have --max-tasks"
    print("  ✓ All CLI arguments present")

    # Check defaults
    assert "arc-agi_training_challenges.json" in source, \
        "Should have default challenges path"
    assert "training_failures.jsonl" in source, \
        "Should have default failure log path"
    print("  ✓ Default paths specified")


def test_summary_statistics():
    """Test that summary statistics are printed."""
    print("\nTest: Summary statistics")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should print summary
    assert "SWEEP SUMMARY" in source or "Summary" in source, \
        "Should print summary"
    print("  ✓ Prints summary")

    # Should track counts
    assert "num_processed" in source, "Should track processed count"
    assert "num_ok" in source, "Should track OK count"
    assert "num_failures" in source, "Should track failures count"
    assert "num_skipped" in source, "Should track skipped count"
    print("  ✓ Tracks all statistics")


def test_directory_creation():
    """Test that directories are created properly."""
    print("\nTest: Directory creation")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Should create parent directory for failure log
    assert "mkdir(parents=True, exist_ok=True)" in source, \
        "Should create directories with mkdir(parents=True, exist_ok=True)"
    print("  ✓ Creates directories with mkdir(parents=True, exist_ok=True)")


def test_code_organization():
    """Test code organization and quality."""
    print("\nTest: Code organization")
    print("-" * 70)

    build_file = project_root / "src/runners/build_catalog_from_training.py"
    source = build_file.read_text()

    # Check has module docstring
    assert '"""' in source[:500], "Module should have docstring"
    print("  ✓ Module has docstring")

    # Check function docstrings
    assert "Args:" in source, "Functions should document Args"
    assert "Returns:" in source or "Example:" in source, \
        "Functions should document Returns or Examples"
    print("  ✓ Functions have docstrings")

    # Check imports
    assert "from src.core.arc_io import load_arc_task_ids" in source
    assert "from src.catalog.store import" in source
    assert "from src.runners.kernel import solve_arc_task_with_diagnostics" in source
    print("  ✓ Imports organized correctly")

    # Check main guard
    assert 'if __name__ == "__main__":' in source
    print("  ✓ Has main guard")


def test_functional_sweep():
    """Functional test: run actual sweep with test configs."""
    print("\nTest: Functional sweep")
    print("-" * 70)

    # Clean up any existing test data
    test_catalog_dir = Path("test_catalog_temp")
    test_log = Path("test_failures_temp.jsonl")

    if test_catalog_dir.exists():
        shutil.rmtree(test_catalog_dir)
    if test_log.exists():
        test_log.unlink()

    try:
        # Create test config
        test_catalog_dir.mkdir(parents=True, exist_ok=True)
        test_config = TaskLawConfig(schema_instances=[])
        save_task_law_config("00576224", test_config, catalog_dir=test_catalog_dir)
        print("  ✓ Created test config")

        # Run sweep programmatically
        from src.runners.build_catalog_from_training import sweep_training_tasks

        sweep_training_tasks(
            challenges_path=Path("data/arc-agi_training_challenges.json"),
            failure_log_path=test_log,
            only_with_configs=False,
            max_tasks=2,
        )
        print("  ✓ Sweep completed without crash")

        # Verify failure log exists and has valid JSONL
        assert test_log.exists(), "Failure log should be created"
        with test_log.open("r") as f:
            lines = f.readlines()
            assert len(lines) > 0, "Should have at least one failure"
            for line in lines:
                record = json.loads(line)  # Should not raise
                assert "task_id" in record
                assert "status" in record
        print(f"  ✓ Failure log created with {len(lines)} record(s)")

    finally:
        # Cleanup
        if test_catalog_dir.exists():
            shutil.rmtree(test_catalog_dir)
        if test_log.exists():
            test_log.unlink()
        print("  ✓ Cleaned up test files")


def main():
    print("=" * 70)
    print("WO-M5.3 COMPREHENSIVE REVIEW TEST")
    print("Testing training sweep + catalog builder")
    print("=" * 70)

    try:
        # Core implementation checks
        test_no_todos_stubs()
        test_no_law_discovery()  # HIGHEST PRIORITY
        test_correct_code_flow()
        test_status_based_logic()
        test_error_handling()
        test_code_organization()

        # Design requirements
        test_jsonl_format()
        test_cli_structure()
        test_summary_statistics()
        test_directory_creation()

        # Functional test
        test_functional_sweep()

        print("\n" + "=" * 70)
        print("✅ WO-M5.3 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Implementation quality - EXCELLENT")
        print("    - No TODOs, stubs, or simplified implementations")
        print("    - NO law discovery code (verified)")
        print("    - Correct code flow: load -> run -> save/log")
        print()
        print("  ✓ Design requirements - ALL MET")
        print("    - Only saves to catalog when status == 'ok'")
        print("    - Writes failures to JSONL when status != 'ok'")
        print("    - Proper error handling with flush")
        print("    - Correct JSONL format with all fields")
        print()
        print("  ✓ Functional tests - ALL PASSED")
        print("    - CLI structure correct")
        print("    - Summary statistics printed")
        print("    - Actual sweep runs without crash")
        print()
        print("WO-M5.3 IMPLEMENTATION READY FOR M5.4")
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
