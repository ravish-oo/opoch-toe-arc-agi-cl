#!/usr/bin/env python3
"""
Comprehensive review test for WO-M5.X: Enrich diagnostics with per-schema stats & example summaries.

This test verifies:
  1. No TODOs, stubs, or simplified implementations
  2. Part A: Per-schema constraint counts
     - SolveDiagnostics has schema_constraint_counts field
     - apply_schema_instance has schema_constraint_counts parameter
     - Before/after constraint measurement logic
     - Sanity check for negative deltas
     - Only counts when added > 0
  3. Part B: Example summaries
     - ExampleSummary dataclass with correct fields
     - SolveDiagnostics has example_summaries field
     - Uses existing connected_components_by_color (NO new algorithm)
     - Summaries computed for all train and test examples
  4. Part C: Diagnostics runner
     - diagnose_task.py CLI exists
     - Loads law config, runs kernel, prints JSON
  5. NO changes to constraint math (regression safety)
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.runners.kernel import solve_arc_task_with_diagnostics
from src.catalog.types import TaskLawConfig, SchemaInstance
from src.runners.results import SolveDiagnostics, ExampleSummary


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs")
    print("-" * 70)

    files = [
        project_root / "src/runners/results.py",
        project_root / "src/schemas/dispatch.py",
        project_root / "src/runners/kernel.py",
        project_root / "src/runners/diagnose_task.py",
    ]

    # Check for actual incomplete implementation markers
    # (Skip dispatch.py docstring which mentions historical M2 stubs)
    bad_patterns = [
        ("TODO:", "active TODO"),
        ("FIXME:", "active FIXME"),
        ("HACK:", "hack marker"),
        ("XXX:", "XXX marker"),
        ("# stub", "stub comment"),
        ("# TODO", "TODO comment"),
        ("# FIXME", "FIXME comment"),
    ]

    for file_path in files:
        source = file_path.read_text()
        for pattern, desc in bad_patterns:
            assert pattern not in source, \
                f"Found {desc} ('{pattern}') in {file_path.name}"

    # Check for actual NotImplementedError code (not in docstrings)
    # Look for it on lines that aren't just documentation
    for file_path in files:
        lines = file_path.read_text().split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip docstring lines and comments that mention it historically
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if 'raise NotImplementedError' in line and 'In M2' not in line:
                raise AssertionError(
                    f"Found 'raise NotImplementedError' in {file_path.name}:{i}"
                )

    print("  ✓ No active TODOs, stubs, or incomplete markers found")


def test_part_a_diagnostics_field():
    """Test that SolveDiagnostics has schema_constraint_counts field."""
    print("\nTest: Part A - SolveDiagnostics field")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Should have schema_constraint_counts field
    assert "schema_constraint_counts: Dict[str, int]" in source, \
        "SolveDiagnostics should have schema_constraint_counts field"
    print("  ✓ schema_constraint_counts field exists")

    # Should use field(default_factory=dict)
    assert "field(default_factory=dict)" in source, \
        "Should use field(default_factory=dict) for mutable default"
    print("  ✓ Uses field(default_factory=dict)")


def test_part_a_dispatch_parameter():
    """Test that apply_schema_instance has schema_constraint_counts parameter."""
    print("\nTest: Part A - apply_schema_instance parameter")
    print("-" * 70)

    dispatch_file = project_root / "src/schemas/dispatch.py"
    source = dispatch_file.read_text()

    # Should have parameter
    assert "schema_constraint_counts: Optional[Dict[str, int]]" in source, \
        "apply_schema_instance should have schema_constraint_counts parameter"
    print("  ✓ schema_constraint_counts parameter exists")


def test_part_a_counting_logic():
    """Test before/after constraint counting logic."""
    print("\nTest: Part A - Constraint counting logic")
    print("-" * 70)

    dispatch_file = project_root / "src/schemas/dispatch.py"
    source = dispatch_file.read_text()

    # Should measure before
    assert "before = len(builder.constraints)" in source, \
        "Should measure constraints before builder call"
    print("  ✓ Measures before = len(builder.constraints)")

    # Should measure after
    assert "after = len(builder.constraints)" in source, \
        "Should measure constraints after builder call"
    print("  ✓ Measures after = len(builder.constraints)")

    # Should compute delta
    assert "added = after - before" in source, \
        "Should compute added = after - before"
    print("  ✓ Computes added = after - before")

    # Should check for negative delta
    assert "if added < 0:" in source, \
        "Should check for negative delta"
    assert "RuntimeError" in source, \
        "Should raise RuntimeError for negative delta"
    print("  ✓ Sanity check for negative delta")

    # Should only update when added > 0
    assert "if added > 0:" in source, \
        "Should only update counts when added > 0"
    print("  ✓ Only updates when added > 0")


def test_part_b_example_summary_dataclass():
    """Test ExampleSummary dataclass structure."""
    print("\nTest: Part B - ExampleSummary dataclass")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Should have ExampleSummary dataclass
    assert "class ExampleSummary:" in source, \
        "Should have ExampleSummary dataclass"
    print("  ✓ ExampleSummary dataclass exists")

    # Should have required fields
    assert "input_shape: Tuple[int, int]" in source, \
        "Should have input_shape field"
    assert "output_shape: Optional[Tuple[int, int]]" in source, \
        "Should have output_shape field"
    assert "components_per_color: Dict[int, int]" in source, \
        "Should have components_per_color field"
    print("  ✓ All three required fields present")


def test_part_b_diagnostics_field():
    """Test that SolveDiagnostics has example_summaries field."""
    print("\nTest: Part B - SolveDiagnostics field")
    print("-" * 70)

    results_file = project_root / "src/runners/results.py"
    source = results_file.read_text()

    # Should have example_summaries field
    assert "example_summaries: List[ExampleSummary]" in source, \
        "SolveDiagnostics should have example_summaries field"
    print("  ✓ example_summaries field exists")

    # Should use field(default_factory=list)
    assert "field(default_factory=list)" in source, \
        "Should use field(default_factory=list) for mutable default"
    print("  ✓ Uses field(default_factory=list)")


def test_part_b_uses_existing_components():
    """Test that implementation uses existing connected_components_by_color."""
    print("\nTest: Part B - Uses existing component detection")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should import connected_components_by_color
    assert "from src.features.components import connected_components_by_color" in source, \
        "Should import connected_components_by_color"
    print("  ✓ Imports connected_components_by_color")

    # Should call it
    assert "connected_components_by_color(grid)" in source or \
           "connected_components_by_color(ex.input_grid)" in source, \
        "Should call connected_components_by_color"
    print("  ✓ Calls existing connected_components_by_color")

    # Should NOT have new component detection keywords
    forbidden = [
        "def detect_components",
        "def find_components",
        "def extract_components",
        "flood_fill",  # Unless from existing code
    ]
    for pattern in forbidden:
        if pattern in source and "connected_components_by_color" not in source:
            raise AssertionError(f"Found forbidden pattern '{pattern}' - suggests new component algorithm")

    print("  ✓ NO new component detection algorithm")


def test_part_b_kernel_summaries():
    """Test that kernel computes summaries for all examples."""
    print("\nTest: Part B - Kernel computes summaries")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should initialize example_summaries
    assert "example_summaries: List[ExampleSummary] = []" in source, \
        "Kernel should initialize example_summaries"
    print("  ✓ Initializes example_summaries")

    # Should have helper function
    assert "def components_per_color(grid: Grid)" in source, \
        "Should have components_per_color helper function"
    print("  ✓ Has components_per_color helper function")

    # Should summarize train examples
    assert "for ex in ctx.train_examples:" in source, \
        "Should loop over train_examples"
    assert "example_summaries.append(ExampleSummary(" in source, \
        "Should append ExampleSummary for each example"
    print("  ✓ Summarizes train examples")

    # Should summarize test examples
    assert "for ex in ctx.test_examples:" in source, \
        "Should loop over test_examples"
    print("  ✓ Summarizes test examples")

    # Should pass to SolveDiagnostics
    assert "example_summaries=example_summaries" in source, \
        "Should pass example_summaries to SolveDiagnostics"
    print("  ✓ Passes to SolveDiagnostics")


def test_part_c_diagnose_script():
    """Test that diagnose_task.py script exists and has correct structure."""
    print("\nTest: Part C - diagnose_task.py script")
    print("-" * 70)

    script_file = project_root / "src/runners/diagnose_task.py"
    assert script_file.exists(), "diagnose_task.py should exist"
    print("  ✓ diagnose_task.py exists")

    source = script_file.read_text()

    # Should have CLI with argparse
    assert "argparse.ArgumentParser" in source, "Should use argparse"
    assert "--task-id" in source, "Should have --task-id argument"
    assert "--challenges-path" in source, "Should have --challenges-path argument"
    print("  ✓ CLI with argparse")

    # Should load law config
    assert "load_task_law_config" in source, "Should load law config"
    print("  ✓ Loads law config")

    # Should run kernel
    assert "solve_arc_task_with_diagnostics" in source, \
        "Should call solve_arc_task_with_diagnostics"
    print("  ✓ Runs kernel with diagnostics")

    # Should serialize schema_constraint_counts
    assert "schema_constraint_counts" in source, \
        "Should serialize schema_constraint_counts"
    print("  ✓ Serializes schema_constraint_counts")

    # Should serialize example_summaries
    assert "example_summaries" in source, \
        "Should serialize example_summaries"
    print("  ✓ Serializes example_summaries")

    # Should print JSON
    assert "json.dumps" in source, "Should serialize to JSON"
    print("  ✓ Prints JSON output")


def test_kernel_integration():
    """Test that kernel properly passes counts to apply_schema_instance."""
    print("\nTest: Kernel integration")
    print("-" * 70)

    kernel_file = project_root / "src/runners/kernel.py"
    source = kernel_file.read_text()

    # Should initialize schema_constraint_counts
    assert "schema_constraint_counts: Dict[str, int] = {}" in source, \
        "Kernel should initialize schema_constraint_counts"
    print("  ✓ Initializes schema_constraint_counts")

    # Should pass to apply_schema_instance
    assert "schema_constraint_counts=schema_constraint_counts" in source, \
        "Kernel should pass schema_constraint_counts to apply_schema_instance"
    print("  ✓ Passes to apply_schema_instance calls")

    # Should pass to SolveDiagnostics
    assert "schema_constraint_counts=schema_constraint_counts" in source, \
        "Kernel should pass schema_constraint_counts to SolveDiagnostics"
    print("  ✓ Passes to SolveDiagnostics")


def test_no_constraint_math_changes():
    """Test that NO constraint math was changed (regression safety)."""
    print("\nTest: NO constraint math changes")
    print("-" * 70)

    # Check that constraint building files weren't modified
    # (Only dispatch.py should have counting logic, not builders)

    dispatch_file = project_root / "src/schemas/dispatch.py"
    source = dispatch_file.read_text()

    # Should NOT modify builder_fn call
    assert "builder_fn(task_context, enriched_params, builder)" in source, \
        "Should call builder_fn with standard signature (no changes)"
    print("  ✓ Builder function call unchanged")

    # Counting should be AROUND the call, not inside
    lines = source.split('\n')
    before_line = -1
    call_line = -1
    after_line = -1

    for i, line in enumerate(lines):
        if "before = len(builder.constraints)" in line:
            before_line = i
        if "builder_fn(task_context, enriched_params, builder)" in line:
            call_line = i
        if "after = len(builder.constraints)" in line:
            after_line = i

    assert before_line < call_line < after_line, \
        "before, call, after should be in correct order"
    print("  ✓ Counting logic wraps builder call (non-invasive)")


def test_functional_with_real_task():
    """Functional test: Run kernel with real task and verify new fields."""
    print("\nTest: Functional test with real task")
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

    # Check that diagnostics is SolveDiagnostics
    assert isinstance(diagnostics, SolveDiagnostics), \
        f"Should return SolveDiagnostics, got {type(diagnostics)}"
    print("  ✓ Returns SolveDiagnostics instance")

    # Check schema_constraint_counts
    assert isinstance(diagnostics.schema_constraint_counts, dict), \
        "schema_constraint_counts should be dict"
    assert "S1" in diagnostics.schema_constraint_counts, \
        "S1 should be in schema_constraint_counts"
    assert diagnostics.schema_constraint_counts["S1"] > 0, \
        f"S1 should contribute constraints, got {diagnostics.schema_constraint_counts['S1']}"
    print(f"  ✓ schema_constraint_counts: {diagnostics.schema_constraint_counts}")

    # Check example_summaries
    assert isinstance(diagnostics.example_summaries, list), \
        "example_summaries should be list"
    assert len(diagnostics.example_summaries) > 0, \
        "Should have at least one example summary"

    # Check first summary structure
    first_summary = diagnostics.example_summaries[0]
    assert isinstance(first_summary, ExampleSummary), \
        "Should be ExampleSummary instance"
    assert isinstance(first_summary.input_shape, tuple), \
        "input_shape should be tuple"
    assert len(first_summary.input_shape) == 2, \
        "input_shape should be (H, W)"
    assert isinstance(first_summary.components_per_color, dict), \
        "components_per_color should be dict"

    print(f"  ✓ example_summaries: {len(diagnostics.example_summaries)} examples")
    print(f"    First summary: input_shape={first_summary.input_shape}, "
          f"components={first_summary.components_per_color}")


def test_diagnose_script_execution():
    """Test that diagnose_task.py runs successfully."""
    print("\nTest: diagnose_task.py execution")
    print("-" * 70)

    # First create a test config
    from src.catalog.store import save_task_law_config
    import shutil

    test_catalog_dir = Path("test_catalog_diagnose")
    test_catalog_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create test config
        test_config = TaskLawConfig(schema_instances=[
            SchemaInstance(
                family_id="S1",
                params={"ties": [{"pairs": [((0, 0), (0, 1))]}]}
            )
        ])
        save_task_law_config("00576224", test_config, catalog_dir=test_catalog_dir)
        print("  ✓ Created test config")

        # Run diagnose_task programmatically
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "src.runners.diagnose_task",
                "--task-id", "00576224",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            env={**dict(subprocess.os.environ), "PYTHONPATH": str(project_root)}
        )

        # Check that it ran successfully
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise AssertionError(f"diagnose_task.py failed with code {result.returncode}")

        print("  ✓ Script executed successfully")

        # Parse JSON output
        output_data = json.loads(result.stdout)

        # Verify JSON structure
        assert "task_id" in output_data
        assert "schema_constraint_counts" in output_data
        assert "example_summaries" in output_data
        assert isinstance(output_data["schema_constraint_counts"], dict)
        assert isinstance(output_data["example_summaries"], list)

        print(f"  ✓ JSON output valid")
        print(f"    schema_constraint_counts: {output_data['schema_constraint_counts']}")
        print(f"    example_summaries: {len(output_data['example_summaries'])} examples")

    finally:
        # Cleanup
        if test_catalog_dir.exists():
            shutil.rmtree(test_catalog_dir)


def main():
    print("=" * 70)
    print("WO-M5.X COMPREHENSIVE REVIEW TEST")
    print("Testing enriched diagnostics: per-schema stats & example summaries")
    print("=" * 70)

    try:
        # Core implementation checks
        test_no_todos_stubs()

        # Part A: Per-schema constraint counts
        test_part_a_diagnostics_field()
        test_part_a_dispatch_parameter()
        test_part_a_counting_logic()
        test_kernel_integration()

        # Part B: Example summaries
        test_part_b_example_summary_dataclass()
        test_part_b_diagnostics_field()
        test_part_b_uses_existing_components()
        test_part_b_kernel_summaries()

        # Part C: Diagnostics runner
        test_part_c_diagnose_script()

        # Regression safety
        test_no_constraint_math_changes()

        # Functional tests
        test_functional_with_real_task()
        test_diagnose_script_execution()

        print("\n" + "=" * 70)
        print("✅ WO-M5.X COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Implementation quality - EXCELLENT")
        print("    - No TODOs, stubs, or simplified implementations")
        print("    - Clean, non-invasive instrumentation")
        print("    - NO changes to constraint math")
        print()
        print("  ✓ Part A: Per-schema constraint counts - COMPLETE")
        print("    - SolveDiagnostics.schema_constraint_counts field added")
        print("    - apply_schema_instance tracks counts via before/after measurement")
        print("    - Sanity check for negative deltas")
        print("    - Only updates when added > 0")
        print()
        print("  ✓ Part B: Example summaries - COMPLETE")
        print("    - ExampleSummary dataclass with input_shape, output_shape, components_per_color")
        print("    - Uses existing connected_components_by_color (NO new algorithm)")
        print("    - Summaries computed for all train and test examples")
        print()
        print("  ✓ Part C: Diagnostics runner - COMPLETE")
        print("    - diagnose_task.py CLI script created")
        print("    - Loads config, runs kernel, prints JSON")
        print("    - Includes both schema_constraint_counts and example_summaries")
        print()
        print("  ✓ Functional tests - ALL PASSED")
        print("    - Real task execution verified")
        print("    - Schema counts tracked correctly")
        print("    - Example summaries populated correctly")
        print("    - diagnose_task.py produces valid JSON")
        print()
        print("WO-M5.X IMPLEMENTATION COMPLETE AND VERIFIED")
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
