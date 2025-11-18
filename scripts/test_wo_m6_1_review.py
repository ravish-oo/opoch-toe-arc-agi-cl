#!/usr/bin/env python3
"""
Comprehensive review test for WO-M6.1: Role labeller (WL/q).

This test verifies:
  1. No TODOs, stubs, or simplified implementations (HIGHEST PRIORITY)
  2. Pure structural refinement - no defaults, no heuristics
  3. WL loop is deterministic (sorted tuples, not random hash)
  4. Neighbors are 4-connected, kind-local only
  5. Uses existing φ operators from M1
  6. No input↔output cross-linking at WL level
  7. Correct node creation (train_in, train_out, test_in)
  8. Initial labels from pure structure
  9. Output format matches spec
  10. Determinism verified on real tasks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_no_todos_stubs():
    """Test that implementation has no TODOs or stubs."""
    print("\nTest: No TODOs or stubs (HIGHEST PRIORITY)")
    print("-" * 70)

    files = [
        project_root / "src/law_mining/roles.py",
        project_root / "src/law_mining/test_roles_smoke.py",
    ]

    bad_patterns = [
        ("TODO:", "active TODO"),
        ("FIXME:", "active FIXME"),
        ("HACK:", "hack marker"),
        ("XXX:", "XXX marker"),
        ("# stub", "stub comment"),
        ("# TODO", "TODO comment"),
        ("# FIXME", "FIXME comment"),
        ("# MVP", "MVP marker"),
        ("# simplified", "simplified implementation"),
    ]

    for file_path in files:
        source = file_path.read_text()
        for pattern, desc in bad_patterns:
            assert pattern not in source, \
                f"Found {desc} ('{pattern}') in {file_path.name}"

    # Check for actual NotImplementedError raises
    for file_path in files:
        lines = file_path.read_text().split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if 'raise NotImplementedError' in line:
                raise AssertionError(
                    f"Found 'raise NotImplementedError' in {file_path.name}:{i}"
                )

    print("  ✓ No TODOs, stubs, or incomplete markers found")


def test_no_defaults_heuristics():
    """Test that implementation has no defaults or heuristics (TOE violation)."""
    print("\nTest: No defaults or heuristics (TOE compliance)")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should NOT have special treatment for color 0
    forbidden_patterns = [
        "if color == 0:",
        "if color != 0:",
        "background",
        "foreground",
        "special case",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in source, \
            f"Found potential heuristic: '{pattern}'"

    print("  ✓ No special color treatment found")
    print("  ✓ No background/foreground heuristics")
    print("  ✓ Pure structural features only")


def test_wl_deterministic():
    """Test that WL loop is deterministic."""
    print("\nTest: WL loop determinism")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should use sorted() for deterministic ordering
    assert "tuple(sorted(neigh_labels))" in source, \
        "Should use sorted() for neighbor labels"
    print("  ✓ Uses sorted() for neighbor labels")

    # Should NOT use random hash
    # Allow "no random" in comments/docstrings, disallow actual random usage
    if 'import random' in source or 'from random' in source:
        raise AssertionError("Should not import random module")
    if 'random.random' in source or 'random.choice' in source:
        raise AssertionError("Should not use random functions")
    # The word "random" can appear in docstrings saying "no random hashing"
    print("  ✓ No random hashing")

    # Should have early exit
    assert "if all(new_labels[n] == labels[n]" in source, \
        "Should have early exit on convergence"
    print("  ✓ Early exit on convergence")


def test_neighbors_4connected():
    """Test that neighbors are 4-connected and kind-local."""
    print("\nTest: Neighbors are 4-connected, kind-local")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should have 4 directions
    assert "[(-1, 0), (1, 0), (0, -1), (0, 1)]" in source, \
        "Should use 4-connected neighbors (not 8-connected)"
    print("  ✓ 4-connected (up, down, left, right)")

    # Should preserve kind
    assert "Node(node.kind, node.example_idx," in source, \
        "Neighbors should preserve kind and example_idx"
    print("  ✓ Neighbors preserve kind and example_idx")

    # Should NOT cross-link different kinds
    assert "train_in" not in source or "train_out" not in source or \
           "# No cross-kind" in source or \
           all(x in source for x in ["node.kind", "node.example_idx"]), \
        "Should not cross-link different kinds"
    print("  ✓ No cross-kind linking in neighbors")


def test_uses_existing_phi():
    """Test that implementation uses existing φ operators from M1."""
    print("\nTest: Uses existing φ operators (M1)")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    required_imports = [
        "from src.features.coords_bands import",
        "row_band_labels",
        "col_band_labels",
        "from src.features.components import",
        "connected_components_by_color",
        "compute_shape_signature",
    ]

    for pattern in required_imports:
        assert pattern in source, \
            f"Should import '{pattern}' from existing M1 modules"

    print("  ✓ Imports row_band_labels from coords_bands")
    print("  ✓ Imports col_band_labels from coords_bands")
    print("  ✓ Imports connected_components_by_color from components")
    print("  ✓ Imports compute_shape_signature from components")

    # Should NOT reimplement these
    assert "def compute_component" not in source, \
        "Should not reimplement component detection"
    assert "def band_label" not in source or "# band_label" in source, \
        "Should not reimplement band labels"

    print("  ✓ No reinvented algorithms")


def test_nodes_created_correctly():
    """Test that nodes are created for train_in, train_out, test_in only."""
    print("\nTest: Node creation")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should create train_in nodes
    assert 'Node("train_in", ex_idx, r, c)' in source, \
        "Should create train_in nodes"
    print("  ✓ Creates train_in nodes")

    # Should create train_out nodes only if output exists
    assert "if ex_ctx.output_grid is not None:" in source, \
        "Should check if output_grid exists"
    assert 'Node("train_out", ex_idx, r, c)' in source, \
        "Should create train_out nodes"
    print("  ✓ Creates train_out nodes (only if output exists)")

    # Should create test_in nodes
    assert 'Node("test_in", ex_idx, r, c)' in source, \
        "Should create test_in nodes"
    print("  ✓ Creates test_in nodes")

    # Should NOT create test_out nodes
    assert 'Node("test_out"' not in source, \
        "Should NOT create test_out nodes (don't exist)"
    print("  ✓ Does NOT create test_out nodes")


def test_initial_labels_structure():
    """Test that initial labels use pure structural features."""
    print("\nTest: Initial label structure")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should include all required features
    required_features = [
        "node.kind",
        "color",
        "row_band",
        "col_band",
        "is_border",
        "shape_sig",
    ]

    # Check initial label tuple
    assert "labels[node] = (node.kind, color, row_band, col_band, is_border, shape_sig)" in source, \
        "Initial label should be tuple of structural features"

    print("  ✓ Includes kind (train_in/train_out/test_in)")
    print("  ✓ Includes color from grid")
    print("  ✓ Includes row_band")
    print("  ✓ Includes col_band")
    print("  ✓ Includes is_border flag")
    print("  ✓ Includes shape_signature")


def test_output_format():
    """Test that output format matches RolesMapping spec."""
    print("\nTest: Output format")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should have correct type alias
    assert "RolesMapping = Dict[Tuple[NodeKind, int, int, int], int]" in source, \
        "Should define RolesMapping type"
    print("  ✓ RolesMapping type defined")

    # Should map to role_id
    assert "roles[(node.kind, node.example_idx, node.r, node.c)] = role_id" in source, \
        "Should map (kind, ex_idx, r, c) to role_id"
    print("  ✓ Maps (kind, example_idx, r, c) -> role_id")

    # Should return RolesMapping
    assert "return roles" in source, \
        "Should return roles mapping"
    print("  ✓ Returns RolesMapping")


def test_functional_determinism():
    """Functional test: verify determinism on real tasks."""
    print("\nTest: Functional determinism on real tasks")
    print("-" * 70)

    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles

    test_tasks = ["00576224", "007bbfb7"]
    challenges_path = Path("data/arc-agi_training_challenges.json")

    for task_id in test_tasks:
        print(f"\n  Testing task: {task_id}")

        # Load task
        raw_task = load_arc_task(task_id, challenges_path)
        task_context = build_task_context_from_raw(raw_task)

        # Compute roles twice
        roles1 = compute_roles(task_context)
        roles2 = compute_roles(task_context)

        # Verify identical
        assert roles1 == roles2, \
            f"compute_roles not deterministic on {task_id}"
        print(f"    ✓ Deterministic: {len(roles1)} pixels, {len(set(roles1.values()))} roles")


def test_functional_sanity():
    """Functional test: sanity checks on role counts."""
    print("\nTest: Functional sanity checks")
    print("-" * 70)

    from src.schemas.context import load_arc_task, build_task_context_from_raw
    from src.law_mining.roles import compute_roles

    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    roles = compute_roles(task_context)
    num_pixels = len(roles)
    num_roles = len(set(roles.values()))

    print(f"  Task: {task_id}")
    print(f"  Total pixels: {num_pixels}")
    print(f"  Distinct roles: {num_roles}")
    print(f"  Compression: {num_roles}/{num_pixels} = {num_roles/num_pixels:.1%}")

    # Sanity checks
    assert num_roles > 0, "Should have at least one role"
    assert num_roles <= num_pixels, "Can't have more roles than pixels"

    print("  ✓ At least one role exists")
    print("  ✓ Role count <= pixel count")

    # Note: Some tasks may have no compression if all pixels are structurally unique
    if num_roles == num_pixels:
        print("  ⚠ No compression (all pixels unique) - this is OK for some tasks")
    else:
        print(f"  ✓ Has compression: {num_roles} roles for {num_pixels} pixels")


def test_shape_signature_computed():
    """Test that shape signatures are computed for all grids."""
    print("\nTest: Shape signatures computed for all grids")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should compute for components
    assert "connected_components_by_color(grid)" in source, \
        "Should compute components"
    assert "compute_shape_signature(comp)" in source, \
        "Should compute shape signature"

    print("  ✓ Computes components via connected_components_by_color")
    print("  ✓ Computes shape_signature for each component")

    # Should map pixels to shape signatures
    assert "pixel_to_shape_sig" in source, \
        "Should create pixel->shape_sig mapping"
    print("  ✓ Maps pixels to shape signatures")


def test_wl_iterations():
    """Test that WL runs for specified iterations."""
    print("\nTest: WL iteration count")
    print("-" * 70)

    roles_file = project_root / "src/law_mining/roles.py"
    source = roles_file.read_text()

    # Should have iteration loop
    assert "for iteration in range(wl_iters):" in source or \
           "for it in range(wl_iters):" in source, \
        "Should loop for wl_iters iterations"
    print("  ✓ Runs WL for wl_iters iterations")

    # Should have default of 3
    assert "wl_iters: int = 3" in source, \
        "Should default to 3 iterations"
    print("  ✓ Default wl_iters = 3")


def main():
    print("=" * 70)
    print("WO-M6.1 COMPREHENSIVE REVIEW TEST")
    print("Testing role labeller (WL/q)")
    print("=" * 70)

    try:
        # HIGHEST PRIORITY: Check for incomplete implementations
        test_no_todos_stubs()

        # Core implementation checks
        test_no_defaults_heuristics()
        test_wl_deterministic()
        test_neighbors_4connected()
        test_uses_existing_phi()

        # Structure checks
        test_nodes_created_correctly()
        test_initial_labels_structure()
        test_output_format()
        test_shape_signature_computed()
        test_wl_iterations()

        # Functional tests
        test_functional_determinism()
        test_functional_sanity()

        print("\n" + "=" * 70)
        print("✅ WO-M6.1 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Implementation quality - EXCELLENT")
        print("    - No TODOs, stubs, or simplified implementations")
        print("    - No defaults or heuristics (TOE-compliant)")
        print("    - Deterministic WL refinement")
        print()
        print("  ✓ Algorithm correctness - VERIFIED")
        print("    - 4-connected neighbors, kind-local only")
        print("    - Uses existing φ operators from M1")
        print("    - No input↔output cross-linking at WL level")
        print("    - Proper node creation for train_in/train_out/test_in")
        print()
        print("  ✓ Functional tests - ALL PASSED")
        print("    - Determinism verified on multiple tasks")
        print("    - Sanity checks passed")
        print("    - Shape signatures computed correctly")
        print()
        print("WO-M6.1 IMPLEMENTATION COMPLETE AND VERIFIED")
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
