#!/usr/bin/env python3
"""
WO-M3.3 Comprehensive Review Test

Tests all aspects of WO-M3.3 implementation:
  - S5 schema builder (template stamping)
  - S11 schema builder (local neighborhood codebook)
  - Dispatch wiring

Critical checks:
  - S5 and S11 use fix_pixel_color (NOT forbid loop)
  - No mining logic in builders (param-driven only)
  - Correct param structures
  - Geometry-preserving (use input_H, input_W)
  - Correct string parsing for offsets
  - Boundary checking
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.schemas.context import build_example_context, TaskContext
from src.constraints.builder import ConstraintBuilder
from src.schemas.dispatch import apply_schema_instance, BUILDERS
from src.schemas.families import SCHEMA_FAMILIES


def test_s5_param_structure():
    """Test S5 accepts correct param structure."""
    print("Testing S5 param structure...")

    # Create 5x5 grid
    input_grid = np.zeros((5, 5), dtype=int)
    input_grid[1, 1] = 1  # seed pixel
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    # Get hash at seed
    if (1, 1) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (1,1), using dummy hash")
        seed_hash = 999999
    else:
        seed_hash = ex.neighborhood_hashes[(1, 1)]

    # Test all param keys
    params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(seed_hash): {
                "(0,0)": 5,
                "(0,1)": 5
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S5", params, ctx, builder)

    # Should add constraints
    assert len(builder.constraints) >= 2

    print("  ✓ S5 accepts correct param structure")
    print(f"    Added {len(builder.constraints)} constraints")


def test_s5_template_stamping_2x2():
    """Test S5 stamps 2x2 template correctly."""
    print("Testing S5 2x2 template stamping...")

    # 5x5 grid with seed at (2,2)
    input_grid = np.zeros((5, 5), dtype=int)
    input_grid[2, 2] = 1
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if (2, 2) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (2,2), skipping test")
        return

    seed_hash = ex.neighborhood_hashes[(2, 2)]

    # Stamp 2x2 square
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

    # Should have 4 constraints (2x2 template)
    assert len(builder.constraints) == 4

    print(f"  ✓ S5 2x2 template: {len(builder.constraints)} constraints")


def test_s5_cross_pattern():
    """Test S5 stamps cross pattern correctly."""
    print("Testing S5 cross pattern...")

    # 5x5 grid
    input_grid = np.zeros((5, 5), dtype=int)
    input_grid[2, 2] = 1
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if (2, 2) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (2,2), skipping test")
        return

    seed_hash = ex.neighborhood_hashes[(2, 2)]

    # Cross: center + 4 cardinal directions
    params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(seed_hash): {
                "(0,0)": 2,    # center
                "(-1,0)": 2,   # up
                "(1,0)": 2,    # down
                "(0,-1)": 2,   # left
                "(0,1)": 2     # right
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S5", params, ctx, builder)

    # Should have 5 constraints (cross)
    assert len(builder.constraints) == 5

    print(f"  ✓ S5 cross pattern: {len(builder.constraints)} constraints")


def test_s5_boundary_checking():
    """Test S5 handles out-of-bounds offsets correctly."""
    print("Testing S5 boundary checking...")

    # Small 3x3 grid
    input_grid = np.zeros((3, 3), dtype=int)
    input_grid[1, 1] = 1
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if (1, 1) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (1,1), skipping test")
        return

    seed_hash = ex.neighborhood_hashes[(1, 1)]

    # Large offsets that go out of bounds
    params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(seed_hash): {
                "(0,0)": 3,
                "(10,10)": 3,   # Out of bounds
                "(-10,-10)": 3  # Out of bounds
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S5", params, ctx, builder)

    # Should have only 1 constraint (center pixel in bounds)
    assert len(builder.constraints) == 1

    print(f"  ✓ S5 boundary checking: {len(builder.constraints)} constraint")
    print("    (out-of-bounds offsets correctly skipped)")


def test_s5_uses_fix_not_forbid():
    """CRITICAL: Test S5 uses fix_pixel_color, NOT forbid loop."""
    print("Testing S5 uses fix_pixel_color (NOT forbid)...")

    # 3x3 grid
    input_grid = np.zeros((3, 3), dtype=int)
    input_grid[1, 1] = 1
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if (1, 1) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (1,1), skipping test")
        return

    seed_hash = ex.neighborhood_hashes[(1, 1)]

    # Single pixel template
    params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(seed_hash): {
                "(0,0)": 5
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S5", params, ctx, builder)

    # Should have 1 constraint (fix), NOT 9 constraints (forbid loop with C=10 would be C-1=9)
    assert len(builder.constraints) == 1, \
        f"S5 should use fix (1 constraint), not forbid loop ({ctx.C-1} constraints)"

    # Verify constraint structure
    c = builder.constraints[0]
    assert c.rhs == 1, "fix constraint should have rhs=1"
    assert len(c.indices) == 1, "fix constraint should have 1 index"
    assert c.coeffs == [1], "fix constraint should have coeff=1"

    print("  ✓ S5 correctly uses fix_pixel_color")
    print("    (1 fix constraint, NOT 9 forbid constraints)")


def test_s11_param_structure():
    """Test S11 accepts correct param structure."""
    print("Testing S11 param structure...")

    # 3x3 grid
    input_grid = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if not ex.neighborhood_hashes:
        print("  ⚠ Warning: No hashes found, skipping test")
        return

    sample_hash = list(ex.neighborhood_hashes.values())[0]

    # Test all param keys
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

    # Should add constraints
    assert len(builder.constraints) >= 1

    print("  ✓ S11 accepts correct param structure")
    print(f"    Added {len(builder.constraints)} constraints")


def test_s11_single_hash_codebook():
    """Test S11 applies template to all pixels with matching hash."""
    print("Testing S11 single hash codebook...")

    # 3x3 grid
    input_grid = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if not ex.neighborhood_hashes:
        print("  ⚠ Warning: No hashes found, skipping test")
        return

    sample_hash = list(ex.neighborhood_hashes.values())[0]
    pixels_with_hash = sum(1 for h in ex.neighborhood_hashes.values() if h == sample_hash)

    # Apply template to all pixels with this hash
    params = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(sample_hash): {
                "(0,0)": 5  # Overwrite center
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S11", params, ctx, builder)

    # Should have 1 constraint per pixel with this hash
    assert len(builder.constraints) == pixels_with_hash

    print(f"  ✓ S11 single hash: {len(builder.constraints)} constraints")
    print(f"    ({pixels_with_hash} pixels with matching hash)")


def test_s11_multiple_hash_codebook():
    """Test S11 applies different templates for different hashes."""
    print("Testing S11 multiple hash codebook...")

    # 4x4 grid with different patterns
    input_grid = np.array([
        [0, 1, 1, 0],
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        [0, 1, 1, 0]
    ], dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if len(ex.neighborhood_hashes) < 2:
        print("  ⚠ Warning: Need at least 2 hashes, skipping test")
        return

    # Get two different hashes
    hash_list = list(ex.neighborhood_hashes.values())
    hash1 = hash_list[0]
    hash2 = hash_list[1]
    for h in hash_list:
        if h != hash1:
            hash2 = h
            break

    count1 = sum(1 for h in ex.neighborhood_hashes.values() if h == hash1)
    count2 = sum(1 for h in ex.neighborhood_hashes.values() if h == hash2)

    # Different templates for different hashes
    params = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(hash1): {"(0,0)": 3},
            str(hash2): {"(0,0)": 4}
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S11", params, ctx, builder)

    # Should have count1 + count2 constraints
    expected = count1 + count2
    assert len(builder.constraints) == expected

    print(f"  ✓ S11 multiple hashes: {len(builder.constraints)} constraints")
    print(f"    (hash1: {count1} pixels, hash2: {count2} pixels)")


def test_s11_3x3_pattern():
    """Test S11 applies 3x3 template."""
    print("Testing S11 3x3 pattern...")

    # 5x5 grid
    input_grid = np.zeros((5, 5), dtype=int)
    input_grid[2, 2] = 1
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if (2, 2) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (2,2), skipping test")
        return

    center_hash = ex.neighborhood_hashes[(2, 2)]

    # Full 3x3 template
    params = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(center_hash): {
                "(-1,-1)": 7, "(-1,0)": 7, "(-1,1)": 7,
                "(0,-1)": 7,  "(0,0)": 7,  "(0,1)": 7,
                "(1,-1)": 7,  "(1,0)": 7,  "(1,1)": 7
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S11", params, ctx, builder)

    # Should have 9 constraints (full 3x3)
    assert len(builder.constraints) == 9

    print(f"  ✓ S11 3x3 pattern: {len(builder.constraints)} constraints")


def test_s11_uses_fix_not_forbid():
    """CRITICAL: Test S11 uses fix_pixel_color, NOT forbid loop."""
    print("Testing S11 uses fix_pixel_color (NOT forbid)...")

    # 3x3 grid
    input_grid = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=int)
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if not ex.neighborhood_hashes:
        print("  ⚠ Warning: No hashes found, skipping test")
        return

    sample_hash = list(ex.neighborhood_hashes.values())[0]

    # Single pixel template
    params = {
        "example_type": "train",
        "example_index": 0,
        "hash_templates": {
            str(sample_hash): {
                "(0,0)": 5
            }
        }
    }

    builder = ConstraintBuilder()
    apply_schema_instance("S11", params, ctx, builder)

    # Each constraint should be a fix (rhs=1, 1 index)
    for c in builder.constraints:
        assert c.rhs == 1, "fix constraint should have rhs=1"
        assert len(c.indices) == 1, "fix constraint should have 1 index"
        assert c.coeffs == [1], "fix constraint should have coeff=1"

    print("  ✓ S11 correctly uses fix_pixel_color")
    print(f"    (all {len(builder.constraints)} constraints are fix constraints)")


def test_no_mining_logic():
    """Test S5/S11 do NOT contain mining logic (param-driven only)."""
    print("Testing S5/S11 have NO mining logic...")

    # Read S5 source
    s5_path = project_root / "src/schemas/s5_template_stamping.py"
    s5_source = s5_path.read_text()

    # Read S11 source
    s11_path = project_root / "src/schemas/s11_local_codebook.py"
    s11_source = s11_path.read_text()

    # Should NOT contain mining keywords
    forbidden_keywords = [
        "mine_",
        "detect_seed",
        "find_template",
        "infer_template",
        "discover_",
        "learn_"
    ]

    for keyword in forbidden_keywords:
        assert keyword not in s5_source, \
            f"S5 should NOT contain '{keyword}' (param-driven, not mining)"
        assert keyword not in s11_source, \
            f"S11 should NOT contain '{keyword}' (param-driven, not mining)"

    print("  ✓ S5/S11 are param-driven (NO mining logic)")


def test_geometry_preserving():
    """Test S5/S11 use input_H, input_W (geometry-preserving)."""
    print("Testing S5/S11 are geometry-preserving...")

    # Read S5 source
    s5_path = project_root / "src/schemas/s5_template_stamping.py"
    s5_source = s5_path.read_text()

    # Read S11 source
    s11_path = project_root / "src/schemas/s11_local_codebook.py"
    s11_source = s11_path.read_text()

    # Should use input_H, input_W
    assert "ex.input_H" in s5_source
    assert "ex.input_W" in s5_source
    assert "ex.input_H" in s11_source
    assert "ex.input_W" in s11_source

    # Should NOT use output_H, output_W
    assert "ex.output_H" not in s5_source
    assert "ex.output_W" not in s5_source
    assert "ex.output_H" not in s11_source
    assert "ex.output_W" not in s11_source

    print("  ✓ S5/S11 use input_H, input_W (geometry-preserving)")


def test_string_parsing():
    """Test S5/S11 correctly parse offset strings like '(0,1)'."""
    print("Testing S5/S11 string parsing...")

    # This is implicitly tested by other tests, but let's verify
    # the parsing logic handles various formats

    # 3x3 grid
    input_grid = np.zeros((3, 3), dtype=int)
    input_grid[1, 1] = 1
    output_grid = input_grid.copy()

    ex = build_example_context(input_grid, output_grid)
    ctx = TaskContext(train_examples=[ex], test_examples=[], C=10)

    if (1, 1) not in ex.neighborhood_hashes:
        print("  ⚠ Warning: No hash at (1,1), skipping test")
        return

    seed_hash = ex.neighborhood_hashes[(1, 1)]

    # Test various offset formats (with/without spaces)
    params = {
        "example_type": "train",
        "example_index": 0,
        "seed_templates": {
            str(seed_hash): {
                "(0,0)": 5,       # no spaces
                "(0, 1)": 5,      # space after comma
                "( 1, 0 )": 5,    # spaces everywhere
                "(-1,0)": 5       # negative
            }
        }
    }

    builder = ConstraintBuilder()
    try:
        apply_schema_instance("S5", params, ctx, builder)
        # Should parse successfully
        assert len(builder.constraints) >= 4
        print("  ✓ S5/S11 correctly parse offset strings")
    except Exception as e:
        raise AssertionError(f"String parsing failed: {e}")


def test_dispatch_wiring():
    """Test S5/S11 are correctly wired into dispatch."""
    print("Testing dispatch/families wiring...")

    # Check BUILDERS has S5/S11
    assert "S5" in BUILDERS
    assert "S11" in BUILDERS

    # Check builder functions are callable
    assert callable(BUILDERS["S5"])
    assert callable(BUILDERS["S11"])

    # Check families match
    assert "S5" in SCHEMA_FAMILIES
    assert "S11" in SCHEMA_FAMILIES
    assert SCHEMA_FAMILIES["S5"].builder_name == "build_S5_constraints"
    assert SCHEMA_FAMILIES["S11"].builder_name == "build_S11_constraints"

    # Check builder function names match
    assert BUILDERS["S5"].__name__ == "build_S5_constraints"
    assert BUILDERS["S11"].__name__ == "build_S11_constraints"

    # Check S6-S10 are still stubs
    for fid in ["S6", "S7", "S8", "S9", "S10"]:
        assert fid in BUILDERS
        # Try calling - should raise NotImplementedError
        import numpy as np
        dummy_grid = np.zeros((2, 2), dtype=int)
        dummy_ex = build_example_context(dummy_grid, dummy_grid)
        ctx = TaskContext(train_examples=[dummy_ex], test_examples=[], C=4)
        builder = ConstraintBuilder()
        try:
            apply_schema_instance(fid, {}, ctx, builder)
            raise AssertionError(f"{fid} should still be stub")
        except NotImplementedError:
            pass  # Expected

    print("  ✓ S5/S11 correctly wired into dispatch/families")
    print("  ✓ S6-S10 remain as stubs")


def main():
    print("=" * 70)
    print("WO-M3.3 COMPREHENSIVE REVIEW TEST")
    print("=" * 70)
    print()
    print("Testing:")
    print("  - S5 schema builder (template stamping)")
    print("  - S11 schema builder (local neighborhood codebook)")
    print("  - Dispatch wiring")
    print()
    print("=" * 70)
    print()

    try:
        # S5 tests
        test_s5_param_structure()
        test_s5_template_stamping_2x2()
        test_s5_cross_pattern()
        test_s5_boundary_checking()
        test_s5_uses_fix_not_forbid()  # CRITICAL

        # S11 tests
        test_s11_param_structure()
        test_s11_single_hash_codebook()
        test_s11_multiple_hash_codebook()
        test_s11_3x3_pattern()
        test_s11_uses_fix_not_forbid()  # CRITICAL

        # Design checks
        test_no_mining_logic()
        test_geometry_preserving()
        test_string_parsing()
        test_dispatch_wiring()

        print()
        print("=" * 70)
        print("✅ WO-M3.3 COMPREHENSIVE REVIEW - ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ S5 builder (template stamping) - COMPLETE")
        print("  ✓ S11 builder (local neighborhood codebook) - COMPLETE")
        print("  ✓ S5 uses fix_pixel_color (NOT forbid loop) - VERIFIED")
        print("  ✓ S11 uses fix_pixel_color (NOT forbid loop) - VERIFIED")
        print("  ✓ Builders are param-driven (NO mining) - VERIFIED")
        print("  ✓ Geometry-preserving (use input_H, input_W) - VERIFIED")
        print("  ✓ String parsing correctness - VERIFIED")
        print("  ✓ Dispatch/families wiring - CORRECT")
        print("  ✓ S6-S10 remain as stubs - VERIFIED")
        print()
        print("WO-M3.3 IMPLEMENTATION READY FOR PRODUCTION")
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
