"""
Integration test for TaskContext construction.

This script validates that all fields in ExampleContext are correctly populated
and align with the math kernel spec requirements.
"""

from pathlib import Path

from src.schemas.context import load_arc_task, build_task_context_from_raw


def test_task_context_structure():
    """Test that TaskContext has correct structure and all features populated."""
    print("Integration test: TaskContext structure validation")
    print("=" * 70)

    # Load a task
    challenges_path = Path("data/arc-agi_training_challenges.json")
    task_id = "007bbfb7"  # Task with 5 train examples
    print(f"Loading task: {task_id}\n")

    task_data = load_arc_task(task_id, challenges_path)
    ctx = build_task_context_from_raw(task_data)

    # Validate task-level structure
    print("Validating task-level structure...")
    assert len(ctx.train_examples) == 5, f"Expected 5 train examples, got {len(ctx.train_examples)}"
    assert len(ctx.test_examples) == 1, f"Expected 1 test example, got {len(ctx.test_examples)}"
    assert ctx.C > 0, f"Palette size C must be > 0, got {ctx.C}"
    print(f"  ✓ Task has {len(ctx.train_examples)} train, {len(ctx.test_examples)} test, C={ctx.C}")

    # Validate train example
    print("\nValidating train example structure...")
    ex = ctx.train_examples[0]

    assert ex.input_grid is not None, "input_grid should not be None"
    assert ex.output_grid is not None, "output_grid should not be None for train"
    assert ex.input_H == ex.input_grid.shape[0], "input_H mismatch"
    assert ex.input_W == ex.input_grid.shape[1], "input_W mismatch"
    assert ex.output_H == ex.output_grid.shape[0], "output_H mismatch"
    assert ex.output_W == ex.output_grid.shape[1], "output_W mismatch"
    print(f"  ✓ Train example has input {ex.input_H}x{ex.input_W}, output {ex.output_H}x{ex.output_W}")

    # Validate components
    assert len(ex.components) > 0, "Should have at least one component"
    for comp in ex.components:
        assert comp.id >= 0, f"Component id should be >= 0, got {comp.id}"
        assert comp.color >= 0, f"Component color should be >= 0, got {comp.color}"
        assert comp.size == len(comp.pixels), f"Component size mismatch"
        assert len(comp.bbox) == 4, f"bbox should have 4 elements"
    print(f"  ✓ Found {len(ex.components)} components with valid structure")

    # Validate object_ids covers all pixels
    N = ex.input_H * ex.input_W
    assert len(ex.object_ids) == N, f"object_ids should cover all {N} pixels, got {len(ex.object_ids)}"
    print(f"  ✓ object_ids covers all {N} pixels")

    # Validate role_bits
    for comp in ex.components:
        assert comp.id in ex.role_bits, f"Component {comp.id} missing from role_bits"
        role = ex.role_bits[comp.id]
        assert "is_small" in role, "role_bits should have 'is_small'"
        assert "is_big" in role, "role_bits should have 'is_big'"
        assert "is_unique_shape" in role, "role_bits should have 'is_unique_shape'"
    print(f"  ✓ role_bits valid for all {len(ex.components)} components")

    # Validate sectors
    assert len(ex.sectors) == N, f"sectors should cover all {N} pixels"
    for pixel, sector in ex.sectors.items():
        assert "vert_sector" in sector, "sector should have 'vert_sector'"
        assert "horiz_sector" in sector, "sector should have 'horiz_sector'"
        assert sector["vert_sector"] in ["top", "center", "bottom"]
        assert sector["horiz_sector"] in ["left", "center", "right"]
    print(f"  ✓ sectors valid for all pixels")

    # Validate border_info
    assert len(ex.border_info) == N, f"border_info should cover all {N} pixels"
    for pixel, info in ex.border_info.items():
        assert "is_border" in info, "border_info should have 'is_border'"
        assert "is_interior" in info, "border_info should have 'is_interior'"
        assert isinstance(info["is_border"], bool)
        assert isinstance(info["is_interior"], bool)
    print(f"  ✓ border_info valid for all pixels")

    # Validate bands
    assert len(ex.row_bands) == ex.input_H, f"row_bands should have {ex.input_H} entries"
    assert len(ex.col_bands) == ex.input_W, f"col_bands should have {ex.input_W} entries"
    for r in range(ex.input_H):
        assert r in ex.row_bands, f"Row {r} missing from row_bands"
        assert ex.row_bands[r] in ["top", "middle", "bottom"]
    for c in range(ex.input_W):
        assert c in ex.col_bands, f"Col {c} missing from col_bands"
        assert ex.col_bands[c] in ["left", "middle", "right"]
    print(f"  ✓ Bands valid: {len(ex.row_bands)} rows, {len(ex.col_bands)} cols")

    # Validate row/col nonzero flags
    assert len(ex.row_nonzero) == ex.input_H, f"row_nonzero should have {ex.input_H} entries"
    assert len(ex.col_nonzero) == ex.input_W, f"col_nonzero should have {ex.input_W} entries"
    print(f"  ✓ Nonzero flags valid")

    # Validate neighborhood hashes
    assert len(ex.neighborhood_hashes) == N, f"neighborhood_hashes should cover all {N} pixels"
    print(f"  ✓ Neighborhood hashes valid for all pixels")

    # Validate coords
    assert len(ex.coords) == N, f"coords should cover all {N} pixels"
    for r in range(ex.input_H):
        for c in range(ex.input_W):
            assert (r, c) in ex.coords, f"Pixel ({r},{c}) missing from coords"
            assert ex.coords[(r, c)] == (r, c), f"coords mismatch at ({r},{c})"
    print(f"  ✓ coords valid for all pixels")

    # Validate residues
    assert len(ex.row_residues) == ex.input_H, f"row_residues should have {ex.input_H} entries"
    assert len(ex.col_residues) == ex.input_W, f"col_residues should have {ex.input_W} entries"
    for r in range(ex.input_H):
        assert r in ex.row_residues, f"Row {r} missing from row_residues"
        for k in [2, 3, 4, 5]:
            assert k in ex.row_residues[r], f"Modulus {k} missing from row_residues[{r}]"
            expected = r % k
            actual = ex.row_residues[r][k]
            assert actual == expected, f"row_residues[{r}][{k}] = {actual}, expected {expected}"
    for c in range(ex.input_W):
        assert c in ex.col_residues, f"Col {c} missing from col_residues"
        for k in [2, 3, 4, 5]:
            assert k in ex.col_residues[c], f"Modulus {k} missing from col_residues[{c}]"
            expected = c % k
            actual = ex.col_residues[c][k]
            assert actual == expected, f"col_residues[{c}][{k}] = {actual}, expected {expected}"
    print(f"  ✓ Residues valid for all rows/cols (mod 2,3,4,5)")

    # Validate test example
    print("\nValidating test example structure...")
    ex_test = ctx.test_examples[0]

    assert ex_test.input_grid is not None, "test input_grid should not be None"
    assert ex_test.output_grid is None, "test output_grid should be None"
    assert ex_test.output_H is None, "test output_H should be None"
    assert ex_test.output_W is None, "test output_W should be None"
    print(f"  ✓ Test example has input {ex_test.input_H}x{ex_test.input_W}, output=None")

    # Test example should also have all features on input
    N_test = ex_test.input_H * ex_test.input_W
    assert len(ex_test.object_ids) == N_test, "test object_ids should cover all pixels"
    assert len(ex_test.coords) == N_test, "test coords should cover all pixels"
    print(f"  ✓ Test example has all φ features computed on input")

    print("\n" + "=" * 70)
    print("✓ All TaskContext integration tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_task_context_structure()
