#!/usr/bin/env python3
"""
Comprehensive test for WO-M2.3 families.py

Tests:
1. Schema count and completeness
2. SchemaFamily structure and fields
3. Alignment with math kernel spec
4. Required features validity
5. Builder name conventions
6. Parameter spec structure
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schemas.families import SchemaFamily, SCHEMA_FAMILIES


def test_schema_count():
    """Test that exactly 11 schema families are defined"""
    print("Testing schema count...")

    assert len(SCHEMA_FAMILIES) == 11, \
        f"Expected exactly 11 schemas, got {len(SCHEMA_FAMILIES)}"

    print(f"  ✓ Exactly 11 schema families defined")


def test_schema_ids():
    """Test that schema IDs are S1 through S11"""
    print("Testing schema IDs...")

    expected_ids = {f"S{i}" for i in range(1, 12)}
    actual_ids = set(SCHEMA_FAMILIES.keys())

    assert actual_ids == expected_ids, \
        f"Expected IDs {expected_ids}, got {actual_ids}"

    # Check all start with "S"
    assert all(sid.startswith("S") for sid in SCHEMA_FAMILIES.keys()), \
        "All schema IDs should start with 'S'"

    print(f"  ✓ Schema IDs are S1 through S11")


def test_schema_family_structure():
    """Test that all SchemaFamily instances have required fields"""
    print("Testing SchemaFamily structure...")

    required_fields = [
        "id", "name", "description",
        "parameter_spec", "required_features", "builder_name"
    ]

    for sid, schema in SCHEMA_FAMILIES.items():
        # Check instance type
        assert isinstance(schema, SchemaFamily), \
            f"{sid}: should be SchemaFamily instance"

        # Check all fields present
        for field in required_fields:
            assert hasattr(schema, field), \
                f"{sid}: missing field '{field}'"
            value = getattr(schema, field)
            assert value is not None, \
                f"{sid}: field '{field}' is None"

        # Check field types
        assert isinstance(schema.id, str), f"{sid}: id should be str"
        assert isinstance(schema.name, str), f"{sid}: name should be str"
        assert isinstance(schema.description, str), f"{sid}: description should be str"
        assert isinstance(schema.parameter_spec, dict), f"{sid}: parameter_spec should be dict"
        assert isinstance(schema.required_features, list), f"{sid}: required_features should be list"
        assert isinstance(schema.builder_name, str), f"{sid}: builder_name should be str"

    print(f"  ✓ All {len(SCHEMA_FAMILIES)} schemas have correct structure")


def test_no_empty_fields():
    """Test that no fields are empty or placeholder"""
    print("Testing for empty/placeholder fields...")

    placeholder_terms = ["todo", "tbd", "fixme", "placeholder", "coming soon", "not implemented"]

    for sid, schema in SCHEMA_FAMILIES.items():
        # Check non-empty strings
        assert len(schema.id) > 0, f"{sid}: id is empty"
        assert len(schema.name) > 0, f"{sid}: name is empty"
        assert len(schema.description) > 0, f"{sid}: description is empty"
        assert len(schema.builder_name) > 0, f"{sid}: builder_name is empty"

        # Check for placeholder text (case-insensitive)
        desc_lower = schema.description.lower()
        for term in placeholder_terms:
            assert term not in desc_lower, \
                f"{sid}: description contains placeholder '{term}'"

        # Check parameter_spec not empty
        assert len(schema.parameter_spec) > 0, \
            f"{sid}: parameter_spec is empty"

        # Check required_features not empty
        assert len(schema.required_features) > 0, \
            f"{sid}: required_features is empty"

    print(f"  ✓ No empty fields or placeholders found")


def test_builder_name_convention():
    """Test that builder names follow convention: build_S{k}_constraints"""
    print("Testing builder name convention...")

    for sid, schema in SCHEMA_FAMILIES.items():
        expected_builder = f"build_{sid}_constraints"
        assert schema.builder_name == expected_builder, \
            f"{sid}: expected builder_name '{expected_builder}', got '{schema.builder_name}'"

    print(f"  ✓ All builder names follow 'build_S{{k}}_constraints' convention")


def test_required_features_validity():
    """Test that required_features reference valid M1 modules"""
    print("Testing required_features validity...")

    # Valid feature names from M1
    valid_features = {
        "coords_bands",      # WO2: src/features/coords_bands.py
        "components",        # WO3: src/features/components.py
        "object_ids",        # WO4: src/features/components.py (assign_object_ids)
        "object_roles",      # WO6: src/features/object_roles.py
        "neighborhood_hashes",  # WO5: src/features/neighborhoods.py
        "line_features",     # WO5: src/features/neighborhoods.py (row/col flags)
    }

    for sid, schema in SCHEMA_FAMILIES.items():
        for feature in schema.required_features:
            assert feature in valid_features, \
                f"{sid}: unknown feature '{feature}' (valid: {valid_features})"

    print(f"  ✓ All required_features reference valid M1 modules")


def test_parameter_spec_structure():
    """Test that parameter_spec has reasonable structure"""
    print("Testing parameter_spec structure...")

    for sid, schema in SCHEMA_FAMILIES.items():
        # Check it's a dict with string keys
        assert isinstance(schema.parameter_spec, dict)

        for param_name, param_type in schema.parameter_spec.items():
            # Keys should be non-empty strings
            assert isinstance(param_name, str)
            assert len(param_name) > 0, f"{sid}: empty parameter name"

            # Values should be non-empty strings (type descriptions)
            assert isinstance(param_type, str)
            assert len(param_type) > 0, f"{sid}: empty type description for '{param_name}'"

    print(f"  ✓ All parameter_spec dicts have valid structure")


def test_alignment_with_math_spec():
    """
    Test alignment with math kernel spec section 2.

    Verifies key descriptions and features for each schema.
    """
    print("Testing alignment with math kernel spec...")

    # S1: Direct pixel color tie
    s1 = SCHEMA_FAMILIES["S1"]
    assert "feature-equivalent" in s1.description.lower() or "equivalence" in s1.description.lower()
    assert "coords_bands" in s1.required_features  # For φ
    assert "components" in s1.required_features

    # S2: Component-wise recolor
    s2 = SCHEMA_FAMILIES["S2"]
    assert "component" in s2.description.lower()
    assert "recolor" in s2.description.lower()
    assert "components" in s2.required_features
    assert "object_ids" in s2.required_features
    assert "color_in" in s2.parameter_spec

    # S3: Band/stripe laws
    s3 = SCHEMA_FAMILIES["S3"]
    assert "band" in s3.description.lower() or "stripe" in s3.description.lower()
    assert "row" in s3.description.lower() or "column" in s3.description.lower()
    assert "coords_bands" in s3.required_features
    assert "line_features" in s3.required_features

    # S4: Periodicity
    s4 = SCHEMA_FAMILIES["S4"]
    assert "residue" in s4.description.lower() or "periodic" in s4.description.lower()
    assert "mod" in s4.description.lower() or "modulo" in s4.description.lower()
    assert "K" in s4.parameter_spec
    assert "coords_bands" in s4.required_features

    # S5: Template stamping
    s5 = SCHEMA_FAMILIES["S5"]
    assert "seed" in s5.description.lower() or "template" in s5.description.lower()
    assert "stamp" in s5.description.lower() or "patch" in s5.description.lower()
    assert "neighborhood_hashes" in s5.required_features

    # S6: Cropping
    s6 = SCHEMA_FAMILIES["S6"]
    assert "crop" in s6.description.lower() or "bbox" in s6.description.lower() or "bounding box" in s6.description.lower()
    assert "components" in s6.required_features

    # S7: Aggregation/summary
    s7 = SCHEMA_FAMILIES["S7"]
    assert "summary" in s7.description.lower() or "aggregat" in s7.description.lower()
    assert "region" in s7.description.lower() or "block" in s7.description.lower()
    assert "coords_bands" in s7.required_features

    # S8: Tiling
    s8 = SCHEMA_FAMILIES["S8"]
    assert "tile" in s8.description.lower() or "replicat" in s8.description.lower()
    assert "tile_size" in s8.parameter_spec

    # S9: Cross propagation
    s9 = SCHEMA_FAMILIES["S9"]
    assert "cross" in s9.description.lower() or "plus" in s9.description.lower()
    assert "spoke" in s9.description.lower() or "propagat" in s9.description.lower()
    assert "neighborhood_hashes" in s9.required_features

    # S10: Frame/border (CRITICAL - requires object_roles)
    s10 = SCHEMA_FAMILIES["S10"]
    assert "border" in s10.description.lower() or "frame" in s10.description.lower()
    assert "interior" in s10.description.lower()
    assert "object_roles" in s10.required_features, \
        "S10 MUST require object_roles (provides border/interior from M1 WO6)"
    assert "border_color" in s10.parameter_spec
    assert "interior_color" in s10.parameter_spec

    # S11: Local neighborhood codebook
    s11 = SCHEMA_FAMILIES["S11"]
    assert "neighborhood" in s11.description.lower() or "codebook" in s11.description.lower()
    assert "3×3" in s11.description or "3x3" in s11.description.lower()
    assert "neighborhood_hashes" in s11.required_features

    print("  ✓ All schemas align with math kernel spec")


def test_description_quality():
    """Test that descriptions are informative and unique"""
    print("Testing description quality...")

    descriptions = {}

    for sid, schema in SCHEMA_FAMILIES.items():
        desc = schema.description

        # Check minimum length (should be substantial)
        assert len(desc) >= 50, \
            f"{sid}: description too short ({len(desc)} chars)"

        # Check uniqueness
        assert desc not in descriptions.values(), \
            f"{sid}: duplicate description"
        descriptions[sid] = desc

        # Check it mentions something specific (not too generic)
        # Generic terms that ALL descriptions might have
        generic_only = (
            desc.lower().count("pixels") +
            desc.lower().count("colors") +
            desc.lower().count("grid")
        )

        # Should have more than just generic terms
        assert len(desc.split()) > 10, \
            f"{sid}: description lacks specificity"

    print(f"  ✓ All descriptions are informative and unique")


def test_id_name_consistency():
    """Test that id field matches the key and has a name"""
    print("Testing id/name consistency...")

    for key, schema in SCHEMA_FAMILIES.items():
        # Check id matches key
        assert schema.id == key, \
            f"Key '{key}' doesn't match id '{schema.id}'"

        # Check name is different from id (should be human-readable)
        assert schema.name != schema.id, \
            f"{key}: name should be human-readable, not just the ID"

        # Check name is capitalized properly
        assert len(schema.name) > 3, \
            f"{key}: name too short"

    print(f"  ✓ All id fields match keys, names are human-readable")


def test_no_duplicate_names():
    """Test that schema names are unique"""
    print("Testing for duplicate names...")

    names = [schema.name for schema in SCHEMA_FAMILIES.values()]
    assert len(names) == len(set(names)), \
        f"Duplicate schema names found"

    print(f"  ✓ All schema names are unique")


def test_comprehensive_feature_coverage():
    """Test that all M1 features are used by at least one schema"""
    print("Testing feature coverage...")

    # All features from M1
    all_m1_features = {
        "coords_bands",
        "components",
        "object_ids",
        "object_roles",
        "neighborhood_hashes",
        "line_features",
    }

    # Collect all used features
    used_features = set()
    for schema in SCHEMA_FAMILIES.values():
        used_features.update(schema.required_features)

    # Check all M1 features are used
    assert all_m1_features == used_features, \
        f"Unused features: {all_m1_features - used_features}"

    print(f"  ✓ All {len(all_m1_features)} M1 features used by schemas")


def test_specific_schema_requirements():
    """Test specific requirements for critical schemas"""
    print("Testing specific schema requirements...")

    # S1 should need multiple feature types (it's the most general)
    s1 = SCHEMA_FAMILIES["S1"]
    assert len(s1.required_features) >= 3, \
        "S1 should require multiple feature types (it's the backbone)"

    # S2 must have object_ids and object_roles
    s2 = SCHEMA_FAMILIES["S2"]
    assert "object_ids" in s2.required_features, "S2 needs object_ids"
    assert "object_roles" in s2.required_features, "S2 needs object_roles (for size)"

    # S10 must have components AND object_roles (for border/interior)
    s10 = SCHEMA_FAMILIES["S10"]
    assert "components" in s10.required_features, "S10 needs components"
    assert "object_roles" in s10.required_features, \
        "S10 needs object_roles (provides border/interior from WO6)"

    # S11 should only need neighborhood_hashes (most local)
    s11 = SCHEMA_FAMILIES["S11"]
    assert "neighborhood_hashes" in s11.required_features, \
        "S11 needs neighborhood_hashes for codebook"

    print("  ✓ Critical schemas have correct requirements")


def test_no_builder_imports():
    """Test that this is metadata only - no builder functions imported"""
    print("Testing metadata-only constraint...")

    # Check the source file doesn't import builder functions
    families_file = Path(project_root) / "src" / "schemas" / "families.py"
    content = families_file.read_text()

    # Should NOT import from src.schemas.s1_copy_tie, etc.
    forbidden_imports = [
        "from src.schemas.s",
        "import build_S",
        "from .s",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in content, \
            f"File should not import builder functions (found '{forbidden}')"

    # Should only import dataclasses and typing
    lines = content.split('\n')
    import_lines = [l for l in lines if l.startswith('import ') or l.startswith('from ')]

    allowed_imports = [
        "from dataclasses import",
        "from typing import",
        "from pprint import",  # Only in __main__ block
    ]

    for import_line in import_lines:
        assert any(allowed in import_line for allowed in allowed_imports), \
            f"Unexpected import: {import_line}"

    print("  ✓ Metadata-only (no builder imports)")


def main():
    print("=" * 60)
    print("WO-M2.3 Comprehensive Test - families.py")
    print("=" * 60)
    print()

    try:
        # Basic structure tests
        test_schema_count()
        test_schema_ids()
        test_schema_family_structure()
        test_no_empty_fields()
        test_builder_name_convention()

        # Feature and parameter tests
        test_required_features_validity()
        test_parameter_spec_structure()

        # Alignment tests
        test_alignment_with_math_spec()
        test_description_quality()
        test_id_name_consistency()
        test_no_duplicate_names()

        # Coverage tests
        test_comprehensive_feature_coverage()
        test_specific_schema_requirements()

        # Metadata-only verification
        test_no_builder_imports()

        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
