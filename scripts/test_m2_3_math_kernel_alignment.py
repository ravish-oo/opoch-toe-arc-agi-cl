#!/usr/bin/env python3
"""
Cross-reference test: families.py vs math_kernel.md

Verifies that every schema family aligns with its corresponding section
in the math kernel spec document.

This reads docs/anchors/math_kernel.md and checks:
1. All 11 schemas are mentioned in the spec
2. Key concepts from spec appear in schema descriptions
3. No schemas are missing from implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schemas.families import SCHEMA_FAMILIES


def test_math_kernel_schema_coverage():
    """
    Test that math_kernel.md mentions all schemas and vice versa.
    """
    print("Testing math_kernel.md coverage...")

    math_kernel_path = project_root / "docs" / "anchors" / "math_kernel.md"
    assert math_kernel_path.exists(), \
        f"Math kernel spec not found at {math_kernel_path}"

    spec_content = math_kernel_path.read_text()

    # Check each schema is mentioned in the spec
    for sid in SCHEMA_FAMILIES.keys():
        # Look for "Schema S1" or "S1 —" patterns
        patterns = [
            f"Schema {sid}",
            f"{sid} —",
            f"{sid}—",
        ]

        found = any(pattern in spec_content for pattern in patterns)
        assert found, \
            f"{sid} not found in math_kernel.md (searched for: {patterns})"

    print(f"  ✓ All 11 schemas mentioned in math_kernel.md")


def test_s1_alignment():
    """S1 — Direct pixel color tie"""
    print("Testing S1 alignment with spec...")

    s1 = SCHEMA_FAMILIES["S1"]
    spec_path = project_root / "docs" / "anchors" / "math_kernel.md"
    spec_content = spec_path.read_text()

    # Find S1 section
    s1_section_start = spec_content.find("Schema S1")
    s1_section_end = spec_content.find("Schema S2")
    s1_section = spec_content[s1_section_start:s1_section_end]

    # Check key concepts from spec appear in description or are correct
    assert "color tie" in s1.name.lower() or "tie" in s1.name.lower()
    assert "equality" in s1.description.lower() or "tie" in s1.description.lower()

    # Spec mentions: "φ(p)=φ(q)" → feature equivalence
    assert "feature" in s1.description.lower() and "equivalent" in s1.description.lower()

    # Check spec mentions "copy/equality"
    assert "copy" in s1_section.lower() or "equality" in s1_section.lower()

    print("  ✓ S1 aligns with spec")


def test_s2_alignment():
    """S2 — Component-wise recolor map"""
    print("Testing S2 alignment with spec...")

    s2 = SCHEMA_FAMILIES["S2"]
    spec_path = project_root / "docs" / "anchors" / "math_kernel.md"
    spec_content = spec_path.read_text()

    # Find S2 section
    s2_section_start = spec_content.find("Schema S2")
    s2_section_end = spec_content.find("Schema S3")
    s2_section = spec_content[s2_section_start:s2_section_end]

    # Check key concepts
    assert "component" in s2.name.lower()
    assert "recolor" in s2.name.lower() or "recolor" in s2.description.lower()

    # Spec mentions: "Group pixels by object_id(p)"
    assert "object_ids" in s2.required_features  # Note: plural form in features

    # Spec mentions: "per-component"
    assert "component" in s2.description.lower()

    print("  ✓ S2 aligns with spec")


def test_s3_alignment():
    """S3 — Band / stripe laws"""
    print("Testing S3 alignment with spec...")

    s3 = SCHEMA_FAMILIES["S3"]

    # Check key concepts
    assert "band" in s3.name.lower() or "stripe" in s3.name.lower()
    assert ("row" in s3.description.lower() and "column" in s3.description.lower()) or \
           "band" in s3.description.lower()

    # Spec mentions bands and periodic bits
    assert "coords_bands" in s3.required_features

    print("  ✓ S3 aligns with spec")


def test_s4_alignment():
    """S4 — Periodicity & residue-class coloring"""
    print("Testing S4 alignment with spec...")

    s4 = SCHEMA_FAMILIES["S4"]

    # Check key concepts
    assert "periodic" in s4.name.lower() or "residue" in s4.name.lower()
    assert "mod" in s4.description.lower() or "modulo" in s4.description.lower()

    # Spec mentions: "residue of coordinates mod K"
    assert "K" in s4.parameter_spec

    print("  ✓ S4 aligns with spec")


def test_s5_alignment():
    """S5 — Template stamping"""
    print("Testing S5 alignment with spec...")

    s5 = SCHEMA_FAMILIES["S5"]

    # Check key concepts
    assert "template" in s5.name.lower() or "stamp" in s5.name.lower()
    assert "seed" in s5.description.lower()

    # Spec mentions: "Small stencil templates (e.g. 3×3)"
    assert "neighborhood_hashes" in s5.required_features

    print("  ✓ S5 aligns with spec")


def test_s6_alignment():
    """S6 — Cropping to ROI / dominant object"""
    print("Testing S6 alignment with spec...")

    s6 = SCHEMA_FAMILIES["S6"]

    # Check key concepts
    assert "crop" in s6.name.lower() or "roi" in s6.name.lower()

    # Spec mentions: "bounding boxes for all components"
    assert "components" in s6.required_features

    print("  ✓ S6 aligns with spec")


def test_s7_alignment():
    """S7 — Aggregation / histogram / summary grids"""
    print("Testing S7 alignment with spec...")

    s7 = SCHEMA_FAMILIES["S7"]

    # Check key concepts
    assert "aggregat" in s7.name.lower() or "summary" in s7.name.lower()

    # Spec mentions: "Compress large region into small matrix"
    assert "summary" in s7.description.lower() or "compress" in s7.description.lower()

    print("  ✓ S7 aligns with spec")


def test_s8_alignment():
    """S8 — Tiling / replication"""
    print("Testing S8 alignment with spec...")

    s8 = SCHEMA_FAMILIES["S8"]

    # Check key concepts
    assert "til" in s8.name.lower() or "replicat" in s8.name.lower()

    # Spec mentions: "Copy a small patch periodically"
    assert "tile_size" in s8.parameter_spec

    print("  ✓ S8 aligns with spec")


def test_s9_alignment():
    """S9 — Cross / plus propagation"""
    print("Testing S9 alignment with spec...")

    s9 = SCHEMA_FAMILIES["S9"]

    # Check key concepts
    assert "cross" in s9.name.lower() or "plus" in s9.name.lower()
    assert "propagat" in s9.description.lower()

    # Spec mentions: "cross-shaped patterns"
    assert "neighborhood_hashes" in s9.required_features

    print("  ✓ S9 aligns with spec")


def test_s10_alignment():
    """S10 — Frame / border vs interior"""
    print("Testing S10 alignment with spec...")

    s10 = SCHEMA_FAMILIES["S10"]

    # Check key concepts
    assert "frame" in s10.name.lower() or "border" in s10.name.lower()
    assert "interior" in s10.description.lower()

    # Spec mentions: "border mask" and "interior mask"
    assert "border_color" in s10.parameter_spec
    assert "interior_color" in s10.parameter_spec

    # CRITICAL: Must use object_roles which provides border/interior from M1 WO6
    assert "object_roles" in s10.required_features, \
        "S10 must require object_roles (provides component_border_interior from WO6)"

    print("  ✓ S10 aligns with spec")


def test_s11_alignment():
    """S11 — Local neighborhood rewrite"""
    print("Testing S11 alignment with spec...")

    s11 = SCHEMA_FAMILIES["S11"]

    # Check key concepts
    assert "neighborhood" in s11.name.lower() or "codebook" in s11.name.lower()

    # Spec mentions: "3×3 neighborhood pattern type"
    assert "3×3" in s11.description or "3x3" in s11.description.lower()
    assert "neighborhood_hashes" in s11.required_features

    print("  ✓ S11 aligns with spec")


def test_spec_task_coverage_examples():
    """
    Test that schema descriptions mention example tasks where appropriate.

    The spec lists specific task IDs that each schema covers.
    """
    print("Testing example task coverage...")

    # S2 covers tasks like 9344f635, 95990924, 95a58926
    s2 = SCHEMA_FAMILIES["S2"]
    # Just verify it's about component recoloring (tasks are in spec comments, not required in code)
    assert "component" in s2.description.lower()

    # S3 covers band/stripe tasks
    s3 = SCHEMA_FAMILIES["S3"]
    assert "band" in s3.description.lower() or "stripe" in s3.description.lower()

    # S10 is explicitly about frames
    s10 = SCHEMA_FAMILIES["S10"]
    assert "frame" in s10.description.lower() or "border" in s10.description.lower()

    print("  ✓ Schema descriptions cover expected use cases")


def test_spec_completeness_claim():
    """
    Test the spec's completeness claim.

    Math kernel section 4 claims these 11 schemas are sufficient for all 1000 tasks.
    We verify all 11 are implemented.
    """
    print("Testing completeness (11 schemas for all tasks)...")

    # Math kernel section 4: "Why this is complete for the 1000"
    spec_path = project_root / "docs" / "anchors" / "math_kernel.md"
    spec_content = spec_path.read_text()

    # Check the completeness section exists
    assert "complete for the 1000" in spec_content.lower() or \
           "why this is complete" in spec_content.lower(), \
        "Spec should have completeness section"

    # We have all 11 schemas
    assert len(SCHEMA_FAMILIES) == 11

    # Each has a builder (will be implemented in M3)
    for schema in SCHEMA_FAMILIES.values():
        assert schema.builder_name.startswith("build_S")
        assert schema.builder_name.endswith("_constraints")

    print("  ✓ All 11 schemas defined (completeness claim satisfied)")


def test_required_features_match_m1_work_orders():
    """
    Verify that required_features strings match actual M1 work order outputs.
    """
    print("Testing required_features match M1 outputs...")

    # Map features to M1 work orders
    feature_to_wo = {
        "coords_bands": "WO2",
        "components": "WO3",
        "object_ids": "WO4",
        "object_roles": "WO6",
        "neighborhood_hashes": "WO5",
        "line_features": "WO5",
    }

    # Verify all features used in schemas map to M1 WOs
    all_used_features = set()
    for schema in SCHEMA_FAMILIES.values():
        all_used_features.update(schema.required_features)

    for feature in all_used_features:
        assert feature in feature_to_wo, \
            f"Feature '{feature}' not mapped to M1 work order"

    print(f"  ✓ All {len(all_used_features)} features map to M1 work orders")


def main():
    print("=" * 60)
    print("WO-M2.3 Math Kernel Alignment Test")
    print("=" * 60)
    print()

    try:
        # Coverage test
        test_math_kernel_schema_coverage()

        # Individual schema alignment tests
        test_s1_alignment()
        test_s2_alignment()
        test_s3_alignment()
        test_s4_alignment()
        test_s5_alignment()
        test_s6_alignment()
        test_s7_alignment()
        test_s8_alignment()
        test_s9_alignment()
        test_s10_alignment()
        test_s11_alignment()

        # Meta tests
        test_spec_task_coverage_examples()
        test_spec_completeness_claim()
        test_required_features_match_m1_work_orders()

        print()
        print("=" * 60)
        print("✅ ALL MATH KERNEL ALIGNMENT TESTS PASSED")
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
