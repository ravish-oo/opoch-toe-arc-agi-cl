"""
Schema family registry (S1-S11 metadata).

This module defines metadata for all 11 schema families from the math kernel spec.
It provides information for Pi-agents and tools about:
  - What each schema does
  - What parameters it needs
  - What features (φ) it requires
  - Which builder function will implement it (in M3)

This is pure metadata with no constraint building logic.
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class SchemaFamily:
    """
    Metadata for a law family (schema type), e.g. S1..S11.

    This is used by Pi-agents and tooling to know:
      - what this family does,
      - which parameters it needs,
      - which features it relies on,
      - which builder function name should be called later.

    Attributes:
        id: Schema identifier (e.g. "S1", "S2", ...)
        name: Short human-readable name
        description: Longer description of what this schema does
        parameter_spec: Schema for parameters (keys=param names, values=type descriptions)
        required_features: List of feature module names from M1 (φ)
        builder_name: Name of builder function to be implemented in M3
    """
    id: str
    name: str
    description: str
    parameter_spec: Dict[str, str]
    required_features: List[str]
    builder_name: str


# Registry of all 11 schema families
SCHEMA_FAMILIES: Dict[str, SchemaFamily] = {
    "S1": SchemaFamily(
        id="S1",
        name="Direct pixel color tie",
        description=(
            "Enforce equality of colors for pixels that are feature-equivalent "
            "(e.g. same object_id, same relative position) across grids. "
            "This is the backbone that propagates input structure everywhere "
            "the kernel recognizes equivalence."
        ),
        parameter_spec={
            "feature_predicate": "str"  # natural language or DSL describing φ-equivalence
        },
        required_features=["coords_bands", "components", "object_ids", "neighborhood_hashes"],
        builder_name="build_S1_constraints"
    ),

    "S2": SchemaFamily(
        id="S2",
        name="Component-wise recolor map",
        description=(
            "For each connected component (object class), recolor its pixels based on "
            "size, input color, or other component attributes. "
            "Learns a mapping: input_color → output_color per object class."
        ),
        parameter_spec={
            "color_in": "int",
            "size_to_color": "dict[str|int->int]"  # e.g. {1:3, 2:2, 'else':1}
        },
        required_features=["components", "object_ids", "object_roles"],
        builder_name="build_S2_constraints"
    ),

    "S3": SchemaFamily(
        id="S3",
        name="Band / stripe laws",
        description=(
            "Partition rows/columns into classes using bands and periodic residues; "
            "enforce shared color patterns within each class. "
            "Handles horizontal/vertical stripes, bands around frames, etc."
        ),
        parameter_spec={
            "row_class_rule": "str",   # description/DSL of row classes
            "col_class_rule": "str"    # description/DSL of column classes or periodicity
        },
        required_features=["coords_bands", "line_features"],
        builder_name="build_S3_constraints"
    ),

    "S4": SchemaFamily(
        id="S4",
        name="Periodicity / residue-class coloring",
        description=(
            "Assign colors based purely on coordinate residues mod K "
            "for rows/columns. Handles checkerboards, alternating stripes, "
            "and any modulo-based repeated pattern."
        ),
        parameter_spec={
            "K": "int",                          # modulus
            "residue_to_color": "dict[int->int]" # mapping residue → color
        },
        required_features=["coords_bands"],
        builder_name="build_S4_constraints"
    ),

    "S5": SchemaFamily(
        id="S5",
        name="Template stamping",
        description=(
            "Detect seed pixels (via local neighborhood patterns) and stamp "
            "a learned template patch around each seed. "
            "Canonical 'pattern → icon' primitive for tasks like digit drawing."
        ),
        parameter_spec={
            "example_type": "str",                         # "train" | "test"
            "example_index": "int",                        # which example
            "seed_templates": "dict[str, dict[str,int]]"   # hash_str -> {offset_str: color}
        },
        required_features=["neighborhood_hashes"],
        builder_name="build_S5_constraints"
    ),

    "S6": SchemaFamily(
        id="S6",
        name="Cropping to ROI / dominant object",
        description=(
            "Select a bounding box (e.g. of largest or special component) and make the "
            "output grid a crop of that region. "
            "Handles single object extraction and focus tasks."
        ),
        parameter_spec={
            "example_type": "str",                  # "train" | "test"
            "example_index": "int",                 # which example
            "output_height": "int",                 # output grid height
            "output_width": "int",                  # output grid width
            "background_color": "int",              # color for unmapped pixels
            "out_to_in": "dict[str,str]"            # output coords -> input coords mapping
        },
        required_features=["input_grid"],
        builder_name="build_S6_constraints"
    ),

    "S7": SchemaFamily(
        id="S7",
        name="Aggregation / summary grid",
        description=(
            "Partition the input into macro-cells (blocks or bands) and summarize each "
            "region (e.g., dominant color) into a smaller output grid. "
            "Compresses large regions into summary matrices."
        ),
        parameter_spec={
            "example_type": "str",                  # "train" | "test"
            "example_index": "int",                 # which example
            "output_height": "int",                 # output grid height
            "output_width": "int",                  # output grid width
            "summary_colors": "dict[str,int]"       # output coords -> summary color
        },
        required_features=[],
        builder_name="build_S7_constraints"
    ),

    "S8": SchemaFamily(
        id="S8",
        name="Tiling / replication",
        description=(
            "Identify a base tile (pattern) and replicate it periodically to fill a region. "
            "Handles motif replication with zero padding or other backgrounds."
        ),
        parameter_spec={
            "example_type": "str",                  # "train" | "test"
            "example_index": "int",                 # which example
            "tile_height": "int",                   # tile height
            "tile_width": "int",                    # tile width
            "tile_pattern": "dict[str,int]",        # offset -> color mapping
            "region_origin": "str",                 # "(r0,c0)" top-left of tiling region
            "region_height": "int",                 # tiling region height
            "region_width": "int"                   # tiling region width
        },
        required_features=[],
        builder_name="build_S8_constraints"
    ),

    "S9": SchemaFamily(
        id="S9",
        name="Cross / plus propagation",
        description=(
            "Detect cross-shaped seeds and propagate spokes along rows/cols with learned "
            "colors and stopping conditions. "
            "Extends plus shapes and crosses along cardinal directions."
        ),
        parameter_spec={
            "example_type": "str",                  # "train" | "test"
            "example_index": "int",                 # which example
            "seeds": "list[dict]"                   # list of seed configs with center, colors, max_steps
        },
        required_features=[],
        builder_name="build_S9_constraints"
    ),

    "S10": SchemaFamily(
        id="S10",
        name="Frame / border vs interior",
        description=(
            "Assign different colors to border vs interior pixels of components or the whole grid. "
            "Draws frames, borders around shapes, or fills interiors differently."
        ),
        parameter_spec={
            "example_type": "str",                  # "train" | "test"
            "example_index": "int",                 # which example
            "border_color": "int",                  # color for border pixels
            "interior_color": "int"                 # color for interior pixels
        },
        required_features=["border_info"],
        builder_name="build_S10_constraints"
    ),

    "S11": SchemaFamily(
        id="S11",
        name="Local neighborhood codebook",
        description=(
            "For each 3×3 neighborhood pattern type, assign a corresponding output pattern "
            "via a learned codebook H → P. "
            "Safety net for any local weirdness that depends on subtle local shape types."
        ),
        parameter_spec={
            "example_type": "str",                         # "train" | "test"
            "example_index": "int",                        # which example
            "hash_templates": "dict[str, dict[str,int]]"   # hash_str -> {offset_str: color}
        },
        required_features=["neighborhood_hashes"],
        builder_name="build_S11_constraints"
    )
}


if __name__ == "__main__":
    # Simple sanity check: we have exactly 11 families S1..S11
    from pprint import pprint

    print("Defined schema families:")
    print("=" * 60)
    for fid, fam in sorted(SCHEMA_FAMILIES.items()):
        print(f"  {fid}: {fam.name}")

    print("\n" + "=" * 60)
    assert len(SCHEMA_FAMILIES) == 11, f"Expected 11 schema families, got {len(SCHEMA_FAMILIES)}"
    assert all(fid.startswith("S") for fid in SCHEMA_FAMILIES.keys()), \
        "All schema IDs should start with 'S'"

    # Spot-check one family's parameter_spec and required_features
    print("\nSpot-checking S2 (Component-wise recolor map):")
    print("-" * 60)
    s2 = SCHEMA_FAMILIES["S2"]
    pprint(s2)

    assert "color_in" in s2.parameter_spec, "S2 should have 'color_in' parameter"
    assert "components" in s2.required_features, "S2 should require 'components' feature"

    print("\n" + "=" * 60)
    print("✓ families.py sanity checks passed.")
