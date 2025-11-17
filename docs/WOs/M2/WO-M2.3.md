## WO-M2.3 – SchemaFamily registry (metadata only)

**File:** `src/schemas/families.py`
**Goal:** define the **law family metadata** for S1–S11 so Pi-agents/tools know:

* what families exist,
* what each one roughly does,
* what parameters it expects,
* what features it needs.

No constraint logic here, no imports of builder functions yet.

---

### 1. Imports & dependencies

Use only standard libs:

```python
from dataclasses import dataclass
from typing import Dict, List, Any
```

No numpy, no solver, no other modules.

---

### 2. SchemaFamily dataclass

Define exactly:

```python
@dataclass
class SchemaFamily:
    """
    Metadata for a law family (schema type), e.g. S1..S11.

    This is used by Pi-agents and tooling to know:
      - what this family does,
      - which parameters it needs,
      - which features it relies on,
      - which builder function name should be called later.
    """
    id: str                    # e.g. "S1"
    name: str                  # short name, e.g. "Direct pixel color tie"
    description: str           # longer, human-readable description
    parameter_spec: Dict[str, str]
    # example: {"feature_predicate": "str"} or {"color_in": "int", "size_to_color": "dict"}

    required_features: List[str]
    # e.g. ["coords_bands", "components", "object_ids"]

    builder_name: str          # e.g. "build_S1_constraints"
```

Notes:

* `parameter_spec` is a **schema for parameters**, not actual values:

  * keys = parameter names,
  * values = simple type descriptions like `"int"`, `"dict[int->int]"`, `"str"`, `"float"`, `"mapping"`.
* `required_features` are strings that refer to the φ modules we already defined, e.g.:

  * `"coords_bands"`,
  * `"components"`,
  * `"object_ids"`,
  * `"neighborhood_hashes"`,
  * `"object_roles"`,
  * `"line_features"`, etc.

We’re not validating them yet; this is just guidance for Pi-agents/tools.

---

### 3. Define SCHEMA_FAMILIES for S1–S11

Create a global dict:

```python
SCHEMA_FAMILIES: Dict[str, SchemaFamily] = {
    "S1": SchemaFamily(...),
    "S2": SchemaFamily(...),
    ...
    "S11": SchemaFamily(...)
}
```

Fill them according to the math spec you shared. Here’s a concrete template for each:

#### S1 – Direct pixel color tie (copy/equality)

* **Idea from spec:** tie colors of feature-equivalent pixels (same φ) across grids.
* **Metadata:**

```python
"S1": SchemaFamily(
    id="S1",
    name="Direct pixel color tie",
    description=(
        "Enforce equality of colors for pixels that are feature-equivalent "
        "(e.g. same object_id, same relative position) across grids."
    ),
    parameter_spec={
        "feature_predicate": "str"  # e.g. natural language or DSL saying which φ-equivalence to use
    },
    required_features=["coords_bands", "components", "object_ids", "neighborhood_hashes"],
    builder_name="build_S1_constraints"
)
```

#### S2 – Component-wise recolor map

* **Idea:** per-component shape/class, map input color to output color, often based on component size.

```python
"SCHEMA_FAMILIES": {
    "S2": SchemaFamily(
        id="S2",
        name="Component-wise recolor map",
        description=(
            "For each connected component (object class), recolor its pixels based on "
            "size, input color, or other component attributes."
        ),
        parameter_spec={
            "color_in": "int",
            "size_to_color": "dict[str|int -> int]"  # e.g. {1:3, 2:2, 'else':1}
        },
        required_features=["components", "object_ids", "object_roles"],
        builder_name="build_S2_constraints"
    ),
    ...
}
```

#### S3 – Band / stripe laws (rows and columns)

* **Idea:** rows/columns partitioned into classes; each class shares a pattern.

```python
"S3": SchemaFamily(
    id="S3",
    name="Band / stripe laws",
    description=(
        "Partition rows/columns into classes using bands and periodic residues; "
        "enforce shared color patterns within each class."
    ),
    parameter_spec={
        "row_class_rule": "str",   # description/DSL of row classes
        "col_class_rule": "str",   # description/DSL of column classes or periodicity
    },
    required_features=["coords_bands", "line_features"],
    builder_name="build_S3_constraints"
)
```

#### S4 – Periodicity & residue-class coloring

```python
"S4": SchemaFamily(
    id="S4",
    name="Periodicity / residue-class coloring",
    description=(
        "Assign colors based purely on coordinate residues mod K "
        "for rows/columns (checkerboards, stripes, etc.)."
    ),
    parameter_spec={
        "K": "int",                      # modulus
        "residue_to_color": "dict[int -> int]"  # mapping residue -> color
    },
    required_features=["coords_bands"],
    builder_name="build_S4_constraints"
)
```

#### S5 – Template stamping (local codebook)

```python
"S5": SchemaFamily(
    id="S5",
    name="Template stamping",
    description=(
        "Detect seed pixels (via local neighborhood patterns) and stamp "
        "a learned template patch around each seed."
    ),
    parameter_spec={
        "seed_type_to_template": "dict[int -> Patch]",  # conceptual; later defined as needed
        "template_size": "tuple[int,int]"               # (h, w)
    },
    required_features=["neighborhood_hashes"],
    builder_name="build_S5_constraints"
)
```

#### S6 – Cropping to ROI / dominant object

```python
"S6": SchemaFamily(
    id="S6",
    name="Cropping to ROI / dominant object",
    description=(
        "Select a bounding box (e.g. of largest or special component) and make the "
        "output grid a crop of that region."
    ),
    parameter_spec={
        "selection_rule": "str"  # e.g. 'largest_nonzero_component', 'topmost_object', etc.
    },
    required_features=["components", "object_roles"],
    builder_name="build_S6_constraints"
)
```

#### S7 – Aggregation / histogram / summary grids

```python
"S7": SchemaFamily(
    id="S7",
    name="Aggregation / summary grid",
    description=(
        "Partition the input into macro-cells (blocks or bands) and summarize each "
        "region (e.g., dominant color) into a smaller output grid."
    ),
    parameter_spec={
        "region_partition_rule": "str",    # how to block-partition or band-partition
        "summary_rule": "str"             # e.g. 'unique_nonzero_color', 'most_frequent_color'
    },
    required_features=["coords_bands", "components"],
    builder_name="build_S7_constraints"
)
```

#### S8 – Tiling / replication

```python
"S8": SchemaFamily(
    id="S8",
    name="Tiling / replication",
    description=(
        "Identify a base tile (pattern) and replicate it periodically to fill a region."
    ),
    parameter_spec={
        "tile_size": "tuple[int,int]",     # (h, w)
        "tiling_region_rule": "str",       # where to tile
        "padding_color": "int"             # background color for incomplete tiles
    },
    required_features=["coords_bands", "neighborhood_hashes"],
    builder_name="build_S8_constraints"
)
```

#### S9 – Cross / plus propagation

```python
"S9": SchemaFamily(
    id="S9",
    name="Cross / plus propagation",
    description=(
        "Detect cross-shaped seeds and propagate spokes along rows/cols with learned "
        "colors and stopping conditions."
    ),
    parameter_spec={
        "seed_type_to_spokes": "dict[int -> dict[str, Any]]"  # mapping seed type to spoke directions/colors
    },
    required_features=["neighborhood_hashes", "coords_bands"],
    builder_name="build_S9_constraints"
)
```

#### S10 – Frame / border vs interior

```python
"S10": SchemaFamily(
    id="S10",
    name="Frame / border vs interior",
    description=(
        "Assign different colors to border vs interior pixels of components or the whole grid."
    ),
    parameter_spec={
        "border_color": "int",
        "interior_color": "int",
        "scope_rule": "str"  # e.g. 'whole_grid', 'per_component_of_color_k'
    },
    required_features=["components", "object_roles"],  # object_roles has per-component border/interior
    builder_name="build_S10_constraints"
)
```

#### S11 – Local neighborhood rewrite (codebook)

```python
"S11": SchemaFamily(
    id="S11",
    name="Local neighborhood codebook",
    description=(
        "For each 3x3 neighborhood pattern type, assign a corresponding output pattern "
        "via a learned codebook H -> P."
    ),
    parameter_spec={
        "hash_to_patch": "dict[int -> Patch]",  # conceptual codebook
        "patch_size": "tuple[int,int]"          # typically (3,3)
    },
    required_features=["neighborhood_hashes"],
    builder_name="build_S11_constraints"
)
```

> **Note:** For `Patch` and similar types in `parameter_spec`, just use `"Patch"` or `"Any"` as a string; we’re not enforcing types here. This is only metadata.

---

### 4. Thin runner for local sanity

At the bottom of `families.py`, add:

```python
if __name__ == "__main__":
    # Simple sanity check: we have exactly 11 families S1..S11
    from pprint import pprint

    print("Defined schema families:")
    for fid, fam in sorted(SCHEMA_FAMILIES.items()):
        print(f"- {fid}: {fam.name}")
    assert len(SCHEMA_FAMILIES) == 11, "Expected 11 schema families (S1..S11)"
    assert all(fid.startswith("S") for fid in SCHEMA_FAMILIES.keys())

    # Spot-check one family's parameter_spec and required_features
    s2 = SCHEMA_FAMILIES["S2"]
    print("\nS2 details:")
    pprint(s2)
    assert "color_in" in s2.parameter_spec
    assert "components" in s2.required_features

    print("\nfamilies.py sanity checks passed.")
```

This runner:

* Prints all schema families,
* Asserts there are exactly 11,
* Checks S2 has expected fields.

---

### 5. Reviewer/tester instructions

For the reviewer/tester:

1. **Code review:**

   * Verify only `dataclasses` and `typing` are imported.
   * Confirm `SchemaFamily` has all fields: `id, name, description, parameter_spec, required_features, builder_name`.
   * Check `SCHEMA_FAMILIES`:

     * has keys `"S1"` through `"S11"`,
     * each value is a `SchemaFamily`,
     * `builder_name` strings follow `"build_Sk_constraints"` convention.

2. **Run thin runner:**

   * `python -m src.schemas.families`
   * Confirm:

     * it prints all 11 families with names,
     * prints S2 details,
     * ends with `families.py sanity checks passed.` and no assertions.

3. **Optional ARC alignment sanity:**

   * Compare the descriptions and required_features against your math spec document in `docs/anchors/`.
   * Confirm that:

     * S2 mentions components & recolor,
     * S6 mentions cropping/ROI,
     * S7 mentions summary/aggregation,
     * S10 uses border vs interior,
     * S11 mentions local neighborhood rewrite.

No solver, no builders, no operators are called here; this is **pure metadata** consistent with your math spec and ready for Pi-agents to inspect later.

---