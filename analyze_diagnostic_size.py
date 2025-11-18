import json

with open('diagnostic_output.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("DIAGNOSTIC OUTPUT SIZE ANALYSIS")
print("=" * 70)
print()

# File size
import os
size_bytes = os.path.getsize('diagnostic_output.json')
print(f"File size: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
print()

# Section breakdown
sections = {
    "Metadata (task_id, status, etc.)": ["task_id", "status", "solver_status", "num_constraints", "num_variables", "schema_ids_used", "error_message"],
    "NEW: schema_constraint_counts": ["schema_constraint_counts"],
    "NEW: example_summaries": ["example_summaries"],
    "train_mismatches": ["train_mismatches"],
}

print("Section sizes:")
print("-" * 70)
for section_name, keys in sections.items():
    section_json = {k: data.get(k) for k in keys if k in data}
    section_str = json.dumps(section_json, indent=2)
    section_bytes = len(section_str.encode('utf-8'))
    section_lines = len(section_str.split('\n'))
    
    marker = "  ← NEW" if "NEW:" in section_name else ""
    print(f"{section_name:40s}: {section_bytes:5,} bytes, {section_lines:4} lines{marker}")

print()
print("Detail on NEW fields:")
print("-" * 70)

# schema_constraint_counts detail
scc = data.get("schema_constraint_counts", {})
print(f"schema_constraint_counts: {len(scc)} schema(s)")
for schema_id, count in scc.items():
    print(f"  {schema_id}: {count} constraints")

# example_summaries detail
es = data.get("example_summaries", [])
print(f"\nexample_summaries: {len(es)} example(s)")
for i, summary in enumerate(es):
    ex_type = "train" if summary["output_shape"] else "test"
    num_colors = len(summary["components_per_color"])
    total_comps = sum(summary["components_per_color"].values())
    print(f"  Example {i} ({ex_type}): {summary['input_shape']} input, "
          f"{num_colors} colors, {total_comps} components")

# train_mismatches detail
tm = data.get("train_mismatches", [])
print(f"\ntrain_mismatches: {len(tm)} failed example(s)")
total_diff_cells = sum(len(m["diff_cells"]) for m in tm)
print(f"  Total mismatched cells: {total_diff_cells}")

print()
print("=" * 70)
print(f"SUMMARY: The new M5.X fields add ~{len(json.dumps({'schema_constraint_counts': scc, 'example_summaries': es}, indent=2).encode('utf-8'))/1024:.1f} KB")
print("         Most size is from train_mismatches (per-cell diffs)")
print("=" * 70)
