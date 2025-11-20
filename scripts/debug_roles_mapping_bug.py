"""Debug the roles_mapping serialization bug."""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.roles import compute_roles
from src.law_mining.role_stats import compute_role_stats
from src.law_mining.mine_s_default import mine_S_Default

task_id = "045e512c"
challenges_path = Path("data/arc-agi_training_challenges.json")

# Load and mine
raw_task = load_arc_task(task_id, challenges_path)
ctx = build_task_context_from_raw(raw_task)
roles = compute_roles(ctx)
role_stats = compute_role_stats(ctx, roles)
schema_instances = mine_S_Default(ctx, roles, role_stats)

if schema_instances:
    s_default = schema_instances[0]
    roles_mapping_raw = s_default.params["roles_mapping"]

    print("Checking roles_mapping structure:")
    print("=" * 70)

    # Check first key
    first_key = list(roles_mapping_raw.keys())[0]
    print(f"First key: {first_key}")
    print(f"Type: {type(first_key)}")
    print(f"Is tuple: {isinstance(first_key, tuple)}")
    print(f"Is string: {isinstance(first_key, str)}")
    print()

    if isinstance(first_key, tuple):
        print("✗ BUG CONFIRMED!")
        print("  roles_mapping has TUPLE keys, but builder expects STRINGS")
        print()
        print("  The builder tries to do:")
        print("    ast.literal_eval(key_str)")
        print("  But key_str is already a tuple, not a string!")
        print()
        print("  This will cause ast.literal_eval to fail silently")
        print("  (caught by except block, skipping all keys)")
        print()
        print("  Result: roles_mapping dict becomes empty")
        print("  Result: No role lookups succeed")
        print("  Result: Zero constraints added")
    else:
        print("✓ Keys are strings (as expected by builder)")
