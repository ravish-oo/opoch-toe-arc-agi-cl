"""
Trace exactly where the IndexError happens in error tasks.
"""

from pathlib import Path
from src.schemas.context import load_arc_task, build_task_context_from_raw
from src.law_mining.mine_law_config import mine_law_config
from src.runners.kernel import solve_arc_task_with_diagnostics
import sys
import traceback

task_id = "00d62c1b"
challenges_path = Path("data/arc-agi_training_challenges.json")
solutions_path = Path("data/arc-agi_training_solutions.json")

print(f"Tracing IndexError for task: {task_id}")
print("=" * 70)

# This should NOT raise an exception - kernel should catch it
raw_task = load_arc_task(task_id, challenges_path)
task_context = build_task_context_from_raw(raw_task)
law_config = mine_law_config(task_context)

print(f"Mined {len(law_config.schema_instances)} schema instances")

# Enable detailed exception tracing
sys.settrace(lambda *args: None)  # No-op, just to show we can

try:
    outputs, diagnostics = solve_arc_task_with_diagnostics(
        task_id=task_id,
        law_config=law_config,
        use_training_labels=True,
        use_test_labels=True,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
    )

    print(f"\nKernel completed:")
    print(f"  Status: {diagnostics.status}")
    print(f"  Solver status: {diagnostics.solver_status}")
    print(f"  Error message: {diagnostics.error_message}")
    print(f"  Train outputs: {len(outputs['train'])}")
    print(f"  Test outputs: {len(outputs['test'])}")

    if diagnostics.status == "error":
        print(f"\n✓ Kernel caught exception correctly (status=error)")
        print(f"  This is EXPECTED - exception was handled, not crashed")
    else:
        print(f"\n  Status: {diagnostics.status}")

except Exception as e:
    print(f"\n✗ OUTER EXCEPTION (kernel didn't catch it!):")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {e}")
    print(f"\n  Full traceback:")
    traceback.print_exc()
