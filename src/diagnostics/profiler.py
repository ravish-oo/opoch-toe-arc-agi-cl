"""
Physics Profiler: The Router.

Analyzes Input→Output transformation *before* mining to determine which
physics schemas are relevant. Based on Conservation Laws:

1. Geometry Preservation: Does output have same shape as input?
2. Mass Conservation: Are pixel counts per color preserved?
3. New Colors: Does output contain colors not in input?

This enables "Thermodynamic Gating" - don't run creation physics (S12, S14)
if the task's energy signature shows no creation occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import numpy as np

from src.schemas.context import TaskContext


@dataclass
class TaskProfile:
    """
    Physics profile for a task.

    Attributes:
        is_geometry_preserving: True if all examples have input.shape == output.shape
        mass_conserved: True if pixel counts per color are identical in/out
        new_colors: Set of colors that appear in output but not in input
    """
    is_geometry_preserving: bool
    mass_conserved: bool
    new_colors: Set[int]

    @property
    def allows_creation(self) -> bool:
        """
        True if creation schemas (S12, S14) can run.

        HARD CONSTRAINT: Geometry must be preserved for pixel-wise operations.
        If geometry changed, pixel-wise schemas don't apply.

        Creation is possible if:
        - Geometry IS preserved (pixel-wise ops can work), AND
        - (Mass is NOT conserved OR New colors appear)
        """
        # Hard constraint: pixel-wise schemas don't work on resizing tasks
        if not self.is_geometry_preserving:
            return False
        # Creation if mass not conserved OR new colors appear
        return not self.mass_conserved or len(self.new_colors) > 0

    @property
    def allows_movement(self) -> bool:
        """
        True if pixel movement is possible (enables S13 Gravity).

        Movement requires:
        - Mass IS conserved (same pixel counts), AND
        - Geometry IS preserved (same shape for pixel-wise ops)
        """
        return self.mass_conserved and self.is_geometry_preserving


def count_colors(grid: np.ndarray) -> Dict[int, int]:
    """
    Count pixels per color in a grid.

    Args:
        grid: 2D numpy array of colors

    Returns:
        Dict mapping color -> count
    """
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def profile_task(task_context: TaskContext) -> TaskProfile:
    """
    Analyze Input→Output transformation to determine relevant physics.

    This function examines all training examples to build a physics profile
    that determines which Heavy Miners (S12, S13, S14) should be activated.

    Args:
        task_context: TaskContext with train/test examples

    Returns:
        TaskProfile with physics characteristics

    Example:
        >>> profile = profile_task(task_context)
        >>> if profile.allows_creation:
        ...     mine_S12(...)  # Rays create pixels
        ...     mine_S14(...)  # Fill creates pixels
        >>> if profile.allows_movement:
        ...     mine_S13(...)  # Gravity moves pixels
    """
    # 1. Check Geometry Preservation
    # All examples must preserve geometry for this to be True
    is_geometry_preserving = True
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            is_geometry_preserving = False
            break

    # 2. Check Mass Conservation
    # All examples must conserve mass (pixel counts per color) for this to be True
    mass_conserved = True
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        in_counts = count_colors(ex.input_grid)
        out_counts = count_colors(ex.output_grid)
        if in_counts != out_counts:
            mass_conserved = False
            break

    # 3. Check New Colors
    # Collect all colors from all training examples
    input_colors: Set[int] = set()
    output_colors: Set[int] = set()
    for ex in task_context.train_examples:
        input_colors.update(np.unique(ex.input_grid).tolist())
        if ex.output_grid is not None:
            output_colors.update(np.unique(ex.output_grid).tolist())

    new_colors = output_colors - input_colors

    return TaskProfile(
        is_geometry_preserving=is_geometry_preserving,
        mass_conserved=mass_conserved,
        new_colors=new_colors
    )


if __name__ == "__main__":
    # Self-test with toy examples
    from src.schemas.context import build_example_context

    print("=" * 70)
    print("Physics Profiler self-test")
    print("=" * 70)

    # Test 1: Geometry preserving, mass conserved (Movement task - S13)
    print("\nTest 1: Gravity task (geometry preserving, mass conserved)")
    print("-" * 70)

    # Pixel moves from (0,1) to (2,1)
    in1 = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=int)
    out1 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ], dtype=int)

    ex1 = build_example_context(in1, out1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=2)
    profile1 = profile_task(ctx1)

    print(f"  is_geometry_preserving: {profile1.is_geometry_preserving}")
    print(f"  mass_conserved: {profile1.mass_conserved}")
    print(f"  new_colors: {profile1.new_colors}")
    print(f"  allows_creation: {profile1.allows_creation}")
    print(f"  allows_movement: {profile1.allows_movement}")

    assert profile1.is_geometry_preserving == True
    assert profile1.mass_conserved == True
    assert profile1.new_colors == set()
    assert profile1.allows_creation == False  # No creation
    assert profile1.allows_movement == True   # Movement allowed
    print("  -> S13 (Gravity) ENABLED, S12/S14 DISABLED")

    # Test 2: Geometry preserving, mass NOT conserved (Fill task - S14)
    print("\nTest 2: Fill task (geometry preserving, mass NOT conserved)")
    print("-" * 70)

    # Hole gets filled with color 2
    in2 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=int)
    out2 = np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ], dtype=int)

    ex2 = build_example_context(in2, out2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=3)
    profile2 = profile_task(ctx2)

    print(f"  is_geometry_preserving: {profile2.is_geometry_preserving}")
    print(f"  mass_conserved: {profile2.mass_conserved}")
    print(f"  new_colors: {profile2.new_colors}")
    print(f"  allows_creation: {profile2.allows_creation}")
    print(f"  allows_movement: {profile2.allows_movement}")

    assert profile2.is_geometry_preserving == True
    assert profile2.mass_conserved == False  # 0 disappeared, 2 appeared
    assert profile2.new_colors == {2}
    assert profile2.allows_creation == True   # Creation allowed
    assert profile2.allows_movement == False  # Mass not conserved
    print("  -> S12/S14 ENABLED, S13 (Gravity) DISABLED")

    # Test 3: Geometry NOT preserving (Crop task - S6)
    print("\nTest 3: Crop task (geometry NOT preserving)")
    print("-" * 70)

    in3 = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=int)
    out3 = np.array([
        [1, 1],
        [1, 1],
    ], dtype=int)

    ex3 = build_example_context(in3, out3)
    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=2)
    profile3 = profile_task(ctx3)

    print(f"  is_geometry_preserving: {profile3.is_geometry_preserving}")
    print(f"  mass_conserved: {profile3.mass_conserved}")
    print(f"  new_colors: {profile3.new_colors}")
    print(f"  allows_creation: {profile3.allows_creation}")
    print(f"  allows_movement: {profile3.allows_movement}")

    assert profile3.is_geometry_preserving == False
    assert profile3.mass_conserved == False  # Different sizes
    assert profile3.new_colors == set()
    assert profile3.allows_creation == False  # Geometry changed = pixel-wise disabled
    assert profile3.allows_movement == False  # Geometry changed
    print("  -> ALL pixel-wise DISABLED (S12/S13/S14), only S6/S7/S8 apply")

    # Test 4: Recolor task (geometry preserving, mass conserved with permutation)
    print("\nTest 4: Recolor task (geometry preserving, colors permuted)")
    print("-" * 70)

    # All 1s become 2s, all 2s become 1s (swap)
    in4 = np.array([
        [1, 1, 2],
        [1, 2, 2],
    ], dtype=int)
    out4 = np.array([
        [2, 2, 1],
        [2, 1, 1],
    ], dtype=int)

    ex4 = build_example_context(in4, out4)
    ctx4 = TaskContext(train_examples=[ex4], test_examples=[], C=3)
    profile4 = profile_task(ctx4)

    print(f"  is_geometry_preserving: {profile4.is_geometry_preserving}")
    print(f"  mass_conserved: {profile4.mass_conserved}")
    print(f"  new_colors: {profile4.new_colors}")
    print(f"  allows_creation: {profile4.allows_creation}")
    print(f"  allows_movement: {profile4.allows_movement}")

    assert profile4.is_geometry_preserving == True
    assert profile4.mass_conserved == True  # Same counts, just swapped
    assert profile4.new_colors == set()
    assert profile4.allows_creation == False
    assert profile4.allows_movement == True
    print("  -> S2 (Recolor) relevant, no Heavy Physics needed")

    print("\n" + "=" * 70)
    print("Physics Profiler self-test passed.")
    print("=" * 70)
