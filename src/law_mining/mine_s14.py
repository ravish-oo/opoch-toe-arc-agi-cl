"""
S14 law miner: Discover topology/flood fill patterns.

This miner finds topological region-filling rules:
  - Fill holes inside enclosed boundaries
  - Fill background regions connected to edges
  - Uses scipy.ndimage for robust topological operations

Mining algorithm (Signal/Noise Separation):
  1. Identify candidate regions (holes or background) in input
  2. Pool output colors in those regions across ALL training examples
  3. Detect DOMINANT color using Signal/Noise ratio >= 2.0
  4. If valid Background (not Texture), emit SchemaInstance

Signal/Noise Logic:
  - Majority Check: dominant_count / total > 0.5 (must be majority)
  - Signal/Noise Check: dominant_count / noise >= 2.0 (signal 2x noise)
  - This distinguishes "Background with Objects" from "Texture (Checkerboard)"
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np

try:
    from scipy.ndimage import label, binary_fill_holes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.schemas.context import TaskContext
from src.catalog.types import SchemaInstance


def find_holes_in_boundary(grid: np.ndarray, boundary_color: int) -> np.ndarray:
    """
    Find holes (enclosed regions) inside a boundary of a specific color.

    Uses scipy.ndimage.binary_fill_holes for robust topological detection.

    Args:
        grid: Input grid
        boundary_color: Color that forms the boundary

    Returns:
        Boolean mask where True = hole pixel (inside but not boundary)
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(grid.shape, dtype=bool)

    # Create binary mask of boundary
    boundary_mask = (grid == boundary_color)

    # Fill holes in the boundary mask
    filled_mask = binary_fill_holes(boundary_mask)

    # Holes are pixels that are filled but not part of boundary
    holes_mask = filled_mask & ~boundary_mask

    return holes_mask


def find_background_region(grid: np.ndarray) -> np.ndarray:
    """
    Find the background region connected to (0,0).

    Uses scipy.ndimage.label for connected component analysis.

    Args:
        grid: Input grid

    Returns:
        Boolean mask where True = background pixel (connected to corner)
    """
    if not SCIPY_AVAILABLE:
        return np.zeros(grid.shape, dtype=bool)

    # Get the color at (0,0) - this is typically background
    background_color = int(grid[0, 0])

    # Create mask of pixels with same color as background
    same_color_mask = (grid == background_color)

    # Label connected components (4-connectivity)
    labeled, num_features = label(same_color_mask)

    # The background component is the one containing (0,0)
    background_label = labeled[0, 0]

    if background_label == 0:
        # (0,0) is not part of any component (shouldn't happen)
        return np.zeros(grid.shape, dtype=bool)

    background_mask = (labeled == background_label)

    return background_mask


def find_enclosed_regions(grid: np.ndarray, boundary_color: int) -> List[np.ndarray]:
    """
    Find individual enclosed regions (holes) inside boundaries.

    Uses scipy.ndimage to find holes and separate them into connected components.

    Args:
        grid: Input grid
        boundary_color: Color that forms the boundary

    Returns:
        List of boolean masks, one per enclosed region
    """
    if not SCIPY_AVAILABLE:
        return []

    # Find all holes as a single mask
    holes_mask = find_holes_in_boundary(grid, boundary_color)

    if not np.any(holes_mask):
        return []

    # Label connected components to separate individual regions
    labeled, num_regions = label(holes_mask)

    # Create a separate mask for each region
    regions = []
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled == region_id)
        regions.append(region_mask)

    return regions


def check_signal_noise_ratio(all_colors: np.ndarray) -> Tuple[Optional[int], bool]:
    """
    Check if the color distribution represents a Background (Signal) vs Texture (Noise).

    Signal/Noise Logic (TOE Law 24 - Signal Separation):
      1. Majority Check: dominant must be > 50% of total
      2. Signal/Noise Check: dominant / noise >= 2.0

    This distinguishes:
      - Background with Objects (70/30 → ratio 2.3 → Accept)
      - Texture/Checkerboard (50/50 → ratio 1.0 → Reject)

    Args:
        all_colors: Array of colors in the region (pooled across examples)

    Returns:
        (dominant_color, is_valid) where is_valid=True if this is Background
    """
    if len(all_colors) == 0:
        return None, False

    # Count color frequencies
    colors, counts = np.unique(all_colors, return_counts=True)

    # Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_color = int(colors[sorted_indices[0]])
    dominant_count = int(counts[sorted_indices[0]])
    total_pixels = len(all_colors)

    # Check 1: Absolute Majority (must be > 50%)
    if dominant_count / total_pixels <= 0.5:
        return None, False  # No majority → Texture/Noise, not Background

    # Check 2: Signal-to-Noise Ratio (must be >= 2.0)
    noise_pixels = total_pixels - dominant_count
    if noise_pixels == 0:
        # 100% Pure - Ideal Case
        return dominant_color, True

    if (dominant_count / noise_pixels) < 2.0:
        # Signal not 2x noise → likely Texture, let S5 handle it
        return None, False

    return dominant_color, True


def detect_dominant_fill_enclosed(
    task_context: TaskContext,
    boundary_color: int
) -> Tuple[Optional[int], bool]:
    """
    Detect the dominant fill color for holes inside a boundary.

    Pools all output colors at hole positions across ALL training examples,
    then applies Signal/Noise ratio check.

    Args:
        task_context: TaskContext with training examples
        boundary_color: Color that forms the boundary

    Returns:
        (fill_color, is_valid) where is_valid=True if a valid fill law exists
    """
    all_hole_colors: List[int] = []

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return None, False  # S14 requires geometry-preserving

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Find holes in input
        holes_mask = find_holes_in_boundary(input_grid, boundary_color)

        if np.any(holes_mask):
            # Pool output colors at hole positions
            hole_colors = output_grid[holes_mask].flatten()
            all_hole_colors.extend(hole_colors.tolist())

    if not all_hole_colors:
        return None, False  # No holes found

    # Apply Signal/Noise ratio check
    return check_signal_noise_ratio(np.array(all_hole_colors))


def detect_dominant_fill_background(
    task_context: TaskContext
) -> Tuple[Optional[int], bool]:
    """
    Detect the dominant fill color for background regions.

    Pools all output colors at background positions across ALL training examples,
    then applies Signal/Noise ratio check.

    Args:
        task_context: TaskContext with training examples

    Returns:
        (fill_color, is_valid) where is_valid=True if a valid fill law exists
    """
    all_bg_colors: List[int] = []

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return None, False  # S14 requires geometry-preserving

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Find background in input
        background_mask = find_background_region(input_grid)

        if np.any(background_mask):
            # Pool output colors at background positions
            bg_colors = output_grid[background_mask].flatten()
            all_bg_colors.extend(bg_colors.tolist())

    if not all_bg_colors:
        return None, False  # No background found

    # Apply Signal/Noise ratio check
    return check_signal_noise_ratio(np.array(all_bg_colors))


def count_affected_pixels_enclosed(
    task_context: TaskContext,
    boundary_color: int,
    fill_color: int
) -> int:
    """
    Count how many pixels are actually changed by fill_enclosed.

    Args:
        task_context: TaskContext with training examples
        boundary_color: Color that forms the boundary
        fill_color: Color that fills the holes

    Returns:
        Total count of pixels changed across all training examples
    """
    total = 0
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        holes_mask = find_holes_in_boundary(input_grid, boundary_color)

        # Count pixels where hole exists AND input color != fill_color
        input_colors = input_grid[holes_mask]
        total += np.sum(input_colors != fill_color)

    return total


def count_affected_pixels_background(
    task_context: TaskContext,
    fill_color: int
) -> int:
    """
    Count how many pixels are actually changed by fill_background.

    Args:
        task_context: TaskContext with training examples
        fill_color: Color that fills the background

    Returns:
        Total count of pixels changed across all training examples
    """
    total = 0
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        background_mask = find_background_region(input_grid)

        # Count pixels where background exists AND input color != fill_color
        input_colors = input_grid[background_mask]
        total += np.sum(input_colors != fill_color)

    return total


def is_destructive_fill_enclosed(
    task_context: TaskContext,
    boundary_color: int,
    fill_color: int,
    roles: Dict[Any, int],
    claimed_roles: Set[int],
    max_error_rate: float = 0.05  # Allow up to 5% noise
) -> bool:
    """
    Check if fill_enclosed would destroy unclaimed Active Information.

    A fill is destructive if it would overwrite MORE THAN max_error_rate of pixels that:
    1. Are in the fill region (holes)
    2. Don't match fill_color in ground truth
    3. Are NOT claimed by higher-authority schemas (S2/S6/S10)

    Args:
        task_context: TaskContext with training examples
        boundary_color: Color that forms the boundary
        fill_color: Color that would fill the holes
        roles: Role mapping (kind, ex_idx, r, c) -> role_id
        claimed_roles: Set of role_ids claimed by higher schemas
        max_error_rate: Maximum allowed error rate (default 5%)

    Returns:
        True if fill would destroy too many unclaimed pixels (destructive)
    """
    total_region_pixels = 0
    unclaimed_mismatch_count = 0

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Get the fill region
        holes_mask = find_holes_in_boundary(input_grid, boundary_color)

        if not np.any(holes_mask):
            continue

        # Count total region pixels
        total_region_pixels += np.sum(holes_mask)

        # Find mismatched pixels: in region AND GT != fill_color
        mismatches = holes_mask & (output_grid != fill_color)

        # Count unclaimed mismatched pixels
        for r, c in zip(*np.where(mismatches)):
            role_key = ("train_out", ex_idx, int(r), int(c))
            if role_key in roles:
                role_id = roles[role_key]
                if role_id not in claimed_roles:
                    # This pixel is unclaimed and would be overwritten
                    unclaimed_mismatch_count += 1

    # Calculate error rate and check threshold (SIZE-AWARE)
    if total_region_pixels == 0:
        return False  # No region to fill

    # SIZE-AWARE TOLERANCE:
    # Small regions (≤50 pixels): 0% tolerance (strict - "noise" might be micro-structure)
    # Large regions (>50 pixels): 5% tolerance (relaxed - likely true noise)
    if total_region_pixels <= 50:
        # STRICT: Any mismatch in small regions is suspicious
        return unclaimed_mismatch_count > 0  # Destructive if ANY unclaimed mismatch
    else:
        # RELAXED: Allow 5% noise in large regions
        error_rate = unclaimed_mismatch_count / total_region_pixels
        return error_rate > max_error_rate  # Destructive if error > 5%


def is_destructive_fill_background(
    task_context: TaskContext,
    fill_color: int,
    roles: Dict[Any, int],
    claimed_roles: Set[int],
    max_error_rate: float = 0.05  # Allow up to 5% noise
) -> bool:
    """
    Check if fill_background would destroy unclaimed Active Information.

    A fill is destructive if it would overwrite MORE THAN max_error_rate of pixels that:
    1. Are in the background region
    2. Don't match fill_color in ground truth
    3. Are NOT claimed by higher-authority schemas (S2/S6/S10)

    Args:
        task_context: TaskContext with training examples
        fill_color: Color that would fill the background
        roles: Role mapping (kind, ex_idx, r, c) -> role_id
        claimed_roles: Set of role_ids claimed by higher schemas
        max_error_rate: Maximum allowed error rate (default 5%)

    Returns:
        True if fill would destroy too many unclaimed pixels (destructive)
    """
    total_region_pixels = 0
    unclaimed_mismatch_count = 0

    for ex_idx, ex in enumerate(task_context.train_examples):
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Get the fill region
        background_mask = find_background_region(input_grid)

        if not np.any(background_mask):
            continue

        # Count total region pixels
        total_region_pixels += np.sum(background_mask)

        # Find mismatched pixels: in region AND GT != fill_color
        mismatches = background_mask & (output_grid != fill_color)

        # Count unclaimed mismatched pixels
        for r, c in zip(*np.where(mismatches)):
            role_key = ("train_out", ex_idx, int(r), int(c))
            if role_key in roles:
                role_id = roles[role_key]
                if role_id not in claimed_roles:
                    # This pixel is unclaimed and would be overwritten
                    unclaimed_mismatch_count += 1

    # Calculate error rate and check threshold (SIZE-AWARE)
    if total_region_pixels == 0:
        return False  # No region to fill

    # SIZE-AWARE TOLERANCE:
    # Small regions (≤50 pixels): 0% tolerance (strict - "noise" might be micro-structure)
    # Large regions (>50 pixels): 5% tolerance (relaxed - likely true noise)
    if total_region_pixels <= 50:
        # STRICT: Any mismatch in small regions is suspicious
        return unclaimed_mismatch_count > 0  # Destructive if ANY unclaimed mismatch
    else:
        # RELAXED: Allow 5% noise in large regions
        error_rate = unclaimed_mismatch_count / total_region_pixels
        return error_rate > max_error_rate  # Destructive if error > 5%


def detect_seeded_fill_enclosed(
    task_context: TaskContext,
    boundary_color: int
) -> Tuple[int, bool]:
    """
    Detect seeded fill pattern: sparse input seeds determine dense output fill.

    Pattern: Input has scattered pixels of color C inside region,
             Output has entire region filled with color C.

    The "noise" pixels are actually SEEDS, not errors!

    Args:
        task_context: TaskContext with training examples
        boundary_color: Color that defines region boundaries

    Returns:
        Tuple of (seed_color, is_valid)
    """
    if not SCIPY_AVAILABLE:
        return (None, False)

    # Collect input colors and output fills across all examples
    seed_color_candidates = {}  # {color: count_of_examples_where_it_seeds}

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        input_grid = ex.input_grid
        output_grid = ex.output_grid

        # Find enclosed regions
        regions = find_enclosed_regions(input_grid, boundary_color)

        for region_mask in regions:
            if not np.any(region_mask):
                continue

            # Get input colors in this region (exclude boundary)
            input_colors_in_region = set()
            for r, c in zip(*np.where(region_mask)):
                color = int(input_grid[r, c])
                if color != 0 and color != boundary_color:
                    input_colors_in_region.add(color)

            # Get output fill color (dominant in region)
            output_colors = output_grid[region_mask]
            if len(output_colors) == 0:
                continue

            from collections import Counter
            color_counts = Counter(output_colors.flatten())
            output_fill, _ = color_counts.most_common(1)[0]

            # Check if output_fill appeared as input seed
            if output_fill in input_colors_in_region:
                seed_color_candidates[output_fill] = seed_color_candidates.get(output_fill, 0) + 1

    # Validate: seed color must work in ALL training examples
    if not seed_color_candidates:
        return (None, False)

    # Pick the most consistent seed color
    best_seed = max(seed_color_candidates, key=seed_color_candidates.get)
    num_examples = len(task_context.train_examples)

    # Must work in at least half of examples
    if seed_color_candidates[best_seed] >= max(1, num_examples // 2):
        return (best_seed, True)

    return (None, False)


def mine_S14(
    task_context: TaskContext,
    roles: Dict[Any, int],
    role_stats: Dict[int, Dict[str, Any]],
    claimed_roles: Set[int] = None
) -> List[SchemaInstance]:
    """
    Mine S14 topology patterns from training examples.

    Algorithm (Signal/Noise Separation + Safety Check):
      1. Check geometry-preserving constraint
      2. For each boundary_color:
         - Detect dominant fill color for holes using Signal/Noise ratio
         - SAFETY CHECK: Reject if would destroy unclaimed pixels
         - If valid, safe, and has kinetic utility >= 1, emit fill_enclosed instance
      3. Detect dominant fill color for background using Signal/Noise ratio
         - SAFETY CHECK: Reject if would destroy unclaimed pixels
         - If valid, safe, and has kinetic utility >= 1, emit fill_background instance

    Signal/Noise Logic:
      - Majority: dominant > 50% of region
      - Ratio: dominant / noise >= 2.0 (dominant >= 66.7%)
      - This accepts "Background with Objects" (70/30)
      - This rejects "Texture/Checkerboard" (50/50) → S5's domain

    Safety Check (Protect Dark Matter):
      - Before emitting a fill rule, verify it doesn't overwrite unclaimed pixels
      - If a mismatched pixel is claimed by S2/S6/S10, it's safe (they'll override)
      - If a mismatched pixel is unclaimed, the fill is destructive → reject

    Args:
        task_context: TaskContext with training examples
        roles: Role mapping (kind, ex_idx, r, c) -> role_id
        role_stats: Role statistics (unused but kept for API consistency)
        claimed_roles: Set of role_ids claimed by higher schemas (S2/S6/S10)

    Returns:
        List of SchemaInstance objects for valid topology patterns
    """
    instances: List[SchemaInstance] = []

    if not SCIPY_AVAILABLE:
        return instances  # Cannot mine without scipy

    # Default to empty set if not provided
    if claimed_roles is None:
        claimed_roles = set()

    # Step 0: S14 only applies to geometry-preserving tasks
    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue
        if ex.input_grid.shape != ex.output_grid.shape:
            return instances  # Not geometry-preserving

    # Collect boundary colors (non-zero colors present in inputs)
    boundary_colors: Set[int] = set()
    for ex in task_context.train_examples:
        boundary_colors.update(set(ex.input_grid.flatten()) - {0})

    # Mine fill_enclosed patterns
    for boundary_color in boundary_colors:
        fill_color, is_valid = detect_dominant_fill_enclosed(task_context, boundary_color)

        if is_valid and fill_color is not None:
            # Check boundary != fill (can't fill with same color as boundary)
            if boundary_color != fill_color:
                # SAFETY CHECK: Don't destroy unclaimed data
                if is_destructive_fill_enclosed(
                    task_context, boundary_color, fill_color, roles, claimed_roles
                ):
                    continue  # Reject destructive fill

                affected = count_affected_pixels_enclosed(task_context, boundary_color, fill_color)
                if affected >= 1:
                    instances.append(SchemaInstance(
                        family_id="S14",
                        params={
                            "operation": "fill_enclosed",
                            "boundary_color": boundary_color,
                            "fill_color": fill_color
                        }
                    ))
        else:
            # Dominance failed - try SEEDED FILL
            # Hypothesis: "Noise" pixels are actually SEEDS that define fill color
            seed_color, is_seeded = detect_seeded_fill_enclosed(task_context, boundary_color)

            if is_seeded and seed_color is not None:
                # Check boundary != seed (can't seed with boundary color)
                if boundary_color != seed_color:
                    # Emit seeded fill rule
                    affected = count_affected_pixels_enclosed(task_context, boundary_color, seed_color)
                    if affected >= 1:
                        instances.append(SchemaInstance(
                            family_id="S14",
                            params={
                                "operation": "fill_seeded",
                                "boundary_color": boundary_color,
                                "seed_color": seed_color
                            }
                        ))

    # Mine fill_background pattern (only one possible)
    fill_color, is_valid = detect_dominant_fill_background(task_context)

    if is_valid and fill_color is not None:
        # SAFETY CHECK: Don't destroy unclaimed data
        if not is_destructive_fill_background(
            task_context, fill_color, roles, claimed_roles
        ):
            affected = count_affected_pixels_background(task_context, fill_color)
            if affected >= 1:
                instances.append(SchemaInstance(
                    family_id="S14",
                    params={
                        "operation": "fill_background",
                        "fill_color": fill_color
                    }
                ))

    return instances


if __name__ == "__main__":
    # Self-test with toy data
    import numpy as np
    from src.schemas.context import build_example_context, TaskContext

    print("Testing S14 miner with toy example...")
    print("=" * 70)

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy not available, skipping test")
        exit(0)

    # Test 1: Fill enclosed hole (100% pure)
    print("Test 1: Fill enclosed hole (100% pure)")
    print("-" * 70)

    input_grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],  # Hole at (2,2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    output_grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],  # Hole filled with Blue (2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    ex1 = build_example_context(input_grid1, output_grid1)
    ctx1 = TaskContext(train_examples=[ex1], test_examples=[], C=3)

    instances1 = mine_S14(ctx1, {}, {})
    print(f"Mined instances: {len(instances1)}")

    found_enclosed = False
    for inst in instances1:
        print(f"  {inst.params}")
        if inst.params.get("operation") == "fill_enclosed":
            if inst.params.get("boundary_color") == 1 and inst.params.get("fill_color") == 2:
                found_enclosed = True

    assert found_enclosed, "Expected to find fill_enclosed(boundary=1, fill=2)"
    print("  Found fill_enclosed(boundary=1, fill=2)")

    # Test 2: Fill background (100% pure)
    print("\nTest 2: Fill background (100% pure)")
    print("-" * 70)

    input_grid2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    output_grid2 = np.array([
        [3, 3, 3, 3, 3],
        [3, 1, 1, 3, 3],
        [3, 1, 1, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
    ], dtype=int)

    ex2 = build_example_context(input_grid2, output_grid2)
    ctx2 = TaskContext(train_examples=[ex2], test_examples=[], C=4)

    instances2 = mine_S14(ctx2, {}, {})
    print(f"Mined instances: {len(instances2)}")

    found_background = False
    for inst in instances2:
        print(f"  {inst.params}")
        if inst.params.get("operation") == "fill_background":
            if inst.params.get("fill_color") == 3:
                found_background = True

    assert found_background, "Expected to find fill_background(fill=3)"
    print("  Found fill_background(fill=3)")

    # Test 3: Dirty background (70% fill, 30% other) - should ACCEPT
    print("\nTest 3: Dirty background (70% fill + 30% other) - should ACCEPT")
    print("-" * 70)

    # 21 background pixels: 15 become 3, 6 become 7 (71.4% / 28.6% → ratio 2.5)
    input_grid3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    output_grid3 = np.array([
        [3, 3, 3, 3, 3],
        [3, 1, 1, 3, 3],
        [3, 1, 1, 3, 3],
        [3, 3, 7, 7, 3],  # Some pixels become 7 instead of 3
        [3, 7, 7, 7, 3],
    ], dtype=int)

    ex3 = build_example_context(input_grid3, output_grid3)
    ctx3 = TaskContext(train_examples=[ex3], test_examples=[], C=8)

    instances3 = mine_S14(ctx3, {}, {})
    print(f"Mined instances: {len(instances3)}")

    found_dirty_bg = False
    for inst in instances3:
        print(f"  {inst.params}")
        if inst.params.get("operation") == "fill_background":
            found_dirty_bg = True

    assert found_dirty_bg, "Expected to find fill_background for dirty background"
    print("  Found fill_background (Signal/Noise ratio accepted 70/30)")

    # Test 4: Checkerboard (50/50) - should REJECT
    print("\nTest 4: Checkerboard (50/50) - should REJECT (Texture, not Background)")
    print("-" * 70)

    input_grid4 = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=int)

    # Checkerboard pattern in background (50% each)
    output_grid4 = np.array([
        [3, 7, 3, 7],
        [7, 1, 1, 3],
        [3, 1, 1, 7],
        [7, 3, 7, 3],
    ], dtype=int)

    ex4 = build_example_context(input_grid4, output_grid4)
    ctx4 = TaskContext(train_examples=[ex4], test_examples=[], C=8)

    instances4 = mine_S14(ctx4, {}, {})
    print(f"Mined instances: {len(instances4)}")

    found_checkerboard_bg = any(
        inst.params.get("operation") == "fill_background"
        for inst in instances4
    )

    assert not found_checkerboard_bg, "Should NOT find fill_background for checkerboard"
    print("  Correctly rejected checkerboard (50/50 = Texture)")

    print("\n" + "=" * 70)
    print("S14 miner self-test passed.")
