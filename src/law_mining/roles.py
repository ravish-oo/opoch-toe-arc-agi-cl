"""
Role labeller (WL/q) over TaskContext.

This module implements WL-style refinement to assign structural role_ids to
each pixel across all grids in a task (train_in, train_out, test_in).

The role labeller uses:
  - φ features (coords, bands, components, shape signatures)
  - 4-connected neighborhood structure
  - Deterministic WL refinement (no random hashing)

Output is a RolesMapping: (kind, example_idx, r, c) -> role_id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal

import numpy as np

from src.schemas.context import TaskContext, ExampleContext
from src.core.grid_types import Grid
from src.features.coords_bands import (
    row_band_labels,
    col_band_labels,
)
from src.features.components import (
    connected_components_by_color,
    compute_shape_signature,
)


# kind ∈ {"train_in", "train_out", "test_in"}
NodeKind = Literal["train_in", "train_out", "test_in"]

# Node key: (kind, example_idx, r, c)
RolesMapping = Dict[Tuple[NodeKind, int, int, int], int]


@dataclass(frozen=True)
class Node:
    """Represents a single pixel position across all grids in a task."""
    kind: NodeKind
    example_idx: int   # index into TaskContext.train_examples or test_examples
    r: int
    c: int


def _neighbors(node: Node, task_context: TaskContext) -> List[Node]:
    """
    Return 4-connected neighbors (up/down/left/right) of this node,
    restricted to the same kind and same example_idx.

    Args:
        node: Node to find neighbors for
        task_context: TaskContext containing grid dimensions

    Returns:
        List of neighboring nodes within grid bounds
    """
    # Get grid shape
    if node.kind == "train_in":
        H = task_context.train_examples[node.example_idx].input_H
        W = task_context.train_examples[node.example_idx].input_W
    elif node.kind == "train_out":
        H = task_context.train_examples[node.example_idx].output_H
        W = task_context.train_examples[node.example_idx].output_W
    else:  # "test_in"
        H = task_context.test_examples[node.example_idx].input_H
        W = task_context.test_examples[node.example_idx].input_W

    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = node.r + dr, node.c + dc
        if 0 <= rr < H and 0 <= cc < W:
            neighbors.append(Node(node.kind, node.example_idx, rr, cc))

    return neighbors


def compute_roles(task_context: TaskContext, wl_iters: int = 3) -> RolesMapping:
    """
    Use φ (coords, bands, components, etc.) + WL-style refinement
    to assign stable role_ids per pixel across all grids in this task.

    Algorithm:
      1. Create nodes for all pixels in train_in, train_out, test_in
      2. Initialize labels from structural φ features
      3. Run WL refinement iterations over 4-neighbor graphs
      4. Map final labels to consecutive role_ids

    Args:
        task_context: TaskContext with train/test examples
        wl_iters: Number of WL refinement iterations (default 3)

    Returns:
        RolesMapping: (kind, example_idx, r, c) -> role_id (0..R-1)

    Example:
        >>> from src.schemas.context import build_task_context_from_raw
        >>> from src.core.arc_io import load_arc_task
        >>> raw_task = load_arc_task("00576224", Path("data/..."))
        >>> ctx = build_task_context_from_raw(raw_task)
        >>> roles = compute_roles(ctx)
        >>> # roles maps each pixel to a structural role_id
    """
    # =========================================================================
    # Step 1: Build nodes
    # =========================================================================
    nodes: List[Node] = []

    # Train inputs and outputs
    for ex_idx, ex_ctx in enumerate(task_context.train_examples):
        H_in, W_in = ex_ctx.input_grid.shape
        for r in range(H_in):
            for c in range(W_in):
                nodes.append(Node("train_in", ex_idx, r, c))

        if ex_ctx.output_grid is not None:
            H_out, W_out = ex_ctx.output_grid.shape
            for r in range(H_out):
                for c in range(W_out):
                    nodes.append(Node("train_out", ex_idx, r, c))

    # Test inputs (no outputs)
    for ex_idx, ex_ctx in enumerate(task_context.test_examples):
        H, W = ex_ctx.input_grid.shape
        for r in range(H):
            for c in range(W):
                nodes.append(Node("test_in", ex_idx, r, c))

    # =========================================================================
    # Step 2: Precompute grid info per (kind, example_idx)
    # =========================================================================
    @dataclass
    class GridInfo:
        """Precomputed structural info for one grid."""
        grid: Grid
        row_bands: Dict[int, str]
        col_bands: Dict[int, str]
        pixel_to_shape_sig: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]

    grid_info_cache: Dict[Tuple[NodeKind, int], GridInfo] = {}

    def get_grid_info(kind: NodeKind, ex_idx: int) -> GridInfo:
        """Get or compute grid info for (kind, example_idx)."""
        key = (kind, ex_idx)
        if key in grid_info_cache:
            return grid_info_cache[key]

        # Get the appropriate grid
        if kind == "train_in":
            grid = task_context.train_examples[ex_idx].input_grid
        elif kind == "train_out":
            grid = task_context.train_examples[ex_idx].output_grid
        else:  # "test_in"
            grid = task_context.test_examples[ex_idx].input_grid

        H, W = grid.shape

        # Compute band labels
        row_bands = row_band_labels(H)
        col_bands = col_band_labels(W)

        # Compute shape signatures per pixel
        components = connected_components_by_color(grid)
        pixel_to_shape_sig: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {}
        for comp in components:
            # Ensure shape_signature is computed
            if comp.shape_signature is None:
                comp.shape_signature = compute_shape_signature(comp)
            # Map each pixel to its component's shape signature
            for (r, c) in comp.pixels:
                pixel_to_shape_sig[(r, c)] = comp.shape_signature

        info = GridInfo(
            grid=grid,
            row_bands=row_bands,
            col_bands=col_bands,
            pixel_to_shape_sig=pixel_to_shape_sig
        )
        grid_info_cache[key] = info
        return info

    # =========================================================================
    # Step 3: Initialize labels
    # =========================================================================
    labels: Dict[Node, Tuple] = {}

    for node in nodes:
        info = get_grid_info(node.kind, node.example_idx)
        H, W = info.grid.shape

        color = int(info.grid[node.r, node.c])
        row_band = info.row_bands[node.r]
        col_band = info.col_bands[node.c]
        is_border = (node.r == 0 or node.r == H - 1 or node.c == 0 or node.c == W - 1)
        shape_sig = info.pixel_to_shape_sig.get((node.r, node.c), None)

        # Initial label is a tuple of structural features
        labels[node] = (node.kind, color, row_band, col_band, is_border, shape_sig)

    # =========================================================================
    # Step 4: WL refinement loop
    # =========================================================================
    for iteration in range(wl_iters):
        new_labels: Dict[Node, Tuple] = {}

        for node in nodes:
            base_label = labels[node]
            neighs = _neighbors(node, task_context)
            neigh_labels = [labels[n] for n in neighs]

            # Canonical multiset: sort by tuple for deterministic ordering
            neigh_labels_sorted = tuple(sorted(neigh_labels))

            # New label = (old_label, multiset(neighbor_labels))
            new_labels[node] = (base_label, neigh_labels_sorted)

        # Early exit if converged
        if all(new_labels[n] == labels[n] for n in nodes):
            labels = new_labels
            break

        labels = new_labels

    # =========================================================================
    # Step 5: Map final labels to role_ids
    # =========================================================================
    label_to_role: Dict[Tuple, int] = {}
    roles: RolesMapping = {}
    next_role_id = 0

    for node in nodes:
        lab = labels[node]
        if lab not in label_to_role:
            label_to_role[lab] = next_role_id
            next_role_id += 1
        role_id = label_to_role[lab]
        roles[(node.kind, node.example_idx, node.r, node.c)] = role_id

    return roles


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    from src.schemas.context import load_arc_task, build_task_context_from_raw

    print("=" * 70)
    print("roles.py self-test")
    print("=" * 70)

    # Use a simple task
    task_id = "00576224"
    challenges_path = Path("data/arc-agi_training_challenges.json")

    print(f"\nLoading task: {task_id}")
    raw_task = load_arc_task(task_id, challenges_path)
    task_context = build_task_context_from_raw(raw_task)

    print(f"Train examples: {len(task_context.train_examples)}")
    print(f"Test examples: {len(task_context.test_examples)}")

    print("\nComputing roles...")
    roles = compute_roles(task_context)

    num_roles = len(set(roles.values()))
    print(f"\n✓ Computed {num_roles} distinct roles across {len(roles)} pixels")

    # Show sample
    print("\nSample role assignments:")
    print("-" * 70)
    for i, (key, role_id) in enumerate(list(roles.items())[:10]):
        kind, ex_idx, r, c = key
        print(f"  {kind:12s} ex={ex_idx} ({r:2d},{c:2d}) -> role_id={role_id}")

    # Test determinism
    print("\nTesting determinism...")
    roles2 = compute_roles(task_context)
    assert roles == roles2, "compute_roles should be deterministic"
    print("✓ Determinism verified (two runs produced identical mappings)")

    print("\n" + "=" * 70)
    print("✓ roles.py self-test passed")
    print("=" * 70)
