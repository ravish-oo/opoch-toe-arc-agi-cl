"""
LP/ILP solver wrapper for constraint-based grid solving.

This module provides a single-grid ILP solver that:
  - Takes constraints from ConstraintBuilder (M2)
  - Creates binary variables y[p,c] ∈ {0,1}
  - Adds all constraints + one-hot constraints
  - Solves using PuLP's CBC solver
  - Returns a (num_pixels, num_colors) numpy array

Uses standard pulp library (no custom solver implementation).
"""

from typing import List

import numpy as np
import pulp

from src.constraints.builder import ConstraintBuilder, LinearConstraint
from src.constraints.indexing import y_index_to_pc


class InfeasibleModelError(Exception):
    """Raised when the ILP model is infeasible or not optimal."""
    pass


def solve_constraints_for_grid(
    builder: ConstraintBuilder,
    num_pixels: int,
    num_colors: int,
    objective: str = "min_sum"
) -> np.ndarray:
    """
    Build and solve an ILP for a single grid.

    This implements the core math kernel solver (Section 0 of math spec):
      - Variables: y ∈ {0,1}^(N*C)
      - Constraints: B(T)y = 0 (all builder constraints)
      - One-hot: Σ_c y[p,c] = 1 ∀p
      - Objective: minimize sum(y) or zero (feasibility only)

    Args:
        builder: ConstraintBuilder with collected LinearConstraint objects.
        num_pixels: number of pixels in this grid (H * W).
        num_colors: number of colors in the palette (C).
        objective: currently supports:
            - "min_sum": minimize sum of all y[p,c]
            - "none":    zero objective (feasibility only)

    Returns:
        y: numpy array of shape (num_pixels, num_colors), with entries 0 or 1.
           Each row is one-hot (exactly one 1 per pixel).

    Raises:
        InfeasibleModelError: if the model is infeasible or no optimal solution is found.

    Example:
        >>> builder = ConstraintBuilder()
        >>> builder.fix_pixel_color(0, 3, C=10)  # pixel 0 → color 3
        >>> y_sol = solve_constraints_for_grid(builder, num_pixels=1, num_colors=10)
        >>> y_sol.shape
        (1, 10)
        >>> y_sol[0, 3]
        1
    """
    # 1. Create model
    prob = pulp.LpProblem("arc_ilp", pulp.LpMinimize)

    # 2. Create binary variables y[p][c]
    # Structure: y[p][c] for p in 0..num_pixels-1, c in 0..num_colors-1
    y = [
        [pulp.LpVariable(f"y_{p}_{c}", lowBound=0, upBound=1, cat=pulp.LpBinary)
         for c in range(num_colors)]
        for p in range(num_pixels)
    ]

    # 3. Add constraints from builder.constraints
    # Each LinearConstraint: sum(coeffs[i] * y[indices[i]]) = rhs
    for lc in builder.constraints:
        assert len(lc.indices) == len(lc.coeffs), \
            f"Constraint has mismatched indices/coeffs: {len(lc.indices)} vs {len(lc.coeffs)}"

        # Build linear expression
        expr = 0
        for idx, coeff in zip(lc.indices, lc.coeffs):
            # Map y_index back to (p_idx, color)
            # y_index_to_pc signature: (y_idx, C, W) -> (p_idx, color)
            # W is not needed for this operation, pass 0 as dummy
            p_idx, color = y_index_to_pc(idx, num_colors, 0)
            expr += coeff * y[p_idx][color]

        # Add equality constraint
        prob += (expr == lc.rhs)

    # 4. Add one-hot constraints per pixel
    # For each pixel p: sum over all colors c: y[p][c] = 1
    for p in range(num_pixels):
        prob += (sum(y[p][c] for c in range(num_colors)) == 1)

    # 5. Set objective
    if objective == "min_sum":
        # Minimize sum of all y variables (trivial objective for TU matrix)
        prob += sum(y[p][c] for p in range(num_pixels) for c in range(num_colors))
    elif objective == "none":
        # Zero objective (feasibility only)
        prob += 0
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # 6. Solve using pulp's CBC solver
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check if optimal solution found
    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleModelError(
            f"Solver status: {pulp.LpStatus[status]}. "
            f"Model may be infeasible or unbounded."
        )

    # 7. Extract solution into numpy array
    y_sol = np.zeros((num_pixels, num_colors), dtype=int)
    for p in range(num_pixels):
        for c in range(num_colors):
            val = pulp.value(y[p][c])
            # Guard against None or float noise (use > 0.5 threshold)
            y_sol[p, c] = 1 if val is not None and val > 0.5 else 0

    # 8. Sanity check: each pixel should be exactly one-hot
    row_sums = y_sol.sum(axis=1)
    if not np.all(row_sums == 1):
        # Find violating pixels for error message
        bad_pixels = np.where(row_sums != 1)[0]
        raise AssertionError(
            f"One-hot constraint violated in solution at pixels: {bad_pixels}. "
            f"Row sums: {row_sums[bad_pixels]}"
        )

    # 9. Return solution
    return y_sol


if __name__ == "__main__":
    # Simple self-test: single pixel, force to color 2 out of 5 colors
    print("Testing lp_solver.py with minimal example...")
    print("=" * 70)

    from src.constraints.builder import ConstraintBuilder

    # Test: 1 pixel, 5 colors, fix to color 2
    builder = ConstraintBuilder()
    builder.fix_pixel_color(p_idx=0, color=2, C=5)

    y_sol = solve_constraints_for_grid(builder, num_pixels=1, num_colors=5)

    print(f"Solution shape: {y_sol.shape}")
    print(f"Solution: {y_sol}")
    print(f"Expected: [[0 0 1 0 0]]")

    # Verify
    expected = np.array([[0, 0, 1, 0, 0]])
    assert np.array_equal(y_sol, expected), f"Expected {expected}, got {y_sol}"

    print("\n✓ lp_solver.py self-test passed.")
    print("=" * 70)
