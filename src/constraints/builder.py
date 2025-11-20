"""
Linear constraint builder for the constraint system.

This module defines core data structures to collect linear equality constraints
over the y-vector (one-hot encoding of pixel colors).

Constraints have the form:
    sum_i coeffs[i] * y[indices[i]] = rhs

This is the generic constraint plumbing used by all schema builders (S1-S11).
No solver logic or schema-specific code here.
"""

from dataclasses import dataclass, field
from typing import List

from src.constraints.indexing import y_index


@dataclass
class LinearConstraint:
    """
    Represents a single linear equality constraint over the y vector:

        sum_i coeffs[i] * y[indices[i]] = rhs

    where y is the flattened (N*C)-dimensional one-hot vector of pixel colors.

    Attributes:
        indices: Indices into y (0 .. N*C-1)
        coeffs: Coefficients (same length as indices)
        rhs: Right-hand side value

    Example:
        # y[5] - y[10] = 0 (pixels 5 and 10 have same color)
        LinearConstraint(indices=[5, 10], coeffs=[1.0, -1.0], rhs=0.0)
    """
    indices: List[int]    # indices into y (0 .. N*C-1)
    coeffs: List[float]   # same length as indices
    rhs: float            # right-hand side


@dataclass
class ConstraintBuilder:
    """
    Collects linear equality constraints and soft preferences over the y vector.

    This is the main interface for schema builders to emit constraints
    that will later be passed to the LP solver.

    Attributes:
        constraints: List of LinearConstraint objects (hard constraints)
        preferences: List of (p_idx, color, weight) tuples (soft preferences)
    """
    constraints: List[LinearConstraint] = field(default_factory=list)
    preferences: List[tuple] = field(default_factory=list)  # List[(p_idx, color, weight)]

    def add_eq(self, indices: List[int], coeffs: List[float], rhs: float) -> None:
        """
        Add a generic linear equality constraint:

            sum_i coeffs[i] * y[indices[i]] = rhs

        Args:
            indices: Indices into the y-vector
            coeffs: Coefficients (must be same length as indices)
            rhs: Right-hand side value

        Raises:
            AssertionError: If indices and coeffs have different lengths
        """
        assert len(indices) == len(coeffs), \
            f"indices and coeffs must have same length, got {len(indices)} != {len(coeffs)}"

        self.constraints.append(
            LinearConstraint(indices=indices, coeffs=coeffs, rhs=rhs)
        )

    def tie_pixel_colors(self, p_idx: int, q_idx: int, C: int) -> None:
        """
        Enforce that pixels p and q have the same color.

        This implements the S1 schema primitive: for all c in 0..C-1:
            y[p,c] - y[q,c] = 0

        Creates C constraints (one per color).

        Args:
            p_idx: Flat pixel index for first pixel (0 <= p_idx < N)
            q_idx: Flat pixel index for second pixel (0 <= q_idx < N)
            C: Number of colors

        Example:
            # Enforce pixels 0 and 5 have the same color
            builder.tie_pixel_colors(0, 5, C=10)
        """
        for c in range(C):
            i_p = y_index(p_idx, c, C)
            i_q = y_index(q_idx, c, C)
            self.add_eq(indices=[i_p, i_q], coeffs=[1.0, -1.0], rhs=0.0)

    def fix_pixel_color(self, p_idx: int, color: int, C: int) -> None:
        """
        Enforce that pixel p has exactly the given color.

        This sets: y[p,color] = 1

        NOTE: This method only enforces y[p,color] = 1.
              Zeroing out other colors is automatically handled by:
              1. One-hot constraints (sum of all colors = 1)
              2. Other schema constraints
              This avoids redundant constraints in the LP.

        Args:
            p_idx: Flat pixel index (0 <= p_idx < N)
            color: Color to fix (0 <= color < C)
            C: Number of colors

        Example:
            # Enforce pixel 3 has color 7
            builder.fix_pixel_color(3, 7, C=10)
        """
        i = y_index(p_idx, color, C)
        self.add_eq(indices=[i], coeffs=[1.0], rhs=1.0)

    def forbid_pixel_color(self, p_idx: int, color: int, C: int) -> None:
        """
        Enforce that pixel p does NOT have the given color.

        This sets: y[p,color] = 0

        Args:
            p_idx: Flat pixel index (0 <= p_idx < N)
            color: Color to forbid (0 <= color < C)
            C: Number of colors

        Example:
            # Forbid pixel 3 from being color 7
            builder.forbid_pixel_color(3, 7, C=10)
        """
        i = y_index(p_idx, color, C)
        self.add_eq(indices=[i], coeffs=[1.0], rhs=0.0)

    def prefer_pixel_color(self, p_idx: int, color: int, weight: float = 1.0) -> None:
        """
        Add a soft preference for pixel p to have the given color.

        This adds a term to the objective function that penalizes
        choosing any color OTHER than the preferred color.

        Unlike fix_pixel_color (hard constraint), this allows the solver
        to choose a different color if required by other hard constraints,
        but at a cost.

        Args:
            p_idx: Flat pixel index (0 <= p_idx < N)
            color: Preferred color (0 <= color < C)
            weight: Penalty weight for violating this preference (default 1.0)

        Example:
            # Prefer pixel 3 to be color 7, but allow override by hard constraints
            builder.prefer_pixel_color(3, 7, weight=10.0)
        """
        self.preferences.append((p_idx, color, weight))


def add_one_hot_constraints(builder: ConstraintBuilder, N: int, C: int) -> None:
    """
    For each pixel index p in 0..N-1, enforce:

        sum_{c=0..C-1} y[p,c] = 1

    This ensures every pixel has exactly one color.

    This is a fundamental constraint from the math kernel spec (section 0):
        "y ∈ {0,1}^{NC}, Σ_c y_{p,c} = 1 ∀ p"

    Creates N constraints (one per pixel).

    Args:
        builder: ConstraintBuilder to add constraints to
        N: Number of pixels (H * W)
        C: Number of colors

    Example:
        >>> builder = ConstraintBuilder()
        >>> add_one_hot_constraints(builder, N=12, C=10)
        >>> len(builder.constraints)
        12
    """
    for p_idx in range(N):
        indices = [y_index(p_idx, c, C) for c in range(C)]
        coeffs = [1.0] * C
        rhs = 1.0
        builder.add_eq(indices, coeffs, rhs)


if __name__ == "__main__":
    # Simple sanity checks with tiny N, C
    N, C = 2, 3  # two pixels, three colors

    print("Testing one-hot constraints...")
    b = ConstraintBuilder()
    add_one_hot_constraints(b, N, C)
    assert len(b.constraints) == N, f"Expected {N} constraints, got {len(b.constraints)}"
    # Check first constraint structure
    lc = b.constraints[0]
    assert len(lc.indices) == C, f"Expected {C} indices per constraint"
    assert len(lc.coeffs) == C
    assert lc.rhs == 1.0
    print(f"  ✓ Created {N} one-hot constraints (one per pixel)")

    print("Testing tie_pixel_colors...")
    b2 = ConstraintBuilder()
    b2.tie_pixel_colors(p_idx=0, q_idx=1, C=C)
    assert len(b2.constraints) == C, f"Expected {C} tie constraints, got {len(b2.constraints)}"
    # Check one tie constraint structure
    lc2 = b2.constraints[0]
    assert len(lc2.indices) == 2
    assert lc2.coeffs == [1.0, -1.0]
    assert lc2.rhs == 0.0
    print(f"  ✓ Created {C} tie constraints (one per color)")

    print("Testing fix_pixel_color...")
    b3 = ConstraintBuilder()
    b3.fix_pixel_color(p_idx=0, color=2, C=C)
    assert len(b3.constraints) == 1, "Expected 1 constraint"
    lc3 = b3.constraints[0]
    assert lc3.rhs == 1.0, "fix_pixel_color should have rhs=1.0"
    assert len(lc3.indices) == 1 and len(lc3.coeffs) == 1
    expected_idx = y_index(0, 2, C)
    assert lc3.indices[0] == expected_idx, f"Expected index {expected_idx}"
    print(f"  ✓ Created 1 fix constraint (y[{expected_idx}] = 1)")

    print("Testing forbid_pixel_color...")
    b4 = ConstraintBuilder()
    b4.forbid_pixel_color(p_idx=0, color=1, C=C)
    lc4 = b4.constraints[0]
    assert lc4.rhs == 0.0, "forbid_pixel_color should have rhs=0.0"
    expected_idx2 = y_index(0, 1, C)
    assert lc4.indices[0] == expected_idx2, f"Expected index {expected_idx2}"
    print(f"  ✓ Created 1 forbid constraint (y[{expected_idx2}] = 0)")

    print("\n✓ builder.py sanity checks passed.")
