"""
Dimension Predictor: Phase 0 Geometry Solver.

Predicts output dimensions for test examples by learning transformation
rules from training examples. This must run BEFORE pixel-level solving.

Supported transformation rules (in priority order):
  1. Identity:  out = in
  2. Scale:     out = k * in  (integer k)
  3. Additive:  out = in + C  (integer C)
  4. Fixed:     out = constant

Each dimension (H, W) is analyzed independently.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.schemas.context import TaskContext


@dataclass
class DimensionRule:
    """A learned dimension transformation rule."""
    rule_type: str  # "identity", "scale", "additive", "fixed"
    param: Optional[int] = None  # k for scale, C for additive/fixed

    def apply(self, input_dim: int) -> int:
        """Apply the rule to an input dimension."""
        if self.rule_type == "identity":
            return input_dim
        elif self.rule_type == "scale":
            return self.param * input_dim
        elif self.rule_type == "additive":
            return input_dim + self.param
        elif self.rule_type == "fixed":
            return self.param
        else:
            # Fallback to identity
            return input_dim


def _try_identity(pairs: List[Tuple[int, int]]) -> Optional[DimensionRule]:
    """Check if all pairs satisfy out = in."""
    for in_dim, out_dim in pairs:
        if in_dim != out_dim:
            return None
    return DimensionRule(rule_type="identity")


def _try_scale(pairs: List[Tuple[int, int]]) -> Optional[DimensionRule]:
    """Check if all pairs satisfy out = k * in for some integer k."""
    if not pairs:
        return None

    # Find k from first pair
    in_dim, out_dim = pairs[0]
    if in_dim == 0:
        return None  # Can't divide by zero

    if out_dim % in_dim != 0:
        return None  # Not integer scale

    k = out_dim // in_dim
    if k <= 0:
        return None  # Only positive scales

    # Verify all pairs
    for in_d, out_d in pairs:
        if in_d == 0 or out_d != k * in_d:
            return None

    return DimensionRule(rule_type="scale", param=k)


def _try_additive(pairs: List[Tuple[int, int]]) -> Optional[DimensionRule]:
    """Check if all pairs satisfy out = in + C for some integer C."""
    if not pairs:
        return None

    # Find C from first pair
    in_dim, out_dim = pairs[0]
    C = out_dim - in_dim

    # Verify all pairs
    for in_d, out_d in pairs:
        if out_d != in_d + C:
            return None

    return DimensionRule(rule_type="additive", param=C)


def _try_fixed(pairs: List[Tuple[int, int]]) -> Optional[DimensionRule]:
    """Check if all outputs are the same constant."""
    if not pairs:
        return None

    # Get constant from first pair
    _, out_dim = pairs[0]

    # Verify all pairs have same output
    for _, out_d in pairs:
        if out_d != out_dim:
            return None

    return DimensionRule(rule_type="fixed", param=out_dim)


def _learn_dimension_rule(pairs: List[Tuple[int, int]]) -> DimensionRule:
    """
    Learn a dimension transformation rule from input/output pairs.

    Tries rules in priority order: Identity > Scale > Additive > Fixed.
    Falls back to identity if no rule fits.

    Args:
        pairs: List of (input_dim, output_dim) pairs from training

    Returns:
        DimensionRule that fits all pairs
    """
    if not pairs:
        return DimensionRule(rule_type="identity")

    # Try rules in priority order
    rule = _try_identity(pairs)
    if rule:
        return rule

    rule = _try_scale(pairs)
    if rule:
        return rule

    rule = _try_additive(pairs)
    if rule:
        return rule

    rule = _try_fixed(pairs)
    if rule:
        return rule

    # No rule fits - fallback to identity (inertia)
    return DimensionRule(rule_type="identity")


def predict_dimensions(task_context: TaskContext) -> List[Tuple[int, int]]:
    """
    Predict output dimensions for all test examples.

    Learns dimension transformation rules from training examples and
    applies them to test inputs. This is "Phase 0" - must run before
    pixel-level solving.

    Algorithm:
      1. Gather (H_in, W_in) -> (H_out, W_out) from all training examples
      2. Learn H rule and W rule independently
      3. Apply rules to each test input to predict (H_out, W_out)

    Args:
        task_context: TaskContext with train/test examples

    Returns:
        List of (predicted_H, predicted_W) for each test example

    Example:
        >>> # Training: 2x2 -> 6x6 (3x scale)
        >>> predictions = predict_dimensions(task_context)
        >>> predictions[0]  # Test input is 3x3
        (9, 9)  # Predicted: 3 * 3 = 9
    """
    # 1. Gather dimension pairs from training
    h_pairs: List[Tuple[int, int]] = []
    w_pairs: List[Tuple[int, int]] = []

    for ex in task_context.train_examples:
        if ex.output_grid is None:
            continue

        h_in, w_in = ex.input_H, ex.input_W
        h_out, w_out = ex.output_H, ex.output_W

        if h_out is not None and w_out is not None:
            h_pairs.append((h_in, h_out))
            w_pairs.append((w_in, w_out))

    # 2. Learn rules for H and W independently
    h_rule = _learn_dimension_rule(h_pairs)
    w_rule = _learn_dimension_rule(w_pairs)

    # 3. Predict for each test example
    predictions: List[Tuple[int, int]] = []

    for ex in task_context.test_examples:
        h_in, w_in = ex.input_H, ex.input_W

        # Apply learned rules
        h_pred = h_rule.apply(h_in)
        w_pred = w_rule.apply(w_in)

        predictions.append((h_pred, w_pred))

    return predictions


if __name__ == "__main__":
    # Self-test with toy examples
    import numpy as np
    from src.schemas.context import build_example_context

    print("=" * 70)
    print("Dimension Predictor self-test")
    print("=" * 70)

    # Test 1: Identity (geometry preserving)
    print("\nTest 1: Identity rule (3x3 -> 3x3)")
    print("-" * 70)

    in1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
    out1 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=int)

    ex1 = build_example_context(in1, out1)
    test_in1 = np.array([[1, 2], [3, 4]], dtype=int)
    test_ex1 = build_example_context(test_in1, None)

    ctx1 = TaskContext(train_examples=[ex1], test_examples=[test_ex1], C=10)

    preds1 = predict_dimensions(ctx1)
    print(f"  Training: {in1.shape} -> {out1.shape}")
    print(f"  Test input: {test_in1.shape}")
    print(f"  Predicted: {preds1[0]}")
    assert preds1[0] == (2, 2), f"Expected (2, 2), got {preds1[0]}"
    print("  Rule: Identity (out = in)")

    # Test 2: Scale (3x scale like task 00576224)
    print("\nTest 2: Scale rule (2x2 -> 6x6, k=3)")
    print("-" * 70)

    in2 = np.zeros((2, 2), dtype=int)
    out2 = np.zeros((6, 6), dtype=int)

    ex2 = build_example_context(in2, out2)
    test_in2 = np.zeros((3, 3), dtype=int)
    test_ex2 = build_example_context(test_in2, None)

    ctx2 = TaskContext(train_examples=[ex2], test_examples=[test_ex2], C=1)

    preds2 = predict_dimensions(ctx2)
    print(f"  Training: {in2.shape} -> {out2.shape}")
    print(f"  Test input: {test_in2.shape}")
    print(f"  Predicted: {preds2[0]}")
    assert preds2[0] == (9, 9), f"Expected (9, 9), got {preds2[0]}"
    print("  Rule: Scale (out = 3 * in)")

    # Test 3: Additive (crop by 2)
    print("\nTest 3: Additive rule (5x5 -> 3x3, C=-2)")
    print("-" * 70)

    in3 = np.zeros((5, 5), dtype=int)
    out3 = np.zeros((3, 3), dtype=int)

    ex3 = build_example_context(in3, out3)
    test_in3 = np.zeros((7, 7), dtype=int)
    test_ex3 = build_example_context(test_in3, None)

    ctx3 = TaskContext(train_examples=[ex3], test_examples=[test_ex3], C=1)

    preds3 = predict_dimensions(ctx3)
    print(f"  Training: {in3.shape} -> {out3.shape}")
    print(f"  Test input: {test_in3.shape}")
    print(f"  Predicted: {preds3[0]}")
    assert preds3[0] == (5, 5), f"Expected (5, 5), got {preds3[0]}"
    print("  Rule: Additive (out = in - 2)")

    # Test 4: Fixed output size
    print("\nTest 4: Fixed rule (various -> 1x1)")
    print("-" * 70)

    in4a = np.zeros((3, 3), dtype=int)
    out4a = np.zeros((1, 1), dtype=int)
    in4b = np.zeros((5, 5), dtype=int)
    out4b = np.zeros((1, 1), dtype=int)

    ex4a = build_example_context(in4a, out4a)
    ex4b = build_example_context(in4b, out4b)
    test_in4 = np.zeros((7, 7), dtype=int)
    test_ex4 = build_example_context(test_in4, None)

    ctx4 = TaskContext(train_examples=[ex4a, ex4b], test_examples=[test_ex4], C=1)

    preds4 = predict_dimensions(ctx4)
    print(f"  Training 1: {in4a.shape} -> {out4a.shape}")
    print(f"  Training 2: {in4b.shape} -> {out4b.shape}")
    print(f"  Test input: {test_in4.shape}")
    print(f"  Predicted: {preds4[0]}")
    assert preds4[0] == (1, 1), f"Expected (1, 1), got {preds4[0]}"
    print("  Rule: Fixed (out = 1)")

    # Test 5: Multiple training examples with consistent scale
    print("\nTest 5: Multiple training examples (consistent 2x scale)")
    print("-" * 70)

    in5a = np.zeros((2, 2), dtype=int)
    out5a = np.zeros((4, 4), dtype=int)
    in5b = np.zeros((3, 3), dtype=int)
    out5b = np.zeros((6, 6), dtype=int)

    ex5a = build_example_context(in5a, out5a)
    ex5b = build_example_context(in5b, out5b)
    test_in5 = np.zeros((4, 4), dtype=int)
    test_ex5 = build_example_context(test_in5, None)

    ctx5 = TaskContext(train_examples=[ex5a, ex5b], test_examples=[test_ex5], C=1)

    preds5 = predict_dimensions(ctx5)
    print(f"  Training 1: {in5a.shape} -> {out5a.shape}")
    print(f"  Training 2: {in5b.shape} -> {out5b.shape}")
    print(f"  Test input: {test_in5.shape}")
    print(f"  Predicted: {preds5[0]}")
    assert preds5[0] == (8, 8), f"Expected (8, 8), got {preds5[0]}"
    print("  Rule: Scale (out = 2 * in)")

    print("\n" + "=" * 70)
    print("Dimension Predictor self-test passed.")
    print("=" * 70)
