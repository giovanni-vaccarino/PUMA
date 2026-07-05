"""
Dataset-specific evaluation utilities for PUMA-VL.

Dispatches to appropriate grading logic based on dataset and question_type.
"""

import os
import re
import sys

# Import DEER math grader from text-only pipeline
_OFFLINE_DIR = os.path.join(os.path.dirname(__file__), "..", "puma")
if _OFFLINE_DIR not in sys.path:
    sys.path.insert(0, _OFFLINE_DIR)
from math_grader import check_is_correct as math_check_is_correct  # noqa: E402


def normalize_number(s: str) -> str:
    """Normalize a numeric answer string for comparison."""
    s = s.strip().rstrip(".").strip()
    # Remove trailing zeros after decimal point
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    # Remove commas
    s = s.replace(",", "")
    return s


def _extract_letter(s: str) -> str:
    """Extract a single choice letter from a string."""
    s = s.strip().upper()
    if len(s) == 1 and s in "ABCDEFGHIJ":
        return s
    m = re.search(r"\b([A-J])\b", s)
    if m:
        return m.group(1)
    return s


def check_mcq_answer(prediction: str, ground_truth: str) -> bool:
    """
    Check multiple-choice answer.

    Handles both cases:
      - Both are letters: direct letter match (A == A)
      - GT is content, pred is letter: try content match via math_grader
      - Both are content: content match via math_grader
    """
    pred = prediction.strip()
    gt = ground_truth.strip()

    if not pred or not gt:
        return False

    pred_letter = _extract_letter(pred)
    gt_letter = _extract_letter(gt)

    # Both are single letters → direct match
    if (len(pred_letter) == 1 and pred_letter in "ABCDEFGHIJ"
            and len(gt_letter) == 1 and gt_letter in "ABCDEFGHIJ"):
        return pred_letter == gt_letter

    # One is letter, other is content → try content comparison
    # (handles case where GT wasn't converted to letter in dataset_utils)
    if len(pred_letter) == 1 and pred_letter in "ABCDEFGHIJ":
        # pred is letter, gt is content — can't compare directly
        # Try if gt also contains the letter
        return pred_letter == gt_letter

    # Both are content → math grader
    return math_check_is_correct(pred, gt)


def check_is_correct_vl(
    prediction: str,
    ground_truth: str,
    dataset: str,
    question_type: str = "",
) -> bool:
    """
    Unified VL evaluation dispatcher.

    Args:
        prediction: Model's extracted answer
        ground_truth: Ground truth answer
        dataset: Dataset name (mathvista, mathvision, mmmu-pro)
        question_type: "free_form" or "multi_choice" (for MathVista)

    Returns:
        True if the prediction matches the ground truth
    """
    if not prediction or not ground_truth:
        return False

    ds = dataset.lower().replace("_", "-")

    # MMMU-Pro: always MCQ
    if "mmmu" in ds:
        return check_mcq_answer(prediction, ground_truth)

    # MathVista: mixed
    if "mathvista" in ds:
        if question_type == "multi_choice":
            return check_mcq_answer(prediction, ground_truth)
        # free_form: try numeric comparison first, then math grader
        pred_norm = normalize_number(prediction)
        gt_norm = normalize_number(ground_truth)
        if pred_norm and gt_norm:
            try:
                if abs(float(pred_norm) - float(gt_norm)) < 1e-6:
                    return True
            except ValueError:
                pass
        return math_check_is_correct(prediction, ground_truth)

    # MathVision: free-form math
    if "mathvision" in ds:
        pred_norm = normalize_number(prediction)
        gt_norm = normalize_number(ground_truth)
        if pred_norm and gt_norm:
            try:
                if abs(float(pred_norm) - float(gt_norm)) < 1e-6:
                    return True
            except ValueError:
                pass
        return math_check_is_correct(prediction, ground_truth)

    # Default: math grader
    return math_check_is_correct(prediction, ground_truth)
