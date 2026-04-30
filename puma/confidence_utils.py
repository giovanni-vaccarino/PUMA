"""Shared confidence and logprob utilities used by gen_trial_answers and gen_prefixed_answers."""

import math
from typing import List


def clamp_logprobs(log_probs: List[float], threshold: float = -0.001) -> List[float]:
    """
    Clamp logprobs close to 0 to exactly 0.0, simulating vLLM V0 (0.3.x) behavior.

    vLLM V0 appears to truncate/round very small negative logprobs to 0.0,
    while V1 returns more precise values. This causes confidence calculation
    differences that affect PUMA compression rates.

    Args:
        log_probs: List of log probabilities (negative values)
        threshold: Logprobs greater than this threshold will be clamped to 0.0
                   Default -0.001 means logprobs > -0.001 become 0.0

    Returns:
        List of clamped log probabilities
    """
    return [0.0 if lp > threshold else lp for lp in log_probs]


def compute_geometric_confidence(log_probs: List[float]) -> float:
    """Geometric mean confidence for answer token log-probabilities."""
    if not log_probs:
        return 0.0
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(avg_log_prob)


def compute_arithmetic_confidence(log_probs: List[float]) -> float:
    """Arithmetic mean confidence for answer token log-probabilities."""
    if not log_probs:
        return 0.0
    return sum(math.exp(lp) for lp in log_probs) / len(log_probs)


def compute_confidence(log_probs: List[float], aggregation: str = "geometric") -> float:
    """Dispatch to geometric or arithmetic mean confidence."""
    if aggregation == "arithmetic":
        return compute_arithmetic_confidence(log_probs)
    return compute_geometric_confidence(log_probs)


def construct_reasoning_prefix(reasoning_steps: List[str]) -> str:
    """Concatenate reasoning steps with spacing."""
    return "\n\n".join(reasoning_steps)


def extract_logprob_from_step(step) -> float:
    """Extract a single logprob value from a vLLM logprobs step entry."""
    if isinstance(step, float):
        return step
    elif isinstance(step, dict):
        val = next(iter(step.values()))
        if hasattr(val, "logprob"):
            return val.logprob
        elif isinstance(val, float):
            return val
        else:
            try:
                return float(val)
            except Exception:
                return 0.0
    else:
        try:
            return float(step)
        except Exception:
            return 0.0
