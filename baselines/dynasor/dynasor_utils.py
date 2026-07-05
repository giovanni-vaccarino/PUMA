"""
Dynasor utility functions.

Ported from the Dynasor repository:
  - dynasor/core/cot.py: obtain_answer, effort_level, uncertain_words
  - dynasor/core/entropy.py: eqaul_group, count_not_empty, should_early_exit, is_certain_answer

Reference: https://github.com/nicholaschenai/Dynasor
"""

from baselines.utils.math_util import fuzzy_code_match


# ---- From dynasor/core/cot.py ----

uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]

# Default probe suffix used to induce an answer mid-reasoning
PROBE_SUFFIX = (
    "... Oh, I suddenly got the answer to the whole problem, "
    "**Final Answer**\n\n\\[ \\boxed{"
)

# Code-specific probe suffix
PROBE_SUFFIX_CODE = (
    "... Oh, I suddenly got the complete solution.\n"
    "### Solution Code\n```python\n"
)


def effort_level(level: str) -> tuple:
    """
    Convert effort level string to (threshold, chunk_size) tuple.

    threshold: number of consecutive consistent answers required for early exit
    chunk_size: number of tokens to generate between probes

    From dynasor/core/cot.py
    """
    levels = {
        "mild":  (8, 64),
        "low":   (5, 64),
        "mid":   (3, 64),
        "high":  (2, 64),
        "crazy": (2, 32),
    }
    if level not in levels:
        raise ValueError(f"Invalid effort level: {level}. Choose from {list(levels.keys())}")
    return levels[level]


def obtain_answer(s):
    """
    Extract answer from probe response text.
    Finds content before the first unpaired '}'.

    The probe suffix ends with '\\boxed{', so the model generates the answer
    followed by '}'. This function extracts everything before that closing '}'.

    From dynasor/core/cot.py
    """
    stack = []
    for i, c in enumerate(s):
        if c == "{":
            stack.append(c)
        elif c == "}":
            if not stack:  # No matching { found — first unpaired }
                return s[:i]
            stack.pop()
    return ""


def obtain_code_answer(s):
    """Extract code from probe response (already inside ```python block).
    Stop at ``` or end of string."""
    if '```' in s:
        return s[:s.index('```')].strip()
    return s.strip()


# ---- From dynasor/core/entropy.py ----

def count_not_empty(answers):
    """Count non-empty answers in list. From dynasor/core/entropy.py"""
    return sum(1 for answer in answers if answer != "")


def eqaul_group(answers):
    """
    Check if all answers in the list are equivalent.

    Note: The original Dynasor uses math_equal() for mathematical equivalence
    (e.g., '1/2' == '0.5'). Here we use simple string comparison, which is
    sufficient for AIME-style integer answers. For more complex benchmarks,
    consider importing a math_equal implementation.

    From dynasor/core/entropy.py (simplified)
    """
    if not answers:
        return False
    first = answers[0]
    return all(a == first for a in answers)


def is_certain_answer(probe_response_text: str, words: list = None) -> bool:
    """
    Check if the probe response does NOT contain any uncertain/hesitation words.
    Returns True if the answer appears certain (no hesitation detected).

    From dynasor/core/entropy.py
    """
    if words is None:
        words = uncertain_words
    return not any(word in probe_response_text.lower() for word in words)


def should_early_exit(
    answers: list,
    is_certains: list,
    threshold: int,
) -> bool:
    """
    Determine whether to early exit based on answer consistency and certainty.

    Conditions (ALL must be true):
    1. Number of answers >= threshold
    2. Last `threshold` answers are all the same (eqaul_group)
    3. Last `threshold` answers are all non-empty
    4. Last `threshold` probe responses are all certain (no hesitation words)

    From dynasor/core/entropy.py (should_early_exit)
    """
    if len(answers) < threshold:
        return False

    answer_window = answers[-threshold:]
    certain_window = is_certains[-threshold:]

    if not eqaul_group(answer_window):
        return False
    if count_not_empty(answer_window) != threshold:
        return False
    if sum(certain_window) != threshold:
        return False

    return True


def should_early_exit_code(
    codes: list,
    is_certains: list,
    threshold: int,
    match_threshold: float = 0.8,
) -> bool:
    """Early exit for code tasks using fuzzy code matching."""
    if len(codes) < threshold:
        return False

    code_window = codes[-threshold:]
    certain_window = is_certains[-threshold:]

    # All must be non-empty
    if any(not c for c in code_window):
        return False
    # All must be certain
    if sum(certain_window) != threshold:
        return False
    # All pairs must fuzzy-match
    for i in range(1, len(code_window)):
        if not fuzzy_code_match(code_window[0], code_window[i], match_threshold):
            return False
    return True