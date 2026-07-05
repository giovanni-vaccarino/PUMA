import difflib
import re
from collections import Counter

from baselines.utils.math_equivalence import is_equiv


# =============================================================================
# Code task constants (shared across baselines)
# =============================================================================

CODE_SYS_PROMPT = (
    "You are a helpful assistant that solves programming problems. "
    "Think step by step, then provide your final solution as a complete "
    "Python program within a ```python code block."
)

CODE_ANSWER_SUFFIX = "\n### Solution Code\n```python\n"


# =============================================================================
# Code answer extraction and matching
# =============================================================================

def extract_code_answer(text):
    """Extract the last Python code block from generated text."""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def normalize_code(code):
    """Normalize code for fuzzy comparison."""
    lines = code.strip().splitlines()
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def fuzzy_code_match(code_a, code_b, threshold=0.8):
    """Check if two code snippets are similar enough."""
    a = normalize_code(code_a)
    b = normalize_code(code_b)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def extract_answer(text, dataset=None):
    """Route to code or math answer extraction based on dataset."""
    if dataset and "livecodebench" in dataset.lower():
        return extract_code_answer(text)
    return my_answer_extraction(text, dataset=dataset)

def remove_boxed(s):
    left = "boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("boxed")
    if idx < 0:
        idx = string.rfind("fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def last_number(output):
    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return output

def extract_multi_choice_answer(solution):
    """Extract multi-choice answer (A/B/C/D/E) with multiple fallback patterns."""
    valid_choices = {"A", "B", "C", "D", "E"}

    # 1. Try \boxed{} first
    boxed = remove_boxed(last_boxed_only_string(solution))
    if boxed and boxed.strip().upper() in valid_choices:
        return boxed.strip().upper()

    # 2. Try "answer is (X)" or "answer is X" pattern
    m = re.search(r"answer\s+is\s*[\(\[]?\s*([A-E])\s*[\)\]]?", solution, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. Try "ANSWER: X" pattern
    m = re.search(r"ANSWER\s*:\s*([A-E])", solution)
    if m:
        return m.group(1)

    # 4. Look in final section (after </think> or last 500 chars) for standalone A/B/C/D
    final_section = solution.split("</think>")[-1] if "</think>" in solution else solution[-500:]
    choices = re.findall(r"\b([A-E])\b", final_section)
    if choices:
        return choices[-1]

    # 5. Return boxed content as-is (may still match via grader)
    return boxed if boxed else ""


def my_answer_extraction(solution, dataset=None):
    """
    Extract answer from solution text.
    For GPQA (multi-choice), uses specialized extraction.
    For math, extracts from \\boxed{} or falls back to last number.
    """
    if dataset and "gpqa" in dataset.lower():
        return extract_multi_choice_answer(solution)

    boxed_answer = remove_boxed(last_boxed_only_string(solution))
    if boxed_answer:
        return boxed_answer
    else:
        return last_number(solution)


def contains_answer(solution, answer):
    """
    solution : str
    """
    pred = my_answer_extraction(solution)
    return int(is_equiv(pred, answer))

if __name__ == "__main__":
    solution = "\\boxed{2x-4} final answer: \\boxed{0.05}"
    pred = my_answer_extraction(solution)
    print(pred)
    print(int(is_equiv(pred, "0.050000001")))