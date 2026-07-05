"""
VL prompt construction utilities for the PUMA-VL pipeline.

Extends puma/prompt_utils.py for vision-language models.
Key difference: prompts include image content blocks, and we use
AutoProcessor (not AutoTokenizer) for chat template application.

Supported model families:
  - Qwen3-VL-*-Thinking (qwen3_vl_thinking)
  - Kimi-VL-A3B-Thinking (kimi_vl_thinking)
"""

import re
import os
import sys

from PIL import Image

# Import shared utilities from text-only pipeline
_OFFLINE_DIR = os.path.join(os.path.dirname(__file__), "..", "puma")
if _OFFLINE_DIR not in sys.path:
    sys.path.insert(0, _OFFLINE_DIR)
from prompt_utils import extract_boxed_answer, extract_code_answer  # noqa: E402


# =============================================================================
# VL Task Type & Model Type
# =============================================================================

VL_DATASETS = {
    "mathvista": "math_vl",
    "mathvision": "math_vl",
    "mmmu-pro": "mcq_vl",
}


def get_task_type_vl(dataset: str) -> str:
    """Determine VL task type from dataset name."""
    ds_lower = dataset.lower().replace("_", "-")
    return VL_DATASETS.get(ds_lower, "math_vl")


def get_model_type_vl(model_name: str) -> str:
    """
    Determine VL model type for prompt construction.

    Returns:
      - "qwen3_vl_thinking": Qwen3-VL-*-Thinking models
      - "kimi_vl_thinking": Kimi-VL-A3B-Thinking models
      - "qwen_vl": Generic Qwen-VL (non-thinking)
      - "other_vl": Unknown VL models
    """
    m = model_name.lower()
    if "qwen3-vl" in m and "thinking" in m:
        return "qwen3_vl_thinking"
    if "kimi" in m and "vl" in m and "thinking" in m:
        return "kimi_vl_thinking"
    if "qwen" in m and "vl" in m:
        return "qwen_vl"
    return "other_vl"


# =============================================================================
# Instructions
# =============================================================================

INSTRUCTION_MATH_VL = (
    "Solve the following math problem based on the given image. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

INSTRUCTION_MCQ_VL = (
    "Answer the following multiple choice question based on the given image. "
    "Please reason step by step, and put your choice letter (e.g. A, B, C, or D) "
    "without any other text within \\boxed{} in the end."
)

INSTRUCTION_MATH_VL_DIRECT = (
    "Solve the following math problem based on the given image. "
    "Directly output your final answer within \\boxed{}. DO NOT say anything else."
)

INSTRUCTION_MCQ_VL_DIRECT = (
    "Answer the following multiple choice question based on the given image. "
    "Directly output your choice letter within \\boxed{}. DO NOT say anything else."
)


def get_instruction_vl(task_type: str, prompt_version: str = "default") -> str:
    """Get instruction for VL task."""
    if task_type == "mcq_vl":
        if prompt_version == "direct":
            return INSTRUCTION_MCQ_VL_DIRECT
        return INSTRUCTION_MCQ_VL
    else:
        if prompt_version == "direct":
            return INSTRUCTION_MATH_VL_DIRECT
        return INSTRUCTION_MATH_VL


# =============================================================================
# Prompt Construction
# =============================================================================

def build_vl_messages(question_text: str, image_path: str, instruction: str):
    """
    Build chat messages with image content block.

    Returns list of message dicts suitable for processor.apply_chat_template().
    """
    content = [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
        {"type": "text", "text": f"{instruction}\n\nQuestion: {question_text}"},
    ]
    return [{"role": "user", "content": content}]


def build_base_prompt_vl(
    processor,
    model_name: str,
    question: str,
    image_path: str,
    task_type: str,
    prompt_version: str = "default",
) -> str:
    """
    Build the base VL prompt using the processor's chat template.

    Returns prompt text string (with image placeholder tokens embedded).
    The actual image must be passed separately via multi_modal_data.

    Strips trailing <think> if auto-added by the template.
    """
    model_type = get_model_type_vl(model_name)
    instruction = get_instruction_vl(task_type, prompt_version)
    messages = build_vl_messages(question, image_path, instruction)

    try:
        if model_type == "qwen3_vl_thinking":
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        elif model_type == "kimi_vl_thinking":
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    except TypeError as e:
        if "enable_thinking" in str(e):
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            raise

    # Strip trailing <think> if auto-added
    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        prompt = stripped[:stripped.rfind("<think>")]

    return prompt


def load_image(image_path: str) -> Image.Image:
    """Load and return a PIL Image, converting to RGB if needed.

    Copies pixel data and closes the file handle to avoid 'Too many open files'.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.load()  # force read pixels into memory, close file handle
    return img


# =============================================================================
# Confident Endings (same logic as text, adapted for VL task types)
# =============================================================================

def get_confident_ending_vl(dataset: str, is_trial: bool = True) -> str:
    """
    Get confident ending string for trial/prefixed answer prompts.

    Args:
        dataset: VL dataset name
        is_trial: True for Step 2 (trial answers), False for Step 4 (prefixed)
    """
    task_type = get_task_type_vl(dataset)

    if is_trial:
        if task_type == "mcq_vl":
            return "\n**Final Answer**\n\nThe answer choice is \\boxed"
        else:
            return "\n**Final Answer**\n\nThe final answer is \\boxed"
    else:
        if task_type == "mcq_vl":
            return "I believe I have obtained the answer. My answer is \\boxed{"
        else:
            return "\nI believe I have obtained the answer. I will provide my answer in \\boxed{}."


# =============================================================================
# Answer Extraction
# =============================================================================

def _extract_mathbf(text: str) -> str:
    """Extract content from the last \\mathbf{...} or \\textbf{...}."""
    for cmd in ["\\mathbf{", "\\textbf{"]:
        idx = text.rfind(cmd)
        if idx == -1:
            continue
        start = idx + len(cmd)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
    return ""


def extract_answer_vl(text: str, dataset: str) -> str:
    """Extract answer from generated text, dataset-aware.

    Tries \\boxed{} first, then falls back to \\mathbf{}/\\textbf{},
    **Answer:** patterns, and MCQ-specific heuristics.
    """
    task_type = get_task_type_vl(dataset)

    # Try \boxed{...} first
    answer = extract_boxed_answer(text)

    # Fallback: \mathbf{} or \textbf{} (Kimi-VL often uses these)
    if not answer:
        answer = _extract_mathbf(text)

    # Fallback: **Answer:** X or **Answer: X** patterns
    if not answer:
        m = re.search(r"\*\*(?:Answer|ANSWER|Final Answer)[:\s]*\*?\*?\s*(.+?)(?:\n|$)", text)
        if m:
            ans_text = m.group(1).strip().rstrip(".")
            # Clean markdown bold
            ans_text = re.sub(r"\*\*", "", ans_text).strip()
            if ans_text:
                answer = ans_text

    if task_type == "mcq_vl":
        # For MCQ, if answer is a valid letter, return it
        if answer and answer.strip().upper() in "ABCDEFGHIJ":
            return answer.strip().upper()
        # Try to extract letter from longer answer
        if answer:
            m = re.search(r"\b([A-J])\b", answer)
            if m:
                return m.group(1).upper()
        # Fallback: ANSWER: X pattern in full text
        m = re.search(r"(?:ANSWER|answer|Answer)\s*[:=]\s*([A-J])", text)
        if m:
            return m.group(1).upper()
        # Fallback: last standalone letter
        m = re.search(r"\b([A-D])\b\s*$", text.strip())
        if m:
            return m.group(1).upper()
        return answer or ""

    return answer or ""
