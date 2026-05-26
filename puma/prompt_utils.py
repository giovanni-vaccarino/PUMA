"""
Shared prompt construction utilities for the PUMA pipeline.

Used by:
- method/offline/run_vllm.py (Step 1a: answer generation)
- method/offline/gen_trial_answers.py (Step 2: trial answers)
- method/offline/gen_prefixed_answers.py (Step 4: prefixed answers)
"""

import difflib
import re


# =============================================================================
# Instruction constants
# =============================================================================

INSTRUCTION_MATH = (
    "Solve the following math problem. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

INSTRUCTION_MATH_DIRECT = (
    "Solve the following math problem. "
    "Directly output your final answer within \\boxed{}. DO NOT say anything else."
)

INSTRUCTION_MATH_WO_STEPBYSTEP = (
    "Solve the following math problem. "
    "Please solve this problem, and put your final answer within \\boxed{}."
)

INSTRUCTION_NQ = (
    "Answer the following question. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

INSTRUCTION_NQ_DIRECT = (
    "Answer the following question. "
    "Directly output your final answer within \\boxed{}. DO NOT say anything else."
)

INSTRUCTION_NQ_WO_STEPBYSTEP = (
    "Answer the following question. "
    "Please solve this problem, and put your final answer within \\boxed{}."
)

INSTRUCTION_GPQA = (
    "Answer the following multiple choice question. "
    "Please reason step by step, and put your choice letter (A, B, C, or D) "
    "without any other text within \\boxed{} in the end."
)

INSTRUCTION_GPQA_DIRECT = (
    "Answer the following multiple choice question. "
    "Directly output your choice letter (A, B, C, or D) within \\boxed{}. DO NOT say anything else."
)

INSTRUCTION_GPQA_WO_STEPBYSTEP = (
    "Answer the following multiple choice question. "
    "Put your choice letter (A, B, C, or D) without any other text within \\boxed{} in the end."
)

INSTRUCTION_CODE = (
    "You are a helpful assistant that solves programming problems. "
    "Think step by step, then provide your final solution as a complete "
    "Python program within a ```python code block."
)

# Aligned with DEER: unified instruction for all datasets
INSTRUCTION_DEER = "Please reason step by step, and put your final answer within \\boxed{}."


# =============================================================================
# Task type and model type detection
# =============================================================================

MATH_DATASETS = {"aime24", "aime25", "amc23", "gsm8k", "math-500", "olympiadbench"}
CODE_DATASETS = {"livecodebench"}


def get_task_type(dataset: str) -> str:
    """Determine task type from dataset name."""
    if "gpqa" in dataset.lower():
        return "gpqa"
    elif dataset in MATH_DATASETS:
        return "math"
    elif any(dataset.startswith(cd) for cd in CODE_DATASETS):
        return "code"
    else:
        return "nq"


def get_model_type(model_name: str) -> str:
    """
    Determine model type for chat template handling.

    Returns:
        - "qwen3_thinking": Qwen3-*-Thinking-* models (always thinking, no enable_thinking needed)
        - "qwen3": Qwen3-*B models (need enable_thinking=True)
        - "qwq": QwQ-32B (pure thinking model)
        - "deepseek_r1": DeepSeek-R1-Distill models
        - "nemotron": Nvidia Nemotron models (need system prompt "detailed thinking on")
        - "other": Unknown models
    """
    model_lower = model_name.lower()

    # DeepSeek-R1-0528-Qwen3-8B is special: it's a DeepSeek R1 model but uses
    # the Qwen3 chat template with enable_thinking=True ("runs like Qwen3-8B").
    # Must check BEFORE general deepseek_r1 match.
    if "deepseek" in model_lower and "r1" in model_lower and "qwen3" in model_lower:
        return "qwen3"
    elif "deepseek" in model_lower and "r1" in model_lower:
        return "deepseek_r1"
    elif "qwen3" in model_lower and "thinking" in model_lower:
        return "qwen3_thinking"
    elif "qwq" in model_lower:
        return "qwq"
    elif "qwen3" in model_lower:
        return "qwen3"
    elif "nemotron" in model_lower:
        return "nemotron"
    else:
        return "other"


def get_instruction(task_type: str, prompt_version: str = "default", is_trial: bool = None) -> str:
    """Get instruction string based on task type and prompt version.

    Args:
        task_type: "math", "gpqa", or "nq"
        prompt_version: "default", "direct", "wo_stepbystep", or "aligned_with_deer"
        is_trial: Unused, kept for API compatibility.
    """
    if prompt_version == "aligned_with_deer":
        return INSTRUCTION_DEER
    if task_type == "code":
        return INSTRUCTION_CODE
    elif task_type == "gpqa":
        if prompt_version == "direct":
            return INSTRUCTION_GPQA_DIRECT
        elif prompt_version == "wo_stepbystep":
            return INSTRUCTION_GPQA_WO_STEPBYSTEP
        else:
            return INSTRUCTION_GPQA
    elif task_type == "nq":
        if prompt_version == "direct":
            return INSTRUCTION_NQ_DIRECT
        elif prompt_version == "wo_stepbystep":
            return INSTRUCTION_NQ_WO_STEPBYSTEP
        else:
            return INSTRUCTION_NQ
    else:
        if prompt_version == "direct":
            return INSTRUCTION_MATH_DIRECT
        elif prompt_version == "wo_stepbystep":
            return INSTRUCTION_MATH_WO_STEPBYSTEP
        else:
            return INSTRUCTION_MATH


# =============================================================================
# Prompt construction
# =============================================================================

def build_base_prompt(
    tokenizer,
    model_name: str,
    question: str,
    task_type: str,
    prompt_version: str = "default",
    is_trial: bool = None,
) -> str:
    """
    Build the base prompt with chat template and instruction.

    Returns a prompt ending at the assistant generation start, WITHOUT <think>.
    This allows callers to consistently append their own <think> block.

    Note: Some models (QwQ-32B, DeepSeek-R1-Distill) auto-add <think> in their
    chat template when add_generation_prompt=True. This function strips it.

    Args:
        tokenizer: HuggingFace tokenizer
        model_name: Model name/path
        question: Raw question text
        task_type: "math", "gpqa", or "nq"
        prompt_version: "default", "direct", "wo_stepbystep", or "aligned_with_deer"
        is_trial: Unused, kept for API compatibility.

    Returns:
        Formatted prompt string (without trailing <think>)
    """
    model_type = get_model_type(model_name)
    instruction = get_instruction(task_type, prompt_version, is_trial=is_trial)

    if prompt_version == "aligned_with_deer":
        # DEER format: system message + user message (question only)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question}
        ]
    else:
        # PUMA default format: instruction + question in user message
        user_content = f"{instruction}\n\nQuestion: {question}"
        messages = [{"role": "user", "content": user_content}]

    # Nemotron models require system prompt to enable thinking mode
    if model_type == "nemotron":
        if messages[0]["role"] == "system":
            messages[0]["content"] = "detailed thinking on\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": "detailed thinking on"})

    try:
        if model_type == "qwen3":
            # Qwen3 models: need enable_thinking=True
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
        else:
            # All other models: standard chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    except TypeError as e:
        # Some tokenizers don't support enable_thinking parameter
        if "enable_thinking" in str(e):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise e

    # Strip trailing <think> if auto-added by chat template
    # (QwQ-32B, DeepSeek-R1-Distill auto-add it; Qwen3 does not)
    # This ensures callers can consistently add their own <think> block
    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        prompt = stripped[: stripped.rfind("<think>")]

    return prompt


def get_confident_ending(dataset: str, is_trial: bool = True) -> str:
    """Get the confident ending string for trial/prefixed answer prompts.

    Args:
        dataset: Dataset name (e.g. "gpqa-diamond", "aime24")
        is_trial: True for Step 2 (trial answers, model completes \\boxed{X}),
                  False for Step 4 (prefixed answers, </think> follows)
    """
    task_type = get_task_type(dataset)

    if task_type == "code":
        if is_trial:
            # Step 2: close think block and start code block for model to fill
            return "</think>\n\n### Solution Code\n```python\n"
        else:
            # Step 4: no confident ending, </think> is added by caller
            return ""

    if is_trial:
        # Step 2: open-ended, model continues with {X}
        if "gpqa" in dataset.lower():
            return "\n**Final Answer**\n\nThe answer choice is \\boxed"
        else:
            return "\n**Final Answer**\n\nThe final answer is \\boxed"
    else:
        # Step 4 (prefixed answers)
        if "gpqa" in dataset.lower():
            # GPQA: placed after </think>, opens \boxed{ for model to fill answer directly
            return "I believe I have obtained the answer. My answer is \\boxed{"
        else:
            # Other datasets: original format, model generates freely after </think>
            return "\nI believe I have obtained the answer. I will provide my answer in \\boxed{}."


def extract_boxed_answer(text: str) -> str:
    """
    Extract content from the last \\boxed{...} with proper brace matching.

    Handles nested braces correctly, e.g. \\boxed{\\frac{1}{2}} -> \\frac{1}{2}
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
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
    return None


def extract_code_answer(text: str) -> str:
    """Extract the last Python code block from generated text."""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def extract_trial_code(text: str) -> str:
    """Extract code from trial answer generation (already inside code block).

    In trial answer mode for code tasks, the prompt ends with ```python\\n
    so the model output is already inside a code block. Extract until ``` or end.
    """
    if '```' in text:
        return text[:text.index('```')].strip()
    return text.strip()


def normalize_code(code: str) -> str:
    """Normalize code for fuzzy comparison."""
    lines = code.strip().splitlines()
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def fuzzy_code_match(code_a: str, code_b: str, threshold: float = 0.8) -> bool:
    """Check if two code snippets are similar enough."""
    a = normalize_code(code_a)
    b = normalize_code(code_b)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold
