"""Prompt and answer utilities for the offline math PUMA pipeline."""


INSTRUCTION_MATH = (
    "Solve the following math problem. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def get_model_type(model_name: str) -> str:
    """Return the model family used for chat-template handling."""
    model_lower = model_name.lower()

    if "deepseek" in model_lower and "r1" in model_lower and "qwen3" in model_lower:
        return "qwen3"
    if "deepseek" in model_lower and "r1" in model_lower:
        return "deepseek_r1"
    if "qwen3" in model_lower and "thinking" in model_lower:
        return "qwen3_thinking"
    if "qwq" in model_lower:
        return "qwq"
    if "qwen3" in model_lower:
        return "qwen3"
    if "nemotron" in model_lower:
        return "nemotron"
    return "other"


def build_base_prompt(tokenizer, model_name: str, question: str) -> str:
    """
    Build the chat-formatted math prompt.

    The returned prompt ends at the assistant generation marker and does not
    include a ``<think>`` block. Callers append the reasoning prefix themselves.
    """
    model_type = get_model_type(model_name)
    user_content = f"{INSTRUCTION_MATH}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": user_content}]

    if model_type == "nemotron":
        messages.insert(0, {"role": "system", "content": "detailed thinking on"})

    try:
        if model_type == "qwen3":
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        return stripped[: stripped.rfind("<think>")]
    return prompt


def get_confident_ending(is_trial: bool = True) -> str:
    """Return the short answer cue used after a reasoning prefix."""
    if is_trial:
        return "\n**Final Answer**\n\nThe final answer is \\boxed"
    return "\nI believe I have obtained the answer. I will provide my answer in \\boxed{}."


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last ``\\boxed{...}``, including nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1].strip()
    return None
