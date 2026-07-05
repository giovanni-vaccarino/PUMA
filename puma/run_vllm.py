import argparse
import json
import os
import re
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add project root to sys.path for baselines import
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from baselines.utils.math_util import my_answer_extraction
from transformers import AutoTokenizer
from prompt_utils import (
    get_model_type,
    get_instruction,
    get_task_type,
    build_base_prompt,
    extract_code_answer,
    INSTRUCTION_DEER,
)


def split_reasoning_and_raw(text, tokenizer, max_tokens=32768):
    # Remove leading <think> if present
    cleaned = text.lstrip()
    if cleaned.startswith("<think>"):
        cleaned = cleaned[len("<think>"):]

    if "</think>" in cleaned:
        reasoning, rest = cleaned.split("</think>", 1)
        return reasoning.strip(), rest.strip()

    # No </think> found: check if output was truncated by max_tokens
    token_count = len(tokenizer.encode(text))
    if token_count >= max_tokens * 0.9:
        # Output likely truncated (>90% of max_tokens), treat as incomplete reasoning
        return text.strip(), ""
    else:
        # Short output without </think>, treat everything as final answer
        return "", text.strip()


# Instruction constants and get_model_type/get_instruction/get_task_type
# are imported from prompt_utils


def extract_answer(generated_text: str, task_type: str) -> str:
    """Extract answer from generated text, with task-specific handling."""
    if task_type == "code":
        return extract_code_answer(generated_text)
    answer = my_answer_extraction(generated_text)
    if task_type == "gpqa" and answer not in ("A", "B", "C", "D"):
        # Fallback: try to extract from "ANSWER: X" pattern
        m = re.search(r"ANSWER\s*:\s*([A-D])", generated_text)
        if m:
            answer = m.group(1)
    return answer


# get_model_type() and get_instruction() imported from prompt_utils


def build_prompts_with_chat_template(questions: list, tokenizer, model_name: str, task_type: str = "math", prompt_version: str = "default") -> list:
    """
    Build prompts using chat template for proper thinking mode activation.

    Args:
        questions: List of question strings
        tokenizer: HuggingFace tokenizer
        model_name: Model name/path
        task_type: "math" or "nq"
        prompt_version: "default", "direct", or "wo_stepbystep"

    Returns:
        List of formatted prompts
    """
    model_type = get_model_type(model_name)

    instruction = get_instruction(task_type, prompt_version)

    prompts = []

    for question in questions:
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
            if model_type == "qwen3_thinking":
                # Qwen3-*-Thinking models: thinking is always on, no enable_thinking needed
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif model_type == "qwen3":
                # Qwen3 models: need enable_thinking=True
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
            elif model_type == "qwq":
                # QwQ-32B: pure thinking model, add_generation_prompt=True enforces <think>
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif model_type == "deepseek_r1":
                # DeepSeek R1 models: use chat template
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: try chat template, if fails use simple format
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"{instruction}\n\nQuestion: {question}"

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

        prompts.append(prompt)

    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--answer_max_tokens", type=int, default=4096,
                        help="Max tokens for answer regeneration when thinking is truncated")
    parser.add_argument("--use_raw_prompt", action="store_true",
                        help="Use raw prompt without chat template (legacy mode)")
    parser.add_argument("--prompt-version", type=str, default="default",
                        choices=["default", "direct", "wo_stepbystep", "aligned_with_deer"],
                        help="Prompt version: default (step by step), direct (answer only), wo_stepbystep (no step by step), aligned_with_deer (DEER system+user format)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: from model's generation_config)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override top_p (default: from model's generation_config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for vLLM engine (default: None)")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    dataset_path = args.dataset_path
    output_path = args.output_path
    dataset_limit = args.limit

    task_type = get_task_type(args.dataset)

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    print(f"Using {tp_size} GPUs for tensor parallelism")

    print(f"Loading model: {model_name}...")
    model_type = get_model_type(model_name)
    print(f"Detected model type: {model_type}")

    # max_model_len needs to accommodate fix prompts for truncated thinking:
    # fix prompt = chat_template(~200) + thinking(~max_tokens) + "</think>\n\n" + answer_max_tokens
    max_model_len = args.max_tokens + 1024 + args.answer_max_tokens

    seed_kwargs = {"seed": args.seed} if args.seed is not None else {}
    if args.seed is not None:
        print(f"Using seed: {args.seed}")

    if "70B" not in model_name:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.85,  # Avoid OOM on long sequences (e.g. math-500)
            **seed_kwargs,
        )
    else:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            dtype="auto",
            max_model_len=8192,
            max_num_seqs=35,
            **seed_kwargs,
        )
        print("Using 70B model")
        print("Using max_num_seqs=35, max_model_len=8192")
        print()

    print("Model loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Tokenizer loaded")

    # Load sampling parameters from model's generation_config.json
    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)  # -1 means disabled in vLLM
        print(f"Loaded generation_config: temperature={temperature}, top_p={top_p}, top_k={top_k}")
    except Exception as e:
        print(f"Failed to load generation_config: {e}")
        print("Using defaults: temperature=0.6, top_p=0.95, top_k=-1")
        temperature = 0.6
        top_p = 0.95
        top_k = -1

    # Override with CLI arguments if provided
    if args.temperature is not None:
        temperature = args.temperature
        print(f"CLI override: temperature={temperature}")
    if args.top_p is not None:
        top_p = args.top_p
        print(f"CLI override: top_p={top_p}")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=args.max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    # Load questions
    questions = []
    ground_truths = []
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            ground_truths.append(item["answer"])

    if dataset_limit > 0:
        questions = questions[:dataset_limit]
        ground_truths = ground_truths[:dataset_limit]

    # Build prompts
    prompt_version = args.prompt_version
    if args.use_raw_prompt:
        # Legacy mode: direct text concatenation
        instruction = get_instruction(task_type, prompt_version)
        prompts = [f"{instruction}\n\nQuestion: {q}" for q in questions]
        print(f"Using raw prompts (legacy mode), prompt_version={prompt_version}")
    else:
        # New mode: use chat template for proper thinking activation
        prompts = build_prompts_with_chat_template(questions, tokenizer, model_name, task_type, prompt_version)
        print(f"Using chat template for {model_type} model, prompt_version={prompt_version}")

    # Debug: show first prompt (first 500 + last 500 chars)
    print(f"\n=== Sample prompt (first question, {len(prompts[0])} chars) ===")
    if len(prompts[0]) <= 1000:
        print(prompts[0])
    else:
        print(prompts[0][:500])
        print(f"\n... [{len(prompts[0]) - 1000} chars omitted] ...\n")
        print(prompts[0][-500:])
    print("=" * 50 + "\n")

    print("Running generation...")
    outputs = llm.generate(prompts, sampling_params)

    print("Processing results...")
    results = []

    for i, output in enumerate(tqdm(outputs)):
        question = questions[i]
        gt_answer = ground_truths[i]
        generated_text = output.outputs[0].text.strip()
        answer = extract_answer(generated_text, task_type)

        reasoning, raw_response = split_reasoning_and_raw(generated_text, tokenizer, max_tokens=args.max_tokens)

        result = {
            "dataset": args.dataset,
            "split": "test",
            "question": question,
            "ground_truth_answer": gt_answer,
            "model_answer": answer,
            "generated_text": generated_text,
            "reasoning": reasoning,
            "raw_response": raw_response,
            "reasoning_steps": []
        }
        results.append(result)

    # =========================================================================
    # Fix truncated thinking (entries without </think>)
    # =========================================================================
    truncated_indices = [i for i, r in enumerate(results)
                         if "</think>" not in r["generated_text"]]

    if truncated_indices:
        print(f"\n=== Fixing {len(truncated_indices)} truncated entries ===")

        # Build fix prompts: original prompt + truncated thinking + </think>
        truncated_questions = [results[i]["question"] for i in truncated_indices]
        base_prompts = build_prompts_with_chat_template(
            truncated_questions, tokenizer, model_name, task_type, prompt_version)

        fix_prompts = []
        for j, idx in enumerate(truncated_indices):
            fix_prompt = base_prompts[j] + results[idx]["generated_text"] + "\n</think>\n\n"
            fix_prompts.append(fix_prompt)
            prompt_tokens = len(tokenizer.encode(fix_prompt))
            print(f"  [{idx}] Prompt tokens: {prompt_tokens}")

        # Debug: show first fix prompt (first 50 + last 50 chars)
        print(f"\n=== Sample fix prompt (first truncated entry, {len(fix_prompts[0])} chars) ===")
        if len(fix_prompts[0]) <= 100:
            print(fix_prompts[0])
        else:
            print(fix_prompts[0][:50])
            print(f"\n... [{len(fix_prompts[0]) - 100} chars omitted] ...\n")
            print(fix_prompts[0][-50:])
        print("=" * 50 + "\n")

        # Generate answers with smaller max_tokens
        fix_sampling = SamplingParams(
            temperature=temperature,
            max_tokens=args.answer_max_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        print(f"Generating answers for {len(fix_prompts)} truncated entries...")
        fix_outputs = llm.generate(fix_prompts, fix_sampling)

        # Update entries
        for j, (idx, output) in enumerate(zip(truncated_indices, fix_outputs)):
            answer_text = output.outputs[0].text.strip()
            old_thinking = results[idx]["generated_text"]
            new_generated_text = old_thinking + "\n</think>\n\n" + answer_text
            new_answer = extract_answer(new_generated_text, task_type)
            new_reasoning, new_raw_response = split_reasoning_and_raw(
                new_generated_text, tokenizer, max_tokens=999999)

            gt = results[idx].get("ground_truth_answer", "")
            old_answer = results[idx].get("model_answer", "")
            print(f"  [{idx}] old_answer=\"{old_answer}\" -> new_answer=\"{new_answer}\" (gt=\"{gt}\")")

            results[idx]["generated_text"] = new_generated_text
            results[idx]["model_answer"] = new_answer
            results[idx]["reasoning"] = new_reasoning
            results[idx]["raw_response"] = new_raw_response
            results[idx]["thinking_truncated"] = True

        print(f"Fixed {len(truncated_indices)} truncated entries.")

    # =========================================================================
    # Write results as JSON
    # =========================================================================
    print(f"Writing results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary statistics
    has_think_open = sum(1 for r in results if "<think>" in r["generated_text"])
    has_think_close = sum(1 for r in results if "</think>" in r["generated_text"])
    has_reasoning = sum(1 for r in results if r["reasoning"])
    truncated_count = sum(1 for r in results if r.get("thinking_truncated", False))

    print(f"\n=== Generation Summary ===")
    print(f"Total entries: {len(results)}")
    print(f"Entries with <think>: {has_think_open}")
    print(f"Entries with </think>: {has_think_close}")
    print(f"Entries with reasoning extracted: {has_reasoning}")
    print(f"Entries with truncated thinking (fixed): {truncated_count}")
    print("=" * 30)


if __name__ == "__main__":
    main()
