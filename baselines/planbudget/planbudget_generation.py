#!/usr/bin/env python3
"""
Plan-and-Budget (P&B) generation with budget-guided reasoning.

Loads pre-computed plans (from decompose_questions.py) and generates
reasoning with per-sub-question budget hints using polynomial decay scheduling.

Based on: "Plan-and-Budget: Effective and Efficient Test-Time Scaling on LLM Reasoning"
          (Lin et al., ICLR 2026)

Usage:
    python -m baselines.planbudget.planbudget_generation \
        <MODEL_NAME> <LIMIT> <BENCHMARK_PATH> <OUTPUT_PATH> \
        --plans <PLANS_JSONL>
"""

import json
import os
import sys
import time
import argparse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from baselines.utils.math_util import my_answer_extraction, extract_code_answer, CODE_SYS_PROMPT
from baselines.utils.generation_config import load_generation_params


# Budget mapping from P&B paper (level -> word budget)
LEVEL_MAP = {1: 200, 2: 250, 3: 350, 4: 450, 5: 600}


def polynomial_decay(i, N, power=2):
    """Polynomial decay schedule: weight proportional to (N - i)^power."""
    return (N - i) ** power


def allocate_tokens(credits, total_tokens):
    """Allocate word budget per sub-question using polynomial decay.

    Args:
        credits: list of credit scores (should sum to 100)
        total_tokens: total word budget for the question
    Returns:
        list of per-sub-question word budgets
    """
    N = len(credits)
    if N == 0:
        return []
    if N == 1:
        return [total_tokens]

    sched_weights = np.array([polynomial_decay(i, N) for i in range(N)], dtype=float)
    combined = np.array(credits, dtype=float) * sched_weights
    total = combined.sum()
    if total == 0:
        combined = np.ones(N)
        total = N
    combined /= total
    tokens = np.floor(combined * total_tokens).astype(int)
    diff = total_tokens - tokens.sum()
    for i in np.argsort(-combined)[:int(diff)]:
        tokens[i] += 1
    return tokens.tolist()


# P&B Planned Local Budget reasoning prompt (from paper Appendix A.8)
SYSTEM_PROMPT = """\
Please reason step by step, and put your final answer within \\boxed{{}}.

The problem is given by an overall description, difficulty level out of 5, followed by a series of sub-questions as a hint.
All the credit is given when you provide a correct final answer for the overall problem.
Please solve the question efficiently and clearly to achieve as much credit as possible."""

SYSTEM_PROMPT_CODE = """\
You are a helpful assistant that solves programming problems.

The problem is given by an overall description, difficulty level out of 5, followed by a series of sub-questions as a hint.
All the credit is given when you provide a correct final solution for the overall problem.
Please solve the question efficiently and clearly. Provide your final solution as a complete Python program within a ```python code block."""

USER_TEMPLATE = """\
Let's start the exam. You are being given this problem:
**Problem (100pt):** {question}
**Level:** {level} out of 5

You may think following these sub-questions or feel free to use other methods that works the best towards getting the final answer:
{decomposed}

Please provide your final answer within \\boxed{{}}."""

USER_TEMPLATE_CODE = """\
Let's start the exam. You are being given this problem:
**Problem (100pt):** {question}
**Level:** {level} out of 5

You may think following these sub-questions or feel free to use other methods that works the best towards getting the final solution:
{decomposed}

Please provide your final solution as a complete Python program within a ```python code block."""


def main():
    parser = argparse.ArgumentParser(
        description="Plan-and-Budget generation with budget-guided reasoning"
    )
    parser.add_argument("model", type=str, help="HuggingFace model identifier")
    parser.add_argument("limit", type=int, help="Max questions to process")
    parser.add_argument("dataset", type=str, help="Path to benchmark JSONL")
    parser.add_argument("output", type=str, help="Path to output JSONL")
    parser.add_argument("--plans", type=str, required=True,
                        help="Path to pre-computed plans JSONL")
    args = parser.parse_args()

    MAX_TOKENS = 8192  # Hard cutoff from P&B paper

    gen_params = load_generation_params(args.model)
    TEMPERATURE = gen_params["temperature"]
    TOP_P = gen_params["top_p"]
    TOP_K = gen_params["top_k"]

    TENSOR_PARALLEL_SIZE = (
        len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else 1
    )
    print(f"Using {TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism")

    is_code = "livecodebench" in args.dataset.lower()
    if is_code:
        print("Code task detected: using code system prompt and user template")

    # Load questions
    questions = []
    question_ids = []
    with open(args.dataset, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            question_ids.append(item.get("question_id", ""))
    questions = questions[:args.limit]
    question_ids = question_ids[:args.limit]

    # Load plans
    plans = []
    with open(args.plans, "r") as f:
        for line in f:
            plans.append(json.loads(line))
    plans = plans[:args.limit]

    assert len(plans) == len(questions), \
        f"Mismatch: {len(plans)} plans vs {len(questions)} questions"

    # Load reasoning model
    print(f"Loading reasoning model: {args.model}...")
    model_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        max_model_len=MAX_TOKENS + 4096,  # prompt headroom + generation
        gpu_memory_utilization=0.85,
    )
    # Larger models may need reduced batch size
    if "32B" in args.model or "30B" in args.model or "70B" in args.model:
        model_kwargs["max_num_seqs"] = 35

    llm = LLM(**model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Model loaded.")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    # Format prompts with budget hints
    print(f"Generating P&B answers for {len(questions)} questions...")
    start_time = time.time()

    prompts = []
    for q, plan in zip(questions, plans):
        level = plan["level"]
        credits = plan["credits"]
        steps = plan["steps"]
        total_budget = LEVEL_MAP.get(level, 350)
        tokens_per_step = allocate_tokens(credits, total_budget)

        decomposed_str = "\n\n".join([
            f"{step} Please only think a little, and directly solve it using up to {budget} words."
            for step, budget in zip(steps, tokens_per_step)
        ])

        if is_code:
            user_msg = USER_TEMPLATE_CODE.format(
                question=q, level=level, decomposed=decomposed_str,
            )
            sys_msg = SYSTEM_PROMPT_CODE
        else:
            user_msg = USER_TEMPLATE.format(
                question=q, level=level, decomposed=decomposed_str,
            )
            sys_msg = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)

    # Process results
    print("Processing results...")
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing")):
        generated_text = output.outputs[0].text.strip()

        result = {
            "question": questions[i],
            "generated_text": generated_text,
            "planning_tokens": plans[i]["planning_tokens"],
        }

        if is_code:
            code = extract_code_answer(generated_text)
            result["answer"] = code
            result["extracted_code"] = code
            result["question_id"] = question_ids[i]
        else:
            result["answer"] = my_answer_extraction(generated_text)

        results.append(result)

    # Save predictions
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"Done! {len(results)} questions in {elapsed:.2f}s")

    # Save walltime
    walltime_file = os.path.join(os.path.dirname(args.output), "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": len(results),
            "method": "planbudget_polynomial",
        }, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Cleanup
    print("Cleaning up GPU memory...")
    del llm
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
