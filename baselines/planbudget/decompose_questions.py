#!/usr/bin/env python3
"""
Offline question decomposition + difficulty assessment for Plan-and-Budget baseline.

Uses a lightweight planner LLM (default: LLaMA-3.1-8B-Instruct) to:
1. Decompose questions into sub-questions
2. Assess difficulty level and assign credits per sub-question

Based on: "Plan-and-Budget: Effective and Efficient Test-Time Scaling on LLM Reasoning"
          (Lin et al., ICLR 2026)

Usage:
    python -m baselines.planbudget.decompose_questions \
        --planner-model "meta-llama/Llama-3.1-8B-Instruct" \
        --benchmark "experiments_mdh/benchmark/math-500_test.jsonl" \
        --output "data/baselines/planbudget/plans/math-500_plans.jsonl" \
        --dataset "math-500"
"""

import json
import os
import sys
import re
import time
import argparse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# Domain mapping for dataset-specific prompts
DOMAIN_MAP = {
    "aime24": "math",
    "aime25": "math",
    "math-500": "math",
    "olympiadbench": "math and science",
    "gpqa-diamond": "graduate-level science",
    "livecodebench": "programming and algorithms",
}

# Default difficulty level per dataset (used when benchmark has no per-question level)
DEFAULT_LEVEL_MAP = {
    "aime24": 5,
    "aime25": 5,
    "math-500": 3,       # overridden by per-question level if available
    "olympiadbench": 4,
    "gpqa-diamond": 4,
    "livecodebench": 3,
}

# --- Decomposition Prompt (from P&B paper, Appendix A.8) ---
DECOMPOSE_SYSTEM = """\
-Goal-
You are an experienced expert in {domain} and exam question designer. Your role is to help students break down challenging {domain} problems into a series of simpler, high-level sub-questions.
We don't want too many detailed sub-questions, which are not beneficial for testing students' ability in an exam. Each sub-question should build on the previous one so that, once all have been answered, the complete solution is clear.
Your output should be a list of sub-questions with brief hints explaining the purpose of each step, but you should not reveal your internal chain-of-thought either the final solution.

Instructions for Decomposition:
First, analyze the problem and identify the key ideas needed to solve it. Then, generate a series of 2 to 5 sub-questions that lead the student step by step to the complete solution.
The difficulty level of the problem is presented out of 5, where 1 is easy, and 5 is hard. Please adjust the number of sub-questions based on the level. Ideally, we want fewer sub-questions for easy problems and more sub-questions for challenging problems.
DO NOT perform reasoning, directly output those sub-questions based on your gut feelings; only output the list of sub-questions with brief hints for each.
Your answer should be a list of numbered sub-questions. Each sub-question should have a brief accompanying hint that explains what the student will achieve by answering that part."""

DECOMPOSE_USER = """\
A student has presented you with the following problem:
Problem: {problem}
Level: {level} out of 5
**REMEMBER**, you are not allowed to think about it, please directly generate the answer in the following:
Decomposed Sub-questions:"""

# --- Assessment Prompt (from P&B paper, Appendix A.8) ---
ASSESS_SYSTEM = """\
You are an experienced expert in {domain} and exam question designer. Your task is to evaluate the difficulty level of a given exam problem and its sub-questions by comparing it against a set of benchmark questions of known levels.
Based on their levels, you will need to assign each subquestion a portion of the credits (assuming the total credit points is 100 for the whole problem).

Each level reflects increasing complexity from 1 (easiest) to 5 (most challenging). Evaluate based on the conceptual depth, steps involved in solving, required knowledge, and potential for misdirection.

1. You will be provided a question and its subquestions. You will evaluate the difficulty level of the problem and its sub-questions.
Assuming the whole problem is worth 100 points, you assign each sub-question a portion of the score points.
- Adhere to the given subquestions, and DO NOT make new subquestions.
- Sum of each subquestion's credits MUST EQUAL to 100.

2. You must return the result in a structured JSON format:
{{"problem": {{"reason": "...", "evaluated_level": level_q}},
"1": {{"reason": "...", "evaluated_level": level_1, "credit": credit_1}},
"2": {{"reason": "...", "evaluated_level": level_2, "credit": credit_2}},
...}}
where
- "reason": a short explanation (up to 50 words) of your level assessment.
- "evaluated_level": an integer from 1 to 5 indicating your judgment.
- "credit": an integer between 1 to 100 indicating when the question is solved correctly, how many credit can be given.

Your response MUST be a valid JSON object and nothing else."""

ASSESS_USER = """\
Evaluate the level of the following question:
Problem: {problem}
Sub-questions: {steps}
Output:"""


def parse_decomposition(text: str) -> list:
    """Parse decomposed sub-questions from LLM output."""
    # Try P&B's original regex: "1. question\nHint: hint"
    pattern = r"(\d+\..*?)(?=\nHint:)(?:\nHint:\s*(.*?))(?:\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [f"{m[0].strip()} Hint: {m[1].strip()}" for m in matches]

    # Fallback: numbered list with inline hints
    pattern2 = r"(\d+\.\s*.+?)(?=\n\d+\.|$)"
    matches2 = re.findall(pattern2, text, re.DOTALL)
    if matches2:
        return [m.strip() for m in matches2]

    # Final fallback
    return ["1. Directly solve the problem. Hint: None."]


def parse_assessment(text: str, num_steps: int) -> tuple:
    """Parse difficulty assessment JSON. Returns (level, credits)."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(text)

        level = int(result.get("problem", {}).get("evaluated_level", 3))
        level = max(1, min(5, level))

        credits = []
        for i in range(1, num_steps + 1):
            key = str(i)
            if key in result:
                credits.append(int(result[key].get("credit", 100 // num_steps)))
            else:
                credits.append(100 // num_steps)

        # Normalize credits to sum to 100
        total = sum(credits)
        if total != 100 and total > 0:
            credits = [round(c * 100 / total) for c in credits]
            diff = 100 - sum(credits)
            credits[0] += diff

        return level, credits
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"  Warning: Failed to parse assessment: {e}")
        equal_credit = 100 // max(num_steps, 1)
        credits = [equal_credit] * num_steps
        if credits:
            credits[0] += 100 - sum(credits)
        return 3, credits


def main():
    parser = argparse.ArgumentParser(
        description="Offline question decomposition + difficulty assessment for P&B"
    )
    parser.add_argument("--planner-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Planner model for decomposition")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output plans JSONL")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (aime24, math-500, gpqa-diamond, etc.)")
    args = parser.parse_args()

    domain = DOMAIN_MAP.get(args.dataset, "math")
    default_level = DEFAULT_LEVEL_MAP.get(args.dataset, 3)
    print(f"Dataset: {args.dataset}, Domain: {domain}, Default level: {default_level}")

    # Load questions (with per-question level if available)
    questions = []
    levels = []
    with open(args.benchmark, "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["question"])
            levels.append(data.get("level", default_level))
    print(f"Loaded {len(questions)} questions from {args.benchmark}")

    # Check if output already exists with correct count
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = sum(1 for _ in f)
        if existing == len(questions):
            print(f"Plans already exist ({existing} entries). Skipping.")
            return
        else:
            print(f"Existing plans incomplete ({existing}/{len(questions)}). Regenerating.")

    TENSOR_PARALLEL_SIZE = (
        len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else 1
    )

    # Load planner model
    print(f"\nLoading planner model: {args.planner_model}...")
    llm = LLM(
        model=args.planner_model,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.planner_model, trust_remote_code=True)
    print("Planner model loaded.")

    # ---- Step 1: Decompose all questions ----
    print(f"\nStep 1/2: Decomposing {len(questions)} questions...")
    start_time = time.time()

    decompose_params = SamplingParams(
        temperature=0.0,  # Deterministic decomposition
        max_tokens=1024,
    )

    decompose_prompts = []
    for q, lvl in zip(questions, levels):
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM.format(domain=domain)},
            {"role": "user", "content": DECOMPOSE_USER.format(problem=q, level=lvl)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        decompose_prompts.append(prompt)

    decompose_outputs = llm.generate(decompose_prompts, decompose_params)

    steps_list = []
    decompose_tokens = []
    for output in tqdm(decompose_outputs, desc="Parsing decompositions"):
        text = output.outputs[0].text.strip()
        steps = parse_decomposition(text)
        steps_list.append(steps)
        decompose_tokens.append(len(output.outputs[0].token_ids))

    decompose_time = time.time() - start_time
    print(f"Decomposition complete: {decompose_time:.1f}s")
    avg_steps = sum(len(s) for s in steps_list) / len(steps_list)
    print(f"Average sub-questions per problem: {avg_steps:.1f}")

    # ---- Step 2: Assess difficulty + credits ----
    print(f"\nStep 2/2: Assessing difficulty for {len(questions)} questions...")
    start_time = time.time()

    assess_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    assess_prompts = []
    for q, steps in zip(questions, steps_list):
        steps_str = "\n".join(steps)
        messages = [
            {"role": "system", "content": ASSESS_SYSTEM.format(domain=domain)},
            {"role": "user", "content": ASSESS_USER.format(problem=q, steps=steps_str)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assess_prompts.append(prompt)

    assess_outputs = llm.generate(assess_prompts, assess_params)

    plans = []
    for i, output in enumerate(tqdm(assess_outputs, desc="Parsing assessments")):
        text = output.outputs[0].text.strip()
        level, credits = parse_assessment(text, len(steps_list[i]))
        assess_tok = len(output.outputs[0].token_ids)

        plans.append({
            "question": questions[i],
            "steps": steps_list[i],
            "level": level,
            "credits": credits,
            "planning_tokens": decompose_tokens[i] + assess_tok,
        })

    assess_time = time.time() - start_time
    print(f"Assessment complete: {assess_time:.1f}s")
    avg_planning = sum(p["planning_tokens"] for p in plans) / len(plans)
    avg_level = sum(p["level"] for p in plans) / len(plans)
    print(f"Average planning tokens per question: {avg_planning:.1f}")
    print(f"Average assessed level: {avg_level:.2f}")

    # ---- Save plans ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for plan in plans:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
    print(f"\nPlans saved to: {args.output}")

    # Clean up
    print("Cleaning up planner model...")
    del llm
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
