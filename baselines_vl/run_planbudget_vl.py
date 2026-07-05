#!/usr/bin/env python3
"""
VL Baseline: Plan-and-Budget — budget-guided reasoning with question decomposition.

Step 1: Decompose questions using a planner LLM (text-only, no images needed).
Step 2: Generate reasoning with per-sub-question budget hints (VL model sees images).

The planner is text-only (Llama-3.1-8B-Instruct by default). Only the reasoning
model is VL and sees images.

Based on: planbudget_generation.py + decompose_questions.py adapted for VL.

Usage:
    python baselines_vl/run_planbudget_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark data/benchmark_vl/mathvista_test.jsonl \
        --output-dir runs/vl/baselines/planbudget/mathvista \
        --limit 500
"""

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VL_DIR = os.path.join(_PROJECT_ROOT, "puma_vl")
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
for _p in [_PROJECT_ROOT, _VL_DIR, _OFFLINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
from tqdm import tqdm

from prompt_utils_vl import (
    get_task_type_vl, get_model_type_vl, get_instruction_vl,
    build_vl_messages, extract_answer_vl, load_image,
)
from eval_utils_vl import check_is_correct_vl


# Budget mapping from P&B paper (level -> word budget)
LEVEL_MAP = {1: 200, 2: 250, 3: 350, 4: 450, 5: 600}

# Domain mapping
DOMAIN_MAP = {
    "mathvista": "math",
    "mathvision": "math",
    "mmmu-pro": "graduate-level science",
}

DEFAULT_LEVEL_MAP = {
    "mathvista": 3,
    "mathvision": 4,
    "mmmu-pro": 4,
}


def polynomial_decay(i, N, power=2):
    return (N - i) ** power


def allocate_tokens(credits, total_tokens):
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


# --- Decomposition prompts (from P&B paper) ---
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

# VL reasoning prompt with budget hints
SYSTEM_PROMPT_VL = """\
Solve the following math problem based on the given image. Please reason step by step, and put your final answer within \\boxed{{}}.

The problem is given by an overall description, difficulty level out of 5, followed by a series of sub-questions as a hint.
All the credit is given when you provide a correct final answer for the overall problem.
Please solve the question efficiently and clearly to achieve as much credit as possible."""

SYSTEM_PROMPT_MCQ_VL = """\
Answer the following multiple choice question based on the given image. Please reason step by step, and put your choice letter (e.g. A, B, C, or D) without any other text within \\boxed{{}} in the end.

The problem is given by an overall description, difficulty level out of 5, followed by a series of sub-questions as a hint.
All the credit is given when you provide a correct final answer for the overall problem.
Please solve the question efficiently and clearly to achieve as much credit as possible."""

USER_TEMPLATE_VL = """\
Let's start the exam. You are being given this problem:
**Problem (100pt):** {question}
**Level:** {level} out of 5

You may think following these sub-questions or feel free to use other methods that works the best towards getting the final answer:
{decomposed}

Please provide your final answer within \\boxed{{}}."""


def parse_decomposition(text):
    pattern = r"(\d+\..*?)(?=\nHint:)(?:\nHint:\s*(.*?))(?:\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [f"{m[0].strip()} Hint: {m[1].strip()}" for m in matches]
    pattern2 = r"(\d+\.\s*.+?)(?=\n\d+\.|$)"
    matches2 = re.findall(pattern2, text, re.DOTALL)
    if matches2:
        return [m.strip() for m in matches2]
    return ["1. Directly solve the problem. Hint: None."]


def parse_assessment(text, num_steps):
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

        total = sum(credits)
        if total != 100 and total > 0:
            credits = [round(c * 100 / total) for c in credits]
            diff = 100 - sum(credits)
            credits[0] += diff

        return level, credits
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        equal_credit = 100 // max(num_steps, 1)
        credits = [equal_credit] * num_steps
        if credits:
            credits[0] += 100 - sum(credits)
        return 3, credits


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: Plan-and-Budget")
    parser.add_argument("--model", type=str, required=True,
                        help="VL reasoning model")
    parser.add_argument("--planner-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Text-only planner model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--plans", type=str, default=None,
                        help="Pre-computed plans JSONL (skip decomposition if provided)")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_type = get_task_type_vl(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    print(f"Using {tp_size} GPUs for tensor parallelism")

    domain = DOMAIN_MAP.get(args.dataset.lower(), "math")
    default_level = DEFAULT_LEVEL_MAP.get(args.dataset.lower(), 3)

    # Load questions
    questions = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions")

    # Step 1: Get plans (decompose + assess)
    plans_path = os.path.join(args.output_dir, "plans.jsonl")
    if args.plans and os.path.exists(args.plans):
        print(f"Loading pre-computed plans from: {args.plans}")
        plans = []
        with open(args.plans, "r") as f:
            for line in f:
                plans.append(json.loads(line))
        plans = plans[:len(questions)]
    elif os.path.exists(plans_path):
        print(f"Loading cached plans from: {plans_path}")
        plans = []
        with open(plans_path, "r") as f:
            for line in f:
                plans.append(json.loads(line))
        if len(plans) != len(questions):
            print(f"Plan count mismatch ({len(plans)} vs {len(questions)}), regenerating...")
            plans = None
        else:
            plans = plans[:len(questions)]
    else:
        plans = None

    if plans is None:
        # Decompose using text-only planner
        print(f"\nStep 1: Decomposing with planner: {args.planner_model}...")
        planner_llm = LLM(
            model=args.planner_model,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            dtype="auto",
            max_model_len=4096 + 18000,  # extra headroom for VL image tokens (~16K)
            gpu_memory_utilization=0.85,
        )
        planner_tokenizer = AutoTokenizer.from_pretrained(
            args.planner_model, trust_remote_code=True,
        )

        decompose_params = SamplingParams(temperature=0.0, max_tokens=1024)
        decompose_prompts = []
        levels = []
        for q in questions:
            lvl = q.get("level", default_level)
            levels.append(lvl)
            messages = [
                {"role": "system", "content": DECOMPOSE_SYSTEM.format(domain=domain)},
                {"role": "user", "content": DECOMPOSE_USER.format(
                    problem=q["question"], level=lvl,
                )},
            ]
            prompt = planner_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            decompose_prompts.append(prompt)

        decompose_outputs = planner_llm.generate(decompose_prompts, decompose_params)

        steps_list = []
        decompose_tokens = []
        for output in tqdm(decompose_outputs, desc="Parsing decompositions"):
            text = output.outputs[0].text.strip()
            steps = parse_decomposition(text)
            steps_list.append(steps)
            decompose_tokens.append(len(output.outputs[0].token_ids))

        # Assess difficulty
        print("Step 1b: Assessing difficulty...")
        assess_params = SamplingParams(temperature=0.0, max_tokens=1024)
        assess_prompts = []
        for q, steps in zip(questions, steps_list):
            steps_str = "\n".join(steps)
            messages = [
                {"role": "system", "content": ASSESS_SYSTEM.format(domain=domain)},
                {"role": "user", "content": ASSESS_USER.format(
                    problem=q["question"], steps=steps_str,
                )},
            ]
            prompt = planner_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            assess_prompts.append(prompt)

        assess_outputs = planner_llm.generate(assess_prompts, assess_params)

        plans = []
        for i, output in enumerate(tqdm(assess_outputs, desc="Parsing assessments")):
            text = output.outputs[0].text.strip()
            level, credits = parse_assessment(text, len(steps_list[i]))
            assess_tok = len(output.outputs[0].token_ids)
            plans.append({
                "question": questions[i]["question"],
                "steps": steps_list[i],
                "level": level,
                "credits": credits,
                "planning_tokens": decompose_tokens[i] + assess_tok,
            })

        # Save plans
        with open(plans_path, "w") as f:
            for plan in plans:
                f.write(json.dumps(plan, ensure_ascii=False) + "\n")
        print(f"Plans saved to: {plans_path}")

        # Free planner
        del planner_llm
        torch.cuda.empty_cache()

    assert len(plans) == len(questions), \
        f"Plan/question mismatch: {len(plans)} vs {len(questions)}"

    # Step 2: Generate budget-guided answers with VL model
    print(f"\nStep 2: Loading VL reasoning model: {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=args.max_tokens + 18000,  # extra headroom for VL image tokens (~16K)
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    print("VL model loaded.")

    # Load generation config
    try:
        gen_config = GenerationConfig.from_pretrained(args.model, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
    except Exception:
        temperature, top_p, top_k = 0.6, 0.95, -1

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=args.max_tokens,
        top_p=top_p, top_k=top_k,
    )

    # Build VL prompts with budget hints
    print(f"Building budget-guided VL prompts for {len(questions)} questions...")
    start_time = time.time()

    model_type = get_model_type_vl(args.model)
    sys_prompt = SYSTEM_PROMPT_MCQ_VL if task_type == "mcq_vl" else SYSTEM_PROMPT_VL

    vllm_inputs = []
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

        user_text = USER_TEMPLATE_VL.format(
            question=q["question"], level=level, decomposed=decomposed_str,
        )

        # Build VL messages with image + budget-guided user text
        content = [
            {"type": "image", "image": f"file://{os.path.abspath(q['image_path'])}"},
            {"type": "text", "text": f"{sys_prompt}\n\n{user_text}"},
        ]
        messages = [{"role": "user", "content": content}]

        try:
            if model_type == "qwen3_vl_thinking":
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True,
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

        # Strip trailing <think>
        stripped = prompt.rstrip()
        if stripped.endswith("<think>"):
            prompt = stripped[:stripped.rfind("<think>")]

        img = load_image(q["image_path"])
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    # Generate
    print(f"Generating P&B answers...")
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Process results
    results = []
    correct = 0
    total_tokens = 0
    for i, (q, plan, out) in enumerate(zip(questions, plans, outputs)):
        text = out.outputs[0].text.strip()
        n_tokens = len(out.outputs[0].token_ids)
        total_tokens += n_tokens

        answer = extract_answer_vl(text, args.dataset)
        is_correct = check_is_correct_vl(
            answer, q["answer"], args.dataset, q.get("question_type", ""),
        )
        if is_correct:
            correct += 1

        results.append({
            "question_idx": i + 1,
            "question": q["question"],
            "ground_truth_answer": q["answer"],
            "image_path": q["image_path"],
            "question_type": q.get("question_type", ""),
            "model_response": text,
            "final_answer": answer or "",
            "total_tokens": n_tokens,
            "planning_tokens": plan.get("planning_tokens", 0),
            "level": plan["level"],
            "num_steps": len(plan["steps"]),
            "correct": is_correct,
        })

    elapsed = time.time() - start_time

    # Save
    out_path = os.path.join(args.output_dir, "planbudget_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    walltime_file = os.path.join(args.output_dir, "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": len(results),
            "method": "planbudget_vl",
        }, f, indent=2)

    acc = correct / len(results) * 100 if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    avg_planning = sum(r["planning_tokens"] for r in results) / len(results) if results else 0
    print(f"\n=== Plan-and-Budget VL Results ===")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Avg planning tokens: {avg_planning:.0f}")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Saved to: {out_path}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
