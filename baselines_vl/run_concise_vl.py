#!/usr/bin/env python3
"""
VL Baseline: Concise CoT (CCoT) — limit thinking tokens.

Generates answers with a hard cap on reasoning tokens (e.g., 8192).
This simulates CCoT/CoD approaches that compress by limiting token budget.

Usage:
    python baselines_vl/run_concise_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark data/benchmark_vl/mathvista_test.jsonl \
        --output-dir runs/vl/baselines/ccot/mathvista \
        --thinking-budget 8192
"""

import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VL_DIR = os.path.join(_PROJECT_ROOT, "puma_vl")
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
for _p in [_PROJECT_ROOT, _VL_DIR, _OFFLINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from tqdm import tqdm

from prompt_utils_vl import (
    get_task_type_vl, build_base_prompt_vl,
    extract_answer_vl, load_image,
)
from eval_utils_vl import check_is_correct_vl

CONCISE_INSTRUCTION_SUFFIX = (
    " Be concise in your reasoning. "
    "Avoid unnecessary elaboration or redundant verification steps."
)


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: CCoT (Concise CoT)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--thinking-budget", type=int, default=8192,
                        help="Max tokens for reasoning (default: 8192)")
    parser.add_argument("--answer-budget", type=int, default=2048,
                        help="Max tokens for answer after reasoning (default: 2048)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_type = get_task_type_vl(args.dataset)
    total_budget = args.thinking_budget + args.answer_budget

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=total_budget + 18000,  # extra headroom for VL image tokens (~16K)
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(args.model, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
    except Exception:
        temperature, top_p, top_k = 0.6, 0.95, -1

    # Use thinking budget as max_tokens for first pass
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=total_budget,
        top_p=top_p, top_k=top_k,
    )

    # Load questions
    questions = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]

    # Build prompts (standard prompts — the budget is enforced by max_tokens)
    vllm_inputs = []
    for q in tqdm(questions, desc="Building CCoT prompts"):
        prompt = build_base_prompt_vl(
            processor, args.model, q["question"], q["image_path"],
            task_type, "default",
        )
        img = load_image(q["image_path"])
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    print(f"Generating {len(vllm_inputs)} CCoT answers (budget={total_budget})...")
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Fix truncated thinking
    truncated = []
    results_text = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        results_text.append(text)
        if "</think>" not in text:
            truncated.append(i)

    if truncated:
        print(f"Fixing {len(truncated)} truncated entries...")
        fix_inputs = []
        for idx in truncated:
            q = questions[idx]
            base = build_base_prompt_vl(
                processor, args.model, q["question"], q["image_path"],
                task_type, "default",
            )
            fix_prompt = base + results_text[idx] + "\n</think>\n\n"
            img = load_image(q["image_path"])
            fix_inputs.append({"prompt": fix_prompt, "multi_modal_data": {"image": img}})

        fix_params = SamplingParams(
            temperature=temperature, max_tokens=args.answer_budget,
            top_p=top_p, top_k=top_k,
        )
        fix_outputs = llm.generate(fix_inputs, fix_params)

        for j, idx in enumerate(truncated):
            answer_text = fix_outputs[j].outputs[0].text.strip()
            results_text[idx] = results_text[idx] + "\n</think>\n\n" + answer_text

    # Evaluate
    results = []
    correct = 0
    for i, (q, text) in enumerate(zip(questions, results_text)):
        answer = extract_answer_vl(text, args.dataset)
        n_tokens = len(tokenizer.encode(text))

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
            "model_response": text,
            "final_answer": answer or "",
            "total_tokens": n_tokens,
            "correct": is_correct,
            "thinking_budget": args.thinking_budget,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"ccot_{args.thinking_budget}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    acc = correct / len(results) * 100 if results else 0
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results) if results else 0
    print(f"\n=== CCoT Baseline Results (budget={args.thinking_budget}) ===")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
