#!/usr/bin/env python3
"""
VL Baseline: Full reasoning (vanilla) — no early exit.

Generates full CoT answers for VL benchmarks. This is the baseline
for computing compression rates and accuracy impact.

Usage:
    python baselines_vl/run_vanilla_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark experiments_mdh/benchmark_vl/mathvista_test.jsonl \
        --output-dir experiments_mdh/data_vl/baselines/vanilla/mathvista
"""

import argparse
import json
import os
import sys

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
    get_model_type_vl, get_task_type_vl,
    build_base_prompt_vl, extract_answer_vl, load_image,
)
from eval_utils_vl import check_is_correct_vl


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: Full Reasoning")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_type = get_task_type_vl(args.dataset)

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=args.max_tokens + 3072,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Load generation config
    from transformers import GenerationConfig
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

    # Load questions
    questions = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]

    # Build prompts
    vllm_inputs = []
    for q in tqdm(questions, desc="Building prompts"):
        prompt = build_base_prompt_vl(
            processor, args.model, q["question"], q["image_path"],
            task_type, "default",
        )
        img = load_image(q["image_path"])
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    # Generate
    print(f"Generating {len(vllm_inputs)} full reasoning answers...")
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Process results
    results = []
    correct = 0
    for i, (q, out) in enumerate(zip(questions, outputs)):
        text = out.outputs[0].text.strip()
        answer = extract_answer_vl(text, args.dataset)
        n_tokens = len(out.outputs[0].token_ids)

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
            "correct": is_correct,
            "success": True,
        })

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "vanilla_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    acc = correct / len(results) * 100 if results else 0
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results) if results else 0
    print(f"\n=== Vanilla Baseline Results ===")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
