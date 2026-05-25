#!/usr/bin/env python3
"""
VL Baseline: Chain-of-Draft (CoD) — encourage concise per-step reasoning.

Based on: "Chain of Draft: Thinking Faster by Writing Less"
          (Xu et al., 2025) -- arXiv:2502.18600

The key idea is a per-step word budget:
    "Think step by step, but only keep a minimum draft for each
     thinking step, with 5 words at most."

Usage:
    python baselines_vl/run_cod_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark experiments_mdh/benchmark_vl/mathvista_test.jsonl \
        --output-dir experiments_mdh/data_vl/baselines/cod/mathvista \
        --limit 500
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VL_DIR = os.path.join(_PROJECT_ROOT, "puma_vl")
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
for _p in [_PROJECT_ROOT, _VL_DIR, _OFFLINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, GenerationConfig
from tqdm import tqdm

from prompt_utils_vl import (
    get_task_type_vl, build_base_prompt_vl,
    extract_answer_vl, load_image,
    get_instruction_vl, get_model_type_vl, build_vl_messages,
)
from eval_utils_vl import check_is_correct_vl

# CoD instruction to append to the standard VL instruction
COD_SUFFIX = (
    " Think step by step, but only keep a minimum draft for each"
    " thinking step, with 5 words at most."
)


def build_cod_prompt_vl(processor, model_name, question, image_path, task_type):
    """Build a VL prompt with the CoD instruction appended."""
    model_type = get_model_type_vl(model_name)
    instruction = get_instruction_vl(task_type) + COD_SUFFIX
    messages = build_vl_messages(question, image_path, instruction)

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

    # Strip trailing <think> if auto-added
    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        prompt = stripped[:stripped.rfind("<think>")]

    return prompt


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: Chain-of-Draft")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MAX_TOKENS = args.max_tokens
    task_type = get_task_type_vl(args.dataset)

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    print(f"Using {tp_size} GPUs for tensor parallelism")

    # Load model
    print(f"Loading model: {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=MAX_TOKENS + 18000,  # extra headroom for VL image tokens (~16K)
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    print("Model loaded.")

    # Load generation config
    try:
        gen_config = GenerationConfig.from_pretrained(args.model, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
    except Exception:
        temperature, top_p, top_k = 0.6, 0.95, -1

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        top_p=top_p,
        top_k=top_k,
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
    print(f"Loaded {len(questions)} questions")

    # Build prompts
    print(f"Generating CoD answers for {len(questions)} questions...")
    start_time = time.time()

    vllm_inputs = []
    for q in tqdm(questions, desc="Building CoD prompts"):
        prompt = build_cod_prompt_vl(
            processor, args.model, q["question"], q["image_path"], task_type,
        )
        img = load_image(q["image_path"])
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    # Generate
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Process results
    results = []
    correct = 0
    total_tokens = 0
    for i, (q, out) in enumerate(zip(questions, outputs)):
        text = out.outputs[0].text.strip()
        n_tokens = len(out.outputs[0].token_ids)
        total_tokens += n_tokens

        # Extract reasoning
        if "</think>" in text:
            reasoning = text.split("</think>")[0].strip()
        else:
            reasoning = text

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
            "reasoning": reasoning,
            "model_response": text,
            "final_answer": answer or "",
            "total_tokens": n_tokens,
            "correct": is_correct,
        })

    elapsed = time.time() - start_time

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cod_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save walltime
    walltime_file = os.path.join(args.output_dir, "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": len(results),
            "method": "cod_vl",
        }, f, indent=2)

    acc = correct / len(results) * 100 if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    print(f"\n=== CoD VL Baseline Results ===")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
