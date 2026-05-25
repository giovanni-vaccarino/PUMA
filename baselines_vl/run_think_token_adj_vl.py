#!/usr/bin/env python3
"""
VL Baseline: Think Token Adjustment — logit_bias on </think> to encourage
earlier exit from thinking phase.

Uses logit_bias to boost the </think> token probability, making the model
exit thinking sooner. After reasoning stops, generates the answer normally.

Based on: boost_sampling.py adapted for vision-language models.

Usage:
    python baselines_vl/run_think_token_adj_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark experiments_mdh/benchmark_vl/mathvista_test.jsonl \
        --output-dir experiments_mdh/data_vl/baselines/think_token_adj/mathvista \
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

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, GenerationConfig
from tqdm import tqdm

from prompt_utils_vl import (
    get_task_type_vl, build_base_prompt_vl,
    extract_answer_vl, load_image,
)
from eval_utils_vl import check_is_correct_vl


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: Think Token Adjustment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--boost-bias", type=float, default=5.0,
                        help="Logit bias for </think> token (default: 5.0)")
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_type = get_task_type_vl(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

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
        max_model_len=args.max_tokens + 18000,  # extra headroom for VL image tokens (~16K)
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

    # Get </think> token id for logit bias
    boost_token_id = tokenizer("</think>", add_special_tokens=False)["input_ids"][0]
    print(f"Boost token id (</think>): {boost_token_id}, bias: {args.boost_bias}")

    # Phase 1 sampling: reasoning with boosted </think>
    sampling_params_reasoning = SamplingParams(
        temperature=temperature,
        max_tokens=args.max_tokens,
        top_p=top_p,
        top_k=top_k,
        logit_bias={boost_token_id: args.boost_bias},
        stop=["</think>"],
        include_stop_str_in_output=True,
    )

    # Phase 2 sampling: answer generation
    sampling_params_answer = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_answer_tokens,
        stop=["\n"],
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
    print("Building prompts...")
    start_time = time.time()

    vllm_inputs = []
    for q in tqdm(questions, desc="Building prompts"):
        prompt = build_base_prompt_vl(
            processor, args.model, q["question"], q["image_path"],
            task_type, "default",
        )
        img = load_image(q["image_path"])
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    # Phase 1: Generate reasoning with boosted </think>
    print(f"Phase 1: Generating reasoning for {len(vllm_inputs)} questions...")
    reasoning_outputs = llm.generate(vllm_inputs, sampling_params_reasoning)

    # Phase 2: Generate answers
    print("Phase 2: Generating answers...")
    max_prompt_len = args.max_tokens - args.max_answer_tokens

    answer_inputs = []
    reasoning_texts = []
    for i, output in enumerate(reasoning_outputs):
        reasoning_text = output.outputs[0].text
        reasoning_texts.append(reasoning_text)

        # Build continuation prompt
        if not reasoning_text.endswith("</think>"):
            reasoning_text_cont = reasoning_text + "</think>"
        else:
            reasoning_text_cont = reasoning_text

        full_prompt = vllm_inputs[i]["prompt"] + reasoning_text_cont + "\n\\boxed"

        img = load_image(questions[i]["image_path"])
        answer_inputs.append({
            "prompt": full_prompt,
            "multi_modal_data": {"image": img},
        })

    answer_outputs = llm.generate(answer_inputs, sampling_params_answer)

    # Process results
    print("Processing results...")
    results = []
    correct = 0
    total_tokens = 0
    for i, (reasoning_text, ans_output) in enumerate(zip(reasoning_texts, answer_outputs)):
        q = questions[i]
        answer_text = "\\boxed" + ans_output.outputs[0].text

        if not reasoning_text.endswith("</think>"):
            reasoning_full = reasoning_text + "</think>"
        else:
            reasoning_full = reasoning_text
        generated_text = reasoning_full + "\n" + answer_text

        reasoning_clean = reasoning_text.replace("</think>", "").strip()
        answer = extract_answer_vl(generated_text, args.dataset)
        n_reasoning_tokens = len(reasoning_outputs[i].outputs[0].token_ids)
        n_answer_tokens = len(ans_output.outputs[0].token_ids)
        n_tokens = n_reasoning_tokens + n_answer_tokens
        total_tokens += n_tokens

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
            "reasoning": reasoning_clean,
            "model_response": generated_text,
            "final_answer": answer or "",
            "total_tokens": n_tokens,
            "reasoning_tokens": n_reasoning_tokens,
            "answer_tokens": n_answer_tokens,
            "correct": is_correct,
        })

    elapsed = time.time() - start_time

    # Save
    out_path = os.path.join(args.output_dir, "boost_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    walltime_file = os.path.join(args.output_dir, "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": len(results),
            "method": "think_token_adj_vl",
        }, f, indent=2)

    acc = correct / len(results) * 100 if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    avg_reasoning = sum(r["reasoning_tokens"] for r in results) / len(results) if results else 0
    print(f"\n=== Think Token Adjustment VL Results ===")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg total tokens: {avg_tokens:.0f}")
    print(f"  Avg reasoning tokens: {avg_reasoning:.0f}")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Saved to: {out_path}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
