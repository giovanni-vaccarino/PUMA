#!/usr/bin/env python3
"""
Step 1a: Generate full chain-of-thought answers for VL benchmarks.

Extends puma/run_vllm.py with multimodal support.
Uses AutoProcessor for chat template + PIL images via multi_modal_data.

Usage:
    python puma_vl/run_vllm_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --dataset_path experiments_mdh/benchmark_vl/mathvista_test.jsonl \
        --output_path experiments_mdh/data_vl/answers/Qwen3-VL-8B-T_mathvista_answers.json \
        --max_tokens 32768 --seed 42
"""

import argparse
import json
import os
import re
import sys

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Add project root and offline dirs to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
_VL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_PROJECT_ROOT, _OFFLINE_DIR, _VL_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prompt_utils_vl import (
    get_model_type_vl, get_task_type_vl, get_instruction_vl,
    build_base_prompt_vl, extract_answer_vl, load_image,
)


def split_reasoning_and_raw(text, tokenizer, max_tokens=32768):
    """Split generated text into reasoning and raw_response."""
    cleaned = text.lstrip()
    if cleaned.startswith("<think>"):
        cleaned = cleaned[len("<think>"):]
    if "</think>" in cleaned:
        reasoning, rest = cleaned.split("</think>", 1)
        return reasoning.strip(), rest.strip()
    token_count = len(tokenizer.encode(text))
    if token_count >= max_tokens * 0.9:
        return text.strip(), ""
    else:
        return "", text.strip()


def main():
    parser = argparse.ArgumentParser(description="VL Step 1a: Generate full CoT answers")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--answer_max_tokens", type=int, default=4096)
    parser.add_argument("--prompt-version", type=str, default="default",
                        choices=["default", "direct"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_name = args.model
    task_type = get_task_type_vl(args.dataset)
    model_type = get_model_type_vl(model_name)

    # Detect tensor parallel size
    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    print(f"Using {tp_size} GPUs for tensor parallelism")

    # max_model_len: image tokens (~1024) + reasoning + answer
    max_model_len = args.max_tokens + 2048 + args.answer_max_tokens

    seed_kwargs = {"seed": args.seed} if args.seed is not None else {}

    # Use spawn instead of fork for multiprocessing to avoid CUDA re-init issues
    # (Qwen3-VL processor may trigger CUDA init at import time)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print(f"Loading VL model: {model_name} (type: {model_type})")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        **seed_kwargs,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Load generation config for temperature/top_p
    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
        print(f"Loaded generation_config: temperature={temperature}, top_p={top_p}, top_k={top_k}")
    except Exception:
        temperature, top_p, top_k = 0.6, 0.95, -1

    if args.temperature is not None:
        temperature = args.temperature
    if args.top_p is not None:
        top_p = args.top_p

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=args.max_tokens,
        top_p=top_p, top_k=top_k,
    )

    # Load questions
    questions = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions from {args.dataset_path}")

    # Build prompts with images
    prompts = []
    images = []
    for q in tqdm(questions, desc="Building prompts"):
        prompt_text = build_base_prompt_vl(
            processor, model_name, q["question"], q["image_path"],
            task_type, args.prompt_version,
        )
        img = load_image(q["image_path"])
        prompts.append(prompt_text)
        images.append(img)

    # Debug: show first prompt
    print(f"\n=== Sample prompt ({len(prompts[0])} chars) ===")
    if len(prompts[0]) <= 500:
        print(prompts[0])
    else:
        print(prompts[0][:250])
        print(f"\n... [{len(prompts[0]) - 500} chars omitted] ...\n")
        print(prompts[0][-250:])
    print("=" * 50 + "\n")

    # Generate (multimodal: pass dicts)
    print("Running VL generation...")
    vllm_inputs = [
        {"prompt": p, "multi_modal_data": {"image": img}}
        for p, img in zip(prompts, images)
    ]
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Process results
    print("Processing results...")
    results = []
    for i, output in enumerate(tqdm(outputs)):
        q = questions[i]
        generated_text = output.outputs[0].text.strip()
        answer = extract_answer_vl(generated_text, args.dataset)
        reasoning, raw_response = split_reasoning_and_raw(
            generated_text, tokenizer, max_tokens=args.max_tokens,
        )

        total_tokens = len(tokenizer.encode(generated_text))
        result = {
            "dataset": args.dataset,
            "split": "test",
            "question": q["question"],
            "ground_truth_answer": q["answer"],
            "image_path": q["image_path"],
            "question_type": q.get("question_type", ""),
            "model_answer": answer,
            "generated_text": generated_text,
            "reasoning": reasoning,
            "raw_response": raw_response,
            "reasoning_steps": [],
            "total_tokens": total_tokens,
        }
        results.append(result)

    # Fix truncated thinking (entries without </think>)
    truncated_indices = [
        i for i, r in enumerate(results)
        if "</think>" not in r["generated_text"]
    ]

    if truncated_indices:
        print(f"\n=== Fixing {len(truncated_indices)} truncated entries ===")

        fix_inputs = []
        for idx in truncated_indices:
            q = questions[idx]
            base_prompt = build_base_prompt_vl(
                processor, model_name, q["question"], q["image_path"],
                task_type, args.prompt_version,
            )
            fix_prompt = base_prompt + results[idx]["generated_text"] + "\n</think>\n\n"
            img = load_image(q["image_path"])
            fix_inputs.append({"prompt": fix_prompt, "multi_modal_data": {"image": img}})

        fix_params = SamplingParams(
            temperature=temperature, max_tokens=args.answer_max_tokens,
            top_p=top_p, top_k=top_k,
        )
        fix_outputs = llm.generate(fix_inputs, fix_params)

        for j, (idx, output) in enumerate(zip(truncated_indices, fix_outputs)):
            answer_text = output.outputs[0].text.strip()
            old_thinking = results[idx]["generated_text"]
            new_text = old_thinking + "\n</think>\n\n" + answer_text
            new_answer = extract_answer_vl(new_text, args.dataset)
            new_reasoning, new_raw = split_reasoning_and_raw(new_text, tokenizer, max_tokens=999999)

            results[idx]["generated_text"] = new_text
            results[idx]["model_answer"] = new_answer
            results[idx]["reasoning"] = new_reasoning
            results[idx]["raw_response"] = new_raw
            results[idx]["thinking_truncated"] = True
            results[idx]["total_tokens"] = len(tokenizer.encode(new_text))

        print(f"Fixed {len(truncated_indices)} truncated entries.")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_tokens = sum(r["total_tokens"] for r in results) / len(results) if results else 0
    print(f"\nSaved {len(results)} results to {args.output_path}")
    print(f"Avg tokens: {avg_tokens:.0f}")
    truncated_count = sum(1 for r in results if r.get("thinking_truncated", False))
    print(f"Truncated thinking (fixed): {truncated_count}")


if __name__ == "__main__":
    main()
