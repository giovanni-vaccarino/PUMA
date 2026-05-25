#!/usr/bin/env python3
"""
Vanilla code generation (full CoT without early exit) for LiveCodeBench.

Usage:
    python -m baselines.think_token_adjustment.vanilla_generation_code \
        <MODEL_NAME> <LIMIT> <DATASET_PATH> <OUTPUT_PATH>
"""

import json
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from baselines.utils.generation_config import load_generation_params


def extract_code(text):
    """Extract the last Python code block from generated text."""
    import re
    # Find all ```python ... ``` blocks
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Fallback: if no code block, try to extract after </think>
    if "</think>" in text:
        after_think = text.split("</think>", 1)[1].strip()
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', after_think, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    return ""


def main():
    MODEL_NAME = sys.argv[1]
    DATASET_LIMIT = int(sys.argv[2])
    DATASET_PATH = sys.argv[3]
    OUTPUT_PATH = sys.argv[4]

    MAX_TOKENS = 32768
    gen_params = load_generation_params(MODEL_NAME)
    TEMPERATURE = gen_params["temperature"]
    TOP_P = gen_params["top_p"]
    TOP_K = gen_params["top_k"]

    TENSOR_PARALLEL_SIZE = (
        len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else 1
    )
    print(f"Using {TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism")

    sys_prompt = (
        "You are a helpful assistant that solves programming problems. "
        "Think step by step, then provide your final solution as a complete "
        "Python program within a ```python code block."
    )

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        max_model_len=MAX_TOKENS + 2048,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    # Load questions
    questions_data = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            questions_data.append(json.loads(line))
    questions_data = questions_data[:DATASET_LIMIT]

    # Build prompts
    print("Building prompts...")
    prompts = []
    for item in questions_data:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": item["question"]},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)

    # Generate
    print(f"Generating code for {len(prompts)} problems...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    print("Processing results...")
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing")):
        generated_text = output.outputs[0].text.strip()
        code = extract_code(generated_text)
        num_tokens = len(output.outputs[0].token_ids)

        results.append({
            "question_id": questions_data[i].get("question_id", i),
            "platform": questions_data[i].get("platform", ""),
            "difficulty": questions_data[i].get("difficulty", ""),
            "question": questions_data[i]["question"],
            "generated_text": generated_text,
            "extracted_code": code,
            "num_tokens": num_tokens,
        })

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed_time = time.time() - start_time

    # Stats
    total_tokens = sum(r["num_tokens"] for r in results)
    code_extracted = sum(1 for r in results if r["extracted_code"])
    print(f"\nDone! {len(results)} problems in {elapsed_time:.2f}s")
    print(f"Code extracted: {code_extracted}/{len(results)}")
    print(f"Avg tokens/problem: {total_tokens / len(results):.0f}")

    # Save wall time
    walltime_file = os.path.join(os.path.dirname(OUTPUT_PATH), "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed_time, 2),
            "num_questions": len(results),
            "method": "vanilla_code",
        }, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
