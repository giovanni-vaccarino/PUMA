#!/usr/bin/env python3
"""
CCoT (Constrained Chain-of-Thought) generation: encourage concise reasoning
by adding a word-budget constraint to the system prompt.

Based on: "Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost"
          (Nayab et al., 2024)

The key idea is to append a length constraint to the standard CoT system prompt:
    "Please reason step by step, and put your final answer within \\boxed{}.
     Limit your reasoning to {budget_words} words."

This is a pure prompt-engineering approach — no model modification, no prefill,
no early exit.  The model still generates freely via <think>…</think>, but the
prompt nudges it toward shorter reasoning chains.

Usage:
    python -m baselines.concise.concise_generation \
        <MODEL_NAME> <LIMIT> <DATASET_PATH> <OUTPUT_PATH> [--budget-words 45]
"""

import json
import os
import sys
import time
import argparse

# Add project root to sys.path so `baselines.*` imports work when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from baselines.utils.math_util import my_answer_extraction, extract_code_answer, CODE_SYS_PROMPT
from baselines.utils.generation_config import load_generation_params


def main():
    parser = argparse.ArgumentParser(
        description="CCoT (Constrained CoT) generation with word-budget prompt",
    )
    parser.add_argument("model", type=str, help="HuggingFace model identifier")
    parser.add_argument("limit", type=int, help="Max number of questions to process")
    parser.add_argument("dataset", type=str, help="Path to benchmark JSONL file")
    parser.add_argument("output", type=str, help="Path to output JSONL file")
    parser.add_argument("--budget-words", type=int, default=45,
                        help="Word budget for CCoT constraint (default: 45)")
    args = parser.parse_args()

    MODEL_NAME = args.model
    DATASET_LIMIT = args.limit
    DATASET_PATH = args.dataset
    OUTPUT_PATH = args.output
    BUDGET_WORDS = args.budget_words

    MAX_TOKENS = 8192
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

    # ---- CCoT system prompt ----
    is_code = "livecodebench" in DATASET_PATH.lower()
    if is_code:
        sys_prompt = CODE_SYS_PROMPT + f" Limit your reasoning to {BUDGET_WORDS} words."
        print(f"Code task detected")
    else:
        sys_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}. "
            f"Limit your reasoning to {BUDGET_WORDS} words."
        )
    print(f"CCoT sys_prompt: {sys_prompt}")

    # ---- Load model ----
    print(f"Loading model: {MODEL_NAME}...")
    if "70B" not in MODEL_NAME:
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            max_model_len=MAX_TOKENS,
            gpu_memory_utilization=0.85,
        )
    else:
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            max_model_len=8192,
            max_num_seqs=35,
            gpu_memory_utilization=0.85,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded.")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    # ---- Load questions ----
    questions = []
    question_ids = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            question_ids.append(item.get("question_id", ""))
    questions = questions[:DATASET_LIMIT]
    question_ids = question_ids[:DATASET_LIMIT]

    # ---- Generate ----
    print(f"Generating CCoT-{BUDGET_WORDS} answers for {len(questions)} questions...")
    start_time = time.time()

    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    outputs = llm.generate(prompts, sampling_params)

    # ---- Process results ----
    print("Processing results...")
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing")):
        generated_text = output.outputs[0].text.strip()

        # Extract reasoning (everything before </think>)
        if "</think>" in generated_text:
            reasoning = generated_text.split("</think>")[0].strip()
        else:
            reasoning = generated_text

        result = {
            "question": questions[i],
            "reasoning": reasoning,
            "generated_text": generated_text,
        }

        if is_code:
            code = extract_code_answer(generated_text)
            result["answer"] = code
            result["extracted_code"] = code
            result["question_id"] = question_ids[i]
        else:
            result["answer"] = my_answer_extraction(generated_text)

        results.append(result)

    # ---- Save ----
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed_time = time.time() - start_time
    print(f"Done! Processed {len(results)} questions in {elapsed_time:.2f} seconds.")

    # Save wall time metadata
    walltime_file = os.path.join(os.path.dirname(OUTPUT_PATH), "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed_time, 2),
            "num_questions": len(results),
            "method": f"concise_ccot{BUDGET_WORDS}",
        }, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Clean up GPU memory
    print("Cleaning up GPU memory...")
    del llm
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()