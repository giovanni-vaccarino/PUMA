#!/usr/bin/env python3
"""
Generate vanilla answers (full reasoning without logits intervention).
Used as baseline to compute compression rate.

Usage:
    python -m baselines.think_token_adjustment.vanilla_generation \
        <MODEL_NAME> <LIMIT> <DATASET_PATH> <OUTPUT_PATH>
"""

import json
import os
import sys
import time

# Add project root to sys.path so `baselines.*` imports work when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from baselines.utils.math_util import my_answer_extraction
from baselines.utils.generation_config import load_generation_params


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

    TENSOR_PARALLEL_SIZE = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
    print(f"Using {TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism")

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

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
    questions = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    questions = questions[:DATASET_LIMIT]

    # Generate
    print("Generating vanilla answers...")
    start_time = time.time()
    prompts = []

    for q in questions:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)
        
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    print("Processing results...")
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing")):
        generated_text = output.outputs[0].text.strip()
        answer = my_answer_extraction(generated_text)
        
        # Extract reasoning
        try:
            if "</think>" in generated_text:
                reasoning = generated_text.split("</think>")[0].strip()
            else:
                reasoning = generated_text
        except:
            reasoning = ""
        
        results.append({
            "question": questions[i],
            "answer": answer,
            "reasoning": reasoning,
            "generated_text": generated_text
        })

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed_time = time.time() - start_time
    print(f"Done! Processed {len(results)} questions in {elapsed_time:.2f} seconds.")

    # Save wall time metadata
    walltime_file = os.path.join(os.path.dirname(OUTPUT_PATH), "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({"wall_time_seconds": round(elapsed_time, 2), "num_questions": len(results), "method": "vanilla"}, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Clean up GPU memory
    print("Cleaning up GPU memory...")
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
