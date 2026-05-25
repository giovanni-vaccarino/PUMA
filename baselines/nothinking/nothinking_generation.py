#!/usr/bin/env python3
"""
NoThinking: bypass the explicit thinking process by prefilling an empty thinking box.

Based on: "Reasoning Models Can Be Effective Without Thinking" (Ma et al., 2025)
https://arxiv.org/abs/2504.09858

The key idea is to prefill the assistant response with a dummy thinking block:
    <think>
    Okay, I think I have finished thinking.
    </think>
Then let the model directly generate the final solution and answer.

This uses the `fake_assistant` role in the r1_qwen.jinja chat template, which
outputs the content without appending an EOS token, allowing continuation.

Usage:
    python -m baselines.nothinking.nothinking_generation \
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
from baselines.utils.math_util import my_answer_extraction, extract_code_answer, CODE_SYS_PROMPT
from baselines.utils.generation_config import load_generation_params


# The dummy thinking block that bypasses reasoning
NOTHINKING_PREFILL = "<think>\nOkay, I think I have finished thinking.\n</think>\n"


def load_chat_template(filepath="baselines/r1_qwen.jinja"):
    """Load chat template from a Jinja file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        template = f.read()
    return template


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

    is_code = "livecodebench" in DATASET_PATH.lower()
    if is_code:
        sys_prompt = CODE_SYS_PROMPT
        print("Code task detected: using CODE_SYS_PROMPT")
    else:
        sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    # Load model
    print(f"Loading model: {MODEL_NAME}...")
    if "70B" not in MODEL_NAME:
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            max_model_len=MAX_TOKENS,
            gpu_memory_utilization=0.85
        )
    else:
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            max_model_len=8192,
            max_num_seqs=35,
            gpu_memory_utilization=0.85
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load the custom chat template that supports fake_assistant role
    tokenizer.chat_template = load_chat_template()

    print("Model loaded.")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    # Load questions
    questions = []
    question_ids = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            question_ids.append(item.get("question_id", ""))
    questions = questions[:DATASET_LIMIT]
    question_ids = question_ids[:DATASET_LIMIT]

    # Build prompts with NoThinking prefill
    print("Building NoThinking prompts...")
    prompts = []

    for q in questions:
        # Use fake_assistant role to prefill the dummy thinking box
        # The template outputs <|Assistant|> + content WITHOUT EOS, allowing continuation
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
            {"role": "fake_assistant", "content": NOTHINKING_PREFILL}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompts.append(formatted_prompt)

    # Generate
    start_time = time.time()
    print(f"Generating NoThinking answers for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    print("Processing results...")
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing")):
        generated_text = output.outputs[0].text.strip()

        result = {
            "question": questions[i],
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

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", exist_ok=True)
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed_time = time.time() - start_time
    print(f"Done! Processed {len(results)} questions in {elapsed_time:.2f} seconds.")

    # Save wall time metadata
    walltime_file = os.path.join(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({"wall_time_seconds": round(elapsed_time, 2), "num_questions": len(results), "method": "nothinking"}, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Clean up GPU memory
    print("Cleaning up GPU memory...")
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()