import json
import time
import re
import os
import sys

# Add project root to sys.path so `baselines.*` imports work when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from baselines.utils.math_util import *
from baselines.utils.generation_config import load_generation_params
from transformers import AutoTokenizer


def endswith(seq, suffix):
    return len(seq) >= len(suffix) and seq[-len(suffix):] == suffix


def main():
    MODEL_NAME = sys.argv[1]
    DATASET_LIMIT = int(sys.argv[2])
    DATASET_PATH = sys.argv[3]
    OUTPUT_PATH = sys.argv[4]
    MAX_TOKENS = 32768
    gen_params = load_generation_params(MODEL_NAME)
    TEMPERATURE_REASONING = gen_params["temperature"]
    TOP_P = gen_params["top_p"]
    TOP_K = gen_params["top_k"]

    # Set tensor parallel size based on CUDA_VISIBLE_DEVICES
    TENSOR_PARALLEL_SIZE = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
    print(f"Using {TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism")

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    model_name = MODEL_NAME
   
    # --- Load VLLM Model ---
    print(f"Loading model: {model_name}...")
    if "70B" not in model_name:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            max_model_len=MAX_TOKENS,
            gpu_memory_utilization=0.85
        )
    else:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            max_model_len=8192,
            max_num_seqs=35,
            gpu_memory_utilization=0.85
        )
    print("Model loaded.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Boost </think> via logit_bias (vLLM V1 does not support per-request logits_processors)
    # This applies a static bias toward </think>, approximating the original dynamic boost.
    boost_token_id = tokenizer("</think>", add_special_tokens=False)["input_ids"][0]

    sampling_params = SamplingParams(
        temperature=TEMPERATURE_REASONING,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
        logit_bias={boost_token_id: 5.0},
        stop=["</think>"],
        include_stop_str_in_output=True,
    )

    questions = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            questions.append(json.loads(line)["question"])

    questions = questions[:DATASET_LIMIT]

    print("Preparing chat inputs...")
    start_time = time.time()
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)

    # Phase 1: Generate reasoning with boosted </think>
    print(f"Phase 1: Generating reasoning for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    # Phase 2: Generate answers by continuing from </think>
    print("Phase 2: Generating answers...")
    max_answer_tokens = 512
    max_prompt_len = MAX_TOKENS - max_answer_tokens  # leave room for answer

    answer_prompts = []
    reasoning_texts = []
    for i, output in enumerate(outputs):
        reasoning_text = output.outputs[0].text
        reasoning_texts.append(reasoning_text)
        # Build continuation prompt: original prompt + reasoning + </think>\n\boxed
        if not reasoning_text.endswith("</think>"):
            reasoning_text += "</think>"
        full_prompt = prompts[i] + reasoning_text + "\n\\boxed"

        # Truncate if exceeds max_model_len
        prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
            full_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        answer_prompts.append(full_prompt)

    sampling_params_answer = SamplingParams(
        temperature=0.0,
        max_tokens=max_answer_tokens,
        stop=["\n"],
    )
    answer_outputs = llm.generate(answer_prompts, sampling_params_answer)

    print("Processing results...")
    results = []
    for i, (reasoning_text, ans_output) in enumerate(zip(reasoning_texts, answer_outputs)):
        question = questions[i]
        answer_text = "\\boxed" + ans_output.outputs[0].text
        # Full generated text for downstream evaluation
        if not reasoning_text.endswith("</think>"):
            reasoning_text_full = reasoning_text + "</think>"
        else:
            reasoning_text_full = reasoning_text
        generated_text = reasoning_text_full + "\n" + answer_text

        reasoning = reasoning_text.replace("</think>", "").strip()
        answer = my_answer_extraction(generated_text)

        result_data = {
            "question": question,
            "answer": answer,
            "confidence": "",
            "reasoning": reasoning,
            "generated_text": generated_text
        }
        results.append(result_data)

    print(f"Writing results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds.")

    # Save wall time metadata
    walltime_file = os.path.join(os.path.dirname(OUTPUT_PATH), "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({"wall_time_seconds": round(elapsed_time, 2), "num_questions": len(results), "method": "boost"}, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Clean up GPU memory
    print("Cleaning up GPU memory...")
    del llm
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
