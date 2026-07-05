#!/usr/bin/env python3
"""
Step 4: Generate prefixed (compressed) answers for VL benchmarks.

Extends puma/gen_prefixed_answers.py with multimodal support.
Each prefixed answer prompt includes the original image.

Usage:
    python puma_vl/gen_prefixed_answers_vl.py \
        --final-candidates data_vl/final_candidates/model_dataset_final_candidates.json \
        --questions-file data_vl/questions_with_steps/model_dataset_with_steps.json \
        --output-file data_vl/prefixed_answers/model_dataset_prefixed_answers.json \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista --seed 42
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
_VL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_PROJECT_ROOT, _OFFLINE_DIR, _VL_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen_prefixed_answers import (
    compute_confidence, clamp_logprobs, find_token_span_in_ids,
    construct_reasoning_prefix,
)
from prompt_utils_vl import (
    get_task_type_vl, build_base_prompt_vl,
    get_confident_ending_vl, extract_answer_vl, load_image,
    extract_boxed_answer,
)

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(__name__ + ".debug")


def build_prefixed_prompt_vl(
    processor, model_name: str, question: str, image_path: str,
    reasoning_prefix: str, dataset: str, task_type: str,
    prompt_version: str = "default",
) -> str:
    """Build a prefixed answer prompt with image placeholder tokens."""
    base_prompt = build_base_prompt_vl(
        processor, model_name, question, image_path,
        task_type, prompt_version,
    )
    confident_ending = get_confident_ending_vl(dataset, is_trial=False)
    if reasoning_prefix.strip():
        if task_type == "mcq_vl":
            # MCQ: put confident ending after </think> for direct answer
            return f"{base_prompt}<think>{reasoning_prefix}\n</think>\n\n{confident_ending}"
        else:
            return f"{base_prompt}<think>{reasoning_prefix} {confident_ending}</think>"
    return base_prompt


def process_vl_prefixed_answers(
    final_candidates: List[Dict],
    questions_data: List[Dict],
    model_name: str,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    seed: int = None,
    dataset: str = "",
    prompt_version: str = "default",
    confidence_aggregation: str = "geometric",
) -> List[Dict]:
    """
    Generate final answers from reasoning prefixes for VL questions.
    """
    # Use spawn to avoid CUDA re-init issues with VL models
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    logger.info(f"Loading VL model {model_name}")
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 38000,
        "gpu_memory_utilization": 0.78,
        "limit_mm_per_prompt": {"image": 1},
    }
    if seed is not None:
        llm_kwargs["seed"] = seed
    llm = LLM(**llm_kwargs)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    task_type = get_task_type_vl(dataset)

    # Load sampling parameters from generation_config
    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
        logger.info(f"Loaded generation_config: temperature={temperature}, top_p={top_p}")
    except Exception:
        temperature, top_p, top_k = 0.6, 0.95, -1

    eos_tokens = [tokenizer.eos_token_id]
    for special in ["<|im_end|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None:
            eos_tokens.append(tid)

    prompts = []
    prompt_images = []
    meta = []
    results = []
    image_cache = {}  # image_path -> PIL Image

    for candidate in final_candidates:
        question_idx = candidate["question_idx"]
        stopped_len = candidate["stopped_len"]
        tokens_trial_answers = candidate.get("tokens_trial_answers", 0)
        tokens_trial_answers_online = candidate.get("tokens_trial_answers_online", tokens_trial_answers)

        question_obj = questions_data[question_idx - 1]
        question_text = question_obj["question"]
        image_path = question_obj.get("image_path", "")

        # full_reasoning: reuse original answer
        if candidate.get("stop_reason") == "full_reasoning":
            original_answer = question_obj.get("model_answer", "")
            original_response = question_obj.get("raw_response", "")
            reasoning_text = question_obj.get("reasoning", "")
            reasoning_tokens = len(tokenizer(reasoning_text, add_special_tokens=False).input_ids) if reasoning_text else 0
            response_tokens = len(tokenizer(original_response, add_special_tokens=False).input_ids) if original_response else 0

            result = {
                "question_idx": question_idx,
                "stopped_len": stopped_len,
                "original_len_reasoning_steps": candidate.get("original_len_reasoning_steps"),
                "saved_steps": 0,
                "question": question_text,
                "image_path": image_path,
                "reasoning_prefix": "",
                "model_response": original_response,
                "final_answer": original_answer,
                "count_reasoning_tokens": reasoning_tokens,
                "count_generated_tokens": response_tokens,
                "total_tokens": reasoning_tokens + response_tokens,
                "generated_trial_answers": candidate.get("generated_trial_answers", 0),
                "tokens_trial_answers": tokens_trial_answers,
                "tokens_trial_answers_online": tokens_trial_answers_online,
                "confidence": candidate.get("stop_confidence") or 0.0,
                "new_thinking": False,
                "success": True,
                "stop_reason": "full_reasoning",
                "stop_confidence": candidate.get("stop_confidence"),
                "stop_threshold": candidate.get("stop_threshold"),
                "consecutive_confidences": candidate.get("consecutive_confidences"),
                "confidence_trajectory": candidate.get("confidence_trajectory"),
                "step_similarities": candidate.get("step_similarities"),
            }
            results.append(result)
            continue

        # Build prefixed prompt with image
        reasoning_steps = question_obj["reasoning_steps"][:stopped_len]
        reasoning_prefix = construct_reasoning_prefix(reasoning_steps)

        prompt = build_prefixed_prompt_vl(
            processor, model_name, question_text, image_path,
            reasoning_prefix, dataset, task_type, prompt_version,
        )
        if image_path not in image_cache:
            image_cache[image_path] = load_image(image_path)
        prompts.append(prompt)
        prompt_images.append(image_cache[image_path])

        reasoning_token_count = len(tokenizer(reasoning_prefix, add_special_tokens=False).input_ids)

        meta.append({
            "question_idx": question_idx,
            "stopped_len": stopped_len,
            "original_len_reasoning_steps": candidate.get("original_len_reasoning_steps"),
            "saved_steps": candidate.get("original_len_reasoning_steps", 0) - stopped_len,
            "question": question_text,
            "image_path": image_path,
            "reasoning_prefix": reasoning_prefix,
            "reasoning_token_count": reasoning_token_count,
            "generated_trial_answers": candidate.get("generated_trial_answers", 0),
            "tokens_trial_answers": tokens_trial_answers,
            "tokens_trial_answers_online": tokens_trial_answers_online,
            "stop_reason": candidate.get("stop_reason"),
            "stop_confidence": candidate.get("stop_confidence"),
            "stop_threshold": candidate.get("stop_threshold"),
            "consecutive_confidences": candidate.get("consecutive_confidences"),
            "confidence_trajectory": candidate.get("confidence_trajectory"),
            "step_similarities": candidate.get("step_similarities"),
            "trial_final_answer": candidate.get("final_answer", ""),
        })

    logger.info(f"Full reasoning reused: {len(results)}/{len(final_candidates)}")

    if not prompts:
        logger.info("All candidates are full_reasoning — no vLLM generation needed.")
        return results

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens,
        top_p=top_p, top_k=top_k,
        logprobs=1, stop_token_ids=eos_tokens,
    )

    vllm_inputs = [
        {"prompt": p, "multi_modal_data": {"image": img}}
        for p, img in zip(prompts, prompt_images)
    ]

    logger.info(f"Submitting {len(vllm_inputs)} VL prefixed prompts to vLLM...")
    all_outputs = llm.generate(vllm_inputs, sampling_params)
    logger.info("vLLM generation complete, processing results...")

    for out, info in zip(all_outputs, meta):
        gen = out.outputs[0]
        response_text = gen.text
        token_ids = gen.token_ids

        # Extract answer
        if task_type == "mcq_vl":
            model_answer = extract_answer_vl("\\boxed{" + response_text, dataset)
        else:
            model_answer = extract_answer_vl(response_text, dataset)

        extra_thinking = "<think>" in response_text

        result = {
            "question_idx": info["question_idx"],
            "stopped_len": info["stopped_len"],
            "original_len_reasoning_steps": info["original_len_reasoning_steps"],
            "saved_steps": info["saved_steps"],
            "question": info["question"],
            "image_path": info["image_path"],
            "reasoning_prefix": info["reasoning_prefix"],
            "model_response": response_text,
            "final_answer": model_answer,
            "count_reasoning_tokens": info["reasoning_token_count"],
            "count_generated_tokens": len(token_ids),
            "total_tokens": info["reasoning_token_count"] + len(token_ids),
            "generated_trial_answers": info["generated_trial_answers"],
            "tokens_trial_answers": info["tokens_trial_answers"],
            "tokens_trial_answers_online": info.get("tokens_trial_answers_online", info["tokens_trial_answers"]),
            "confidence": 0.0,
            "new_thinking": extra_thinking,
            "success": bool(model_answer),
            "stop_reason": info.get("stop_reason"),
            "stop_confidence": info.get("stop_confidence"),
            "stop_threshold": info.get("stop_threshold"),
            "consecutive_confidences": info.get("consecutive_confidences"),
            "confidence_trajectory": info.get("confidence_trajectory"),
            "step_similarities": info.get("step_similarities"),
        }

        if not model_answer:
            logger.warning(f"Q{info['question_idx']}: final_answer empty (new_thinking={extra_thinking})")

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="VL Step 4: Generate prefixed answers")
    parser.add_argument("--final-candidates", type=str, required=True)
    parser.add_argument("--questions-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--prompt-version", type=str, default="default", choices=["default", "direct"])
    parser.add_argument("--confidence-aggregation", type=str, default="geometric",
                        choices=["geometric", "arithmetic"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_dir = "logs/gen_prefixed_answers_vl_debug"
    os.makedirs(debug_log_dir, exist_ok=True)
    debug_handler = logging.FileHandler(f"{debug_log_dir}/debug_{timestamp}.log")
    debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    debug_logger.addHandler(debug_handler)
    debug_logger.setLevel(logging.INFO)
    debug_logger.propagate = False

    with open(args.final_candidates, "r") as f:
        final_candidates = json.load(f)
    with open(args.questions_file, "r") as f:
        questions_data = json.load(f)

    logger.info(f"Processing {len(final_candidates)} VL candidates")

    results = process_vl_prefixed_answers(
        final_candidates=final_candidates,
        questions_data=questions_data,
        model_name=args.model,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        dataset=args.dataset,
        prompt_version=args.prompt_version,
        confidence_aggregation=args.confidence_aggregation,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Done! Total: {len(results)}, Successful: {successful}")


if __name__ == "__main__":
    main()
