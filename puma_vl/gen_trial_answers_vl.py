#!/usr/bin/env python3
"""
Step 2: Generate trial answers for VL benchmarks.

Extends puma/gen_trial_answers.py with multimodal support.
Each trial answer prompt includes the original image via multi_modal_data.

Usage:
    python puma_vl/gen_trial_answers_vl.py \
        --questions-file data_vl/questions_with_steps/model_dataset_with_steps.json \
        --output-file data_vl/trial_answers/model_dataset_trial_answers.json \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --max-tokens 30 --seed 42
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Add project root and offline dirs to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
_VL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_PROJECT_ROOT, _OFFLINE_DIR, _VL_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import confidence computation from text pipeline (modality-independent)
from gen_trial_answers import (
    compute_confidence, find_boxed_content_token_span,
    extract_first_braced_content, parse_response,
    construct_reasoning_prefix, extract_logprob_from_step,
    compute_gpqa_confidence,
)
from prompt_utils_vl import (
    get_model_type_vl, get_task_type_vl,
    build_base_prompt_vl, get_confident_ending_vl, load_image,
)

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(__name__ + ".debug")


def build_trial_prompt_vl(
    processor, model_name: str, question: str, image_path: str,
    reasoning_prefix: str, dataset: str, task_type: str,
    prompt_version: str = "default",
) -> str:
    """Build a trial answer prompt with image placeholder tokens."""
    base_prompt = build_base_prompt_vl(
        processor, model_name, question, image_path,
        task_type, prompt_version,
    )
    confident_ending = get_confident_ending_vl(dataset, is_trial=True)
    if reasoning_prefix.strip():
        return f"{base_prompt}<think>{reasoning_prefix} {confident_ending}"
    return base_prompt


def process_vl_trial_answers(
    questions_data: List[Dict],
    model_name: str,
    max_tokens: int = 30,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 30,
    tensor_parallel_size: int = 1,
    respect_embedding_filter: bool = False,
    seed: int = None,
    dataset: str = "",
    trial_decoding: str = "sampling",
    prompt_version: str = "default",
    confidence_mode: str = "token_in_boxed",
    confidence_aggregation: str = "geometric",
) -> List[Dict]:
    """
    Batch generate trial answers for VL questions.

    Same logic as text gen_trial_answers, but prompts include images.
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

    skipped_results = []

    # Collect non-filtered steps
    candidate_steps = []
    for q_idx, question_obj in enumerate(questions_data, start=1):
        question_text = question_obj["question"]
        image_path = question_obj.get("image_path", "")
        reasoning_steps = question_obj.get("reasoning_steps", [])
        original_len = len(reasoning_steps)

        should_generate_trial = question_obj.get("should_generate_trial", None)
        step_similarities = question_obj.get("step_similarities", None)

        for stopped_len in range(1, original_len + 1):
            step_idx = stopped_len - 1

            if respect_embedding_filter and should_generate_trial is not None:
                if not should_generate_trial[step_idx]:
                    similarity = step_similarities[step_idx] if step_similarities else None
                    skipped_results.append({
                        "question_idx": q_idx,
                        "stopped_len": stopped_len,
                        "original_len_reasoning_steps": original_len,
                        "question": question_text,
                        "skipped": True,
                        "skip_reason": "embedding_filter",
                        "similarity": similarity,
                        "success": True,
                    })
                    continue

            reasoning_prefix = construct_reasoning_prefix(reasoning_steps[:stopped_len])
            candidate_steps.append({
                "question_idx": q_idx,
                "stopped_len": stopped_len,
                "original_len_reasoning_steps": original_len,
                "question": question_text,
                "image_path": image_path,
                "reasoning_prefix": reasoning_prefix,
            })

    if respect_embedding_filter:
        total_steps = len(candidate_steps) + len(skipped_results)
        logger.info(f"Embedding filter: {len(skipped_results)}/{total_steps} steps skipped")

    # Build prompts with images (cache per question to avoid redundant loads)
    prompts = []
    prompt_images = []
    meta = []
    image_cache = {}  # image_path -> PIL Image

    for info in candidate_steps:
        prompt = build_trial_prompt_vl(
            processor, model_name, info["question"], info["image_path"],
            info["reasoning_prefix"], dataset, task_type, prompt_version,
        )
        img_path = info["image_path"]
        if img_path not in image_cache:
            image_cache[img_path] = load_image(img_path)
        prompts.append(prompt)
        prompt_images.append(image_cache[img_path])
        meta.append(info)

    logger.info(f"Loaded {len(image_cache)} unique images for {len(candidate_steps)} prompts")

    n_logprobs = 5 if task_type == "mcq_vl" else 1

    logger.info(f"Total prompts to generate: {len(prompts)}")
    logger.info(f"Trial decoding mode: {trial_decoding}")

    if trial_decoding == "greedy":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=n_logprobs)
    elif trial_decoding == "default":
        sampling_params = SamplingParams(max_tokens=max_tokens, logprobs=n_logprobs)
    else:
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, top_k=top_k, logprobs=n_logprobs,
        )

    # Generate (multimodal)
    vllm_inputs = [
        {"prompt": p, "multi_modal_data": {"image": img}}
        for p, img in zip(prompts, prompt_images)
    ]

    logger.info(f"Submitting {len(vllm_inputs)} VL prompts to vLLM...")
    all_outputs = llm.generate(vllm_inputs, sampling_params)
    logger.info("vLLM generation complete, processing results...")

    results = []
    for out, info in zip(all_outputs, meta):
        gen = out.outputs[0]
        text = gen.text
        token_ids = gen.token_ids
        token_logprobs = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []

        # Answer extraction and confidence (same logic as text)
        if confidence_mode == "token_in_boxed":
            boxed_content = extract_first_braced_content(text)
            answer = boxed_content if boxed_content else parse_response(text)
            start_idx, end_idx = find_boxed_content_token_span(token_ids, tokenizer)
            if start_idx < end_idx:
                answer_token_logprobs = token_logprobs[start_idx:end_idx]
            else:
                answer_token_logprobs = token_logprobs
        else:
            answer = parse_response(text)
            if answer:
                answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
                answer_token_logprobs = token_logprobs[:len(answer_ids)]
            else:
                answer_token_logprobs = token_logprobs

        geo_conf = compute_confidence(answer_token_logprobs, confidence_aggregation)

        # MCQ: softmax over choice logprobs (like GPQA)
        if task_type == "mcq_vl" and confidence_mode == "token_in_boxed":
            mcq_conf = compute_gpqa_confidence(gen.logprobs, token_ids, tokenizer, temperature=0.5)
            if mcq_conf > 0:
                geo_conf = mcq_conf

        result = {
            "question_idx": info["question_idx"],
            "stopped_len": info["stopped_len"],
            "original_len_reasoning_steps": info["original_len_reasoning_steps"],
            "question": info["question"],
            "reasoning_prefix": info["reasoning_prefix"],
            "model_response": text,
            "final_answer": answer,
            "count_reasoning_tokens": len(
                tokenizer(info["reasoning_prefix"], add_special_tokens=False).input_ids
            ),
            "count_generated_tokens": len(token_ids),
            "confidence": geo_conf,
            "count_answer_tokens": len(answer_token_logprobs),
            "new_thinking": "<think>" in text,
            "success": True,
        }
        results.append(result)

    all_results = results + skipped_results
    all_results.sort(key=lambda x: (x["question_idx"], x["stopped_len"]))
    return all_results


def main():
    parser = argparse.ArgumentParser(description="VL Step 2: Generate trial answers")
    parser.add_argument("--questions-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--respect-embedding-filter", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--trial-decoding", type=str, choices=["greedy", "sampling", "default"], default="sampling")
    parser.add_argument("--prompt-version", type=str, default="default", choices=["default", "direct"])
    parser.add_argument("--confidence-mode", type=str, default="token_in_boxed",
                        choices=["first_line", "token_in_boxed"])
    parser.add_argument("--confidence-aggregation", type=str, default="geometric",
                        choices=["geometric", "arithmetic"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_dir = "logs/gen_trial_answers_vl_debug"
    os.makedirs(debug_log_dir, exist_ok=True)
    debug_handler = logging.FileHandler(f"{debug_log_dir}/debug_{timestamp}.log")
    debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    debug_logger.addHandler(debug_handler)
    debug_logger.setLevel(logging.INFO)
    debug_logger.propagate = False

    logger.info(f"Loading questions from {args.questions_file}")
    with open(args.questions_file, "r") as f:
        questions_data = json.load(f)

    logger.info(f"Processing {len(questions_data)} VL questions")
    results = process_vl_trial_answers(
        questions_data=questions_data,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        respect_embedding_filter=args.respect_embedding_filter,
        seed=args.seed,
        dataset=args.dataset,
        trial_decoding=args.trial_decoding,
        prompt_version=args.prompt_version,
        confidence_mode=args.confidence_mode,
        confidence_aggregation=args.confidence_aggregation,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped", False))
    logger.info(f"Done! Total: {total}, Generated: {total - skipped}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
