import json
import logging
import argparse
import sys
from typing import List, Dict
from .prompt_utils import build_base_prompt, get_confident_ending, extract_boxed_answer
from .confidence_utils import (
    clamp_logprobs,
    compute_confidence,
    construct_reasoning_prefix,
    extract_logprob_from_step,
)

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(__name__ + ".debug")


def parse_response(message_content: str):
    """Extract the final math answer from a generated response."""
    return extract_boxed_answer(message_content)


def load_json(filepath: str) -> List[Dict]:
    """Load JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: List[Dict], output_path: str):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def build_prompt(
    tokenizer,
    model_name: str,
    question: str,
    reasoning_prefix: str,
    is_trial_answer: bool = False,
) -> tuple:
    """
    Build the vLLM prompt with chat template, instruction, and reasoning prefix.

    Returns:
        tuple: (prompt_text, reasoning_tokens_count)
    """
    base_prompt = build_base_prompt(tokenizer, model_name, question)
    confident_ending = get_confident_ending(is_trial=is_trial_answer)

    # If a reasoning prefix exists, append <think> block
    think_block = ""
    reasoning_tokens = 0

    if reasoning_prefix.strip():
        if is_trial_answer:
            think_block = f"<think>{reasoning_prefix} {confident_ending}"
        else:
            think_block = f"<think>{reasoning_prefix} {confident_ending}</think>"
        prompt = base_prompt + think_block

        # Count reasoning tokens
        reasoning_ids = tokenizer(think_block, return_tensors="pt").input_ids
        reasoning_tokens = reasoning_ids.shape[1]
    else:
        prompt = base_prompt

    return prompt, reasoning_tokens


def find_token_span_in_ids(full_ids, answer_ids: List[int]) -> tuple:
    """Find the span of answer_ids within full_ids."""
    # Convert to list to ensure consistent comparison
    # (vLLM v0.15+ returns token_ids as tuple, tokenizer returns list)
    full_ids = list(full_ids)
    for i in range(len(full_ids) - len(answer_ids) + 1):
        if full_ids[i:i+len(answer_ids)] == answer_ids:
            return i, i + len(answer_ids)
    return 0, 0


def process_questions_with_prefixes(
    final_candidates: List[Dict],
    questions_data: List[Dict],
    model_name: str,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    clamp_logprobs_threshold: float = None,
    seed: int = None,
    confidence_aggregation: str = "geometric",
) -> List[Dict]:
    """
    Process questions with reasoning prefixes using vLLM batched inference.

    Temperature/top_p are loaded from the model's generation_config.json,
    so final-answer decoding follows the released model's default settings.

    Args:
        final_candidates: Array of final candidate objects with question_idx, stopped_len, tokens_trial_answers, etc.
        questions_data: Array of question objects with reasoning_steps
        model_name: Model to use for vLLM
        max_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        clamp_logprobs_threshold: If set, clamp logprobs > threshold to 0.0 (for vLLM V0 compatibility)

    Returns:
        List of results with model responses
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logger.info(f"Loading vLLM model {model_name}")
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 38000,
        "gpu_memory_utilization": 0.78,
    }
    if seed is not None:
        llm_kwargs["seed"] = seed
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load sampling parameters from the model's generation_config.json.
    from transformers import GenerationConfig
    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
        logger.info(f"Loaded generation_config: temperature={temperature}, top_p={top_p}, top_k={top_k}")
    except Exception as e:
        logger.info(f"Failed to load generation_config: {e}")
        logger.info("Using defaults: temperature=0.6, top_p=0.95, top_k=-1")
        temperature = 0.6
        top_p = 0.95
        top_k = -1

    # Get EOS token IDs matching call_local logic
    eos_tokens = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    ]
    eos_tokens = [t for t in eos_tokens if t is not None]

    prompts = []
    meta = []
    results = []  # Collect results (full_reasoning reuse + vLLM generation)

    # Build batch of prompts
    for candidate in final_candidates:
        question_idx = candidate["question_idx"]
        stopped_len = candidate["stopped_len"]
        tokens_trial_answers = candidate.get("tokens_trial_answers", 0)

        logger.info(f"Processing question {question_idx} with {stopped_len} reasoning steps as prefix")

        # question_idx is 1-indexed, so we need to subtract 1 for 0-indexed array
        question_obj = questions_data[question_idx - 1]

        # Extract the question text
        question_text = question_obj["question"]

        # full_reasoning: reuse original answer directly (no re-generation needed)
        if candidate.get("stop_reason") == "full_reasoning":
            original_answer = question_obj.get("model_answer", "")
            original_response = question_obj.get("raw_response", "")
            response_tokens = len(tokenizer(original_response, add_special_tokens=False).input_ids) if original_response else 0

            result = {
                "question_idx": question_idx,
                "stopped_len": stopped_len,
                "original_len_reasoning_steps": candidate.get("original_len_reasoning_steps"),
                "saved_steps": 0,
                "question": question_text,
                "reasoning_prefix": "",
                "model_response": original_response,
                "final_answer": original_answer,
                "count_reasoning_tokens": 0,
                "count_generated_tokens": response_tokens,
                "total_tokens": response_tokens,
                "generated_trial_answers": candidate.get("generated_trial_answers", 0),
                "tokens_trial_answers": tokens_trial_answers,
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
            logger.info(f"Q{question_idx}: full_reasoning — reusing original answer (skip re-generation)")
            continue

        # Get the first stopped_len reasoning steps
        reasoning_steps = question_obj["reasoning_steps"][:stopped_len]

        # Construct the reasoning prefix
        reasoning_prefix = construct_reasoning_prefix(reasoning_steps)

        logger.info(f"Question: {question_text[:100]}...")
        logger.info(f"Using {len(reasoning_steps)} reasoning steps as prefix")
        
        # Build the truncated-reasoning prompt for final answer generation.
        prompt, reasoning_token_count = build_prompt(
            tokenizer,
            model_name,
            question_text,
            reasoning_prefix,
            is_trial_answer=False,
        )
        prompts.append(prompt)
        
        meta.append({
            "question_idx": question_idx,
            "stopped_len": stopped_len,
            "original_len_reasoning_steps": candidate.get("original_len_reasoning_steps"),
            "saved_steps": candidate.get("original_len_reasoning_steps", 0) - stopped_len,
            "question": question_text,
            "reasoning_prefix": reasoning_prefix,
            "reasoning_token_count": reasoning_token_count,
            "generated_trial_answers": candidate.get("generated_trial_answers", 0),
            "tokens_trial_answers": tokens_trial_answers,
            # Propagate Step 3 metadata for Step 5 statistics
            "stop_reason": candidate.get("stop_reason"),
            "stop_confidence": candidate.get("stop_confidence"),
            "stop_threshold": candidate.get("stop_threshold"),
            "consecutive_confidences": candidate.get("consecutive_confidences"),
            "confidence_trajectory": candidate.get("confidence_trajectory"),
            "step_similarities": candidate.get("step_similarities"),
            "trial_final_answer": candidate.get("final_answer", ""),
        })

    # Debug: show first prompt (first 500 + last 500 chars)
    if prompts:
        logger.info(f"\n=== Sample prompt (first prefixed, {len(prompts[0])} chars) ===")
        if len(prompts[0]) <= 1000:
            logger.info(prompts[0])
        else:
            logger.info(prompts[0][:500])
            logger.info(f"\n... [{len(prompts[0]) - 1000} chars omitted] ...\n")
            logger.info(prompts[0][-500:])
        logger.info("=" * 50 + "\n")

    logger.info(f"Total prompts to generate: {len(prompts)}")
    logger.info(f"Sampling params: temperature={temperature}, top_p={top_p}, top_k={top_k}")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        logprobs=1,
        stop_token_ids=eos_tokens,
    )

    logger.info(f"Full reasoning reused: {len(results)}/{len(final_candidates)} (skipped re-generation)")

    if not prompts:
        logger.info("All candidates are full_reasoning — no vLLM generation needed.")
        return results

    logger.info(f"Submitting {len(prompts)} prompts to vLLM (continuous batching)...")
    all_outputs = llm.generate(prompts, sampling_params)
    logger.info("vLLM generation complete, processing results...")

    for out, info in zip(all_outputs, meta):
        gen = out.outputs[0]
        response_text = gen.text
        token_ids = gen.token_ids

        token_logprobs = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []

        # Parse the model answer.
        model_answer = parse_response(response_text)

        debug_logger.debug(f"Token log probs length: {len(token_logprobs)}")
        debug_logger.debug(f"Response text: {response_text[:100]}")
        debug_logger.debug(f"Model Answer: {str(model_answer)[:100] if model_answer else None}")

        # Find answer token span
        answer_token_logprobs = []
        answer_start, answer_end = 0, 0
        if model_answer:
            answer_ids = tokenizer(model_answer, add_special_tokens=False).input_ids
            answer_start, answer_end = find_token_span_in_ids(token_ids, answer_ids)
            debug_logger.debug(f"Answer Start, End: {answer_start} ; {answer_end}")
            if answer_start < len(token_logprobs) and answer_end <= len(token_logprobs):
                answer_token_logprobs = token_logprobs[answer_start:answer_end]
            debug_logger.debug(f"Answer log probs: {answer_token_logprobs}")

        # Apply logprobs clamping if enabled (for vLLM V0 compatibility)
        if clamp_logprobs_threshold is not None:
            answer_logprobs_for_conf = clamp_logprobs(answer_token_logprobs, clamp_logprobs_threshold)
            debug_logger.debug(f"Clamped answer log probs (threshold={clamp_logprobs_threshold}): {answer_logprobs_for_conf}")
        else:
            answer_logprobs_for_conf = answer_token_logprobs

        geo_conf = compute_confidence(answer_logprobs_for_conf, confidence_aggregation)
        debug_logger.debug(f"Confidence: {geo_conf:.4f}")

        # Detect if new thinking was generated (matching call_local logic)
        extra_thinking = "<think>" in response_text

        # Store the result
        result = {
            "question_idx": info["question_idx"],
            "stopped_len": info["stopped_len"],
            "original_len_reasoning_steps": info["original_len_reasoning_steps"],
            "saved_steps": info["saved_steps"],
            "question": info["question"],
            "reasoning_prefix": info["reasoning_prefix"],
            "model_response": response_text,
            "final_answer": model_answer,
            "count_reasoning_tokens": info["reasoning_token_count"],
            "count_generated_tokens": len(token_ids),
            "total_tokens": info["reasoning_token_count"] + len(token_ids),
            "generated_trial_answers": info["generated_trial_answers"],
            "tokens_trial_answers": info["tokens_trial_answers"],
            "confidence": geo_conf,
            "new_thinking": extra_thinking,
            "success": True,
            # Propagated from Step 3
            "stop_reason": info.get("stop_reason"),
            "stop_confidence": info.get("stop_confidence"),
            "stop_threshold": info.get("stop_threshold"),
            "consecutive_confidences": info.get("consecutive_confidences"),
            "confidence_trajectory": info.get("confidence_trajectory"),
            "step_similarities": info.get("step_similarities"),
        }

        # Mark outputs without a boxed answer as unsuccessful.
        if not model_answer:
            result["success"] = False
            logger.warning(f"Q{result['question_idx']}: final_answer is empty/None (new_thinking={extra_thinking}, "
                          f"generated_tokens={len(token_ids)}). Marking success=False.")

        logger.info(f"Successfully computed entry: question_idx={result['question_idx']}, stopped_len={result['stopped_len']}, saved_steps={result['saved_steps']}")
        debug_logger.debug(f"Full entry details:\n{result}")
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process questions with reasoning prefixes using vLLM"
    )
    parser.add_argument(
        "--final-candidates",
        type=str,
        required=True,
        help="Path to JSON file containing final candidates information (question_idx, stopped_len, tokens_trial_answers, etc.)"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        required=True,
        help="Path to JSON file containing question objects with reasoning_steps"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate for final answer"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--clamp-logprobs-threshold",
        type=float,
        default=None,
        help="Clamp logprobs > threshold to 0.0 for vLLM V0 compatibility (e.g., -0.001)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for vLLM reproducibility (default: None for non-deterministic)"
    )
    parser.add_argument(
        "--confidence-aggregation",
        type=str,
        choices=["geometric", "arithmetic"],
        default="geometric",
        help="Aggregation method: 'geometric' (exp(mean(logprobs))) or 'arithmetic' (mean(exp(logprobs)))",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger.info(f"Loading final candidates from {args.final_candidates}")
    final_candidates = load_json(args.final_candidates)
    
    logger.info(f"Loading questions data from {args.questions_file}")
    questions_data = load_json(args.questions_file)
    
    logger.info(f"Processing {len(final_candidates)} questions with prefixes using vLLM")
    
    results = process_questions_with_prefixes(
        final_candidates=final_candidates,
        questions_data=questions_data,
        model_name=args.model,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        clamp_logprobs_threshold=args.clamp_logprobs_threshold,
        seed=args.seed,
        confidence_aggregation=args.confidence_aggregation,
    )

    save_results(results, args.output_file)
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    logger.info("\nProcessing complete:")
    logger.info(f"  Total: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
