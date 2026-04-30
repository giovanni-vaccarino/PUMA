import json
import logging
import argparse
import sys
from typing import List, Dict

from .prompt_utils import build_base_prompt, get_confident_ending
from .confidence_utils import (
    clamp_logprobs,
    compute_confidence,
    compute_geometric_confidence,
    compute_arithmetic_confidence,
    construct_reasoning_prefix,
    extract_logprob_from_step,
)

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(__name__ + ".debug")


def build_prompt(
    tokenizer,
    model_name: str,
    question: str,
    reasoning_prefix: str,
    is_trial_answer: bool = True,
) -> str:
    """Build the vLLM prompt with chat template, instruction, and reasoning prefix."""
    confident_ending = get_confident_ending(is_trial=is_trial_answer)
    base_prompt = build_base_prompt(tokenizer, model_name, question)
    if reasoning_prefix.strip():
        if is_trial_answer:
            return f"{base_prompt}<think>{reasoning_prefix} {confident_ending}"
        else:
            return f"{base_prompt}<think>{reasoning_prefix} {confident_ending}</think>"
    else:
        return base_prompt


def parse_response(message):
    """
    Extracts everything before the first '.\\n'
    Keeps curly braces intact.
    """
    content = message

    idx = content.find(".\n")
    if idx == -1:
        return content.strip()

    answer = content[:idx].strip()
    return answer


def extract_first_braced_content(text):
    """
    Extract content inside the first {...} using balanced brace matching.
    E.g. '{330} minutes' -> '330', '{\\frac{1}{2}} ...' -> '\\frac{1}{2}'
    Returns None if no balanced braces found.
    """
    idx = text.find("{")
    if idx == -1:
        return None
    start = idx + 1
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


def find_boxed_content_token_span(token_ids, tokenizer):
    """
    Find token indices of content inside first {...} in generated tokens.

    Since the prompt ends with \\boxed, the response starts with {answer}...
    This function tracks brace depth at the token level to find the matching
    closing brace and returns the token span of the content between braces.

    Confidence is computed only over the answer tokens, excluding braces.

    Returns:
        tuple: (start_idx, end_idx) — token_ids[start_idx:end_idx] are content tokens.
               Returns (0, 0) if no valid boxed content found.
    """
    if not token_ids:
        return 0, 0

    depth = 0
    content_start = None

    for i, tid in enumerate(token_ids):
        token_text = tokenizer.decode([tid])

        opens = token_text.count('{')
        closes = token_text.count('}')

        old_depth = depth
        depth += opens - closes

        # First time depth goes positive: opening brace found
        if old_depth == 0 and opens > 0 and content_start is None:
            content_start = i + 1  # Content starts after this token

        # Depth drops to 0: matching closing brace found
        if depth == 0 and content_start is not None:
            return content_start, i  # exclusive end (don't include '}' token)

    # No matching closing brace found
    if content_start is not None:
        return content_start, len(token_ids)
    return 0, 0



def process_questions_with_prefixes_vllm(
    questions_data: List[Dict],
    model_name: str,
    max_tokens: int = 30,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 30,
    tensor_parallel_size: int = 1,
    logprobs_mode: str = None,
    clamp_logprobs_threshold: float = None,
    respect_embedding_filter: bool = False,
    seed: int = None,
    trial_decoding: str = "sampling",
    confidence_mode: str = "token_in_boxed",
    confidence_aggregation: str = "geometric",
) -> List[Dict]:
    """
    Batch-run vLLM on all question + prefix combinations.
    Generates trial answers for each reasoning step of each question.

    confidence_mode: "first_line" | "token_in_boxed"
        - first_line: parse_response (text before '.\n'), confidence on all those tokens
        - token_in_boxed: only tokens inside {answer}, skipping '{' and '}'

    trial_decoding: "greedy" | "sampling" | "default"
        - greedy: temperature=0.0 (deterministic)
        - sampling: uses temperature/top_p/top_k as provided
        - default: no explicit temperature/top_p/top_k (vLLM SamplingParams defaults)
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logger.info(f"Loading vLLM model {model_name}")
    # Build LLM kwargs - logprobs_mode is an engine argument for vLLM >= 0.10.x
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 38000,
        "gpu_memory_utilization": 0.78,
    }
    if logprobs_mode:
        llm_kwargs["logprobs_mode"] = logprobs_mode
    if seed is not None:
        llm_kwargs["seed"] = seed
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    skipped_results = []  # Track skipped steps (embedding filter)

    # =========================================================================
    # Collect non-embedding-filtered steps (candidates for trial answer)
    # =========================================================================
    candidate_steps = []  # Steps that pass embedding filter

    for q_idx, question_obj in enumerate(questions_data, start=1):
        question_text = question_obj["question"]
        reasoning_steps = question_obj.get("reasoning_steps", [])
        original_len_reasoning_steps = len(reasoning_steps)

        # Get embedding filter flags if available
        should_generate_trial = question_obj.get("should_generate_trial", None)
        step_similarities = question_obj.get("step_similarities", None)

        for stopped_len in range(1, original_len_reasoning_steps + 1):
            step_idx = stopped_len - 1  # 0-indexed

            # Check if this step should be skipped based on embedding filter
            if respect_embedding_filter and should_generate_trial is not None:
                if not should_generate_trial[step_idx]:
                    similarity = step_similarities[step_idx] if step_similarities else None
                    skipped_results.append({
                        "question_idx": q_idx,
                        "stopped_len": stopped_len,
                        "original_len_reasoning_steps": original_len_reasoning_steps,
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
                "original_len_reasoning_steps": original_len_reasoning_steps,
                "question": question_text,
                "reasoning_prefix": reasoning_prefix,
            })

    # Log embedding filter statistics
    if respect_embedding_filter:
        total_steps = len(candidate_steps) + len(skipped_results)
        logger.info(f"Embedding filter: {len(skipped_results)}/{total_steps} steps skipped "
                    f"({len(skipped_results)/total_steps*100:.1f}%)")

    # =========================================================================
    # Build trial answer prompts (only for steps passing all filters)
    # =========================================================================
    prompts = []
    meta = []

    for info in candidate_steps:
        prompt = build_prompt(
            tokenizer, model_name, info["question"], info["reasoning_prefix"],
            is_trial_answer=True,
        )
        prompts.append(prompt)
        meta.append(info)

    logger.info(f"Total prompts to generate: {len(prompts)}")
    logger.info(f"Trial decoding mode: {trial_decoding}")
    logger.info(f"Confidence mode: {confidence_mode}")

    if trial_decoding == "greedy":
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=1,
        )
    elif trial_decoding == "default":
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            logprobs=1,
        )
    else:  # sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            logprobs=1,
        )

    # Debug: show first prompt (first 500 + last 500 chars)
    if prompts:
        logger.info(f"\n=== Sample prompt (first trial, {len(prompts[0])} chars) ===")
        if len(prompts[0]) <= 1000:
            logger.info(prompts[0])
        else:
            logger.info(prompts[0][:500])
            logger.info(f"\n... [{len(prompts[0]) - 1000} chars omitted] ...\n")
            logger.info(prompts[0][-500:])
        logger.info("=" * 50 + "\n")

    results = []

    logger.info(f"Submitting all {len(prompts)} prompts to vLLM (continuous batching)...")
    all_outputs = llm.generate(prompts, sampling_params)
    logger.info("vLLM generation complete, processing results...")

    for out, info in zip(all_outputs, meta):
        gen = out.outputs[0]
        text = gen.text
        token_ids = gen.token_ids
        token_logprobs = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []

        if confidence_mode == "token_in_boxed":
            # Use only tokens inside the generated {answer}, skipping braces.
            boxed_content = extract_first_braced_content(text)
            answer = boxed_content if boxed_content else parse_response(text)
            start_idx, end_idx = find_boxed_content_token_span(token_ids, tokenizer)
            if start_idx < end_idx:
                answer_token_logprobs = token_logprobs[start_idx:end_idx]
            else:
                # Fallback to first_line if no braces found
                answer_token_logprobs = token_logprobs
            debug_logger.debug(f"[token_in_boxed] boxed_content='{boxed_content}', span=[{start_idx}:{end_idx}]")
        else:
            # first_line: parse_response (text before '.\n')
            answer = parse_response(text)
            if answer:
                answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
                answer_len = len(answer_ids)
                answer_token_logprobs = token_logprobs[:answer_len]
            else:
                answer_token_logprobs = token_logprobs

        debug_logger.debug(f"Trial Answer Result: {answer}")
        debug_logger.debug(f"Full token logprobs of all trial answer: {token_logprobs}")
        debug_logger.debug(f"Confidence mode: {confidence_mode}, answer logprobs ({len(answer_token_logprobs)} tokens): {answer_token_logprobs}")

        # Apply logprobs clamping if enabled (for vLLM V0 compatibility)
        if clamp_logprobs_threshold is not None:
            token_logprobs_for_conf = clamp_logprobs(token_logprobs, clamp_logprobs_threshold)
            answer_logprobs_for_conf = clamp_logprobs(answer_token_logprobs, clamp_logprobs_threshold)
            debug_logger.debug(f"Clamped logprobs (threshold={clamp_logprobs_threshold}): {answer_logprobs_for_conf}")
        else:
            token_logprobs_for_conf = token_logprobs
            answer_logprobs_for_conf = answer_token_logprobs

        all_geo_conf = compute_confidence(token_logprobs_for_conf, confidence_aggregation)
        geo_conf = compute_confidence(answer_logprobs_for_conf, confidence_aggregation)

        debug_logger.debug(f"Confidence with token logprobs of all trial answer: {all_geo_conf}")
        debug_logger.debug(f"Confidence with token logprobs of only answer: {geo_conf}")

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

    # Combine generated results with skipped results
    all_results = results + skipped_results

    # Sort by question_idx and stopped_len to maintain order
    all_results.sort(key=lambda x: (x["question_idx"], x["stopped_len"]))

    return all_results



def load_json(filepath: str) -> List[Dict]:
    """Load JSON list from file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_results(results: List[Dict], output_path: str):
    """Save results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process questions with reasoning prefixes using batched vLLM"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        required=True,
        help="JSON with question objects (must include reasoning_steps)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save batched vLLM results JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name to use for vLLM",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens generated per trial answer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism",
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument(
        "--logprobs-mode",
        type=str,
        default=None,
        help="Logprobs mode for vLLM >= 0.10.x (e.g., 'processed_logprobs')",
    )
    parser.add_argument(
        "--clamp-logprobs-threshold",
        type=float,
        default=None,
        help="Clamp logprobs > threshold to 0.0 for vLLM V0 compatibility. "
             "Recommended: -0.001. Disabled by default (None).",
    )
    parser.add_argument(
        "--respect-embedding-filter",
        action="store_true",
        help="Skip steps where should_generate_trial=False (from embedding filter)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for vLLM reproducibility (default: None for non-deterministic)",
    )
    parser.add_argument(
        "--trial-decoding",
        type=str,
        choices=["greedy", "sampling", "default"],
        default="sampling",
        help="Trial answer decoding mode: "
             "'greedy' (temp=0.0), "
             "'sampling' (uses --temperature/--top_p/--top_k), "
             "'default' (vLLM SamplingParams defaults, no explicit temp/top_p/top_k)",
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        choices=["first_line", "token_in_boxed"],
        default="token_in_boxed",
        help="Which tokens to use for confidence: "
             "'first_line' (text before .\\n, includes braces/units), "
             "'token_in_boxed' (only tokens inside {answer})",
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
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Loading questions data from {args.questions_file}")
    questions_data = load_json(args.questions_file)

    if args.respect_embedding_filter:
        logger.info("Embedding filter: ENABLED (will skip steps with should_generate_trial=False)")
    else:
        logger.info("Embedding filter: DISABLED (will process all steps)")

    logger.info(f"Processing {len(questions_data)} questions with vLLM batched inference")
    results = process_questions_with_prefixes_vllm(
        questions_data=questions_data,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        logprobs_mode=args.logprobs_mode,
        clamp_logprobs_threshold=args.clamp_logprobs_threshold,
        respect_embedding_filter=args.respect_embedding_filter,
        seed=args.seed,
        trial_decoding=args.trial_decoding,
        confidence_mode=args.confidence_mode,
        confidence_aggregation=args.confidence_aggregation,
    )

    save_results(results, args.output_file)

    # Log statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    skipped_emb = sum(1 for r in results if r.get("skip_reason") == "embedding_filter")
    generated = total - skipped_emb

    logger.info("Done!")
    logger.info(f"  Total entries: {total}")
    logger.info(f"  Generated trial answers: {generated}")
    logger.info(f"  Skipped (embedding filter): {skipped_emb}")
    logger.info(f"  Successful: {successful}")


if __name__ == "__main__":
    main()
