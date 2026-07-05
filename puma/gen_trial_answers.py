import json
import math
import logging
import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt_utils import build_base_prompt, get_task_type, get_confident_ending, extract_trial_code

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(__name__ + ".debug")


def clamp_logprobs(log_probs: List[float], threshold: float = -0.001) -> List[float]:
    """
    Clamp logprobs close to 0 to exactly 0.0, simulating vLLM V0 (0.3.x) behavior.

    vLLM V0 appears to truncate/round very small negative logprobs to 0.0,
    while V1 returns more precise values. This causes confidence calculation
    differences that affect PUMA compression rates.

    Args:
        log_probs: List of log probabilities (negative values)
        threshold: Logprobs greater than this threshold will be clamped to 0.0
                   Default -0.001 means logprobs > -0.001 become 0.0

    Returns:
        List of clamped log probabilities
    """
    return [0.0 if lp > threshold else lp for lp in log_probs]


def compute_geometric_confidence(log_probs: List[float]) -> float:
    """Geometric mean confidence for answer token log-probabilities."""
    if not log_probs:
        return 0.0
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(avg_log_prob)


def compute_arithmetic_confidence(log_probs: List[float]) -> float:
    """Arithmetic mean confidence for answer token log-probabilities."""
    if not log_probs:
        return 0.0
    return sum(math.exp(lp) for lp in log_probs) / len(log_probs)


def compute_confidence(log_probs: List[float], aggregation: str = "geometric") -> float:
    """Dispatch to geometric or arithmetic mean confidence."""
    if aggregation == "arithmetic":
        return compute_arithmetic_confidence(log_probs)
    return compute_geometric_confidence(log_probs)


def compute_gpqa_confidence(logprobs_steps, token_ids, tokenizer, temperature=0.5):
    """
    GPQA-specific confidence: temperature-scaled softmax over A/B/C/D logprobs.

    For GPQA multiple-choice, the answer token competes with the entire vocabulary,
    making it hard to reach high confidence thresholds (e.g., 0.98). This function
    renormalizes among just A/B/C/D to get a more meaningful confidence score.

    Args:
        logprobs_steps: raw gen.logprobs (list of dicts with top-K entries per position)
        token_ids: generated token IDs
        tokenizer: HF tokenizer
        temperature: softmax temperature (< 1.0 sharpens, default 0.5)

    Returns:
        float: renormalized top-1 probability among A/B/C/D
    """
    # Find answer token position (inside \boxed{})
    start_idx, end_idx = find_boxed_content_token_span(token_ids, tokenizer)
    if start_idx >= end_idx:
        return 0.0

    # Get A/B/C/D token IDs
    abcd_ids = {
        tokenizer.encode(letter, add_special_tokens=False)[0]: letter
        for letter in ["A", "B", "C", "D"]
    }

    # For each answer token position, extract ABCD logprobs from top-K
    confidences = []
    for pos in range(start_idx, end_idx):
        step = logprobs_steps[pos]

        # Extract logprobs for A/B/C/D from this position's top-K
        abcd_logprobs = {}
        if isinstance(step, dict):
            for tid, val in step.items():
                if tid in abcd_ids:
                    lp = val.logprob if hasattr(val, "logprob") else float(val)
                    abcd_logprobs[tid] = lp

        if not abcd_logprobs:
            confidences.append(0.0)
            continue

        # Temperature-scaled softmax among available ABCD tokens
        logits = list(abcd_logprobs.values())
        scaled = [lp / temperature for lp in logits]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        total = sum(exps)
        top1_prob = max(exps) / total
        confidences.append(top1_prob)

    if not confidences:
        return 0.0
    return math.exp(sum(math.log(max(c, 1e-10)) for c in confidences) / len(confidences))


def construct_reasoning_prefix(reasoning_steps: List[str]) -> str:
    """Concatenate reasoning steps with spacing."""
    return "\n\n".join(reasoning_steps)


def build_prompt(
    tokenizer,
    model_name: str,
    question: str,
    reasoning_prefix: str,
    task_type: str,
    prompt_version: str = "default",
    is_trial_answer: bool = True,
    dataset: str = "",
) -> str:
    """Build the vLLM prompt with chat template, instruction, and reasoning prefix."""
    confident_ending = get_confident_ending(dataset, is_trial=is_trial_answer)
    base_prompt = build_base_prompt(tokenizer, model_name, question, task_type, prompt_version, is_trial=is_trial_answer)
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

    Similar to DEER's approach of only using tokens inside \\boxed{} for
    confidence calculation (skipping the braces themselves).

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



def extract_logprob_from_step(step):
    """Extract a single logprob value from a vLLM logprobs step entry."""
    if isinstance(step, float):
        return step
    elif isinstance(step, dict):
        val = next(iter(step.values()))
        if hasattr(val, "logprob"):
            return val.logprob
        elif isinstance(val, float):
            return val
        else:
            try:
                return float(val)
            except Exception:
                return 0.0
    else:
        try:
            return float(step)
        except Exception:
            return 0.0


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
    dataset: str = "",
    trial_decoding: str = "sampling",
    prompt_version: str = "default",
    confidence_mode: str = "token_in_boxed",
    confidence_aggregation: str = "geometric",
    gpqa_softmax_temperature: float = None,
) -> List[Dict]:
    """
    Batch-run vLLM on all question + prefix combinations.
    Generates trial answers for each reasoning step of each question.

    confidence_mode: "first_line" | "token_in_boxed"
        - first_line: parse_response (text before '.\n'), confidence on all those tokens
        - token_in_boxed: only tokens inside {answer} (like DEER), skip '{' and '}'

    trial_decoding: "greedy" | "sampling" | "default"
        - greedy: temperature=0.0 (deterministic)
        - sampling: uses temperature/top_p/top_k as provided
        - default: no explicit temperature/top_p/top_k (vLLM SamplingParams defaults)
    """
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
    task_type = get_task_type(dataset)

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
            task_type, prompt_version, is_trial_answer=True, dataset=dataset,
        )
        prompts.append(prompt)
        meta.append(info)

    # GPQA needs top-5 logprobs to extract A/B/C/D probabilities
    n_logprobs = 5 if task_type == "gpqa" else 1

    logger.info(f"Total prompts to generate: {len(prompts)}")
    logger.info(f"Trial decoding mode: {trial_decoding}")
    logger.info(f"Confidence mode: {confidence_mode}")
    if n_logprobs > 1:
        logger.info(f"GPQA mode: logprobs={n_logprobs}, softmax_temperature={gpqa_softmax_temperature}")

    if trial_decoding == "greedy":
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=n_logprobs,
        )
    elif trial_decoding == "default":
        # No explicit temperature/top_p/top_k — use vLLM SamplingParams defaults
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            logprobs=n_logprobs,
        )
    else:  # sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            logprobs=n_logprobs,
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
    logger.info(f"vLLM generation complete, processing results...")

    for out, info in zip(all_outputs, meta):
        gen = out.outputs[0]
        text = gen.text
        token_ids = gen.token_ids
        token_logprobs = [extract_logprob_from_step(step) for step in gen.logprobs] if gen.logprobs else []

        if task_type == "code":
            # Code tasks: model output is already inside ```python block,
            # extract code until ``` or end of text
            answer = extract_trial_code(text)
            # Use all generated token logprobs for confidence
            answer_token_logprobs = token_logprobs
            debug_logger.info(f"[code] extracted trial code ({len(answer)} chars)")
        elif confidence_mode == "token_in_boxed":
            # DEER-style: only tokens inside {answer}, skip braces
            boxed_content = extract_first_braced_content(text)
            answer = boxed_content if boxed_content else parse_response(text)
            start_idx, end_idx = find_boxed_content_token_span(token_ids, tokenizer)
            if start_idx < end_idx:
                answer_token_logprobs = token_logprobs[start_idx:end_idx]
            else:
                # Fallback to first_line if no braces found
                answer_token_logprobs = token_logprobs
            debug_logger.info(f"[token_in_boxed] boxed_content='{boxed_content}', span=[{start_idx}:{end_idx}]")
        else:
            # first_line: parse_response (text before '.\n')
            answer = parse_response(text)
            if answer:
                answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
                answer_len = len(answer_ids)
                answer_token_logprobs = token_logprobs[:answer_len]
            else:
                answer_token_logprobs = token_logprobs

        debug_logger.info(f"Trial Answer Result: {answer}")
        debug_logger.info(f"Full token logprobs of all trial answer: {token_logprobs}")
        debug_logger.info(f"Confidence mode: {confidence_mode}, answer logprobs ({len(answer_token_logprobs)} tokens): {answer_token_logprobs}")

        # Apply logprobs clamping if enabled (for vLLM V0 compatibility)
        if clamp_logprobs_threshold is not None:
            token_logprobs_for_conf = clamp_logprobs(token_logprobs, clamp_logprobs_threshold)
            answer_logprobs_for_conf = clamp_logprobs(answer_token_logprobs, clamp_logprobs_threshold)
            debug_logger.info(f"Clamped logprobs (threshold={clamp_logprobs_threshold}): {answer_logprobs_for_conf}")
        else:
            token_logprobs_for_conf = token_logprobs
            answer_logprobs_for_conf = answer_token_logprobs

        all_geo_conf = compute_confidence(token_logprobs_for_conf, confidence_aggregation)
        geo_conf = compute_confidence(answer_logprobs_for_conf, confidence_aggregation)

        # GPQA: override confidence with temperature-scaled softmax over A/B/C/D
        if task_type == "gpqa" and gpqa_softmax_temperature and confidence_mode == "token_in_boxed":
            gpqa_conf = compute_gpqa_confidence(
                gen.logprobs, token_ids, tokenizer,
                temperature=gpqa_softmax_temperature
            )
            debug_logger.info(f"GPQA softmax confidence: {gpqa_conf:.4f} (original: {geo_conf:.4f})")
            geo_conf = gpqa_conf

        debug_logger.info(f"Confidence with token logprobs of all trial answer: {all_geo_conf}")
        debug_logger.info(f"Confidence with token logprobs of only answer: {geo_conf}")

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
        "--dataset",
        type=str,
        default="",
        help="Dataset name (e.g., gpqa-diamond). Used to adjust prompt for GPQA.",
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
        "--prompt-version",
        type=str,
        choices=["default", "direct", "wo_stepbystep", "aligned_with_deer"],
        default="default",
        help="Prompt version for instruction and chat template format",
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        choices=["first_line", "token_in_boxed"],
        default="token_in_boxed",
        help="Which tokens to use for confidence: "
             "'first_line' (text before .\\n, includes braces/units), "
             "'token_in_boxed' (only tokens inside {answer}, like DEER)",
    )
    parser.add_argument(
        "--confidence-aggregation",
        type=str,
        choices=["geometric", "arithmetic"],
        default="geometric",
        help="Aggregation method: 'geometric' (exp(mean(logprobs))) or 'arithmetic' (mean(exp(logprobs)))",
    )
    parser.add_argument(
        "--gpqa-softmax-temperature",
        type=float,
        default=None,
        help="Temperature for GPQA ABCD softmax confidence (e.g., 0.5). "
             "When set, GPQA confidence is computed via temperature-scaled softmax "
             "over A/B/C/D logprobs instead of raw token probability. "
             "Only affects GPQA datasets with token_in_boxed confidence mode.",
    )

    args = parser.parse_args()

    # Main logger - outputs to stdout (goes to .out file in SLURM)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Debug logger - outputs only to debug file with timestamp (not to console)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_dir = "logs/gen_trial_answers_debug"
    os.makedirs(debug_log_dir, exist_ok=True)
    debug_log_file = f"{debug_log_dir}/gen_trial_answers_debug_{timestamp}.log"
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    debug_logger.addHandler(debug_handler)
    debug_logger.setLevel(logging.INFO)
    debug_logger.propagate = False
    logger.info(f"Debug log file: {debug_log_file}")

    logger.info(f"Loading questions data from {args.questions_file}")
    questions_data = load_json(args.questions_file)

    if args.respect_embedding_filter:
        logger.info("Embedding filter: ENABLED (will skip steps with should_generate_trial=False)")
    else:
        logger.info("Embedding filter: DISABLED (will process all steps)")

    if args.gpqa_softmax_temperature is not None:
        logger.info(f"GPQA softmax confidence: ENABLED (temperature={args.gpqa_softmax_temperature})")
    else:
        logger.info("GPQA softmax confidence: DISABLED")

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
        dataset=args.dataset,
        trial_decoding=args.trial_decoding,
        prompt_version=args.prompt_version,
        confidence_mode=args.confidence_mode,
        confidence_aggregation=args.confidence_aggregation,
        gpqa_softmax_temperature=args.gpqa_softmax_temperature,
    )

    save_results(results, args.output_file)

    # Log statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    skipped_emb = sum(1 for r in results if r.get("skip_reason") == "embedding_filter")
    generated = total - skipped_emb

    logger.info(f"Done!")
    logger.info(f"  Total entries: {total}")
    logger.info(f"  Generated trial answers: {generated}")
    logger.info(f"  Skipped (embedding filter): {skipped_emb}")
    logger.info(f"  Successful: {successful}")


if __name__ == "__main__":
    main()