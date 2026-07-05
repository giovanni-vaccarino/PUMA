import json
import logging
import os
import argparse
import math
import re
import sys
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt_utils import (
    build_base_prompt, get_task_type, get_confident_ending, extract_boxed_answer,
    extract_code_answer,
)

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
    """
    Compute the geometric mean confidence for a sequence of token log-probabilities.

    Args:
        log_probs: List of token log-probs (natural log) for the answer tokens.

    Returns:
        geo_conf (float): The geometric mean confidence score in [0, 1].
                          Larger means the model is more confident.
    """
    if not log_probs:
        return 0.0

    avg_log_prob = sum(log_probs) / len(log_probs)
    geo_conf = math.exp(avg_log_prob)

    return geo_conf


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


def parse_response(message_content: str, task_type: str = "math"):
    """
    Extracts the answer from the message content.
    For code: extracts last Python code block.
    For math: tries \\boxed{...}. For GPQA, falls back to 'ANSWER: X' pattern.
    """
    if task_type == "code":
        return extract_code_answer(message_content)
    answer = extract_boxed_answer(message_content)
    if task_type == "gpqa" and answer not in ("A", "B", "C", "D"):
        m = re.search(r"ANSWER\s*:\s*([A-D])", message_content)
        if m:
            answer = m.group(1)
    return answer


def load_json(filepath: str) -> List[Dict]:
    """Load JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: List[Dict], output_path: str):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def construct_reasoning_prefix(reasoning_steps: List[str]) -> str:
    """
    Construct a reasoning prefix from a list of reasoning steps.
    
    Args:
        reasoning_steps: List of reasoning step strings
        
    Returns:
        Concatenated reasoning chain as a single string
    """
    return "\n\n".join(reasoning_steps)


def build_prompt(
    tokenizer,
    model_name: str,
    question: str,
    reasoning_prefix: str,
    task_type: str,
    prompt_version: str = "default",
    is_trial_answer: bool = False,
    dataset: str = "",
    **kwargs,
) -> tuple:
    """
    Build the vLLM prompt with chat template, instruction, and reasoning prefix.

    Returns:
        tuple: (prompt_text, reasoning_tokens_count)
    """
    base_prompt = build_base_prompt(tokenizer, model_name, question, task_type, prompt_version, is_trial=is_trial_answer)
    confident_ending = get_confident_ending(dataset, is_trial=is_trial_answer)

    # If a reasoning prefix exists, append <think> block
    think_block = ""
    reasoning_tokens = 0

    if reasoning_prefix.strip():
        if is_trial_answer:
            think_block = f"<think>{reasoning_prefix} {confident_ending}"
        else:
            if task_type == "code" and kwargs.get("code_greedy_guided"):
                # Code greedy guided: close think block + format guide
                think_block = f"<think>{reasoning_prefix}\n</think>\n\n### Solution Code\n```python\n"
            elif task_type == "code":
                # Code: close think block, model generates code freely after </think>
                think_block = f"<think>{reasoning_prefix}\n</think>\n\n"
            elif task_type == "gpqa":
                # GPQA: close think block, put confident ending outside to prevent re-thinking
                think_block = f"<think>{reasoning_prefix}\n</think>\n\n{confident_ending}"
            else:
                # Other datasets: confident ending inside think block (original format)
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
    dataset: str = "",
    prompt_version: str = "default",
    confidence_aggregation: str = "geometric",
    code_greedy_guided: bool = False,
) -> List[Dict]:
    """
    Process questions with reasoning prefixes using vLLM batched inference.

    Temperature/top_p are loaded from the model's generation_config.json,
    matching the behavior of run_vllm.py (Step 1a).

    Args:
        final_candidates: Array of final candidate objects with question_idx, stopped_len, tokens_trial_answers, etc.
        questions_data: Array of question objects with reasoning_steps
        model_name: Model to use for vLLM
        max_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        clamp_logprobs_threshold: If set, clamp logprobs > threshold to 0.0 (for vLLM V0 compatibility)
        prompt_version: Prompt version for instruction and chat template format

    Returns:
        List of results with model responses
    """
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

    # Load sampling parameters from model's generation_config.json (matching run_vllm.py)
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

    task_type = get_task_type(dataset)
    
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
        tokens_trial_answers_online = candidate.get("tokens_trial_answers_online", tokens_trial_answers)

        logger.info(f"Processing question {question_idx} with {stopped_len} reasoning steps as prefix")

        # question_idx is 1-indexed, so we need to subtract 1 for 0-indexed array
        question_obj = questions_data[question_idx - 1]

        # Extract the question text
        question_text = question_obj["question"]

        # full_reasoning: reuse original answer directly (no re-generation needed)
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
            logger.info(f"Q{question_idx}: full_reasoning — reusing original answer (skip re-generation)")
            continue

        # Get the first stopped_len reasoning steps
        reasoning_steps = question_obj["reasoning_steps"][:stopped_len]

        # Construct the reasoning prefix
        reasoning_prefix = construct_reasoning_prefix(reasoning_steps)

        logger.info(f"Question: {question_text[:100]}...")
        logger.info(f"Using {len(reasoning_steps)} reasoning steps as prefix")
        
        # Build prompt matching call_local logic
        prompt, reasoning_token_count = build_prompt(
            tokenizer,
            model_name,
            question_text,
            reasoning_prefix,
            task_type,
            prompt_version=prompt_version,
            is_trial_answer=False,
            dataset=dataset,
            code_greedy_guided=code_greedy_guided,
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
            "tokens_trial_answers_online": tokens_trial_answers_online,
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

    # Override to greedy for code tasks with code_greedy_guided
    if code_greedy_guided and task_type == "code":
        temperature = 0.0
        top_p = 1.0
        top_k = -1
        logger.info(f"Code greedy guided mode: temperature=0, greedy decoding + format guide")

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
    logger.info(f"vLLM generation complete, processing results...")

    for out, info in zip(all_outputs, meta):
        gen = out.outputs[0]
        response_text = gen.text
        token_ids = gen.token_ids

        # Extract token log probabilities
        token_logprobs = []
        for step in gen.logprobs:
            if isinstance(step, float):
                token_logprobs.append(step)
            elif isinstance(step, dict):
                val = next(iter(step.values()))
                if hasattr(val, "logprob"):
                    token_logprobs.append(val.logprob)
                elif isinstance(val, float):
                    token_logprobs.append(val)
                else:
                    try:
                        token_logprobs.append(float(val))
                    except Exception:
                        token_logprobs.append(0.0)
            else:
                try:
                    token_logprobs.append(float(step))
                except Exception:
                    token_logprobs.append(0.0)

        # Parse the model answer
        if task_type == "code" and code_greedy_guided:
            # Greedy guided: response starts with code body (after ```python\n in prompt)
            # Extract up to closing ``` or use extract_code_answer as fallback
            from prompt_utils import extract_trial_code
            model_answer = extract_trial_code(response_text)
            if not model_answer:
                model_answer = parse_response(response_text, task_type=task_type)
        elif task_type == "code":
            model_answer = parse_response(response_text, task_type=task_type)
        elif task_type == "gpqa":
            # GPQA: prompt ends with \boxed{, prepend it so extract_boxed_answer can find the pattern
            model_answer = parse_response("\\boxed{" + response_text, task_type=task_type)
        else:
            model_answer = parse_response(response_text, task_type=task_type)

        debug_logger.info(f"Token log probs length: {len(token_logprobs)}")
        debug_logger.info(f"Response text: {response_text[:100]}")
        debug_logger.info(f"Model Answer: {str(model_answer)[:100] if model_answer else None}")

        # Find answer token span
        answer_token_logprobs = []
        answer_start, answer_end = 0, 0
        if task_type == "code":
            # Code: use all generated token logprobs for confidence
            answer_token_logprobs = token_logprobs
        elif model_answer:
            answer_ids = tokenizer(model_answer, add_special_tokens=False).input_ids
            answer_start, answer_end = find_token_span_in_ids(token_ids, answer_ids)
            debug_logger.info(f"Answer Start, End: {answer_start} ; {answer_end}")
            if answer_start < len(token_logprobs) and answer_end <= len(token_logprobs):
                answer_token_logprobs = token_logprobs[answer_start:answer_end]
            debug_logger.info(f"Answer log probs: {answer_token_logprobs}")

        # Apply logprobs clamping if enabled (for vLLM V0 compatibility)
        if clamp_logprobs_threshold is not None:
            answer_logprobs_for_conf = clamp_logprobs(answer_token_logprobs, clamp_logprobs_threshold)
            debug_logger.info(f"Clamped answer log probs (threshold={clamp_logprobs_threshold}): {answer_logprobs_for_conf}")
        else:
            answer_logprobs_for_conf = answer_token_logprobs

        geo_conf = compute_confidence(answer_logprobs_for_conf, confidence_aggregation)
        debug_logger.info(f"Confidence: {geo_conf:.4f}")

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
            "tokens_trial_answers_online": info.get("tokens_trial_answers_online", info["tokens_trial_answers"]),
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

        # Fix success field: if final_answer is None or empty (e.g., new_thinking hit max_tokens
        # without producing \boxed{} or code block), mark as unsuccessful
        if not model_answer:
            # For code tasks: fallback to last trial answer from Step 3
            if task_type == "code" and info.get("trial_final_answer"):
                model_answer = info["trial_final_answer"]
                result["final_answer"] = model_answer
                result["success"] = True
                logger.info(f"Q{info['question_idx']}: empty code from Step 4, fallback to trial answer ({len(model_answer)} chars)")
            else:
                result["success"] = False
            logger.warning(f"Q{result['question_idx']}: final_answer is empty/None (new_thinking={extra_thinking}, "
                          f"generated_tokens={len(token_ids)}). Marking success=False.")

        logger.info(f"Successfully computed entry: question_idx={result['question_idx']}, stopped_len={result['stopped_len']}, saved_steps={result['saved_steps']}")
        debug_logger.info(f"Full entry details:\n{result}")
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
        "--dataset",
        type=str,
        default="",
        help="Dataset name (e.g., gpqa-diamond). Used to adjust prompt for GPQA.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="default",
        choices=["default", "direct", "wo_stepbystep", "aligned_with_deer"],
        help="Prompt version for instruction and chat template format (default: default)",
    )
    parser.add_argument(
        "--confidence-aggregation",
        type=str,
        choices=["geometric", "arithmetic"],
        default="geometric",
        help="Aggregation method: 'geometric' (exp(mean(logprobs))) or 'arithmetic' (mean(exp(logprobs)))",
    )
    parser.add_argument(
        "--code-greedy-guided",
        action="store_true",
        default=False,
        help="For code tasks: use greedy decoding + format guide (### Solution Code + ```python) in Step 4",
    )

    args = parser.parse_args()
    
    # Set up logging
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_dir = "logs/process_prefixes_vllm_debug"
    os.makedirs(debug_log_dir, exist_ok=True)

    # Main logger - outputs to stdout (goes to .out file in SLURM)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Debug logger - outputs only to debug file with timestamp (not to console)
    debug_log_file = f"{debug_log_dir}/process_prefixes_vllm_debug_{timestamp}.log"
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    debug_logger.addHandler(debug_handler)
    debug_logger.setLevel(logging.INFO)
    debug_logger.propagate = False  # Don't propagate to root logger (avoid duplicate output)
    logger.info(f"Debug log file: {debug_log_file}")
    
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
        dataset=args.dataset,
        prompt_version=args.prompt_version,
        confidence_aggregation=args.confidence_aggregation,
        code_greedy_guided=args.code_greedy_guided,
    )

    save_results(results, args.output_file)
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Total: {len(results)}")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")


if __name__ == "__main__":
    main()