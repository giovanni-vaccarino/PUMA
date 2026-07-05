#!/usr/bin/env python3
"""
Dynasor Baseline: Probe-In-The-Middle with Answer Consistency Early Exit.

Implements Dynasor's core algorithm (from dynasor/core/cot.py):
  1. Generate reasoning in chunks of `chunk_size` tokens
  2. At each chunk boundary, inject a probe suffix to induce an answer
  3. Check if the last `threshold` probe answers are consistent and certain
  4. If yes, early exit with the consistent answer
  5. If no, discard the probe and continue reasoning

Uses vLLM Python API directly (consistent with the rest of the project).
Output format is compatible with check_accuracy_baseline.py (DEER branch).

Usage:
    python -m baselines.dynasor.eval_dynasor \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --benchmark ../data/aime24_test.jsonl \
        --output-dir data/baselines/dynasor/aime24 \
        --effort mid
"""

import argparse
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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from baselines.utils.generation_config import load_generation_params

from baselines.utils.math_util import extract_code_answer, CODE_SYS_PROMPT
from baselines.dynasor.dynasor_utils import (
    PROBE_SUFFIX,
    PROBE_SUFFIX_CODE,
    effort_level,
    obtain_answer,
    obtain_code_answer,
    uncertain_words,
    is_certain_answer,
    should_early_exit,
    should_early_exit_code,
)


def main():
    parser = argparse.ArgumentParser(
        description="Dynasor Probe-In-The-Middle inference"
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output files")

    # Dynasor parameters
    parser.add_argument("--effort", type=str, default="mid",
                        choices=["mild", "low", "mid", "high", "crazy"],
                        help="Dynasor effort level preset (default: mid)")
    parser.add_argument("--effort-custom", type=str, default=None,
                        help="Custom effort as 'threshold,chunk_size', e.g. '5,128'. "
                             "Overrides --effort if provided.")
    parser.add_argument("--probe-suffix", type=str, default=PROBE_SUFFIX,
                        help="Probe suffix to inject for answer extraction")
    parser.add_argument("--probe-max-tokens", type=int, default=20,
                        help="Max tokens for probe response (default: 20)")

    # Generation
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Total token budget per question (default: 32768)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for reasoning generation (default: 0.6)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=10000,
                        help="Maximum number of questions to process")

    args = parser.parse_args()

    # Load generation params from model's generation_config.json
    gen_params = load_generation_params(args.model)
    args.temperature = gen_params["temperature"]
    args.top_p = gen_params["top_p"]
    args.top_k = gen_params["top_k"]

    # ---- Parse effort level ----
    if args.effort_custom:
        parts = tuple(map(int, args.effort_custom.split(",")))
        assert len(parts) == 2, "effort-custom must be 'threshold,chunk_size'"
        threshold, chunk_size = parts
    else:
        threshold, chunk_size = effort_level(args.effort)

    is_code = "livecodebench" in args.benchmark.lower()
    if is_code:
        args.probe_suffix = PROBE_SUFFIX_CODE
        args.probe_max_tokens = 200  # code answers are longer
        print("Code task detected: using code probe suffix and 200 max probe tokens")

    print(f"\n{'='*60}")
    print(f"  Dynasor Probe-In-The-Middle Inference")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Effort:      threshold={threshold}, chunk_size={chunk_size}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Probe max:   {args.probe_max_tokens}")
    print(f"Code task:   {is_code}")
    print(f"{'='*60}\n")

    # ---- Setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "final_answers.jsonl")

    # ---- Resume support: load previously completed questions ----
    completed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                item = json.loads(line)
                completed_questions.add(item["question"])
        print(f"Resuming: found {len(completed_questions)} completed questions in {output_file}")

    tensor_parallel_size = (
        len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else 1
    )
    print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")

    # ---- Load model ----
    print(f"Loading model: {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        max_model_len=args.max_tokens + 2048,  # extra room for probe
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,  # important for iterative generation
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.")

    # Sampling params for reasoning chunks
    sampling_params_chunk = SamplingParams(
        temperature=args.temperature,
        max_tokens=chunk_size,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Sampling params for probes (use model defaults, matching reasoning)
    sampling_params_probe = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.probe_max_tokens,
        top_p=args.top_p,
    )

    sys_prompt = CODE_SYS_PROMPT if is_code else "Please reason step by step, and put your final answer within \\boxed{}."

    # ---- Load benchmark ----
    print(f"Loading benchmark: {args.benchmark}...")
    all_questions = []
    all_gt_answers = []
    all_question_ids = []
    with open(args.benchmark, "r") as f:
        for line in f:
            item = json.loads(line)
            all_questions.append(item["question"])
            all_gt_answers.append(item.get("answer", ""))
            all_question_ids.append(item.get("question_id", ""))
    all_questions = all_questions[: args.limit]
    all_gt_answers = all_gt_answers[: args.limit]
    all_question_ids = all_question_ids[: args.limit]
    total_benchmark = len(all_questions)

    # Filter out already completed questions
    questions = []
    gt_answers = []
    question_ids = []
    original_indices = []  # track original index for logging
    for i, (q, a, qid) in enumerate(zip(all_questions, all_gt_answers, all_question_ids)):
        if q not in completed_questions:
            questions.append(q)
            gt_answers.append(a)
            question_ids.append(qid)
            original_indices.append(i)
    print(f"Loaded {total_benchmark} questions, {len(completed_questions)} already done, "
          f"{len(questions)} remaining")

    # ---- Dynasor Probe-In-The-Middle inference (batched across questions) ----
    start_time = time.time()
    n_questions = len(questions)

    # Per-question state
    current_prompts = []
    accumulated_responses = [""] * n_questions
    all_probe_answers = [[] for _ in range(n_questions)]
    all_probe_certains = [[] for _ in range(n_questions)]
    num_probe_tokens = [0] * n_questions
    tokens_generated = [0] * n_questions
    early_exits = [False] * n_questions
    finished = [False] * n_questions

    # Build initial prompts for all questions
    for question in questions:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        current_prompts.append(prompt)

    active_indices = list(range(n_questions))  # questions still being processed
    max_rounds = args.max_tokens // chunk_size + 1
    total_early_exit = 0

    # Open output file in append mode for incremental saving
    out_f = open(output_file, "a")

    def _save_question(q_idx):
        """Save a single completed question result immediately."""
        result = {
            "question": questions[q_idx],
            "generated_text": accumulated_responses[q_idx],
            "gold_answer": gt_answers[q_idx],
            "thinking_steps": len(all_probe_answers[q_idx]),
            "num_trial_answer_tokens": num_probe_tokens[q_idx],
            "early_exit": early_exits[q_idx],
            "probe_answers": all_probe_answers[q_idx],
        }
        if is_code:
            result["question_id"] = question_ids[q_idx]
            result["extracted_code"] = extract_code_answer(accumulated_responses[q_idx])
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

    for round_idx in range(max_rounds):
        if not active_indices:
            break

        done_count = n_questions - len(active_indices) + len(completed_questions)
        print(f"\r  Round {round_idx+1}: {len(active_indices)} active questions "
              f"({done_count}/{total_benchmark} done)", end="", flush=True)

        # ---- Step 1: Batch generate chunks for all active questions ----
        chunk_prompts = [current_prompts[i] for i in active_indices]
        chunk_outputs = llm.generate(chunk_prompts, sampling_params_chunk, use_tqdm=False)

        # Process chunk results, determine which need probing
        need_probe_indices = []  # indices into active_indices
        newly_finished = []  # q_idx of questions finished this round
        for idx, (q_idx, output) in enumerate(zip(active_indices, chunk_outputs)):
            chunk_text = output.outputs[0].text
            chunk_token_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason

            accumulated_responses[q_idx] += chunk_text
            current_prompts[q_idx] += chunk_text
            tokens_generated[q_idx] += len(chunk_token_ids)

            if finish_reason != "length" or not chunk_text:
                # Model finished naturally (EOS) or empty output
                finished[q_idx] = True
                newly_finished.append(q_idx)
            else:
                need_probe_indices.append(idx)

        # ---- Step 2: Batch generate probes for questions that need them ----
        if need_probe_indices:
            probe_q_indices = [active_indices[idx] for idx in need_probe_indices]
            probe_prompts = [current_prompts[q_idx] + args.probe_suffix for q_idx in probe_q_indices]
            probe_outputs = llm.generate(probe_prompts, sampling_params_probe, use_tqdm=False)

            for q_idx, output in zip(probe_q_indices, probe_outputs):
                probe_text = output.outputs[0].text
                probe_token_ids = output.outputs[0].token_ids
                num_probe_tokens[q_idx] += len(probe_token_ids)

                # Extract answer and check certainty
                if is_code:
                    answer = obtain_code_answer(probe_text)
                else:
                    answer = obtain_answer(probe_text)
                all_probe_answers[q_idx].append(answer)

                is_certain = is_certain_answer(probe_text, uncertain_words)
                all_probe_certains[q_idx].append(is_certain)

                # Check early exit condition
                if is_code:
                    do_exit = should_early_exit_code(
                        all_probe_answers[q_idx], all_probe_certains[q_idx], threshold
                    )
                else:
                    do_exit = should_early_exit(
                        all_probe_answers[q_idx], all_probe_certains[q_idx], threshold
                    )

                if do_exit:
                    final_answer = all_probe_answers[q_idx][-1]
                    if is_code:
                        accumulated_responses[q_idx] += (
                            "\n\n...</think>\n### Solution Code\n```python\n"
                            + final_answer
                            + "\n```"
                        )
                    elif "</think>" in accumulated_responses[q_idx]:
                        accumulated_responses[q_idx] += (
                            "\n\n... Oh, I have got the answer to the whole problem\n"
                            "**Final Answer:**\n\\[\n \\boxed{"
                            + final_answer
                            + "}\n\\]"
                        )
                    else:
                        accumulated_responses[q_idx] += (
                            "\n\n...</think>\n Oh, I have got the answer to the whole problem\n"
                            "**Final Answer:**\n\\[\n \\boxed{"
                            + final_answer
                            + "}\n\\]"
                        )
                    early_exits[q_idx] = True
                    finished[q_idx] = True
                    newly_finished.append(q_idx)
                    total_early_exit += 1

        # Incrementally save newly finished questions
        for q_idx in newly_finished:
            _save_question(q_idx)

        # Update active list: remove finished questions
        active_indices = [i for i in active_indices if not finished[i]]

    # Save any remaining active questions that hit max_rounds without finishing
    if active_indices:
        print(f"\n  Warning: {len(active_indices)} questions hit max token budget without early exit or EOS")
        for q_idx in active_indices:
            _save_question(q_idx)

    out_f.close()
    last_round = round_idx + 1 if n_questions > 0 else 0
    print(f"\r  Completed all {n_questions} questions in {last_round} rounds.          ")

    # ---- Re-order results to match benchmark order ----
    # Incremental saving may produce out-of-order results; check_accuracy_baseline.py
    # requires predictions[i] to align with ground_truth[i] by position.
    result_by_question = {}
    with open(output_file, "r") as f:
        for line in f:
            r = json.loads(line)
            result_by_question[r["question"]] = r

    all_results = []
    for q in all_questions:
        if q in result_by_question:
            all_results.append(result_by_question[q])

    # Rewrite file in correct order
    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(all_results)
    total_ee = sum(1 for r in all_results if r["early_exit"])
    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"Total questions:     {n} (resumed {len(completed_questions)} + new {n_questions})")
    print(f"Early exits:         {total_ee} ({100*total_ee/n:.1f}%)")
    avg_probes = sum(r["thinking_steps"] for r in all_results) / n if n else 0
    avg_probe_tokens = sum(r["num_trial_answer_tokens"] for r in all_results) / n if n else 0
    print(f"Avg probes/question: {avg_probes:.1f}")
    print(f"Avg probe tokens:    {avg_probe_tokens:.1f}")
    print(f"Wall time (this run): {elapsed_time:.2f}s")
    print(f"Output: {output_file}")

    # Save wall time metadata
    walltime_file = os.path.join(os.path.dirname(output_file), "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed_time, 2),
            "num_questions": n,
            "num_resumed": len(completed_questions),
            "method": "dynasor",
        }, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

    # Clean up GPU memory
    print("\nCleaning up GPU memory...")
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
