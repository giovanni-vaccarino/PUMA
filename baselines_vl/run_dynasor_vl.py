#!/usr/bin/env python3
"""
VL Baseline: Dynasor — Probe-In-The-Middle with Answer Consistency Early Exit.

Generates reasoning in chunks, injects probe suffix at each boundary to
extract answers, and checks consistency for early exit.

Based on: Dynasor (eval_dynasor.py) adapted for vision-language models.

Usage:
    python baselines_vl/run_dynasor_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark experiments_mdh/benchmark_vl/mathvista_test.jsonl \
        --output-dir experiments_mdh/data_vl/baselines/dynasor/mathvista \
        --limit 500
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VL_DIR = os.path.join(_PROJECT_ROOT, "puma_vl")
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
for _p in [_PROJECT_ROOT, _VL_DIR, _OFFLINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, GenerationConfig
from tqdm import tqdm

from prompt_utils_vl import (
    get_task_type_vl, build_base_prompt_vl,
    extract_answer_vl, load_image,
)
from eval_utils_vl import check_is_correct_vl

# Import dynasor utilities
sys.path.insert(0, _PROJECT_ROOT)
from baselines.dynasor.dynasor_utils import (
    PROBE_SUFFIX,
    effort_level,
    obtain_answer,
    uncertain_words,
    is_certain_answer,
    should_early_exit,
)


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: Dynasor")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--effort", type=str, default="mid",
                        choices=["mild", "low", "mid", "high", "crazy"])
    parser.add_argument("--effort-custom", type=str, default=None,
                        help="Custom 'threshold,chunk_size' overriding --effort")
    parser.add_argument("--probe-suffix", type=str, default=PROBE_SUFFIX)
    parser.add_argument("--probe-max-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse effort level
    if args.effort_custom:
        parts = tuple(map(int, args.effort_custom.split(",")))
        assert len(parts) == 2
        threshold, chunk_size = parts
    else:
        threshold, chunk_size = effort_level(args.effort)

    task_type = get_task_type_vl(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "final_answers.jsonl")

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1

    print(f"\n{'='*60}")
    print(f"  Dynasor VL Probe-In-The-Middle Inference")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Effort:      threshold={threshold}, chunk_size={chunk_size}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"{'='*60}\n")

    # Resume support
    completed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                item = json.loads(line)
                completed_questions.add(item["question"])
        print(f"Resuming: {len(completed_questions)} completed")

    # Load model
    print(f"Loading model: {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=args.max_tokens + 18000,  # extra headroom for VL image tokens (~16K)
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=True,
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    print("Model loaded.")

    # Load generation config
    try:
        gen_config = GenerationConfig.from_pretrained(args.model, trust_remote_code=True)
        temperature = getattr(gen_config, "temperature", 0.6)
        top_p = getattr(gen_config, "top_p", 0.95)
        top_k = getattr(gen_config, "top_k", -1)
    except Exception:
        temperature, top_p, top_k = 0.6, 0.95, -1

    sampling_params_chunk = SamplingParams(
        temperature=temperature, max_tokens=chunk_size,
        top_p=top_p, top_k=top_k,
    )
    sampling_params_probe = SamplingParams(
        temperature=temperature, max_tokens=args.probe_max_tokens,
        top_p=top_p,
    )

    # Load questions
    all_questions = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_questions.append(json.loads(line))
    if args.limit > 0:
        all_questions = all_questions[:args.limit]

    # Filter completed
    questions = [q for q in all_questions if q["question"] not in completed_questions]
    print(f"Loaded {len(all_questions)} total, {len(questions)} remaining")

    # Pre-cache images
    print("Pre-caching images...")
    image_cache = {}
    for i, q in enumerate(tqdm(questions, desc="Loading images")):
        image_cache[i] = load_image(q["image_path"])

    # Build initial prompts
    n = len(questions)
    current_prompts = []
    accumulated = [""] * n
    probe_answers = [[] for _ in range(n)]
    probe_certains = [[] for _ in range(n)]
    num_probe_tokens = [0] * n
    tokens_generated = [0] * n
    early_exits = [False] * n
    finished = [False] * n

    for i, q in enumerate(questions):
        prompt = build_base_prompt_vl(
            processor, args.model, q["question"], q["image_path"],
            task_type, "default",
        )
        current_prompts.append(prompt)

    active_indices = list(range(n))
    max_rounds = args.max_tokens // chunk_size + 1
    total_early_exit = 0

    start_time = time.time()
    out_f = open(output_file, "a")

    def _save(q_idx):
        result = {
            "question": questions[q_idx]["question"],
            "generated_text": accumulated[q_idx],
            "gold_answer": questions[q_idx]["answer"],
            "thinking_steps": len(probe_answers[q_idx]),
            "num_trial_answer_tokens": num_probe_tokens[q_idx],
            "early_exit": early_exits[q_idx],
            "probe_answers": probe_answers[q_idx],
            "image_path": questions[q_idx]["image_path"],
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

    for round_idx in range(max_rounds):
        if not active_indices:
            break

        done = n - len(active_indices) + len(completed_questions)
        print(f"\r  Round {round_idx+1}: {len(active_indices)} active "
              f"({done}/{len(all_questions)} done)", end="", flush=True)

        # Step 1: Generate chunks
        chunk_inputs = []
        for i in active_indices:
            chunk_inputs.append({
                "prompt": current_prompts[i],
                "multi_modal_data": {"image": image_cache[i]},
            })
        chunk_outputs = llm.generate(chunk_inputs, sampling_params_chunk, use_tqdm=False)

        need_probe = []
        newly_finished = []
        for idx, (q_idx, output) in enumerate(zip(active_indices, chunk_outputs)):
            chunk_text = output.outputs[0].text
            chunk_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason

            accumulated[q_idx] += chunk_text
            current_prompts[q_idx] += chunk_text
            tokens_generated[q_idx] += len(chunk_ids)

            if finish_reason != "length" or not chunk_text:
                finished[q_idx] = True
                newly_finished.append(q_idx)
            else:
                need_probe.append(idx)

        # Step 2: Probe
        if need_probe:
            probe_q_indices = [active_indices[idx] for idx in need_probe]
            probe_inputs = []
            for q_idx in probe_q_indices:
                probe_inputs.append({
                    "prompt": current_prompts[q_idx] + args.probe_suffix,
                    "multi_modal_data": {"image": image_cache[q_idx]},
                })
            probe_outputs = llm.generate(probe_inputs, sampling_params_probe, use_tqdm=False)

            for q_idx, output in zip(probe_q_indices, probe_outputs):
                probe_text = output.outputs[0].text
                probe_ids = output.outputs[0].token_ids
                num_probe_tokens[q_idx] += len(probe_ids)

                answer = obtain_answer(probe_text)
                probe_answers[q_idx].append(answer)

                certain = is_certain_answer(probe_text, uncertain_words)
                probe_certains[q_idx].append(certain)

                if should_early_exit(probe_answers[q_idx], probe_certains[q_idx], threshold):
                    final_ans = probe_answers[q_idx][-1]
                    if "</think>" in accumulated[q_idx]:
                        accumulated[q_idx] += (
                            "\n\n... Oh, I have got the answer to the whole problem\n"
                            "**Final Answer:**\n\\[\n \\boxed{" + final_ans + "}\n\\]"
                        )
                    else:
                        accumulated[q_idx] += (
                            "\n\n...</think>\n Oh, I have got the answer to the whole problem\n"
                            "**Final Answer:**\n\\[\n \\boxed{" + final_ans + "}\n\\]"
                        )
                    early_exits[q_idx] = True
                    finished[q_idx] = True
                    newly_finished.append(q_idx)
                    total_early_exit += 1

        for q_idx in newly_finished:
            _save(q_idx)
        active_indices = [i for i in active_indices if not finished[i]]

    # Save remaining
    if active_indices:
        for q_idx in active_indices:
            _save(q_idx)
    out_f.close()

    # Re-order results
    result_by_q = {}
    with open(output_file, "r") as f:
        for line in f:
            r = json.loads(line)
            result_by_q[r["question"]] = r

    all_results = []
    for q in all_questions:
        if q["question"] in result_by_q:
            all_results.append(result_by_q[q["question"]])

    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time

    # Evaluate
    correct = 0
    total_tok = 0
    for r in all_results:
        answer = extract_answer_vl(r["generated_text"], args.dataset)
        r["final_answer"] = answer or ""
        q_data = next((q for q in all_questions if q["question"] == r["question"]), None)
        if q_data:
            is_correct = check_is_correct_vl(
                answer, q_data["answer"], args.dataset,
                q_data.get("question_type", ""),
            )
            r["correct"] = is_correct
            if is_correct:
                correct += 1
        total_tok += len(tokenizer.encode(r["generated_text"]))

    nr = len(all_results)
    acc = correct / nr * 100 if nr else 0
    avg_tok = total_tok / nr if nr else 0
    total_ee = sum(1 for r in all_results if r.get("early_exit", False))

    # Write back with answer/correct fields
    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  Dynasor VL Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{nr})")
    print(f"  Avg tokens: {avg_tok:.0f}")
    print(f"  Early exits: {total_ee} ({100*total_ee/nr:.1f}%)" if nr else "")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Saved to: {output_file}")

    walltime_file = os.path.join(args.output_dir, "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": nr,
            "method": "dynasor_vl",
        }, f, indent=2)

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
