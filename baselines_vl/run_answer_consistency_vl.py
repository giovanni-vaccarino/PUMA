#!/usr/bin/env python3
"""
VL Baseline: Answer Consistency — streaming early exit via sentence-level
consistency checking on precomputed full reasoning.

For each question:
1. Load precomputed vanilla VL reasoning (from PUMA pipeline Step 1a)
2. Split reasoning into cumulative sentence prefixes
3. At each prefix, generate a trial answer (max 100 tokens, temp=0)
4. Check consecutive answer consistency (default threshold=10)
5. Stop early when threshold reached

Usage:
    python baselines_vl/run_answer_consistency_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark experiments_mdh/benchmark_vl/mathvista_test.jsonl \
        --output-dir experiments_mdh/data_vl/baselines/answer_consistency/mathvista \
        --vanilla-answers experiments_mdh/data_vl/step1a/mathvista/vanilla_results.json \
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

from nltk import sent_tokenize
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, GenerationConfig
from tqdm import tqdm

from prompt_utils_vl import (
    get_task_type_vl, build_base_prompt_vl,
    extract_answer_vl, load_image,
)
from eval_utils_vl import check_is_correct_vl


def cumulative_sentences_from_text(text):
    """Split text into cumulative sentence prefixes."""
    sentences = sent_tokenize(text)
    cumulative = []
    end = 0
    for sentence in sentences:
        idx = text.find(sentence, end)
        if idx == -1:
            end = len(text)
            cumulative.append(text[:end].strip())
            break
        end = idx + len(sentence)
        cumulative.append(text[:end].strip())
    return cumulative


def extract_reasoning(generated_text):
    """Extract reasoning from generated text."""
    if "<think>" in generated_text and "</think>" in generated_text:
        return generated_text.split("<think>")[1].split("</think>")[0].strip()
    elif "</think>" in generated_text:
        return generated_text.split("</think>")[0].strip()
    return generated_text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="VL Baseline: Answer Consistency with sentence-level early exit"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vanilla-answers", type=str, required=True,
                        help="Path to precomputed vanilla VL answers (JSON or JSONL)")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Consecutive identical answers to stop (default: 10)")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens-answer", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_type = get_task_type_vl(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "final_answers.jsonl")

    tp_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) \
        if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    print(f"Using {tp_size} GPUs for tensor parallelism")

    # Load model
    print(f"Loading model: {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=32768 + 18000,  # extra headroom for VL image tokens (~16K)
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    print("Model loaded.")

    # Sampling params for trial answer generation
    sampling_params_answer = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens_answer,
        stop=["\n"],
        presence_penalty=1.0,
    )

    # Load benchmark questions
    print(f"Loading benchmark: {args.benchmark}...")
    questions = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions")

    # Load precomputed vanilla answers (JSON list or JSONL)
    print(f"Loading vanilla answers: {args.vanilla_answers}...")
    vanilla_data = []
    with open(args.vanilla_answers, "r") as f:
        content = f.read().strip()
        if content.startswith("["):
            vanilla_data = json.loads(content)
        else:
            for line in content.split("\n"):
                if line.strip():
                    vanilla_data.append(json.loads(line))

    # Build question -> vanilla data mapping
    question2data = {}
    for item in vanilla_data:
        q_text = item.get("question", "")
        reasoning = item.get("reasoning", "")
        if not reasoning and item.get("model_response"):
            reasoning = extract_reasoning(item["model_response"])
        item["_reasoning"] = reasoning
        question2data[q_text] = item
    print(f"Loaded {len(question2data)} vanilla answers")

    # Pre-cache images
    print("Pre-caching images...")
    image_cache = {}
    for i, q in enumerate(tqdm(questions, desc="Loading images")):
        image_cache[i] = load_image(q["image_path"])

    # Pre-compute partial reasoning prompts
    print("Building partial reasoning prompts...")
    all_partial_reasonings = {}
    all_partial_prompts = {}
    q_state = {}
    results_dict = {}
    total_sentences = 0
    total_stopped_early = 0

    for q_idx, q in enumerate(tqdm(questions, desc="Building prompts")):
        if q["question"] not in question2data:
            print(f"Warning: No vanilla answer for question {q_idx}, skipping")
            results_dict[q_idx] = None
            continue

        data = question2data[q["question"]]
        reasoning = data.get("_reasoning", "")

        if not reasoning:
            results_dict[q_idx] = {
                "question": q["question"],
                "answer": data.get("answer", data.get("final_answer", "")),
                "partial_reasoning": "",
                "generated_text": data.get("model_response", ""),
                "num_trial_answers": 0,
                "num_trial_answer_tokens": 0,
                "stopped_early": False,
                "total_sentences": 0,
            }
            continue

        partial_reasonings = cumulative_sentences_from_text(reasoning)
        num_sentences = len(partial_reasonings)
        total_sentences += num_sentences

        # Build base prompt for this question
        base_prompt = build_base_prompt_vl(
            processor, args.model, q["question"], q["image_path"],
            task_type, "default",
        )

        prompts = []
        for partial in partial_reasonings:
            trial_prompt = base_prompt + partial + "\n</think>\n\\boxed"
            prompts.append(trial_prompt)

        all_partial_reasonings[q_idx] = partial_reasonings
        all_partial_prompts[q_idx] = prompts
        q_state[q_idx] = {
            "last_answer": None,
            "repeat_count": 0,
            "trial_tokens": 0,
            "last_text": "",
        }

    # Iterative batch processing
    active_indices = sorted(all_partial_prompts.keys())
    max_rounds = max(len(v) for v in all_partial_prompts.values()) if all_partial_prompts else 0

    print(f"\n{'='*60}")
    print(f"  Online Answer Consistency VL (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"Questions to process: {len(active_indices)}, max sentences: {max_rounds}")
    start_time = time.time()

    for round_idx in range(max_rounds):
        if not active_indices:
            break

        # Build batch for this round
        batch_inputs = []
        batch_q_indices = []
        for q_idx in active_indices:
            if round_idx < len(all_partial_prompts[q_idx]):
                prompt = all_partial_prompts[q_idx][round_idx]
                img = image_cache[q_idx]
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": img},
                })
                batch_q_indices.append(q_idx)

        if not batch_inputs:
            break

        # Batch generate
        outputs = llm.generate(batch_inputs, sampling_params_answer)

        # Process results
        next_active = []
        for q_idx, output in zip(batch_q_indices, outputs):
            raw_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)

            text = "\\boxed" + raw_text
            answer = extract_answer_vl(text, args.dataset)

            state = q_state[q_idx]
            state["trial_tokens"] += token_count
            state["last_text"] = text

            if answer == state["last_answer"]:
                state["repeat_count"] += 1
            else:
                state["last_answer"] = answer
                state["repeat_count"] = 1

            num_sentences = len(all_partial_reasonings[q_idx])
            trial_num = round_idx + 1

            if state["repeat_count"] >= args.threshold:
                results_dict[q_idx] = {
                    "question": questions[q_idx]["question"],
                    "answer": answer,
                    "partial_reasoning": all_partial_reasonings[q_idx][round_idx],
                    "generated_text": text,
                    "num_trial_answers": trial_num,
                    "num_trial_answer_tokens": state["trial_tokens"],
                    "stopped_early": True,
                    "total_sentences": num_sentences,
                }
                total_stopped_early += 1
            elif round_idx >= num_sentences - 1:
                results_dict[q_idx] = {
                    "question": questions[q_idx]["question"],
                    "answer": state["last_answer"],
                    "partial_reasoning": all_partial_reasonings[q_idx][-1],
                    "generated_text": state["last_text"],
                    "num_trial_answers": num_sentences,
                    "num_trial_answer_tokens": state["trial_tokens"],
                    "stopped_early": False,
                    "total_sentences": num_sentences,
                }
            else:
                next_active.append(q_idx)

        # Retire questions not in batch
        for q_idx in active_indices:
            if q_idx not in batch_q_indices and q_idx not in results_dict:
                num_sentences = len(all_partial_reasonings[q_idx])
                state = q_state[q_idx]
                results_dict[q_idx] = {
                    "question": questions[q_idx]["question"],
                    "answer": state["last_answer"],
                    "partial_reasoning": all_partial_reasonings[q_idx][-1],
                    "generated_text": state["last_text"],
                    "num_trial_answers": num_sentences,
                    "num_trial_answer_tokens": state["trial_tokens"],
                    "stopped_early": False,
                    "total_sentences": num_sentences,
                }

        active_indices = next_active
        print(f"  Round {round_idx+1}: batch={len(batch_inputs)}, "
              f"converged={total_stopped_early}, active={len(active_indices)}")

    elapsed = time.time() - start_time

    # Collect results in order and evaluate
    results = []
    correct = 0
    for q_idx in range(len(questions)):
        if q_idx in results_dict and results_dict[q_idx] is not None:
            r = results_dict[q_idx]
            q = questions[q_idx]
            # total_tokens = full output (reasoning prefix + answer)
            full_output = r.get("partial_reasoning", "") + "\n</think>\n" + r.get("generated_text", "")
            r["total_tokens"] = len(tokenizer.encode(full_output))
            is_correct = check_is_correct_vl(
                r.get("answer", ""), q["answer"], args.dataset,
                q.get("question_type", ""),
            )
            if is_correct:
                correct += 1
            r["correct"] = is_correct
            r["ground_truth_answer"] = q["answer"]
            r["image_path"] = q["image_path"]
            results.append(r)

    # Save
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Save walltime
    walltime_file = os.path.join(args.output_dir, "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": len(results),
            "method": "answer_consistency_vl",
        }, f, indent=2)

    # Print statistics
    acc = correct / len(results) * 100 if results else 0
    print(f"\n{'='*60}")
    print(f"  Answer Consistency VL Results")
    print(f"{'='*60}")
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results) if results else 0
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Stopped early: {total_stopped_early} ({100*total_stopped_early/len(results):.1f}%)" if results else "")
    if results:
        avg_trial = sum(r["num_trial_answers"] for r in results) / len(results)
        avg_total = sum(r["total_sentences"] for r in results) / len(results)
        print(f"  Avg trial answers: {avg_trial:.1f}")
        print(f"  Avg total sentences: {avg_total:.1f}")
        if avg_total > 0:
            print(f"  Avg savings: {100*(1 - avg_trial/avg_total):.1f}% fewer generations")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()
