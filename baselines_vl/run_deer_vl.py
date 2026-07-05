#!/usr/bin/env python3
"""
VL Baseline: DEER — Deliberative Reasoning with Confidence-Based Early Exit.

Generates reasoning incrementally, checking confidence at action transition
points ("Wait", "Alternatively"). Uses sigmoid confidence on logprobs to
determine when to exit early.

Based on: DEER (vllm_deer.py) adapted for vision-language models.

Usage:
    python baselines_vl/run_deer_vl.py \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista \
        --benchmark data/benchmark_vl/mathvista_test.jsonl \
        --output-dir runs/vl/baselines/deer/mathvista \
        --limit 500
"""

import argparse
import json
import math
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


def calculate_average_max_prob_from_logprobs(logprobs_list, policy='avg2'):
    """Calculate average max token probability from logprobs list."""
    num_tokens = len(logprobs_list)
    start_index = 1
    end_index = num_tokens

    if num_tokens < 1:
        return 0.0

    total_prob_sum = 0.0
    log_prob_sum = 0.0
    count = 0
    min_prob = 1.0

    for i in range(start_index, end_index):
        if i < len(logprobs_list) and logprobs_list[i]:
            try:
                logprob_obj = list(logprobs_list[i].values())[0]
                if hasattr(logprob_obj, 'logprob'):
                    prob = torch.exp(torch.tensor(logprob_obj.logprob)).item()
                    if prob < min_prob:
                        min_prob = prob
                    total_prob_sum += prob
                    log_prob_sum += math.log(max(prob, 1e-10))
                    count += 1
            except (IndexError, KeyError, AttributeError):
                pass

    if count == 0:
        return 0.0
    if policy == 'min':
        return min_prob
    elif policy == 'avg1':
        return total_prob_sum / count
    elif policy == 'avg2':
        return math.exp(log_prob_sum / count)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="VL Baseline: DEER")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Confidence threshold for early exit (default: 0.95)")
    parser.add_argument("--policy", type=str, default="avg1",
                        choices=["avg1", "avg2", "min"],
                        help="Confidence aggregation policy")
    parser.add_argument("--max-judge-steps", type=int, default=10,
                        help="Max number of confidence checks (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Total token budget per question")
    parser.add_argument("--think-ratio", type=float, default=0.9)
    parser.add_argument("--prob-check-max-tokens", type=int, default=20)
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

    # Load questions
    questions = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions")

    # Pre-cache images
    print("Pre-caching images...")
    image_cache = {}
    for i, q in enumerate(tqdm(questions, desc="Loading images")):
        image_cache[i] = load_image(q["image_path"])

    # DEER parameters
    continue_str = "Wait"
    last_token_strs = ["</think>"]
    answer_prompt_str = "\n**Final Answer**\n\\boxed"
    pred_prob_stop_tokens = [' }', '}\n', '}\n\n', '}.', '}.\n', '}\\', '}}', ')}', ')}.', ')}\n']
    answer_stop_tokens = [tokenizer.eos_token] if tokenizer.eos_token else []

    last_token_ids = []
    for s in last_token_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            last_token_ids.extend(ids)
    last_token_ids = list(set(last_token_ids))

    generation_stop_tokens = [continue_str] + last_token_strs
    if tokenizer.eos_token:
        generation_stop_tokens.append(tokenizer.eos_token)

    think_limit_tokens = int(args.max_tokens * args.think_ratio)

    # Build initial state for each question
    print("Building initial prompts...")
    start_time = time.time()
    n_questions = len(questions)

    questions_state = {}
    for i, q in enumerate(questions):
        base_prompt = build_base_prompt_vl(
            processor, args.model, q["question"], q["image_path"],
            task_type, "default",
        )
        questions_state[i] = {
            'question': q,
            'state': 'needs_thought_chunk',
            'base_prompt': base_prompt,
            'current_full_sequence': base_prompt,
            'generated_thinking_history': "",
            'generated_answer_history': "",
            'pred_prob': 0.0,
            'too_long': 0,
            'high_prob': 0,
            'regular_end': 0,
            'thinking_steps': 0,
            'num_trial_answer_tokens': 0,
            'output_dict': {},
        }

    active_indices = list(range(n_questions))
    pbar = tqdm(total=n_questions, desc="Processing questions")

    # Main DEER loop
    while active_indices:
        batch_inputs = []
        batch_sampling_params = []
        batch_request_info = []

        for q_idx in list(active_indices):
            state = questions_state[q_idx]
            if state['state'] in ['finished', 'error']:
                continue

            img = image_cache[q_idx]
            current_len = len(tokenizer.encode(state['current_full_sequence'], add_special_tokens=False))
            thinking_len = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))

            if state['state'] == 'needs_thought_chunk':
                max_new = min(
                    think_limit_tokens - thinking_len,
                    (args.max_tokens + 8000) - current_len,
                )
                if max_new <= 0:
                    state['state'] = 'needs_answer'
                    state['too_long'] = 1
                    continue

                if state['thinking_steps'] < args.max_judge_steps:
                    sp = SamplingParams(
                        max_tokens=max_new, temperature=temperature,
                        top_p=top_p, top_k=top_k,
                        stop=generation_stop_tokens,
                    )
                else:
                    sp = SamplingParams(
                        max_tokens=max_new, temperature=temperature,
                        top_p=top_p, top_k=top_k,
                        stop=last_token_strs,
                    )

                batch_inputs.append({
                    "prompt": state['current_full_sequence'],
                    "multi_modal_data": {"image": img},
                })
                batch_sampling_params.append(sp)
                batch_request_info.append((q_idx, 'think'))

            elif state['state'] == 'needs_prob_check':
                probe_prompt = state['current_full_sequence'] + answer_prompt_str
                sp = SamplingParams(
                    max_tokens=args.prob_check_max_tokens,
                    temperature=temperature,
                    stop=pred_prob_stop_tokens,
                    logprobs=1,
                )
                batch_inputs.append({
                    "prompt": probe_prompt,
                    "multi_modal_data": {"image": img},
                })
                batch_sampling_params.append(sp)
                batch_request_info.append((q_idx, 'prob_check'))

            elif state['state'] == 'needs_answer':
                answer_prompt = (state['base_prompt'] +
                                 state['generated_thinking_history'] +
                                 '\n</think>\n\n')
                state['generated_thinking_history'] += '\n</think>\n\n'

                answer_len = len(tokenizer.encode(answer_prompt, add_special_tokens=False))
                max_new = min(
                    args.max_tokens - thinking_len,
                    (args.max_tokens + 8000) - answer_len,
                )
                if max_new <= 0:
                    state['state'] = 'error'
                    state['output_dict'] = {
                        'question': state['question']['question'],
                        'generated_text': state['generated_thinking_history'],
                        'gold_answer': state['question']['answer'],
                    }
                    active_indices.remove(q_idx)
                    pbar.update(1)
                    continue

                sp = SamplingParams(
                    max_tokens=max_new, temperature=temperature,
                    top_p=top_p, stop=answer_stop_tokens,
                )
                batch_inputs.append({
                    "prompt": answer_prompt,
                    "multi_modal_data": {"image": img},
                })
                batch_sampling_params.append(sp)
                batch_request_info.append((q_idx, 'answer'))

        if not batch_inputs:
            # Check if all remaining are stuck
            remaining = [i for i in active_indices
                         if questions_state[i]['state'] not in ['finished', 'error']]
            if not remaining:
                break
            # Force-finish stuck questions
            for q_idx in remaining:
                state = questions_state[q_idx]
                state['state'] = 'error'
                state['output_dict'] = {
                    'question': state['question']['question'],
                    'generated_text': state['generated_thinking_history'] +
                                      state['generated_answer_history'],
                    'gold_answer': state['question']['answer'],
                }
                active_indices.remove(q_idx)
                pbar.update(1)
            break

        # Generate batch
        outputs = llm.generate(batch_inputs, batch_sampling_params, use_tqdm=False)

        for i, output in enumerate(outputs):
            q_idx, step_type = batch_request_info[i]
            state = questions_state[q_idx]

            if not output.outputs:
                state['state'] = 'error'
                state['output_dict'] = {
                    'question': state['question']['question'],
                    'generated_text': state['generated_thinking_history'],
                    'gold_answer': state['question']['answer'],
                }
                if q_idx in active_indices:
                    active_indices.remove(q_idx)
                    pbar.update(1)
                continue

            completion = output.outputs[0]
            gen_text = completion.text
            gen_ids = completion.token_ids
            last_id = gen_ids[-1] if gen_ids else -1

            if step_type == 'think':
                if last_id in last_token_ids:
                    state['state'] = 'needs_answer'
                    state['generated_thinking_history'] += gen_text
                    state['current_full_sequence'] = state['base_prompt'] + state['generated_thinking_history']
                    state['regular_end'] = 1
                else:
                    state['generated_thinking_history'] += gen_text
                    state['current_full_sequence'] = state['base_prompt'] + state['generated_thinking_history']
                    state['state'] = 'needs_prob_check'
                    state['thinking_steps'] += 1

            elif step_type == 'prob_check':
                if completion.logprobs:
                    state['pred_prob'] = calculate_average_max_prob_from_logprobs(
                        completion.logprobs, args.policy,
                    )
                else:
                    state['pred_prob'] = 0.0

                state['num_trial_answer_tokens'] += len(gen_ids)
                thinking_len = len(tokenizer.encode(
                    state['generated_thinking_history'], add_special_tokens=False,
                ))
                limit_reached = thinking_len >= think_limit_tokens - 50

                if state['pred_prob'] > args.threshold or limit_reached:
                    state['state'] = 'needs_answer'
                    if limit_reached:
                        state['too_long'] = 1
                    else:
                        state['high_prob'] = 1
                else:
                    state['state'] = 'needs_thought_chunk'
                    if not state['current_full_sequence'].strip().endswith(continue_str):
                        state['current_full_sequence'] += continue_str
                        state['generated_thinking_history'] += continue_str

            elif step_type == 'answer':
                state['generated_answer_history'] += gen_text
                state['state'] = 'finished'
                final_text = state['generated_thinking_history'] + state['generated_answer_history']
                state['output_dict'] = {
                    'question': state['question']['question'],
                    'generated_text': final_text,
                    'gold_answer': state['question']['answer'],
                    'too_long': state['too_long'],
                    'thinking_steps': state['thinking_steps'],
                    'high_prob': state['high_prob'],
                    'regular_end': state['regular_end'],
                    'num_trial_answer_tokens': state['num_trial_answer_tokens'],
                }
                if q_idx in active_indices:
                    active_indices.remove(q_idx)
                    pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time

    # Collect and evaluate results
    results = []
    correct = 0
    total_tokens = 0
    for q_idx in range(n_questions):
        state = questions_state[q_idx]
        od = state.get('output_dict', {})
        if not od:
            continue
        q = questions[q_idx]
        gen_text = od.get('generated_text', '')
        answer = extract_answer_vl(gen_text, args.dataset)
        n_tok = len(tokenizer.encode(gen_text))
        total_tokens += n_tok

        is_correct = check_is_correct_vl(
            answer, q["answer"], args.dataset, q.get("question_type", ""),
        )
        if is_correct:
            correct += 1

        od['final_answer'] = answer or ""
        od['correct'] = is_correct
        od['total_tokens'] = n_tok
        od['image_path'] = q['image_path']
        od['early_exit'] = bool(state.get('high_prob', 0))
        results.append(od)

    # Save
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    walltime_file = os.path.join(args.output_dir, "walltime.json")
    with open(walltime_file, "w") as f:
        json.dump({
            "wall_time_seconds": round(elapsed, 2),
            "num_questions": len(results),
            "method": "deer_vl",
        }, f, indent=2)

    acc = correct / len(results) * 100 if results else 0
    avg_tok = total_tokens / len(results) if results else 0
    total_ee = sum(1 for r in results if r.get('early_exit', False))
    print(f"\n{'='*60}")
    print(f"  DEER VL Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.2f}% ({correct}/{len(results)})")
    print(f"  Avg tokens: {avg_tok:.0f}")
    print(f"  Early exits: {total_ee} ({100*total_ee/len(results):.1f}%)" if results else "")
    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Saved to: {output_file}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
