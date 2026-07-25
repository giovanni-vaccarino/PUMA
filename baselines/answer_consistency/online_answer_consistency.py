#!/usr/bin/env python3
"""
Online Answer Consistency - Streaming baseline for token efficiency.

This module performs answer consistency checking on precomputed vanilla answers.
For each question, it:
1. Loads precomputed vanilla answers (full reasoning)
2. Incrementally generates answers from partial reasoning prefixes
3. Checks consistency after each partial answer
4. Stops early when consistency threshold is reached (e.g., 3 consecutive same answers)

Vanilla answers must be precomputed separately using vanilla_generation.py.

Usage:
    python -m baselines.answer_consistency.online_answer_consistency \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/answer_consistency/aime24" \
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl" \
        --threshold 3
"""

import json
import argparse
import os
import sys

# Add project root to sys.path so `baselines.*` imports work when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nltk import sent_tokenize
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from baselines.utils.math_util import (
    my_answer_extraction, extract_code_answer, fuzzy_code_match,
    CODE_SYS_PROMPT, CODE_ANSWER_SUFFIX,
)


def cumulative_sentences_from_text(text):
    """Split text into cumulative sentence prefixes."""
    sentences = sent_tokenize(text)
    cumulative = []
    end = 0
    for sentence in sentences:
        idx = text.find(sentence, end)
        if idx == -1:
            # Fallback: just append
            end = len(text)
            cumulative.append(text[:end].strip())
            break
        end = idx + len(sentence)
        cumulative.append(text[:end].strip())
    return cumulative


def extract_reasoning(generated_text):
    """Extract reasoning from generated text, handling various formats."""
    # Case 1: Both <think> and </think> are present
    if "<think>" in generated_text and "</think>" in generated_text:
        reasoning = generated_text.split("<think>")[1].split("</think>")[0].strip()
    # Case 2: Only </think> is present (reasoning starts immediately)
    elif "</think>" in generated_text:
        reasoning = generated_text.split("</think>")[0].strip()
    # Case 3: Neither tag is present (entire text is reasoning)
    else:
        reasoning = generated_text.strip()
    return reasoning

def main():
    parser = argparse.ArgumentParser(
        description="Online Answer Consistency - streaming baseline for token efficiency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on AIME24 benchmark
    python -m baselines.answer_consistency.online_answer_consistency \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/answer_consistency/aime24" \\
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl"

    # With custom threshold
    python -m baselines.answer_consistency.online_answer_consistency \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/answer_consistency/aime24" \\
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl" \\
        --threshold 5
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file with questions")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--vanilla-answers", type=str, required=True,
                        help="Path to precomputed vanilla answers (from vanilla_generation.py)")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Number of consecutive identical answers to stop (default: 10)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Maximum number of questions to process (default: 10000)")
    parser.add_argument("--max-tokens-answer", type=int, default=100,
                        help="Max tokens for partial answer generation")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "final_answers.jsonl")

    is_code = "livecodebench" in args.benchmark.lower()
    if is_code:
        print("Code task detected: using fuzzy code match for convergence")

    # Determine tensor parallel size from CUDA_VISIBLE_DEVICES
    tensor_parallel_size = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
    print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")

    sys_prompt = CODE_SYS_PROMPT if is_code else "Please reason step by step, and put your final answer within \\boxed{}."
    
    # Load model
    print(f"\nLoading model: {args.model}...")
    if "70B" not in args.model:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            max_model_len=32768,
            gpu_memory_utilization=0.90,
        )
    else:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            max_model_len=8192,
            max_num_seqs=35,
        )
    print("Model loaded.")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Sampling params for partial answer generation
    if is_code:
        # Code: generate up to 200 tokens, stop at ``` (end of code block)
        sampling_params_answer = SamplingParams(
            temperature=0.0,
            max_tokens=200,
            stop=["```"],
        )
    else:
        # Math: stop at newline, max 100 tokens
        sampling_params_answer = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens_answer,
            stop=["\n"],
            presence_penalty=1.0
        )
    
    # Load benchmark questions
    print(f"\nLoading benchmark: {args.benchmark}...")
    questions = []
    gt_answers = []
    question_ids = []
    with open(args.benchmark, "r") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
            gt_answers.append(item.get("answer", ""))
            question_ids.append(item.get("question_id", ""))
    questions = questions[:args.limit]
    gt_answers = gt_answers[:args.limit]
    question_ids = question_ids[:args.limit]
    print(f"Loaded {len(questions)} questions")
    
    # Load precomputed vanilla answers
    print(f"\nLoading precomputed vanilla answers from: {args.vanilla_answers}")
    question2data = {}
    with open(args.vanilla_answers, "r") as f:
        for line in f:
            item = json.loads(line)
            # For code tasks, vanilla_code has no "reasoning" field;
            # use generated_text as the reasoning (it IS the chain-of-thought)
            if "reasoning" not in item and item.get("generated_text"):
                item["reasoning"] = item["generated_text"]
            question2data[item["question"]] = item
    print(f"Loaded {len(question2data)} vanilla answers")
    
    # Build base prompt (up to assistant turn) using standard chat template
    # This will be reused for all partial reasoning prompts
    def build_partial_prompt(question, partial_reasoning):
        """Build prompt with partial reasoning using standard chat template + manual concatenation."""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ]
        base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_code:
            # Code: close thinking, start code block
            full_prompt = base_prompt + partial_reasoning + CODE_ANSWER_SUFFIX
        else:
            # Math: close thinking, start \boxed for answer extraction
            full_prompt = base_prompt + partial_reasoning + "\n</think>\n\\boxed"
        return full_prompt
    
    # Online answer consistency - iterative batch processing
    print(f"\n{'='*60}")
    print(f"  Online Answer Consistency (threshold={args.threshold}, iterative batch)")
    print(f"{'='*60}")

    # Pre-compute partial reasonings and prompts for all questions
    all_partial_reasonings = {}  # q_idx -> list of partial reasonings
    all_partial_prompts = {}     # q_idx -> list of prompts
    q_state = {}                 # q_idx -> {last_answer, repeat_count, trial_tokens, last_text}
    results_dict = {}            # q_idx -> result dict (filled when done)

    total_sentences = 0
    total_stopped_early = 0

    print("Pre-computing partial reasoning prompts...")
    for q_idx, question in enumerate(tqdm(questions, desc="Building prompts")):
        if question not in question2data:
            print(f"Warning: No vanilla answer for question {q_idx}, skipping")
            results_dict[q_idx] = None
            continue

        data = question2data[question]
        reasoning = data.get("reasoning", "")

        if not reasoning:
            results_dict[q_idx] = {
                "question": question,
                "answer": data.get("answer", ""),
                "partial_reasoning": "",
                "generated_text": data.get("generated_text", ""),
                "num_trial_answers": 0,
                "num_trial_answer_tokens": 0,
                "stopped_early": False,
                "total_sentences": 0
            }
            continue

        partial_reasonings = cumulative_sentences_from_text(reasoning)
        num_sentences = len(partial_reasonings)
        total_sentences += num_sentences

        prompts = []
        for partial in partial_reasonings:
            prompts.append(build_partial_prompt(question, partial))

        all_partial_reasonings[q_idx] = partial_reasonings
        all_partial_prompts[q_idx] = prompts
        q_state[q_idx] = {
            "last_answer": None,
            "repeat_count": 0,
            "trial_tokens": 0,
            "last_text": "",
        }

    # Active set: questions that still need processing
    active_indices = sorted(all_partial_prompts.keys())
    max_rounds = max(len(v) for v in all_partial_prompts.values()) if all_partial_prompts else 0

    print(f"Questions to process: {len(active_indices)}, max sentences: {max_rounds}")
    print(f"Starting iterative batch processing...\n")

    for round_idx in range(max_rounds):
        if not active_indices:
            break

        # Build batch for this round
        batch_prompts = []
        batch_q_indices = []
        for q_idx in active_indices:
            if round_idx < len(all_partial_prompts[q_idx]):
                batch_prompts.append(all_partial_prompts[q_idx][round_idx])
                batch_q_indices.append(q_idx)

        if not batch_prompts:
            break

        # Batch generate
        outputs = llm.generate(batch_prompts, sampling_params_answer)

        # Process results and update state
        next_active = []
        for q_idx, output in zip(batch_q_indices, outputs):
            raw_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)

            if is_code:
                # Code: the model generated code after ```python\n prefix
                answer = raw_text.strip()
                text = raw_text
            else:
                text = "\\boxed" + raw_text
                answer = my_answer_extraction(text)

            state = q_state[q_idx]
            state["trial_tokens"] += token_count
            state["last_text"] = text

            if is_code:
                # Fuzzy code match for convergence
                if state["last_answer"] and answer and fuzzy_code_match(answer, state["last_answer"]):
                    state["repeat_count"] += 1
                else:
                    state["last_answer"] = answer
                    state["repeat_count"] = 1 if answer else 0
            else:
                if answer == state["last_answer"]:
                    state["repeat_count"] += 1
                else:
                    state["last_answer"] = answer
                    state["repeat_count"] = 1

            num_sentences = len(all_partial_reasonings[q_idx])
            trial_num = round_idx + 1

            if state["repeat_count"] >= args.threshold:
                result_entry = {
                    "question": questions[q_idx],
                    "answer": answer,
                    "partial_reasoning": all_partial_reasonings[q_idx][round_idx],
                    "generated_text": text,
                    "num_trial_answers": trial_num,
                    "num_trial_answer_tokens": state["trial_tokens"],
                    "stopped_early": True,
                    "total_sentences": num_sentences,
                }
                if is_code:
                    result_entry["extracted_code"] = answer
                    result_entry["question_id"] = question_ids[q_idx]
                results_dict[q_idx] = result_entry
                total_stopped_early += 1
            elif round_idx >= num_sentences - 1:
                result_entry = {
                    "question": questions[q_idx],
                    "answer": state["last_answer"],
                    "partial_reasoning": all_partial_reasonings[q_idx][-1],
                    "generated_text": state["last_text"],
                    "num_trial_answers": num_sentences,
                    "num_trial_answer_tokens": state["trial_tokens"],
                    "stopped_early": False,
                    "total_sentences": num_sentences,
                }
                if is_code:
                    result_entry["extracted_code"] = state["last_answer"]
                    result_entry["question_id"] = question_ids[q_idx]
                results_dict[q_idx] = result_entry
            else:
                next_active.append(q_idx)

        # Retire questions not in this batch (exhausted earlier)
        for q_idx in active_indices:
            if q_idx not in batch_q_indices and q_idx not in results_dict:
                num_sentences = len(all_partial_reasonings[q_idx])
                state = q_state[q_idx]
                result_entry = {
                    "question": questions[q_idx],
                    "answer": state["last_answer"],
                    "partial_reasoning": all_partial_reasonings[q_idx][-1],
                    "generated_text": state["last_text"],
                    "num_trial_answers": num_sentences,
                    "num_trial_answer_tokens": state["trial_tokens"],
                    "stopped_early": False,
                    "total_sentences": num_sentences,
                }
                if is_code:
                    result_entry["extracted_code"] = state["last_answer"]
                    result_entry["question_id"] = question_ids[q_idx]
                results_dict[q_idx] = result_entry

        active_indices = next_active
        print(f"  Round {round_idx+1}: batch={len(batch_prompts)}, converged={total_stopped_early}, active={len(active_indices)}")

    # Collect results in original question order
    results = []
    for q_idx in range(len(questions)):
        if q_idx in results_dict and results_dict[q_idx] is not None:
            results.append(results_dict[q_idx])
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"Total questions processed: {len(results)}")
    print(f"Questions stopped early: {total_stopped_early} ({100*total_stopped_early/len(results):.1f}%)")
    print(f"Total sentences across all questions: {total_sentences}")
    
    if results:
        avg_trial_answers = sum(r["num_trial_answers"] for r in results) / len(results)
        avg_total_sentences = sum(r["total_sentences"] for r in results) / len(results)
        print(f"Average trial answers per question: {avg_trial_answers:.1f}")
        print(f"Average total sentences per question: {avg_total_sentences:.1f}")
        if avg_total_sentences > 0:
            print(f"Average savings: {100*(1 - avg_trial_answers/avg_total_sentences):.1f}% fewer generations")

if __name__ == "__main__":
    main()