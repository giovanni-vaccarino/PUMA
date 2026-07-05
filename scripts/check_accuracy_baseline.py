#!/usr/bin/env python3
"""
Compute accuracy, CR (Compression Rate), and CRT (Compression Rate with Trial answers).

CR  = compressed_tokens / original_vanilla_tokens
CRT = (compressed_tokens + trial_answer_tokens) / original_vanilla_tokens

Usage:
    python scripts/check_accuracy_baseline.py \
        --model <MODEL_NAME> \
        --predictions <PRED_FILE> \
        --ground-truth <GT_FILE> \
        --vanilla <VANILLA_FILE> \
        --dataset <DATASET_NAME>
"""

import json
import os
import sys
import argparse

# Add project root to sys.path so `baselines.*` and `method.*` imports work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoTokenizer
from baselines.utils.math_util import my_answer_extraction, extract_code_answer
from puma.math_grader import check_is_correct


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens for a given text."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy, CR and CRT for baseline outputs")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model identifier")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL file")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSONL file")
    parser.add_argument("--vanilla", type=str, required=True, help="Path to vanilla answers JSONL")
    parser.add_argument("--tokens-after-think", type=str, default="",
                        help="Path to tokens_after_think.jsonl (for think token adjustment)")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. 'gpqa') for dataset-specific answer extraction")
    parser.add_argument("--verbose", action="store_true",
                        help="(accepted for compatibility with baseline runners; no effect)")
    args = parser.parse_args()

    # Auto-detect dataset from ground-truth path if not specified
    if not args.dataset:
        gt_lower = args.ground_truth.lower()
        for name in ["livecodebench", "gpqa", "aime", "math", "olympiadbench"]:
            if name in gt_lower:
                args.dataset = name
                break
    if args.dataset:
        print(f"Dataset: {args.dataset}")

    is_code = "livecodebench" in args.dataset.lower() if args.dataset else False
    if is_code:
        print("Code task detected: skipping math accuracy grading, computing CR/CRT only.")
        print("Accuracy: evaluate with eval_livecodebench.py")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load data
    preds, gts, vanilla = [], [], []
    with open(args.predictions) as f:
        for line in f:
            preds.append(json.loads(line))
    with open(args.ground_truth) as f:
        for line in f:
            gts.append(json.loads(line))
    with open(args.vanilla) as f:
        for line in f:
            vanilla.append(json.loads(line))

    tokens_after_think = []
    if args.tokens_after_think:
        with open(args.tokens_after_think) as f:
            for line in f:
                tokens_after_think.append(json.loads(line))

    n = len(preds)
    assert n == len(gts), f"Mismatch: {n} predictions vs {len(gts)} ground truths"
    assert n == len(vanilla), f"Mismatch: {n} predictions vs {len(vanilla)} vanilla"
    if tokens_after_think:
        assert n == len(tokens_after_think), f"Mismatch: {n} predictions vs {len(tokens_after_think)} tokens_after_think"

    correct_compressed = 0
    correct_original = 0
    total_original_tokens = 0
    total_compressed_tokens = 0
    total_trial_answer_tokens = 0

    dataset = args.dataset

    for i in range(n):
        pred, gt, van = preds[i], gts[i], vanilla[i]

        if not is_code:
            # Math/GPQA: compute accuracy via math grader
            gt_ans = str(gt["answer"]).strip()

            pred_text = pred.get("generated_text", "") or pred.get("answer", "")
            pred_ans = my_answer_extraction(pred_text, dataset=dataset)
            if check_is_correct(pred_ans, gt_ans):
                correct_compressed += 1

            van_text = van.get("generated_text", "") or van.get("answer", "")
            van_ans = my_answer_extraction(van_text, dataset=dataset)
            if check_is_correct(van_ans, gt_ans):
                correct_original += 1

        original_tokens = count_tokens(tokenizer, van.get("generated_text", ""))
        total_original_tokens += original_tokens

        if tokens_after_think: # Think Token Adjustment Method
            reasoning_tokens = count_tokens(tokenizer, pred.get("reasoning", ""))
            answer_tokens = tokens_after_think[i].get("tokens_after_think", 0)
            compressed_tokens = reasoning_tokens + answer_tokens
            trial_tokens = 0
        elif "thinking_steps" in pred: # DEER Method & dynasor method
            compressed_tokens = count_tokens(tokenizer, pred.get("generated_text", ""))
            trial_tokens = pred.get("num_trial_answer_tokens", 0)
        elif "partial_reasoning" in pred: # Answer Consistency Method
            partial_reasoning_tokens = count_tokens(tokenizer, pred.get("partial_reasoning", ""))
            answer_tokens = count_tokens(tokenizer, pred.get("generated_text", ""))
            compressed_tokens = partial_reasoning_tokens + answer_tokens
            trial_tokens = pred.get("num_trial_answer_tokens", 0)
        elif "planning_tokens" in pred: # Plan-and-Budget Method
            compressed_tokens = count_tokens(tokenizer, pred.get("generated_text", ""))
            trial_tokens = pred.get("planning_tokens", 0)
        else: # nothinking method
            compressed_tokens = count_tokens(tokenizer, pred.get("generated_text", ""))
            trial_tokens = 0


        total_compressed_tokens += compressed_tokens
        total_trial_answer_tokens += trial_tokens

    avg_original = total_original_tokens / n
    avg_compressed = total_compressed_tokens / n
    avg_trial = total_trial_answer_tokens / n
    avg_total = avg_compressed + avg_trial

    cr = total_compressed_tokens / total_original_tokens * 100 if total_original_tokens > 0 else 0
    crt = (total_compressed_tokens + total_trial_answer_tokens) / total_original_tokens * 100 if total_original_tokens > 0 else 0

    # Print results
    print(f"\n{'='*70}")
    print(f"  COMPRESSION STATISTICS")
    print(f"{'='*70}")
    print(f"Evaluated samples: {n}")
    print()
    print(f"Token Statistics:")
    print(f"    Average tokens (original):               {avg_original:.2f}")
    print(f"    Average tokens (compressed):             {avg_compressed:.2f}")
    print(f"    Average trial answer tokens:             {avg_trial:.2f}")
    print(f"    Average tokens (compressed + trial answers): {avg_total:.2f}")
    print()
    print(f"Compression (excluding trial answers):")
    print(f"    Compression rate: {cr:.2f}%")
    print(f"    Reduction:        {100-cr:.2f}%")
    print()
    print(f"Compression (including trial answers):")
    print(f"    Compression rate: {crt:.2f}%")
    print(f"    Reduction:        {100-crt:.2f}%")

    if is_code:
        print()
        print(f"Accuracy: N/A (code task — use eval_livecodebench.py for pass@1)")
        accuracy_compressed, accuracy_original, accuracy_impact = 0, 0, 0
    else:
        accuracy_compressed = (correct_compressed / n) * 100
        accuracy_original = (correct_original / n) * 100
        accuracy_impact = accuracy_compressed - accuracy_original
        print()
        print(f"Accuracy:")
        print(f"    Original:   {accuracy_original:.2f}% ({correct_original}/{n})")
        print(f"    Compressed: {accuracy_compressed:.2f}% ({correct_compressed}/{n})")
        print(f"    Impact:     {accuracy_impact:+.2f}%")

    # Wall time (read from walltime.json in predictions directory)
    walltime_file = os.path.join(os.path.dirname(args.predictions), "walltime.json")
    if os.path.exists(walltime_file):
        with open(walltime_file) as f:
            wt = json.load(f)
        wall_secs = wt.get("wall_time_seconds", 0)
        print()
        print(f"Wall Time:")
        print(f"    Inference:  {wall_secs:.2f}s ({wall_secs/60:.1f}min)")
        if n > 0:
            print(f"    Per question: {wall_secs/n:.2f}s")

    print(f"{'='*70}")

    return {"accuracy_compressed": accuracy_compressed, "accuracy_original": accuracy_original, "cr": cr, "crt": crt}


if __name__ == "__main__":
    main()