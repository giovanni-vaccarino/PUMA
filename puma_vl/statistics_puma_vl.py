#!/usr/bin/env python3
"""
Step 5: Compute compression statistics for VL benchmarks.

Extends puma/statistics_puma.py with VL-specific evaluation.
Uses eval_utils_vl for dataset-aware answer grading.

Usage:
    python puma_vl/statistics_puma_vl.py \
        --original data_vl/answers/model_dataset_answers.json \
        --compressed data_vl/prefixed_answers/model_dataset_prefixed_answers.json \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --dataset mathvista
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List
from multiprocessing import Pool

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OFFLINE_DIR = os.path.join(_PROJECT_ROOT, "puma")
_VL_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_PROJECT_ROOT, _OFFLINE_DIR, _VL_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_utils_vl import check_is_correct_vl
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_LOCAL_TOKENIZER = None


def load_tokenizer(model_name: str):
    global _LOCAL_TOKENIZER
    if _LOCAL_TOKENIZER is not None:
        return _LOCAL_TOKENIZER
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _LOCAL_TOKENIZER = getattr(proc, "tokenizer", proc)
    except Exception:
        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _LOCAL_TOKENIZER


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]


def _grade_one(args):
    """Worker function for parallel grading."""
    qidx, answer, gt, tag, dataset, question_type = args
    try:
        return (qidx, tag, check_is_correct_vl(answer, gt, dataset, question_type))
    except Exception:
        return (qidx, tag, False)


def compute_statistics_vl(
    original_json_path: str,
    compressed_json_path: str,
    model_name: str,
    dataset: str = "",
    confidence_threshold: float = None,
    epsilon: float = None,
    consecutive: int = None,
    enable_embedding_filter: bool = False,
    trial_answers_json_path: str = None,
    args_output: str = None,
    num_workers: int = 1,
):
    """Compute and print compression statistics for VL experiments."""
    logger.info("=" * 70)
    logger.info("VL EXPERIMENT CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Model: {model_name}")
    if confidence_threshold is not None:
        logger.info(f"CT: {confidence_threshold}, EPS: {epsilon}, CON: {consecutive}")
    logger.info(f"Embedding Filter: {'ENABLED' if enable_embedding_filter else 'disabled'}")
    logger.info("")

    tokenizer = load_tokenizer(model_name)

    with open(original_json_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    with open(compressed_json_path, "r", encoding="utf-8") as f:
        compressed_data = json.load(f)

    # Index original data
    original_by_idx = {}
    for idx, obj in enumerate(original_data):
        reasoning_text = obj.get("reasoning", "")
        raw_response_text = obj.get("raw_response", "")
        reasoning_tokens = count_tokens(tokenizer, reasoning_text)
        raw_response_tokens = count_tokens(tokenizer, raw_response_text)
        enriched = dict(obj)
        enriched["reasoning_token_count"] = reasoning_tokens
        enriched["raw_response_token_count"] = raw_response_tokens
        enriched["total_token_count"] = reasoning_tokens + raw_response_tokens
        original_by_idx[idx + 1] = enriched

    # Compute statistics
    total_original_tokens = 0
    total_compressed_tokens = 0
    total_trial_answer_tokens = 0
    evaluated = 0
    original_correct = 0
    compressed_correct = 0

    grade_tasks = []
    eval_data = []

    for obj in compressed_data:
        qidx = obj.get("question_idx")
        if qidx not in original_by_idx:
            continue

        orig = original_by_idx[qidx]
        original_total = orig.get("total_token_count")
        compressed_total = obj.get("total_tokens")
        trial_tokens = obj.get("tokens_trial_answers", 0)

        if original_total is None or compressed_total is None:
            continue

        total_original_tokens += original_total
        total_compressed_tokens += compressed_total
        total_trial_answer_tokens += trial_tokens
        evaluated += 1

        orig_answer = str(orig.get("model_answer", "")).strip()
        gt = str(orig.get("ground_truth_answer", "")).strip()
        comp_answer = str(obj.get("final_answer", "")).strip()
        question_type = orig.get("question_type", "")

        grade_tasks.append((qidx, orig_answer, gt, "orig", dataset, question_type))
        grade_tasks.append((qidx, comp_answer, gt, "comp", dataset, question_type))

        eval_data.append({
            "qidx": qidx,
            "question_type": question_type,
            "original_tokens": original_total,
            "compressed_tokens": compressed_total,
            "stopped_len": obj.get("stopped_len"),
            "original_len": obj.get("original_len_reasoning_steps"),
            "stop_reason": obj.get("stop_reason"),
        })

    # Grade
    logger.info(f"Grading {len(grade_tasks)} answer pairs with {num_workers} workers...")
    if num_workers > 1 and len(grade_tasks) > 4:
        with Pool(num_workers) as pool:
            grade_results = pool.map(_grade_one, grade_tasks, chunksize=4)
    else:
        grade_results = [_grade_one(t) for t in grade_tasks]

    grade_lookup = {(qidx, tag): correct for qidx, tag, correct in grade_results}

    for item in eval_data:
        qidx = item["qidx"]
        if grade_lookup.get((qidx, "orig"), False):
            original_correct += 1
        if grade_lookup.get((qidx, "comp"), False):
            compressed_correct += 1

    if evaluated == 0:
        logger.error("No overlapping questions found.")
        return

    avg_original = total_original_tokens / evaluated
    avg_compressed = total_compressed_tokens / evaluated
    avg_trial = total_trial_answer_tokens / evaluated
    avg_comp_rate = avg_compressed / avg_original
    avg_comp_incl_trial = (total_compressed_tokens + total_trial_answer_tokens) / total_original_tokens

    orig_acc = original_correct / evaluated * 100
    comp_acc = compressed_correct / evaluated * 100

    logger.info("=" * 70)
    logger.info("VL COMPRESSION STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Evaluated samples: {evaluated}")
    logger.info("")
    logger.info("Token Statistics:")
    logger.info(f"  Average tokens (original):     {avg_original:.2f}")
    logger.info(f"  Average tokens (compressed):   {avg_compressed:.2f}")
    logger.info(f"  Average trial answer tokens:   {avg_trial:.2f}")
    logger.info("")
    logger.info("Compression (excluding trial answers):")
    logger.info(f"  Compression rate: {avg_comp_rate * 100:.2f}%")
    logger.info(f"  Reduction:        {(1 - avg_comp_rate) * 100:.2f}%")
    logger.info("")
    logger.info("Compression (including trial answers):")
    logger.info(f"  Compression rate: {avg_comp_incl_trial * 100:.2f}%")
    logger.info(f"  Reduction:        {(1 - avg_comp_incl_trial) * 100:.2f}%")
    logger.info("")
    logger.info("Accuracy:")
    logger.info(f"  Original:   {orig_acc:.2f}% ({original_correct}/{evaluated})")
    logger.info(f"  Compressed: {comp_acc:.2f}% ({compressed_correct}/{evaluated})")
    logger.info(f"  Impact:     {comp_acc - orig_acc:+.2f}%")
    logger.info("")

    # Accuracy transition
    rr = rw = wr = ww = 0
    for item in eval_data:
        qidx = item["qidx"]
        oc = grade_lookup.get((qidx, "orig"), False)
        cc = grade_lookup.get((qidx, "comp"), False)
        if oc and cc: rr += 1
        elif oc and not cc: rw += 1
        elif not oc and cc: wr += 1
        else: ww += 1

    logger.info("Accuracy Transition:")
    logger.info(f"  R→R: {rr}/{evaluated} ({rr/evaluated*100:.1f}%)")
    logger.info(f"  R→W: {rw}/{evaluated} ({rw/evaluated*100:.1f}%)")
    logger.info(f"  W→R: {wr}/{evaluated} ({wr/evaluated*100:.1f}%)")
    logger.info(f"  W→W: {ww}/{evaluated} ({ww/evaluated*100:.1f}%)")
    logger.info("")

    # Stop reason distribution
    stop_reasons = {}
    for item in eval_data:
        r = item.get("stop_reason") or "unknown"
        stop_reasons[r] = stop_reasons.get(r, 0) + 1
    if stop_reasons:
        logger.info("Stop Reason Distribution:")
        for reason, count in sorted(stop_reasons.items(), key=lambda x: -x[1]):
            correct = sum(1 for it in eval_data
                          if it.get("stop_reason") == reason
                          and grade_lookup.get((it["qidx"], "comp"), False))
            acc = correct / count * 100 if count > 0 else 0
            logger.info(f"  {reason}: {count}/{evaluated} ({count/evaluated*100:.1f}%) — accuracy: {acc:.1f}%")

    # Step statistics
    stopped_lens = [it["stopped_len"] for it in eval_data if it.get("stopped_len")]
    orig_lens = [it["original_len"] for it in eval_data if it.get("original_len")]
    if stopped_lens and orig_lens:
        avg_sl = sum(stopped_lens) / len(stopped_lens)
        avg_ol = sum(orig_lens) / len(orig_lens)
        logger.info("")
        logger.info("Step Statistics:")
        logger.info(f"  Average stopped at step: {avg_sl:.1f}")
        logger.info(f"  Average original steps:  {avg_ol:.1f}")
        logger.info(f"  Average saved steps:     {avg_ol - avg_sl:.1f} ({(1 - avg_sl/avg_ol)*100:.1f}%)")

    logger.info("=" * 70)

    # Save comparison JSON
    if args_output:
        comparison = []
        for item in eval_data:
            qidx = item["qidx"]
            comparison.append({
                "question_idx": qidx,
                "original_correct": grade_lookup.get((qidx, "orig"), False),
                "compressed_correct": grade_lookup.get((qidx, "comp"), False),
                "original_tokens": item["original_tokens"],
                "compressed_tokens": item["compressed_tokens"],
                "stop_reason": item.get("stop_reason"),
            })
        comp_path = args_output.replace("_statistics_", "_comparison_").replace(".txt", ".json")
        os.makedirs(os.path.dirname(comp_path) if os.path.dirname(comp_path) else ".", exist_ok=True)
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparison saved to: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="VL Step 5: Compute compression statistics")
    parser.add_argument("--original", required=True)
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--consecutive", type=int, default=None)
    parser.add_argument("--enable-embedding-filter", action="store_true")
    parser.add_argument("--trial-answers", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.workers == 0:
        try:
            args.workers = min(len(os.sched_getaffinity(0)), 16)
        except AttributeError:
            args.workers = min(os.cpu_count() or 1, 16)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.output))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    compute_statistics_vl(
        original_json_path=args.original,
        compressed_json_path=args.compressed,
        model_name=args.model,
        dataset=args.dataset,
        confidence_threshold=args.confidence_threshold,
        epsilon=args.epsilon,
        consecutive=args.consecutive,
        enable_embedding_filter=args.enable_embedding_filter,
        trial_answers_json_path=args.trial_answers,
        args_output=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
