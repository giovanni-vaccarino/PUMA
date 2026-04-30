import json
import os
import re
import logging
import sys
from typing import Dict, List
from multiprocessing import Pool
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_LOCAL_TOKENIZER = None


def _grade_one(args):
    qidx, answer, gt, tag = args
    try:
        from .math_grader import check_is_correct
        return (qidx, tag, check_is_correct(answer, gt))
    except Exception:
        return (qidx, tag, False)


def load_tokenizer(model_name: str):
    global _LOCAL_TOKENIZER
    if _LOCAL_TOKENIZER is not None:
        return _LOCAL_TOKENIZER
    logger.info(f"Loading tokenizer '{model_name}'")
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _LOCAL_TOKENIZER


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_original_data(original_data: List[Dict], tokenizer) -> Dict[int, Dict]:
    indexed = {}
    for idx, obj in enumerate(original_data):
        enriched = dict(obj)
        enriched["total_token_count"] = count_tokens(tokenizer, obj.get("raw_response", ""))
        indexed[idx + 1] = enriched
    return indexed


def compute_statistics(
    original_json_path: str,
    compressed_json_path: str,
    model_name: str,
    args_output: str = None,
    num_workers: int = 1,
):
    tokenizer = load_tokenizer(model_name)

    logger.info(f"Loading original data from {original_json_path}")
    original_data = load_json(original_json_path)

    logger.info(f"Loading compressed data from {compressed_json_path}")
    compressed_data = load_json(compressed_json_path)

    original_by_idx = process_original_data(original_data, tokenizer)

    total_original_tokens = 0
    total_compressed_tokens_including_trial_answers = 0
    evaluated = 0
    grade_tasks = []
    eval_data = []

    for obj in compressed_data:
        qidx = obj.get("question_idx")
        if qidx not in original_by_idx:
            continue
        original_obj = original_by_idx[qidx]

        original_total = original_obj.get("total_token_count")
        compressed_total = obj.get("total_tokens")
        if original_total is None or compressed_total is None:
            continue

        trial_answer_tokens = obj.get("tokens_trial_answers", 0)
        total_original_tokens += original_total
        total_compressed_tokens_including_trial_answers += compressed_total + trial_answer_tokens
        evaluated += 1

        original_answer = str(original_obj.get("model_answer", "")).strip()
        ground_truth = str(original_obj.get("ground_truth_answer", "")).strip()
        m = re.fullmatch(r"\\boxed\{(.+)\}", ground_truth)
        if m:
            ground_truth = m.group(1).strip()
        compressed_answer = str(obj.get("final_answer", "")).strip()

        grade_tasks.append((qidx, original_answer, ground_truth, "orig"))
        grade_tasks.append((qidx, compressed_answer, ground_truth, "comp"))

        eval_data.append({
            "qidx": qidx,
            "question": original_obj.get("question", ""),
            "ground_truth": ground_truth,
            "original_answer": original_answer,
            "compressed_answer": compressed_answer,
            "original_tokens": original_total,
            "compressed_tokens": compressed_total,
        })

    if evaluated == 0:
        logger.error("No overlapping questions found.")
        return

    logger.info(f"Grading {len(grade_tasks)} answer pairs with {num_workers} workers...")
    if num_workers > 1 and len(grade_tasks) > 4:
        with Pool(num_workers) as pool:
            grade_results = pool.map(_grade_one, grade_tasks, chunksize=4)
    else:
        grade_results = [_grade_one(t) for t in grade_tasks]
    grade_lookup = {(qidx, tag): correct for qidx, tag, correct in grade_results}

    original_correct = sum(grade_lookup.get((d["qidx"], "orig"), False) for d in eval_data)
    compressed_correct = sum(grade_lookup.get((d["qidx"], "comp"), False) for d in eval_data)

    compression_rate = (total_compressed_tokens_including_trial_answers / total_original_tokens)
    original_accuracy = original_correct / evaluated * 100
    compressed_accuracy = compressed_correct / evaluated * 100

    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Samples: {evaluated}")
    logger.info("")
    logger.info("Accuracy:")
    logger.info(f"  Original:   {original_accuracy:.2f}% ({original_correct}/{evaluated})")
    logger.info(f"  Compressed: {compressed_accuracy:.2f}% ({compressed_correct}/{evaluated})")
    logger.info(f"  Impact:     {compressed_accuracy - original_accuracy:+.2f}%")
    logger.info("")
    logger.info("Compression:")
    logger.info(f"  Rate:      {compression_rate * 100:.2f}%")
    logger.info(f"  Reduction: {(1 - compression_rate) * 100:.2f}%")
    logger.info("=" * 70)

    if args_output:
        comparison_results = []
        for item in eval_data:
            qidx = item["qidx"]
            comparison_results.append({
                "question_idx": qidx,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "original_answer": item["original_answer"],
                "compressed_answer": item["compressed_answer"],
                "original_correct": grade_lookup.get((qidx, "orig"), False),
                "compressed_correct": grade_lookup.get((qidx, "comp"), False),
                "original_tokens": item["original_tokens"],
                "compressed_tokens": item["compressed_tokens"],
            })
        os.makedirs(os.path.dirname(args_output) if os.path.dirname(args_output) else ".", exist_ok=True)
        with open(args_output, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Per-question results saved to: {args_output}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate accuracy and compression of PUMA-compressed answers"
    )
    parser.add_argument("--original", required=True, help="Path to original answers JSON")
    parser.add_argument("--compressed", required=True, help="Path to compressed answers JSON")
    parser.add_argument("--model", type=str, required=True, help="Model name or path (for tokenizer)")
    parser.add_argument("--output", type=str, default=None, help="Path to save per-question results JSON")
    parser.add_argument("--workers", type=int, default=0, help="Parallel grading workers (0=auto)")

    args = parser.parse_args()

    if args.workers == 0:
        try:
            args.workers = min(len(os.sched_getaffinity(0)), 16)
        except AttributeError:
            args.workers = min(os.cpu_count() or 1, 16)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    compute_statistics(
        original_json_path=args.original,
        compressed_json_path=args.compressed,
        model_name=args.model,
        args_output=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
