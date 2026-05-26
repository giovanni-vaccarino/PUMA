import json
import os
import re
import logging
import sys
from typing import Dict, List
from multiprocessing import Pool
from transformers import AutoTokenizer
# Add this script's directory to path for the math_grader import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from math_grader import check_is_correct
from prompt_utils import get_task_type

logger = logging.getLogger(__name__)


def _grade_one(args):
    """Worker function for parallel grading. Must be top-level for pickling."""
    qidx, answer, gt, tag = args
    try:
        from math_grader import check_is_correct
        return (qidx, tag, check_is_correct(answer, gt))
    except Exception:
        return (qidx, tag, False)

# Cache tokenizer globally
_LOCAL_TOKENIZER = None


def load_tokenizer(model_name: str):
    """Lazy-load tokenizer (cached)."""
    global _LOCAL_TOKENIZER

    if _LOCAL_TOKENIZER is not None:
        return _LOCAL_TOKENIZER

    logger.info(f"Loading tokenizer '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    _LOCAL_TOKENIZER = tokenizer
    return tokenizer


def count_tokens(tokenizer, text: str) -> int:
    """
    Count tokens for a given text using the model tokenizer.
    """
    if not text:
        return 0

    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids

    return tokens.shape[1]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_original_data(
    original_data: List[Dict],
    tokenizer
) -> Dict[int, Dict]:
    """
    Process original data to count tokens and index by question number.
    Returns dict mapping question_idx -> enriched object.
    """
    logger.info("Processing original data and counting tokens...")
    
    indexed_data = {}
    
    for idx, obj in enumerate(original_data):
        reasoning_text = obj.get("reasoning", "")
        raw_response_text = obj.get("raw_response", "")

        reasoning_tokens = count_tokens(tokenizer, reasoning_text)
        raw_response_tokens = count_tokens(tokenizer, raw_response_text)

        enriched_obj = dict(obj)
        enriched_obj["reasoning_token_count"] = reasoning_tokens
        enriched_obj["raw_response_token_count"] = raw_response_tokens
        enriched_obj["total_token_count"] = reasoning_tokens + raw_response_tokens
        
        # Index by 1-based question number
        indexed_data[idx + 1] = enriched_obj
    
    logger.info(f"Processed {len(indexed_data)} original questions")
    return indexed_data


def compute_statistics(
    original_json_path: str,
    compressed_json_path: str,
    model_name: str,
    dataset: str = None,
    confidence_threshold: float = None,
    epsilon: float = None,
    consecutive: int = None,
    confidence_mode: str = None,
    confidence_aggregation: str = None,
    min_stop_step: int = None,
    trial_decoding: str = None,
    # Embedding filter parameters
    enable_embedding_filter: bool = False,
    embedding_model: str = None,
    similarity_threshold: float = None,
    always_check_first_n: int = None,
    embedding_window_size: int = None,
    trigger_mode: str = None,
    consecutive_redundancy_stop: int = None,
    consecutive_redundancy_min_step: int = None,
    trial_answers_json_path: str = None,
    filtered_steps_json_path: str = None,
    questions_with_steps_path: str = None,
    args_output: str = None,
    num_workers: int = 1,
):
    """
    Compute and print compression statistics comparing original and compressed answers.

    Args:
        original_json_path: Path to original answers JSON
        compressed_json_path: Path to compressed/stopped answers JSON
        model_name: Model name for tokenizer
        dataset: Dataset name
        confidence_threshold: Confidence threshold for two-stage detection
        epsilon: Maximum confidence drop allowed
        consecutive: Number of consecutive points required
        enable_embedding_filter: Whether embedding filter was enabled
        embedding_model: Path to embedding model
        similarity_threshold: Similarity threshold (τ_sim)
        always_check_first_n: Number of first steps to always check
        embedding_window_size: Window size (K) for similarity comparison
        trigger_mode: Trigger mode ("current" or "previous")
        trial_answers_json_path: Path to trial answers JSON (for counting skipped)
    """
    # Print experiment configuration
    logger.info("=" * 70)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 70)
    if dataset:
        logger.info(f"Dataset: {dataset}")
    logger.info(f"Model: {model_name}")
    if confidence_threshold is not None:
        logger.info(f"Confidence Threshold: {confidence_threshold}")
    if epsilon is not None:
        logger.info(f"Epsilon: {epsilon}")
    if consecutive is not None:
        logger.info(f"Consecutive: {consecutive}")
    if confidence_mode is not None:
        logger.info(f"Confidence Mode: {confidence_mode}")
    if confidence_aggregation is not None:
        logger.info(f"Confidence Aggregation: {confidence_aggregation}")
    if min_stop_step is not None and min_stop_step > 0:
        logger.info(f"Min Stop Step: {min_stop_step}")
    if trial_decoding is not None:
        logger.info(f"Trial Decoding: {trial_decoding}")
    if enable_embedding_filter:
        logger.info(f"Embedding Filter: ENABLED")
        if embedding_model is not None:
            logger.info(f"  Model: {embedding_model}")
        if similarity_threshold is not None:
            logger.info(f"  Similarity Threshold (τ_sim): {similarity_threshold}")
        if always_check_first_n is not None:
            logger.info(f"  Always Check First N: {always_check_first_n}")
        if embedding_window_size is not None:
            logger.info(f"  Window Size (K): {embedding_window_size}")
        if trigger_mode is not None:
            logger.info(f"  Trigger Mode: {trigger_mode}")
        if consecutive_redundancy_stop is not None:
            logger.info(f"  Consecutive Redundancy Stop (m): {consecutive_redundancy_stop}")
        if consecutive_redundancy_min_step is not None:
            logger.info(f"  Consecutive Redundancy Min Step: {consecutive_redundancy_min_step}")
    else:
        logger.info(f"Embedding Filter: disabled")
    logger.info(f"Original File: {original_json_path}")
    logger.info(f"Compressed File: {compressed_json_path}")
    if trial_answers_json_path:
        logger.info(f"Trial Answers File: {trial_answers_json_path}")
    logger.info("")

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)

    # Load data
    logger.info(f"Loading original data from {original_json_path}")
    original_data: List[Dict] = load_json(original_json_path)

    logger.info(f"Loading compressed data from {compressed_json_path}")
    compressed_data: List[Dict] = load_json(compressed_json_path)

    # Load trial answers if provided (for embedding filter statistics)
    trial_answers_data = None
    if trial_answers_json_path:
        logger.info(f"Loading trial answers from {trial_answers_json_path}")
        trial_answers_data: List[Dict] = load_json(trial_answers_json_path)

    # Load questions_with_steps if provided (for online simulation overhead)
    steps_by_idx = None
    if questions_with_steps_path and enable_embedding_filter and trigger_mode == "previous":
        logger.info(f"Loading questions_with_steps from {questions_with_steps_path}")
        questions_with_steps_data = load_json(questions_with_steps_path)
        steps_by_idx = {}
        for idx, item in enumerate(questions_with_steps_data, start=1):
            steps_by_idx[idx] = item.get("reasoning_steps", [])

    # Process original data with token counts
    original_by_idx = process_original_data(original_data, tokenizer)
    
    # Compute statistics
    total_original_tokens = 0
    total_compressed_tokens = 0
    total_compressed_tokens_including_trial_answers = 0
    total_trial_answer_tokens = 0
    total_online_overhead_tokens = 0  # Extra step tokens for online simulation (previous mode)
    total_trial_answer_tokens_online = 0  # Trial answer tokens excluding steps before MSS
    compression_rates = []

    evaluated = 0
    original_correct_answers = 0
    compressed_correct_answers = 0
    original_boxed_count = 0
    compressed_boxed_count = 0
    comparison_results = []

    # First pass: collect token stats and grading tasks
    grade_tasks = []  # (qidx, answer, ground_truth, tag)
    eval_data = []  # per-question data for second pass

    for obj in compressed_data:
        qidx = obj.get("question_idx")

        if qidx not in original_by_idx:
            continue

        original_obj = original_by_idx[qidx]

        original_total = original_obj.get("total_token_count")
        compressed_total = obj.get("total_tokens")
        trial_answer_tokens = obj.get("tokens_trial_answers", 0)
        trial_answer_tokens_online = obj.get("tokens_trial_answers_online", trial_answer_tokens)

        if original_total is None or compressed_total is None:
            continue

        if original_total > 0:
            compression_rates.append(compressed_total / original_total)

        total_original_tokens += original_total
        total_compressed_tokens += compressed_total
        total_trial_answer_tokens += trial_answer_tokens
        total_trial_answer_tokens_online += trial_answer_tokens_online
        total_compressed_tokens_including_trial_answers += compressed_total + trial_answer_tokens

        # Online simulation overhead (per-question)
        per_question_overhead = 0
        if steps_by_idx is not None:
            stopped_len = obj.get("stopped_len", 0)
            original_len = obj.get("original_len_reasoning_steps", 0)
            steps = steps_by_idx.get(qidx, [])
            if stopped_len < original_len and stopped_len < len(steps):
                extra_step_text = steps[stopped_len]
                per_question_overhead = count_tokens(tokenizer, extra_step_text)
                total_online_overhead_tokens += per_question_overhead

        evaluated += 1

        # Prepare grading tasks
        original_answer = str(original_obj.get("model_answer", "")).strip()
        ground_truth = str(original_obj.get("ground_truth_answer", "")).strip()

        m = re.fullmatch(r"\\boxed\{(.+)\}", ground_truth)
        if m:
            ground_truth = m.group(1).strip()

        compressed_answer = str(obj.get("final_answer", "")).strip()
        # GPQA fallback: extract from "ANSWER: X" if final_answer is missing/invalid
        if dataset and "gpqa" in dataset.lower() and compressed_answer not in ("A", "B", "C", "D"):
            m_ans = re.search(r"ANSWER\s*:\s*([A-D])", obj.get("model_response", ""))
            if m_ans:
                compressed_answer = m_ans.group(1)

        grade_tasks.append((qidx, original_answer, ground_truth, "orig"))
        grade_tasks.append((qidx, compressed_answer, ground_truth, "comp"))

        # Check if raw model output contains \boxed{}
        original_raw = original_obj.get("raw_response", "")
        if "\\boxed{" in original_raw:
            original_boxed_count += 1
        compressed_raw = obj.get("model_response", "")
        if "\\boxed{" in compressed_raw:
            compressed_boxed_count += 1

        eval_data.append({
            "qidx": qidx,
            "question": original_obj.get("question", ""),
            "ground_truth": ground_truth,
            "original_answer": original_answer,
            "compressed_answer": compressed_answer,
            "original_tokens": original_total,
            "compressed_tokens": compressed_total,
            "stopped_len": obj.get("stopped_len"),
            "original_len_reasoning_steps": obj.get("original_len_reasoning_steps"),
            "stop_reason": obj.get("stop_reason"),
            "stop_confidence": obj.get("stop_confidence"),
            "tokens_trial_answers": trial_answer_tokens,
            "generated_trial_answers": obj.get("generated_trial_answers", 0),
            "online_overhead_tokens": per_question_overhead,
            "confidence_trajectory": obj.get("confidence_trajectory"),
            "step_similarities": obj.get("step_similarities"),
        })

    # Determine task type for grading
    task_type = get_task_type(dataset) if dataset else "math"
    is_code_task = (task_type == "code")

    # Batch grading (parallel if workers > 1) — skip for code tasks
    if is_code_task:
        logger.info(f"Code task: skipping math grading (accuracy evaluated separately via eval_livecodebench.py)")
        grade_lookup = {}
    else:
        logger.info(f"Grading {len(grade_tasks)} answer pairs with {num_workers} workers...")
        if num_workers > 1 and len(grade_tasks) > 4:
            with Pool(num_workers) as pool:
                grade_results = pool.map(_grade_one, grade_tasks, chunksize=4)
        else:
            grade_results = [_grade_one(t) for t in grade_tasks]

        # Index grading results by (qidx, tag)
        grade_lookup = {}
        for qidx, tag, correct in grade_results:
            grade_lookup[(qidx, tag)] = correct

    # Second pass: build comparison results
    for item in eval_data:
        qidx = item["qidx"]
        orig_correct = grade_lookup.get((qidx, "orig"), False)
        comp_correct = grade_lookup.get((qidx, "comp"), False)

        if orig_correct:
            original_correct_answers += 1
        if comp_correct:
            compressed_correct_answers += 1

        transition = f"{'R' if orig_correct else 'W'}\u2192{'R' if comp_correct else 'W'}"
        result_item = {
            "question_idx": qidx,
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "original_answer": item["original_answer"],
            "compressed_answer": item["compressed_answer"],
            "original_correct": orig_correct,
            "compressed_correct": comp_correct,
            "transition": transition,
            "original_tokens": item["original_tokens"],
            "compressed_tokens": item["compressed_tokens"],
            "stopped_len": item.get("stopped_len"),
            "original_len_reasoning_steps": item.get("original_len_reasoning_steps"),
            "stop_reason": item.get("stop_reason"),
            "stop_confidence": item.get("stop_confidence"),
            "tokens_trial_answers": item.get("tokens_trial_answers", 0),
            "generated_trial_answers": item.get("generated_trial_answers", 0),
            "online_overhead_tokens": item.get("online_overhead_tokens", 0),
        }
        # Include confidence trajectory and step similarities if available
        if item.get("confidence_trajectory") is not None:
            result_item["confidence_trajectory"] = item["confidence_trajectory"]
        if item.get("step_similarities") is not None:
            result_item["step_similarities"] = item["step_similarities"]
        comparison_results.append(result_item)

    if evaluated == 0:
        logger.error("No overlapping questions found.")
        return

    # Calculate averages and rates
    avg_original = total_original_tokens / evaluated
    avg_compressed = total_compressed_tokens / evaluated
    avg_trial_answer_tokens = total_trial_answer_tokens / evaluated
    avg_compressed_including_trial_answers = total_compressed_tokens_including_trial_answers / evaluated
    
    avg_compression = avg_compressed / avg_original
    avg_compression_including_trial_answers = avg_compressed_including_trial_answers / avg_original

    original_accuracy = (original_correct_answers / evaluated) * 100
    compressed_accuracy = (compressed_correct_answers / evaluated) * 100

    # Print statistics
    logger.info("=" * 70)
    logger.info("COMPRESSION STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Evaluated samples: {evaluated}")
    logger.info("")
    
    logger.info("Token Statistics:")
    logger.info(f"  Average tokens (original):     {avg_original:.2f}")
    logger.info(f"  Average tokens (compressed):   {avg_compressed:.2f}")
    logger.info(f"  Average trial answer tokens:   {avg_trial_answer_tokens:.2f}")
    logger.info(f"  Average tokens (compressed + trial answers): {avg_compressed_including_trial_answers:.2f}")
    logger.info("")
    
    logger.info("Compression (excluding trial answers):")
    logger.info(f"  Compression rate: {avg_compression * 100:.2f}%")
    logger.info(f"  Reduction:        {(1 - avg_compression) * 100:.2f}%")
    logger.info("")

    logger.info("Compression (including trial answers):")
    logger.info(f"  Compression rate: {avg_compression_including_trial_answers * 100:.2f}%")
    logger.info(f"  Reduction:        {(1 - avg_compression_including_trial_answers) * 100:.2f}%")
    logger.info("")

    # Online simulation: compressed + online trial answers + overhead from "previous" trigger mode
    # Uses tokens_trial_answers_online which excludes trial answers from steps before MSS
    if steps_by_idx is not None and evaluated > 0:
        avg_online_overhead = total_online_overhead_tokens / evaluated
        total_online_tokens = total_compressed_tokens + total_trial_answer_tokens_online + total_online_overhead_tokens
        avg_online = total_online_tokens / evaluated
        avg_online_compression = avg_online / avg_original
        logger.info("Online Simulation (including trial answers + previous-mode overhead):")
        logger.info(f"  Average overhead tokens (trigger step): {avg_online_overhead:.2f}")
        logger.info(f"  Average tokens (online total):          {avg_online:.2f}")
        logger.info(f"  Compression rate: {avg_online_compression * 100:.2f}%")
        logger.info(f"  Reduction:        {(1 - avg_online_compression) * 100:.2f}%")
        if total_trial_answer_tokens != total_trial_answer_tokens_online:
            saved = total_trial_answer_tokens - total_trial_answer_tokens_online
            logger.info(f"  (MSS saved {saved} trial answer tokens, {saved / evaluated:.1f} avg/question)")
        logger.info("")

    if is_code_task:
        logger.info("Accuracy: (code task — evaluate separately with eval_livecodebench.py)")
        logger.info("")
    else:
        logger.info("Accuracy:")
        logger.info(f"  Original:   {original_accuracy:.2f}% ({original_correct_answers}/{evaluated})")
        logger.info(f"  Compressed: {compressed_accuracy:.2f}% ({compressed_correct_answers}/{evaluated})")
        logger.info(f"  Impact:     {compressed_accuracy - original_accuracy:+.2f}%")
        logger.info("")

    # Accuracy Transition Analysis (4 categories) — skip for code tasks
    if not is_code_task:
        transitions = {u"R\u2192R": [], u"R\u2192W": [], u"W\u2192R": [], u"W\u2192W": []}
        for item in eval_data:
            qidx = item["qidx"]
            oc = grade_lookup.get((qidx, "orig"), False)
            cc = grade_lookup.get((qidx, "comp"), False)
            key = f"{'R' if oc else 'W'}\u2192{'R' if cc else 'W'}"
            transitions[key].append(item)

        logger.info("Accuracy Transition:")
        for key in [u"R\u2192R", u"R\u2192W", u"W\u2192R", u"W\u2192W"]:
            items = transitions[key]
            n = len(items)
            if n == 0:
                logger.info(f"  {key}: 0/{evaluated} (0.0%)")
                continue

            pct = n / evaluated * 100

            # Compression rate (total compressed / total original)
            total_orig_tok = sum(i["original_tokens"] for i in items)
            total_comp_tok = sum(i["compressed_tokens"] for i in items)
            comp_rate = total_comp_tok / total_orig_tok * 100 if total_orig_tok > 0 else 0

            # Steps
            sl = [i["stopped_len"] for i in items if i.get("stopped_len") is not None]
            ol = [i["original_len_reasoning_steps"] for i in items if i.get("original_len_reasoning_steps") is not None]
            avg_sl = sum(sl) / len(sl) if sl else 0
            avg_ol = sum(ol) / len(ol) if ol else 0
            avg_saved_pct = (1 - avg_sl / avg_ol) * 100 if avg_ol > 0 else 0

            # Stop reason distribution
            reasons = {}
            for i in items:
                r = i.get("stop_reason") or "unknown"
                reasons[r] = reasons.get(r, 0) + 1
            reason_str = ", ".join(f"{r}={c}" for r, c in sorted(reasons.items(), key=lambda x: -x[1]))

            # Average stop confidence
            confs = [i["stop_confidence"] for i in items if i.get("stop_confidence") is not None]
            avg_conf = sum(confs) / len(confs) if confs else None

            # Average similarity at stop step
            sims_at_stop = []
            for i in items:
                ss = i.get("step_similarities")
                sl_i = i.get("stopped_len")
                if ss and sl_i:
                    if trigger_mode == "previous":
                        if sl_i < len(ss):
                            sim_val = ss[sl_i]
                            if sim_val is not None:
                                sims_at_stop.append(sim_val)
                    else:
                        if 0 < sl_i <= len(ss):
                            sim_val = ss[sl_i - 1]
                            if sim_val is not None:
                                sims_at_stop.append(sim_val)
            avg_sim = sum(sims_at_stop) / len(sims_at_stop) if sims_at_stop else None

            logger.info(f"  {key}: {n}/{evaluated} ({pct:.1f}%)")
            logger.info(f"    compression: {comp_rate:.1f}%, steps: {avg_sl:.1f}/{avg_ol:.1f} (saved {avg_saved_pct:.1f}%)")
            logger.info(f"    stop_reason: {reason_str}")
            conf_str = f"{avg_conf:.3f}" if avg_conf is not None else "N/A"
            sim_str = f"{avg_sim:.3f}" if avg_sim is not None else "N/A"
            logger.info(f"    avg stop_confidence: {conf_str}, avg similarity_at_stop: {sim_str}")

            # For R→W: list affected questions
            if key == u"R\u2192W":
                qidxs = sorted(i["qidx"] for i in items)
                logger.info(f"    questions: {qidxs}")
        logger.info("")

        logger.info("Boxed Answer Format (\\boxed{{}} in raw output):")
        logger.info(f"  Original:   {original_boxed_count}/{evaluated} ({original_boxed_count / evaluated * 100:.1f}%)")
        logger.info(f"  Compressed: {compressed_boxed_count}/{evaluated} ({compressed_boxed_count / evaluated * 100:.1f}%)")

    # Stop reason distribution and per-reason accuracy
    stop_reason_counts = {}
    stop_reason_correct = {}
    for item in eval_data:
        reason = item.get("stop_reason") or "unknown"
        qidx = item["qidx"]
        comp_correct = grade_lookup.get((qidx, "comp"), False)
        stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1
        if comp_correct:
            stop_reason_correct[reason] = stop_reason_correct.get(reason, 0) + 1

    if any(r != "unknown" for r in stop_reason_counts):
        logger.info("")
        logger.info("Stop Reason Distribution:")
        for reason, count in sorted(stop_reason_counts.items(), key=lambda x: -x[1]):
            if is_code_task:
                # Code tasks: accuracy evaluated separately in eval_livecodebench.py
                logger.info(f"  {reason}: {count}/{evaluated} ({count / evaluated * 100:.1f}%)")
            else:
                correct = stop_reason_correct.get(reason, 0)
                acc = (correct / count * 100) if count > 0 else 0
                logger.info(f"  {reason}: {count}/{evaluated} ({count / evaluated * 100:.1f}%) — accuracy: {acc:.1f}% ({correct}/{count})")

    # Stopped steps distribution
    stopped_lens = [item.get("stopped_len") for item in eval_data if item.get("stopped_len") is not None]
    orig_lens = [item.get("original_len_reasoning_steps") for item in eval_data if item.get("original_len_reasoning_steps") is not None]
    if stopped_lens and orig_lens:
        avg_stopped = sum(stopped_lens) / len(stopped_lens)
        avg_orig = sum(orig_lens) / len(orig_lens)
        saved_steps = [o - s for s, o in zip(stopped_lens, orig_lens)]
        avg_saved = sum(saved_steps) / len(saved_steps)
        logger.info("")
        logger.info("Step Statistics:")
        logger.info(f"  Average stopped at step: {avg_stopped:.1f}")
        logger.info(f"  Average original steps:  {avg_orig:.1f}")
        logger.info(f"  Average saved steps:     {avg_saved:.1f} ({avg_saved / avg_orig * 100:.1f}%)")

    # Embedding filter statistics
    if trial_answers_data and enable_embedding_filter:
        logger.info("")
        logger.info("Embedding Filter Statistics:")
        total_entries = len(trial_answers_data)
        skipped_entries = sum(1 for e in trial_answers_data if e.get("skipped", False))
        generated_entries = total_entries - skipped_entries
        skip_rate = (skipped_entries / total_entries * 100) if total_entries > 0 else 0
        logger.info(f"  Total trial answer entries: {total_entries}")
        logger.info(f"  Generated: {generated_entries} ({100 - skip_rate:.1f}%)")
        logger.info(f"  Skipped:   {skipped_entries} ({skip_rate:.1f}%)")

        # Embedding filter skip statistics
        emb_skipped = [e for e in trial_answers_data if e.get("skip_reason") == "embedding_filter"]
        if emb_skipped:
            logger.info(f"  Skipped by embedding filter: {len(emb_skipped)}")

        # Consecutive redundancy stop statistics
        if filtered_steps_json_path and consecutive_redundancy_stop and consecutive_redundancy_stop > 0:
            try:
                filtered_steps_data = load_json(filtered_steps_json_path)
                total_questions = len(filtered_steps_data)
                consecutive_stops = sum(
                    1 for q in filtered_steps_data
                    if q.get("consecutive_redundancy_detected", False)
                )
                logger.info(f"  Consecutive redundancy stops: {consecutive_stops}/{total_questions} questions")
            except Exception as e:
                logger.warning(f"  Could not load filtered steps for consecutive redundancy stats: {e}")

    logger.info("=" * 70)

    # Save per-question comparison results as JSON
    if args_output:
        comparison_path = args_output.replace("_statistics_", "_comparison_").replace(".txt", ".json")
        os.makedirs(os.path.dirname(comparison_path) if os.path.dirname(comparison_path) else ".", exist_ok=True)
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Comparison results saved to: {comparison_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute compression statistics between original and compressed answers"
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Path to original answers JSON file"
    )
    parser.add_argument(
        "--compressed",
        required=True,
        help="Path to compressed/stopped answers JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name for tokenizer (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save statistics output (optional, also prints to terminal)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., aime24, aime25, amc23)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold used in two-stage detection"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon value used in two-stage detection"
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=None,
        help="Number of consecutive points required"
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        default=None,
        help="Confidence calculation mode ('first_line' or 'token_in_boxed')"
    )
    parser.add_argument(
        "--confidence-aggregation",
        type=str,
        default=None,
        help="Confidence aggregation method ('geometric' or 'arithmetic')"
    )
    parser.add_argument(
        "--min-stop-step",
        type=int,
        default=0,
        help="Minimum step before allowing early stop (for logging)"
    )
    parser.add_argument(
        "--trial-decoding",
        type=str,
        default=None,
        help="Trial decoding mode ('greedy', 'sampling', or 'default')"
    )
    # Embedding filter parameters (for logging)
    parser.add_argument(
        "--enable-embedding-filter",
        action="store_true",
        help="Log that embedding filter was enabled"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Path to embedding model"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold (τ_sim)"
    )
    parser.add_argument(
        "--always-check-first-n",
        type=int,
        default=None,
        help="Number of first steps to always check"
    )
    parser.add_argument(
        "--embedding-window-size",
        type=int,
        default=None,
        help="Window size (K) for similarity comparison"
    )
    parser.add_argument(
        "--trigger-mode",
        type=str,
        default=None,
        help="Trigger mode ('current' or 'previous')"
    )
    parser.add_argument(
        "--consecutive-redundancy-stop",
        type=int,
        default=None,
        help="Consecutive redundancy stop threshold (m)"
    )
    parser.add_argument(
        "--consecutive-redundancy-min-step",
        type=int,
        default=None,
        help="Only start detecting consecutive redundancy after this step"
    )
    parser.add_argument(
        "--trial-answers",
        type=str,
        default=None,
        help="Path to trial answers JSON (for counting skipped steps)"
    )
    parser.add_argument(
        "--filtered-steps",
        type=str,
        default=None,
        help="Path to filtered steps JSON (for consecutive redundancy stats)"
    )
    parser.add_argument(
        "--questions-with-steps",
        type=str,
        default=None,
        help="Path to questions_with_steps JSON (for online simulation overhead in 'previous' trigger mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for grading (0=auto, 1=single-process)"
    )

    args = parser.parse_args()

    if args.workers == 0:
        try:
            args.workers = min(len(os.sched_getaffinity(0)), 16)
        except AttributeError:
            args.workers = min(os.cpu_count() or 1, 16)

    # Set up logging - use stdout so logs go to .out file in SLURM
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.output))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    compute_statistics(
        original_json_path=args.original,
        compressed_json_path=args.compressed,
        model_name=args.model,
        dataset=args.dataset,
        confidence_threshold=args.confidence_threshold,
        epsilon=args.epsilon,
        consecutive=args.consecutive,
        confidence_mode=args.confidence_mode,
        confidence_aggregation=args.confidence_aggregation,
        min_stop_step=args.min_stop_step,
        trial_decoding=args.trial_decoding,
        enable_embedding_filter=args.enable_embedding_filter,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        always_check_first_n=args.always_check_first_n,
        embedding_window_size=args.embedding_window_size,
        trigger_mode=args.trigger_mode,
        consecutive_redundancy_stop=args.consecutive_redundancy_stop,
        consecutive_redundancy_min_step=args.consecutive_redundancy_min_step,
        trial_answers_json_path=args.trial_answers,
        filtered_steps_json_path=args.filtered_steps,
        questions_with_steps_path=args.questions_with_steps,
        args_output=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()