import json
import os
import argparse
import logging
import sys
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Dict, Tuple
from .prompt_utils import extract_boxed_answer

logger = logging.getLogger(__name__)


def load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: List[Dict], path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def extract_answer(exp):
    """
    Extract a math answer from a trial-answer entry.

    The trial-answer prompt asks the model to continue from ``\\boxed``. Most
    outputs therefore start with ``{answer}``, but this function also accepts a
    complete ``\\boxed{answer}`` response.
    """
    final_answer = exp.get("final_answer", "")
    if final_answer and not exp.get("model_response", ""):
        return final_answer

    model_response = exp.get("model_response", "")
    if not model_response:
        return final_answer or ""

    # Try \boxed{...} first (balanced braces)
    boxed_answer = extract_boxed_answer(model_response)
    if boxed_answer:
        return boxed_answer

    # Fallback: first {...} with balanced braces
    idx = model_response.find("{")
    if idx == -1:
        return final_answer or ""
    start = idx + 1
    depth = 1
    i = start
    while i < len(model_response) and depth > 0:
        if model_response[i] == '{':
            depth += 1
        elif model_response[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return model_response[start:i - 1].strip()
    return final_answer or ""


def consecutive_same_answer_and_conf_delta_method(
    entries: List[Dict],
    confidence_threshold: float,
    epsilon: float,
    consecutive: int = 2,
    forced_stop_step: int = None,
    forced_stop_min_confidence: float = 0.0,
    min_stop_step: int = 0,
) -> Tuple[Dict, int, int, Dict]:
    """
    Stop when first point has confidence >= threshold,
    and all consecutive have same answer AND confidence >= (first_conf - epsilon).
    For 3+ consecutive, check delta from first point, not previous.

    Supports skipped entries from embedding filter:
    - Entries with skipped=True are excluded from consecutive matching
    - When looking for consecutive entries, skip over skipped entries

    Args:
        entries: List of entries for a question
        confidence_threshold: Minimum confidence for first point
        epsilon: Maximum confidence drop allowed from first point
        consecutive: Number of consecutive points to check
        forced_stop_step: If set, acts as an UPPER BOUND (latest possible stop point).
            Normal logic runs first; if it finds a stop point before forced_stop_step,
            use that (earlier is better). If no stop point is found before forced_stop_step,
            use forced_stop_step as the cap.
        forced_stop_min_confidence: Minimum confidence gate for forced_stop_redundancy.
            If the best trial answer before forced_stop_len has confidence below this,
            skip forced_stop and fall through to full_reasoning. (default: 0.0 = no gate)
        min_stop_step: Minimum step count before allowing early stop.
            Steps before this are still evaluated but cannot trigger stopping.
            (default: 0 = no minimum)

    Returns:
        (chosen_entry, num_trial_answers, tokens_trial_answers, stop_info)
    """
    entries.sort(key=lambda x: x["stopped_len"])

    # Filter to get non-skipped entries for consecutive matching
    non_skipped_entries = [e for e in entries if not e.get("skipped", False)]

    chosen = None
    generated_trial_answers = 0
    tokens_trial_answers = 0
    stop_info = {
        "stop_reason": "full_reasoning",
        "stop_confidence": None,
        "stop_threshold": confidence_threshold,
        "consecutive_confidences": [],
    }

    # forced_stop_step acts as an UPPER BOUND (latest possible stop point)
    # We use min(normal_stop_point, forced_stop_step)
    # Convert to 1-indexed for comparison with stopped_len
    forced_stop_len = (forced_stop_step + 1) if forced_stop_step is not None else None

    for idx in range(len(non_skipped_entries)):
        first = non_skipped_entries[idx]
        step_count = first["stopped_len"]

        # If we've passed the forced_stop_len, stop searching
        # (forced_stop is handled separately after the loop)
        if forced_stop_len is not None and step_count > forced_stop_len:
            break

        current_threshold = confidence_threshold
        current_epsilon = epsilon

        if first["confidence"] >= current_threshold:
            first_ans = extract_answer(first)
            first_conf = first["confidence"]

            # Find next consecutive-1 non-skipped entries (within forced_stop_len bound)
            consecutive_entries = [first]
            search_idx = idx + 1
            while len(consecutive_entries) < consecutive and search_idx < len(non_skipped_entries):
                candidate = non_skipped_entries[search_idx]
                # Don't include entries beyond forced_stop_len
                if forced_stop_len is not None and candidate["stopped_len"] > forced_stop_len:
                    break
                consecutive_entries.append(candidate)
                search_idx += 1

            if len(consecutive_entries) < consecutive:
                # Not enough non-skipped entries left (within forced_stop_len bound)
                continue

            # Check if all consecutive have same answer and confidence within epsilon of FIRST
            all_match = True
            for j in range(1, consecutive):
                next_entry = consecutive_entries[j]
                next_ans = extract_answer(next_entry)
                next_conf = next_entry["confidence"]

                answers_match = first_ans and first_ans == next_ans

                if not answers_match or next_conf < (first_conf - current_epsilon):
                    all_match = False
                    break

            if all_match:
                # Check min_stop_step: don't stop before this step
                if min_stop_step > 0 and consecutive_entries[-1]["stopped_len"] < min_stop_step:
                    continue

                # Use the last consecutive entry as the offline stopping point:
                # this is where answer consistency has actually been verified.
                chosen = consecutive_entries[-1]
                # Count trial answers up to and including the last consecutive entry
                last_stopped_len = consecutive_entries[-1]["stopped_len"]
                generated_trial_answers = len([e for e in entries if e["stopped_len"] <= last_stopped_len and not e.get("skipped", False)])
                # Sum count_answer_tokens for all generated (non-skipped) trial answers
                tokens_trial_answers = sum(
                    e.get("count_answer_tokens", 0)
                    for e in entries
                    if e["stopped_len"] <= last_stopped_len and not e.get("skipped", False)
                )
                stop_info = {
                    "stop_reason": "confidence_consecutive",
                    "stop_confidence": chosen["confidence"],
                    "stop_threshold": current_threshold,
                    "consecutive_confidences": [e["confidence"] for e in consecutive_entries],
                }
                break

    # Fallback: no confidence_consecutive found
    if chosen is None and forced_stop_len is not None:
        # Skip forced_stop if it's before min_stop_step
        if min_stop_step > 0 and forced_stop_len < min_stop_step:
            pass  # Fall through to full_reasoning
        else:
            # Quality gate: check if the best trial answer before forced_stop_len
            # has sufficient confidence. If not, skip forced_stop → full_reasoning.
            nearby = [e for e in non_skipped_entries if e["stopped_len"] <= forced_stop_len]
            best_conf = max((e["confidence"] for e in nearby), default=None) if nearby else None

            if best_conf is not None and best_conf >= forced_stop_min_confidence:
                # forced_stop_redundancy: stop directly at forced_stop_len.
                # Construct a synthetic entry — Step 4 only needs question_idx + stopped_len.
                original_len = entries[0].get("original_len_reasoning_steps", 0) if entries else 0
                generated_trial_answers = len([
                    e for e in entries
                    if e["stopped_len"] <= forced_stop_len and not e.get("skipped", False)
                ])
                tokens_trial_answers = sum(
                    e.get("count_answer_tokens", 0)
                    for e in entries
                    if e["stopped_len"] <= forced_stop_len and not e.get("skipped", False)
                )
                chosen = {
                    "question_idx": entries[0]["question_idx"],
                    "stopped_len": forced_stop_len,
                    "original_len_reasoning_steps": original_len,
                    "success": True,
                }
                stop_info = {
                    "stop_reason": "forced_stop_redundancy",
                    "stop_confidence": best_conf,
                    "stop_threshold": forced_stop_min_confidence,
                    "consecutive_confidences": [],
                }

    # Fallback: no forced_stop, use full reasoning
    if chosen is None:
        # Find the full reasoning entry (stopped_len == original_len)
        full_entry = next(
            (e for e in entries if e["stopped_len"] == e.get("original_len_reasoning_steps")),
            None
        )
        if full_entry is not None:
            chosen = full_entry
            generated_trial_answers = len([e for e in entries if not e.get("skipped", False)])
            tokens_trial_answers = sum(
                e.get("count_answer_tokens", 0)
                for e in entries
                if not e.get("skipped", False)
            )
            stop_info = {
                "stop_reason": "full_reasoning",
                "stop_confidence": full_entry.get("confidence"),
                "stop_threshold": confidence_threshold,
                "consecutive_confidences": [],
            }

    return chosen, generated_trial_answers, tokens_trial_answers, stop_info


def _process_question(args):
    """Worker function for parallel candidate extraction. Must be top-level for pickling."""
    question_idx, entries, forced_stop_step, method_kwargs, step_similarities = args
    chosen, generated_trial_answers, tokens_trial_answers, stop_info = consecutive_same_answer_and_conf_delta_method(
        entries, forced_stop_step=forced_stop_step, **method_kwargs
    )
    if chosen is None:
        raise RuntimeError(f"No full-reasoning entry found for question {question_idx}")
    chosen_copy = chosen.copy()
    chosen_copy["generated_trial_answers"] = generated_trial_answers
    chosen_copy["tokens_trial_answers"] = tokens_trial_answers

    # Add stop metadata
    chosen_copy["stop_reason"] = stop_info["stop_reason"]
    chosen_copy["stop_confidence"] = stop_info["stop_confidence"]
    chosen_copy["stop_threshold"] = stop_info["stop_threshold"]
    chosen_copy["consecutive_confidences"] = stop_info["consecutive_confidences"]

    # Build confidence trajectory: per-step confidence from all entries
    confidence_trajectory = []
    for e in sorted(entries, key=lambda x: x["stopped_len"]):
        item = {"step": e["stopped_len"]}
        if e.get("skipped", False):
            item["skipped"] = True
        else:
            item["confidence"] = e.get("confidence")
        confidence_trajectory.append(item)
    chosen_copy["confidence_trajectory"] = confidence_trajectory

    # Attach step_similarities from filtered_steps (embedding filter)
    if step_similarities is not None:
        chosen_copy["step_similarities"] = step_similarities

    return chosen_copy


def extract_final_candidates(
    results: List[Dict],
    confidence_threshold: float = 0.96,
    epsilon: float = 0.05,
    consecutive: int = 2,
    forced_stop_min_confidence: float = 0.0,
    min_stop_step: int = 0,
    filtered_steps_data: List[Dict] = None,
    num_workers: int = 1,
) -> List[Dict]:
    """
    For each question_idx:
      - Use consecutive + epsilon method to find stopping point
      - If none exist, pick the full-reasoning entry
        (stopped_len == original_len_reasoning_steps)

    Args:
        results: List of all trial answer results
        confidence_threshold: Minimum confidence for first point
        epsilon: Maximum confidence drop allowed from first point
        consecutive: Number of consecutive points that must match criteria
        forced_stop_min_confidence: Minimum confidence for forced_stop quality gate
        min_stop_step: Minimum step count before allowing early stop (default: 0 = no minimum)
        filtered_steps_data: List of question dicts with forced_stop_step info (from embed_and_filter_steps.py)

    Returns:
        List of selected entries (one per question)
    """
    # Build lookups by question index (1-indexed)
    forced_stop_lookup = {}
    step_similarities_lookup = {}
    if filtered_steps_data:
        for idx, q in enumerate(filtered_steps_data, start=1):
            if q.get("forced_stop_step") is not None:
                forced_stop_lookup[idx] = q["forced_stop_step"]
            if q.get("step_similarities") is not None:
                step_similarities_lookup[idx] = q["step_similarities"]

    grouped = defaultdict(list)

    # Group entries by question_idx
    # Include both generated entries (with confidence) and skipped entries (from embedding filter)
    for obj in results:
        if not obj.get("success", False):
            continue

        # Include skipped entries (they have no confidence but are still valid)
        if obj.get("skipped", False):
            grouped[obj["question_idx"]].append(obj)
            continue

        # For non-skipped entries, require confidence
        if obj.get("confidence") is None:
            continue

        grouped[obj["question_idx"]].append(obj)

    method_kwargs = dict(
        confidence_threshold=confidence_threshold,
        epsilon=epsilon,
        consecutive=consecutive,
        forced_stop_min_confidence=forced_stop_min_confidence,
        min_stop_step=min_stop_step,
    )

    tasks = [
        (qidx, entries, forced_stop_lookup.get(qidx, None), method_kwargs, step_similarities_lookup.get(qidx, None))
        for qidx, entries in grouped.items()
    ]

    if num_workers > 1 and len(tasks) > 1:
        logger.info("Using %d workers for %d questions", num_workers, len(tasks))
        with Pool(num_workers) as pool:
            selected = pool.map(_process_question, tasks)
    else:
        selected = [_process_question(t) for t in tasks]

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract final candidates using consecutive + epsilon method"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="JSON file produced by the prefix-processing script"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Where to save the filtered JSON"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.96,
        help="Minimum confidence for first point (default: 0.96)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Maximum confidence drop allowed from first point (default: 0.05)"
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Number of consecutive points that must match criteria (default: 2)"
    )
    parser.add_argument(
        "--forced-stop-min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence gate for forced_stop_redundancy. "
             "If the best trial answer before forced_stop has confidence below this, "
             "skip forced_stop and use full reasoning instead. (default: 0.0 = no gate)"
    )
    parser.add_argument(
        "--min-stop-step",
        type=int,
        default=0,
        help="Minimum step count before allowing early stop. "
             "Steps before this are still evaluated but cannot trigger stopping. "
             "(default: 0 = no minimum)"
    )
    parser.add_argument(
        "--filtered-steps",
        type=str,
        default=None,
        help="Path to filtered steps JSON file (from embed_and_filter_steps.py) for forced_stop_step handling"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, 1=single-process)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.workers == 0:
        try:
            args.workers = min(len(os.sched_getaffinity(0)), 16)
        except AttributeError:
            args.workers = min(os.cpu_count() or 1, 16)

    logger.info("Loading data...")
    results = load_json(args.input_file)
    total_entries = len(results)
    skipped_entries = sum(1 for r in results if r.get("skipped", False))
    generated_entries = total_entries - skipped_entries
    logger.info("Loaded %d entries:", total_entries)
    logger.info("  - Generated trial answers: %d", generated_entries)
    logger.info("  - Skipped (embedding filter): %d", skipped_entries)

    # Load filtered steps file for forced_stop_step handling
    filtered_steps_data = None
    if args.filtered_steps:
        logger.info("Loading filtered steps from: %s", args.filtered_steps)
        filtered_steps_data = load_json(args.filtered_steps)
        forced_stop_count = sum(1 for q in filtered_steps_data if q.get("forced_stop_step") is not None)
        logger.info("  - Questions with forced_stop_step: %d", forced_stop_count)

    logger.info("Extracting final candidates using:")
    logger.info("  - Consecutive points: %d", args.consecutive)
    logger.info("  - Confidence threshold: %s", args.confidence_threshold)
    logger.info("  - Epsilon: %s", args.epsilon)
    if args.forced_stop_min_confidence > 0:
        logger.info("  - Forced stop min confidence: %s", args.forced_stop_min_confidence)
    else:
        logger.info("  - Forced stop min confidence: disabled (0.0)")
    if args.min_stop_step > 0:
        logger.info("  - Min stop step: %d", args.min_stop_step)

    filtered = extract_final_candidates(
        results,
        confidence_threshold=args.confidence_threshold,
        epsilon=args.epsilon,
        consecutive=args.consecutive,
        forced_stop_min_confidence=args.forced_stop_min_confidence,
        min_stop_step=args.min_stop_step,
        filtered_steps_data=filtered_steps_data,
        num_workers=args.workers,
    )

    save_json(filtered, args.output_file)

    logger.info("Extracted %d final candidates", len(filtered))

    # Print some statistics
    avg_trial_answers = sum(e["generated_trial_answers"] for e in filtered) / len(filtered) if filtered else 0
    avg_tokens_trial_answers = sum(e["tokens_trial_answers"] for e in filtered) / len(filtered) if filtered else 0
    logger.info("Average trial answers generated: %.2f", avg_trial_answers)
    logger.info("Average tokens in trial answers: %.2f", avg_tokens_trial_answers)

    # Stop reason distribution
    from collections import Counter
    stop_reasons = Counter(e.get("stop_reason", "unknown") for e in filtered)
    logger.info("Stop reason distribution:")
    for reason, count in stop_reasons.most_common():
        logger.info("  %s: %d/%d (%.1f%%)", reason, count, len(filtered), count / len(filtered) * 100)


if __name__ == "__main__":
    main()
