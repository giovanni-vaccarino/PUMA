#!/usr/bin/env python3
"""
Evaluate LiveCodeBench code generation results.

Uses official LCB testing_util.run_test() for correctness checking,
but with a robust single-level multiprocessing wrapper to avoid hangs.

Usage:
    python scripts/eval_livecodebench.py \
        --predictions runs/baselines/vanilla_code/DeepSeek-R1-Distill-Qwen-7B/livecodebench/vanilla_answers.jsonl \
        [--num-workers 8] [--timeout 6]
"""

import argparse
import json
import sys
import os
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.set_int_max_str_digits(50000)

# Code (LiveCodeBench) evaluation is optional and only needed for code datasets.
# It depends on the external LiveCodeBench repo (https://github.com/LiveCodeBench/LiveCodeBench):
# clone it and point LCB_REPO at the checkout.
LCB_REPO = os.environ.get("LCB_REPO", "")
if LCB_REPO:
    sys.path.insert(0, LCB_REPO)

import numpy as np
from tqdm import tqdm
try:
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
except ImportError as e:
    raise ImportError(
        "Could not import lcb_runner. Code evaluation requires the external "
        "LiveCodeBench repo: clone https://github.com/LiveCodeBench/LiveCodeBench "
        "and set the LCB_REPO environment variable to its path. This is only "
        "needed for code datasets, not for the paper's math/GPQA results."
    ) from e


def _evaluate_single(args):
    """Evaluate a single problem in a worker process.
    Uses subprocess with hard timeout to prevent hangs."""
    idx, sample_json, code, timeout = args

    # Fork a child process for actual code execution (isolation from segfaults)
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_run_in_child, args=(child_conn, sample_json, code, timeout))
    p.start()

    # Hard wall-clock timeout: timeout * num_tests + 30s buffer
    in_outs = json.loads(sample_json)
    num_tests = len(in_outs["inputs"])
    wall_timeout = (timeout + 1) * num_tests + 15

    p.join(timeout=wall_timeout)
    if p.is_alive():
        p.kill()
        p.join(timeout=5)
        return idx, [-1] * num_tests

    if parent_conn.poll():
        result = parent_conn.recv()
        return idx, result
    else:
        return idx, [-1] * num_tests


def _run_in_child(conn, sample_json, code, timeout):
    """Run in forked child. Calls official run_test."""
    try:
        from lcb_runner.evaluation.testing_util import run_test
        sample = {"input_output": sample_json}
        results, _ = run_test(sample, test=code, debug=False, timeout=timeout)
        # Normalize results
        fixed = []
        for e in results:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        conn.send(fixed)
    except Exception:
        # Send failure for all tests
        try:
            in_outs = json.loads(sample_json)
            conn.send([-1] * len(in_outs["inputs"]))
        except:
            conn.send([-1])
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Predictions file: JSONL (question_id+extracted_code) or PUMA JSON (prefixed_answers)")
    parser.add_argument("--benchmark", default=None,
                        help="Benchmark JSONL file (required for PUMA JSON format to map question->question_id)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file for eval results (default: eval_results.json in predictions dir)")
    parser.add_argument("--candidates", default=None,
                        help="Final candidates JSON file (for per-stop-reason accuracy breakdown)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()

    # Load official dataset
    print("Loading LiveCodeBench v5 dataset...")
    dataset = load_code_generation_dataset(release_version="release_v5")
    problem_map = {p.question_id: p for p in dataset}

    # Load predictions (support both JSONL and PUMA JSON formats)
    print(f"Loading predictions from {args.predictions}...")
    predictions = []
    with open(args.predictions) as f:
        content = f.read().strip()
    if content.startswith("["):
        # PUMA JSON format (list of dicts with "question" and "final_answer")
        puma_data = json.loads(content)
        # Build question -> question_id mapping from benchmark file
        if not args.benchmark:
            print("ERROR: --benchmark required for PUMA JSON format")
            sys.exit(1)
        q2id = {}
        with open(args.benchmark) as bf:
            for line in bf:
                item = json.loads(line)
                q2id[item["question"]] = item["question_id"]
        for entry in puma_data:
            qid = q2id.get(entry["question"])
            if qid:
                # Try final_answer (prefixed_answers), then model_answer (original answers)
                code = entry.get("final_answer", "") or entry.get("model_answer", "")
                predictions.append({"question_id": qid, "extracted_code": code})
        print(f"Loaded {len(predictions)} predictions from PUMA JSON (matched {len(predictions)}/{len(puma_data)})")
    else:
        # JSONL format
        for line in content.split("\n"):
            if line.strip():
                predictions.append(json.loads(line))
        print(f"Loaded {len(predictions)} predictions")

    # Build evaluation tasks
    tasks = []
    question_ids = []
    no_code = 0
    for pred in predictions:
        qid = pred["question_id"]
        if qid not in problem_map:
            continue
        code = pred.get("extracted_code", "")
        if not code:
            no_code += 1
            code = "pass"
        problem = problem_map[qid]
        eval_sample = problem.get_evaluation_sample()
        sample_json = eval_sample["input_output"]
        tasks.append((len(tasks), sample_json, code, args.timeout))
        question_ids.append(qid)

    print(f"Matched: {len(tasks)}, No code: {no_code}")
    print(f"Evaluating with {args.num_workers} workers, timeout={args.timeout}s per test case...")

    # Run evaluation
    results = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_evaluate_single, task): task[0] for task in tasks}
        with tqdm(total=len(tasks)) as pbar:
            for future in as_completed(futures):
                try:
                    idx, res = future.result(timeout=120)
                    results[idx] = res
                except Exception as e:
                    idx = futures[future]
                    in_outs = json.loads(tasks[idx][1])
                    results[idx] = [-1] * len(in_outs["inputs"])
                pbar.update(1)

    # Compute pass@1
    total = len(results)
    passed = sum(1 for idx, res in results.items() if all(r == True or r == 1 for r in res))
    pass_at_1 = passed / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"pass@1 = {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    print(f"Total: {total}, Passed: {passed}")
    print(f"{'='*50}")

    # Per-difficulty breakdown
    diff_stats = {}
    plat_stats = {}
    for idx, qid in enumerate(question_ids):
        problem = problem_map[qid]
        diff = problem.difficulty.value
        plat = problem.platform.value
        res = results.get(idx, [-1])
        is_pass = all(r == True or r == 1 for r in res)

        diff_stats.setdefault(diff, {"total": 0, "passed": 0})
        diff_stats[diff]["total"] += 1
        if is_pass:
            diff_stats[diff]["passed"] += 1

        plat_stats.setdefault(plat, {"total": 0, "passed": 0})
        plat_stats[plat]["total"] += 1
        if is_pass:
            plat_stats[plat]["passed"] += 1

    print("\nPer-difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff in diff_stats:
            d = diff_stats[diff]
            print(f"  {diff:8s}: {d['passed']:3d}/{d['total']:3d} = {d['passed']/d['total']*100:.1f}%")

    print("\nPer-platform:")
    for plat in ["leetcode", "codeforces", "atcoder"]:
        if plat in plat_stats:
            d = plat_stats[plat]
            print(f"  {plat:12s}: {d['passed']:3d}/{d['total']:3d} = {d['passed']/d['total']*100:.1f}%")

    # Per-question results
    per_question = {}
    for idx, qid in enumerate(question_ids):
        res = results.get(idx, [-1])
        per_question[qid] = all(r == True or r == 1 for r in res)

    # Per-stop-reason accuracy breakdown (if candidates provided)
    stop_reason_stats = {}
    if args.candidates and args.benchmark:
        candidates = json.load(open(args.candidates))
        bench_data = [json.loads(l) for l in open(args.benchmark)]
        idx_to_qid = {str(i+1): b["question_id"] for i, b in enumerate(bench_data)}

        from collections import defaultdict
        reason_stats = defaultdict(lambda: [0, 0])  # [total, correct]
        for c in candidates:
            qid = idx_to_qid.get(str(c["question_idx"]), "")
            if not qid:
                continue
            reason = c.get("stop_reason") or "unknown"
            is_correct = per_question.get(qid, False)
            reason_stats[reason][0] += 1
            if is_correct:
                reason_stats[reason][1] += 1

        print(f"\nPer Stop-Reason Accuracy:")
        for reason, (cnt, correct) in sorted(reason_stats.items(), key=lambda x: -x[1][0]):
            acc = correct / cnt * 100 if cnt > 0 else 0
            print(f"  {reason}: {correct}/{cnt} ({acc:.1f}%)")
        stop_reason_stats = {r: {"total": v[0], "correct": v[1], "accuracy": v[1]/v[0] if v[0] > 0 else 0} for r, v in reason_stats.items()}

    # Save results
    if args.output:
        eval_file = args.output
    else:
        output_dir = os.path.dirname(args.predictions)
        eval_file = os.path.join(output_dir, "eval_results.json")
    with open(eval_file, "w") as f:
        json.dump({
            "pass@1": pass_at_1,
            "num_problems": total,
            "passed": passed,
            "no_code_extracted": no_code,
            "difficulty": {k: {**v, "pass_rate": v["passed"]/v["total"]} for k, v in diff_stats.items()},
            "platform": {k: {**v, "pass_rate": v["passed"]/v["total"]} for k, v in plat_stats.items()},
            "per_question": per_question,
            "stop_reason": stop_reason_stats,
        }, f, indent=2)
    print(f"\nSaved to {eval_file}")


if __name__ == "__main__":
    main()
