#!/usr/bin/env python3
"""
Plan-and-Budget (P&B) Pipeline

A single script to run the full P&B baseline pipeline:
1. Generate plans (decomposition + assessment) using planner LLM — if not cached
2. Run P&B reasoning with budget hints (polynomial decay scheduling)
3. Compute accuracy, CR, and CRT

Based on: "Plan-and-Budget: Effective and Efficient Test-Time Scaling on LLM Reasoning"
          (Lin et al., ICLR 2026)

Usage:
    python -m baselines.run_planbudget \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/planbudget/DeepSeek-R1-Distill-Qwen-7B/aime24" \
        --vanilla-answers "runs/baselines/vanilla/.../vanilla_answers.jsonl" \
        --dataset "aime24"

    # Skip inference, only run evaluation:
    python -m baselines.run_planbudget \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/planbudget/DeepSeek-R1-Distill-Qwen-7B/aime24" \
        --eval-only
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  {description} failed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Plan-and-Budget (P&B) pipeline",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier (reasoning model)")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--planner-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Planner model for decomposition (default: Llama-3.1-8B)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max questions to process (default: 10000 = all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation")
    parser.add_argument("--vanilla-answers", type=str, default="",
                        help="Path to precomputed vanilla answers")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (aime24, math-500, gpqa-diamond, etc.)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"

    # Plans are shared across reasoning models (same planner + dataset = same plans)
    plans_dir = Path("runs/baselines/planbudget/plans")
    plans_dir.mkdir(parents=True, exist_ok=True)
    plans_file = plans_dir / f"{args.dataset}_plans.jsonl"

    print(f"\n{'#'*60}")
    print(f"  Plan-and-Budget (P&B) Pipeline")
    print(f"{'#'*60}")
    print(f"Model:        {args.model}")
    print(f"Planner:      {args.planner_model}")
    print(f"Benchmark:    {args.benchmark}")
    print(f"Output:       {output_dir}")
    print(f"Plans:        {plans_file}")
    print(f"Limit:        {args.limit}")

    if not args.eval_only:
        # Step 1: Generate plans (if not cached)
        need_plans = True
        if plans_file.exists():
            with open(args.benchmark) as f:
                num_questions = sum(1 for _ in f)
            with open(plans_file) as f:
                num_plans = sum(1 for _ in f)
            if num_plans >= min(num_questions, args.limit):
                print(f"\nPlans already exist ({num_plans} entries). Skipping decomposition.")
                need_plans = False
            else:
                print(f"\nPlans incomplete ({num_plans}/{num_questions}). Regenerating.")

        if need_plans:
            cmd = [
                sys.executable, "-m", "baselines.planbudget.decompose_questions",
                "--planner-model", args.planner_model,
                "--benchmark", args.benchmark,
                "--output", str(plans_file),
                "--dataset", args.dataset,
            ]
            if run_command(cmd, "Step 1/3: Decomposing questions + assessing difficulty") != 0:
                sys.exit(1)

        # Step 2: Run P&B generation
        cmd = [
            sys.executable, "-m", "baselines.planbudget.planbudget_generation",
            args.model, str(args.limit), args.benchmark, str(predictions_file),
            "--plans", str(plans_file),
        ]
        if run_command(cmd, "Step 2/3: Running P&B reasoning with budget hints") != 0:
            sys.exit(1)
    else:
        print(f"\nSkipping inference (--eval-only)")
        if not predictions_file.exists():
            print(f"Predictions not found: {predictions_file}")
            sys.exit(1)

    # Step 3: Compute accuracy, CR, CRT
    cmd = [
        sys.executable, "scripts/check_accuracy_baseline.py",
        "--model", args.model,
        "--predictions", str(predictions_file),
        "--ground-truth", args.benchmark,
        "--vanilla", str(vanilla_file),
        "--dataset", args.dataset,
    ]
    if run_command(cmd, "Step 3/3: Computing accuracy, CR, CRT") != 0:
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"Output files:")
    print(f"  - Plans:       {plans_file}")
    print(f"  - Predictions: {predictions_file}")
    print(f"  - Vanilla:     {vanilla_file}")


if __name__ == "__main__":
    main()
