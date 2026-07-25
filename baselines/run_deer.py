#!/usr/bin/env python3
"""
DEER Pipeline (Dynamic Early Exit for Reasoning)

Run the full DEER baseline and compute accuracy, CR, and CRT.

Usage:
    python -m baselines.run_deer \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --dataset_dir "data" \
        --dataset "aime24" \
        --output-dir "runs/baselines/deer/aime24"

Pipeline:
    1. Generate vanilla answers (vanilla_generation.py — full reasoning)
    2. Run DEER inference (vllm_deer.py — confidence-based early exit)
    3. Compute accuracy, CR, and CRT
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
        print(f"\n✗ {description} failed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full DEER pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on AIME24
    python -m baselines.run_deer \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --dataset_dir "data" \\
        --dataset "aime24" \\
        --output-dir "runs/baselines/deer/aime24"

    # With custom threshold and policy
    python -m baselines.run_deer \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --dataset_dir "data" \\
        --dataset "aime24" \\
        --output-dir "runs/baselines/deer/aime24" \\
        --threshold 0.90 --policy avg1

    # Skip inference, only evaluate
    python -m baselines.run_deer \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --dataset_dir "data" \\
        --dataset "aime24" \\
        --output-dir "runs/baselines/deer/aime24" \\
        --eval-only

    # Use precomputed vanilla answers
    python -m baselines.run_deer \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --dataset_dir "data" \\
        --dataset "aime24" \\
        --output-dir "runs/baselines/deer/aime24" \\
        --vanilla-answers "runs/baselines/deer/aime24/vanilla_answers.jsonl"
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier")
    parser.add_argument("--dataset_dir", type=str, default="data",
                        help="Root directory containing dataset files (default: data)")
    parser.add_argument("--dataset", type=str, default="aime24",
                        help="Dataset name, expects {dataset_dir}/{dataset}_test.jsonl")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max questions to process (default: 10000 = all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation")
    parser.add_argument("--vanilla-answers", type=str, default="",
                        help="Path to precomputed vanilla answers")

    # DEER-specific parameters
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Confidence threshold for early exit (default: 0.95)")
    parser.add_argument("--atp", type=str, default="Wait",
                        choices=["Wait", "Alternatively"],
                        help="Action Transition Point string (default: Wait)")
    parser.add_argument("--policy", type=str, default="avg1",
                        choices=["avg1", "avg2", "min"],
                        help="Confidence calculation policy (default: avg1)")
    parser.add_argument("--max-judge-steps", type=int, default=10,
                        help="Max confidence checks per question (default: 10)")
    parser.add_argument("--enable-repetition-exit", action="store_true",
                        help="Enable early exit on repetition detection")

    # Generation parameters (passed to vllm_deer.py)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for vLLM (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for generation (default: 0.6)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p for generation (default: 0.95)")
    parser.add_argument("--max-generated-tokens", type=int, default=32768,
                        help="Max total generated tokens (default: 32768)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "final_answers.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"
    benchmark_file = f"{args.dataset_dir}/{args.dataset}_test.jsonl"

    print(f"\n{'#'*60}")
    print(f"  DEER Pipeline")
    print(f"{'#'*60}")
    print(f"Model:      {args.model}")
    print(f"Dataset:    {benchmark_file}")
    print(f"Output:     {output_dir}")
    print(f"Threshold:  {args.threshold}")
    print(f"ATP:        {args.atp}")
    print(f"Policy:     {args.policy}")

    if not args.eval_only:
        # Step 1: Generate vanilla answers
        if not args.vanilla_answers:
            cmd = [
                sys.executable, "-m",
                "baselines.think_token_adjustment.vanilla_generation",
                args.model, str(args.limit), benchmark_file, str(vanilla_file)
            ]
            if run_command(cmd, "Step 1/3: Generating vanilla answers") != 0:
                sys.exit(1)
        else:
            print(f"\n⭐ Using precomputed vanilla answers: {vanilla_file}")

        # Step 2: Run DEER inference
        cmd = [
            sys.executable, "baselines/deer/vllm_deer.py",
            "--model_name_or_path", args.model,
            "--dataset_dir", args.dataset_dir,
            "--dataset", args.dataset,
            "--output_path", str(predictions_file),
            "--threshold", str(args.threshold),
            "--points", "1" if args.atp == "Wait" else "0",
            "--policy", args.policy,
            "--max_judge_steps", str(args.max_judge_steps),
            "--seed", str(args.seed),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--max_generated_tokens", str(args.max_generated_tokens),
        ]
        if args.enable_repetition_exit:
            cmd.extend(["--rep", "1"])

        if run_command(cmd, "Step 2/3: Running DEER inference") != 0:
            sys.exit(1)
    else:
        print(f"\n⭐ Skipping inference (--eval-only)")
        if not predictions_file.exists():
            print(f"✗ Predictions not found: {predictions_file}")
            sys.exit(1)

    # Step 3: Compute accuracy, CR, CRT
    cmd = [
        sys.executable, "scripts/check_accuracy_baseline.py",
        "--model", args.model,
        "--predictions", str(predictions_file),
        "--ground-truth", benchmark_file,
        "--vanilla", str(vanilla_file),
        "--dataset", args.dataset,
    ]
    if run_command(cmd, "Step 3/3: Computing accuracy, CR, CRT") != 0:
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"Output files:")
    print(f"  - Vanilla:     {vanilla_file}")
    print(f"  - Predictions: {predictions_file}")


if __name__ == "__main__":
    main()