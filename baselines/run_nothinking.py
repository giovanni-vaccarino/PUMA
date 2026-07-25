#!/usr/bin/env python3
"""
NoThinking Pipeline

A single script to run the full NoThinking baseline pipeline:
1. Run vanilla_generation.py to generate full-thinking baseline (if not provided)
2. Run nothinking_generation.py to generate NoThinking predictions
3. Run check_accuracy_baseline.py to compute accuracy and statistics

Based on: "Reasoning Models Can Be Effective Without Thinking" (Ma et al., 2025)
https://arxiv.org/abs/2504.09858

Usage:
    python -m baselines.run_nothinking \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/nothinking/aime24"

    # With dataset limit:
    python -m baselines.run_nothinking \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/nothinking/aime24" \
        --limit 10

    # Skip inference, only run evaluation:
    python -m baselines.run_nothinking \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/nothinking/aime24" \
        --eval-only

    # Use existing vanilla answers:
    python -m baselines.run_nothinking \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/nothinking/aime24" \
        --vanilla-answers "runs/baselines/vanilla/aime24/predictions.jsonl"
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with exit code {result.returncode}")
    else:
        print(f"\n✓ {description} completed successfully")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full NoThinking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on AIME24 with DeepSeek-R1-Distill-Qwen-7B
    python -m baselines.run_nothinking \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/nothinking/aime24"

    # Run with a limit on number of samples (for testing)
    python -m baselines.run_nothinking \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/nothinking/aime24" \\
        --limit 5

    # Skip inference and only run evaluation
    python -m baselines.run_nothinking \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/nothinking/aime24" \\
        --eval-only
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file (e.g., data/aime24_test.jsonl)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save all output files")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Maximum number of samples to process (default: 10000 = all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation on existing predictions")
    parser.add_argument("--vanilla-answers", type=str, default=None,
                        help="Path to existing vanilla answers JSONL (skip vanilla generation if provided)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print individual predictions during evaluation")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. 'gpqa') for dataset-specific answer extraction")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    predictions_file = output_dir / "predictions.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"

    print(f"\n{'#'*60}")
    print(f"  NoThinking Pipeline")
    print(f"{'#'*60}")
    print(f"Model:      {args.model}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Output dir: {args.output_dir}")
    print(f"Limit:      {args.limit}")

    if not args.eval_only:
        # Step 1: Generate vanilla answers (for CR/CRT computation)
        if args.vanilla_answers and Path(args.vanilla_answers).exists():
            print(f"\n⭐ Using existing vanilla answers: {args.vanilla_answers}")
        else:
            cmd = [
                sys.executable, "-m", "baselines.think_token_adjustment.vanilla_generation",
                args.model,
                str(args.limit),
                args.benchmark,
                str(vanilla_file)
            ]
            exit_code = run_command(cmd, "Step 1/3: Generating vanilla (full thinking) answers")
            if exit_code != 0:
                sys.exit(exit_code)

        # Step 2: Run NoThinking generation
        cmd = [
            sys.executable, "-m", "baselines.nothinking.nothinking_generation",
            args.model,
            str(args.limit),
            args.benchmark,
            str(predictions_file)
        ]
        exit_code = run_command(cmd, "Step 2/3: Running NoThinking inference")
        if exit_code != 0:
            sys.exit(exit_code)
    else:
        print(f"\n⭐ Skipping inference (--eval-only), using existing predictions: {predictions_file}")
        if not predictions_file.exists():
            print(f"✗ Error: Predictions file not found: {predictions_file}")
            sys.exit(1)

    # Step 3: Compute accuracy and statistics
    cmd = [
        sys.executable, "scripts/check_accuracy_baseline.py",
        "--model", args.model,
        "--predictions", str(predictions_file),
        "--ground-truth", args.benchmark,
        "--vanilla", str(vanilla_file),
        "--dataset", args.dataset,
    ]
    if args.verbose:
        cmd.append("--verbose")

    exit_code = run_command(cmd, "Step 3/3: Computing accuracy and statistics")
    if exit_code != 0:
        sys.exit(exit_code)

    # Summary
    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"\nOutput files:")
    print(f"  - Predictions:     {predictions_file}")
    print(f"  - Vanilla answers: {vanilla_file}")
    print(f"\nTo re-run evaluation only:")
    print(f"  python -m baselines.run_nothinking \\")
    print(f"      --model \"{args.model}\" \\")
    print(f"      --benchmark \"{args.benchmark}\" \\")
    print(f"      --output-dir \"{args.output_dir}\" \\")
    print(f"      --eval-only")


if __name__ == "__main__":
    main()