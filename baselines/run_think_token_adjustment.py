#!/usr/bin/env python3
"""
Think Token Adjustment Pipeline

A single script to run the full Think Token Adjustment baseline pipeline:
1. Generate vanilla answers (full reasoning without intervention) — for CR/CRT calculation
2. Run boost_sampling.py to generate predictions with logits intervention
3. Run count_tokens_after_think.py to count answer tokens
4. Run check_accuracy_baseline.py to compute accuracy, CR, and CRT

Usage:
    python -m baselines.run_think_token_adjustment \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/think_token_adjustment/aime24"

    # Use precomputed vanilla answers:
    python -m baselines.run_think_token_adjustment \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/think_token_adjustment/aime24" \
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl"
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
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
    else:
        print(f"\n✓ {description} completed successfully")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Think Token Adjustment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on AIME24 with DeepSeek-R1-Distill-Qwen-7B
    python -m baselines.run_think_token_adjustment \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/think_token_adjustment/aime24"

    # Run with a limit on number of samples (for testing)
    python -m baselines.run_think_token_adjustment \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/think_token_adjustment/aime24" \\
        --limit 5

    # Use precomputed vanilla answers
    python -m baselines.run_think_token_adjustment \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/think_token_adjustment/aime24" \\
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl"

    # Skip inference and only run evaluation (if predictions already exist)
    python -m baselines.run_think_token_adjustment \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/think_token_adjustment/aime24" \\
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl" \\
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
    parser.add_argument("--vanilla-answers", type=str, default="",
                        help="Path to precomputed vanilla answers (skip vanilla generation if provided)")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. 'gpqa') for dataset-specific answer extraction")

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    predictions_file = output_dir / "predictions.jsonl"
    tokens_file = output_dir / "tokens_after_think.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"
    
    print(f"\n{'#'*60}")
    print(f"  Think Token Adjustment Pipeline")
    print(f"{'#'*60}")
    print(f"Model:      {args.model}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Output dir: {args.output_dir}")
    print(f"Limit:      {args.limit}")
    print(f"Vanilla:    {vanilla_file}")
    
    if not args.eval_only:
        # Step 1: Generate vanilla answers (for CR/CRT calculation)
        if not args.vanilla_answers:
            cmd = [
                sys.executable, "-m", "baselines.think_token_adjustment.vanilla_generation",
                args.model, str(args.limit), args.benchmark, str(vanilla_file)
            ]
            if run_command(cmd, "Step 1/4: Generating vanilla answers") != 0:
                sys.exit(1)
        else:
            print(f"\n⭢ Using precomputed vanilla answers: {vanilla_file}")

        # Step 2: Run boost_sampling.py (inference with logits intervention)
        cmd = [
            sys.executable, "-m", "baselines.think_token_adjustment.boost_sampling",
            args.model,
            str(args.limit),
            args.benchmark,
            str(predictions_file)
        ]
        if run_command(cmd, "Step 2/4: Running Think Token Adjustment inference") != 0:
            sys.exit(1)
    else:
        print(f"\n⭢ Skipping inference (--eval-only)")
        if not predictions_file.exists():
            print(f"✗ Error: Predictions file not found: {predictions_file}")
            sys.exit(1)
        if not vanilla_file.exists():
            print(f"✗ Error: Vanilla file not found: {vanilla_file}")
            print("  Provide --vanilla-answers or remove --eval-only to generate them.")
            sys.exit(1)
    
    # Step 3: Count tokens after </think>
    cmd = [
        sys.executable, "-m", "baselines.think_token_adjustment.count_tokens_after_think",
        args.model,
        str(predictions_file),
        str(tokens_file)
    ]
    if run_command(cmd, "Step 3/4: Counting tokens after </think>") != 0:
        sys.exit(1)
    
    # Step 4: Compute accuracy, CR, and CRT
    cmd = [
        sys.executable, "scripts/check_accuracy_baseline.py",
        "--model", args.model,
        "--predictions", str(predictions_file),
        "--ground-truth", args.benchmark,
        "--vanilla", str(vanilla_file),
        "--tokens-after-think", str(tokens_file),
        "--dataset", args.dataset,
    ]
    if run_command(cmd, "Step 4/4: Computing accuracy, CR, and CRT") != 0:
        sys.exit(1)
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete!")
    print(f"{'#'*60}")
    print(f"\nOutput files:")
    print(f"  - Vanilla answers:    {vanilla_file}")
    print(f"  - Predictions:        {predictions_file}")
    print(f"  - Tokens after think: {tokens_file}")
    print(f"\nTo re-run evaluation only:")
    print(f"  python -m baselines.run_think_token_adjustment \\")
    print(f"      --model \"{args.model}\" \\")
    print(f"      --benchmark \"{args.benchmark}\" \\")
    print(f"      --output-dir \"{args.output_dir}\" \\")
    print(f"      --vanilla-answers \"{vanilla_file}\" \\")
    print(f"      --eval-only")


if __name__ == "__main__":
    main()