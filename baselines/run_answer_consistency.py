#!/usr/bin/env python3
"""
Answer Consistency Pipeline

Run the full Answer Consistency baseline pipeline:
1. Generate vanilla answers (full reasoning without intervention)
2. Run online_answer_consistency.py (incremental answer consistency with early stopping)
3. Compute accuracy, CR, and CRT

Usage:
    python -m baselines.run_answer_consistency \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/answer_consistency/aime24"
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
        print(f"\n✗ {description} failed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Answer Consistency pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on AIME24
    python -m baselines.run_answer_consistency \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/answer_consistency/aime24"

    # With custom threshold
    python -m baselines.run_answer_consistency \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/answer_consistency/aime24" \\
        --threshold 5

    # Skip inference, only evaluate
    python -m baselines.run_answer_consistency \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/answer_consistency/aime24" \\
        --eval-only

    # Use precomputed vanilla answers
    python -m baselines.run_answer_consistency \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/answer_consistency/aime24" \\
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl"
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Number of consecutive identical answers to stop (default: 10)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max questions to process")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation")
    parser.add_argument("--vanilla-answers", type=str, default="",
                        help="Path to precomputed vanilla answers")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. 'gpqa') for dataset-specific answer extraction")

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = output_dir / "final_answers.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"
    
    print(f"\n{'#'*60}")
    print(f"  Answer Consistency Pipeline")
    print(f"{'#'*60}")
    print(f"Model:      {args.model}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Output:     {output_dir}")
    print(f"Threshold:  {args.threshold}")
    
    if not args.eval_only:
        # Step 1: Generate vanilla answers (for CR calculation)
        if not args.vanilla_answers:
            cmd = [
                sys.executable, "-m", "baselines.think_token_adjustment.vanilla_generation",
                args.model, str(args.limit), args.benchmark, str(vanilla_file)
            ]
            if run_command(cmd, "Step 1/3: Generating vanilla answers") != 0:
                sys.exit(1)
        else:
            print(f"\n⭐ Using precomputed vanilla answers: {vanilla_file}")
        
        # Step 2: Run online answer consistency
        cmd = [
            sys.executable, "-m", "baselines.answer_consistency.online_answer_consistency",
            "--model", args.model,
            "--benchmark", args.benchmark,
            "--output-dir", str(output_dir),
            "--vanilla-answers", str(vanilla_file),
            "--threshold", str(args.threshold),
            "--limit", str(args.limit),
        ]
        if run_command(cmd, "Step 2/3: Running online answer consistency") != 0:
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
    print(f"  - {vanilla_file}")
    print(f"  - {predictions_file}")


if __name__ == "__main__":
    main()