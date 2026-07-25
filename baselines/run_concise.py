#!/usr/bin/env python3
"""
CCoT (Constrained Chain-of-Thought) Pipeline

A single script to run the full CCoT baseline pipeline:
1. Generate vanilla answers (full reasoning without intervention)
2. Run concise_generation.py (CCoT with word-budget prompt)
3. Compute accuracy, CR, and CRT

Based on: "Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost"
          (Nayab et al., 2024)

Usage:
    python -m baselines.run_concise \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/concise/aime24"

    # With custom word budget (default: 45):
    python -m baselines.run_concise \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/concise/aime24" \
        --budget-words 30

    # Use existing vanilla answers:
    python -m baselines.run_concise \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/concise/aime24" \
        --vanilla-answers "runs/baselines/vanilla/aime24/predictions.jsonl"

    # Skip inference, only run evaluation:
    python -m baselines.run_concise \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/concise/aime24" \
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
        print(f"\n✗ {description} failed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full CCoT (Constrained Chain-of-Thought) pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on AIME24
    python -m baselines.run_concise \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/concise/aime24"

    # With word budget 30
    python -m baselines.run_concise \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/concise/aime24" \\
        --budget-words 30

    # Skip inference, only evaluate
    python -m baselines.run_concise \\
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
        --benchmark "data/aime24_test.jsonl" \\
        --output-dir "runs/baselines/concise/aime24" \\
        --eval-only
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument("--budget-words", type=int, default=45,
                        help="Word budget for CCoT constraint (default: 45)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max questions to process (default: 10000 = all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation")
    parser.add_argument("--vanilla-answers", type=str, default="",
                        help="Path to precomputed vanilla answers")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. 'gpqa') for dataset-specific answer extraction")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"

    print(f"\n{'#'*60}")
    print(f"  CCoT (Constrained Chain-of-Thought) Pipeline")
    print(f"{'#'*60}")
    print(f"Model:        {args.model}")
    print(f"Benchmark:    {args.benchmark}")
    print(f"Output:       {output_dir}")
    print(f"Budget words: {args.budget_words}")
    print(f"Limit:        {args.limit}")

    if not args.eval_only:
        # Step 1: Generate vanilla answers (for CR/CRT computation)
        if args.vanilla_answers and Path(args.vanilla_answers).exists():
            print(f"\n⭐ Using precomputed vanilla answers: {vanilla_file}")
        else:
            cmd = [
                sys.executable, "-m", "baselines.think_token_adjustment.vanilla_generation",
                args.model, str(args.limit), args.benchmark, str(vanilla_file),
            ]
            if run_command(cmd, "Step 1/3: Generating vanilla answers") != 0:
                sys.exit(1)

        # Step 2: Run CCoT generation
        cmd = [
            sys.executable, "-m", "baselines.concise.concise_generation",
            args.model, str(args.limit), args.benchmark, str(predictions_file),
            "--budget-words", str(args.budget_words),
        ]
        if run_command(cmd, f"Step 2/3: Running CCoT-{args.budget_words} generation") != 0:
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
    print(f"  - Vanilla:     {vanilla_file}")
    print(f"  - Predictions: {predictions_file}")


if __name__ == "__main__":
    main()