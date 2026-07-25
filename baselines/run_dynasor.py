#!/usr/bin/env python3
"""
Dynasor Pipeline

A single script to run the full Dynasor baseline pipeline:
1. Generate vanilla answers (using vanilla_generation.py, for CR calculation)
2. Run Dynasor inference (probe-in-the-middle with answer consistency early exit)
3. Compute accuracy, CR, and CRT

Usage:
    python -m baselines.run_dynasor \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/dynasor/aime24"

    # With custom effort level:
    python -m baselines.run_dynasor \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/dynasor/aime24" \
        --effort high

    # With custom threshold and chunk size:
    python -m baselines.run_dynasor \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/dynasor/aime24" \
        --effort-custom "5,128"

    # Use precomputed vanilla answers:
    python -m baselines.run_dynasor \
        --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --benchmark "data/aime24_test.jsonl" \
        --output-dir "runs/baselines/dynasor/aime24" \
        --vanilla-answers "runs/baselines/vanilla/aime24.jsonl"
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
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with exit code {result.returncode}")
    else:
        print(f"\n✓ {description} completed successfully")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Dynasor pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Effort levels (threshold, chunk_size):
    mild:  (8, 64) — most conservative, fewest early exits
    low:   (5, 64)
    mid:   (3, 64) — default
    high:  (2, 64)
    crazy: (2, 32) — most aggressive, most early exits
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model identifier (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark JSONL file (e.g., data/aime24_test.jsonl)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save all output files")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Maximum number of questions to process (default: 10000 = all)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation on existing predictions")
    parser.add_argument("--vanilla-answers", type=str, default="",
                        help="Path to precomputed vanilla answers (skip vanilla generation)")

    # Dynasor-specific parameters
    parser.add_argument("--effort", type=str, default="mid",
                        choices=["mild", "low", "mid", "high", "crazy"],
                        help="Dynasor effort level preset (default: mid)")
    parser.add_argument("--effort-custom", type=str, default=None,
                        help="Custom effort as 'threshold,chunk_size', e.g. '5,128'. "
                             "Overrides --effort if provided.")
    parser.add_argument("--probe-max-tokens", type=int, default=20,
                        help="Max tokens for probe response (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Total token budget per question (default: 32768)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for reasoning generation (default: 0.6)")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. 'gpqa') for dataset-specific answer extraction")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "final_answers.jsonl"
    vanilla_file = Path(args.vanilla_answers) if args.vanilla_answers else output_dir / "vanilla_answers.jsonl"

    print(f"\n{'#'*60}")
    print(f"  Dynasor Pipeline")
    print(f"{'#'*60}")
    print(f"Model:      {args.model}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Output:     {output_dir}")
    print(f"Effort:     {args.effort_custom if args.effort_custom else args.effort}")

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

        # Step 2: Run Dynasor inference
        cmd = [
            sys.executable, "-m", "baselines.dynasor.eval_dynasor",
            "--model", args.model,
            "--benchmark", args.benchmark,
            "--output-dir", str(output_dir),
            "--effort", args.effort,
            "--probe-max-tokens", str(args.probe_max_tokens),
            "--max-tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--limit", str(args.limit),
        ]
        if args.effort_custom:
            cmd.extend(["--effort-custom", args.effort_custom])

        if run_command(cmd, "Step 2/3: Running Dynasor inference") != 0:
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
