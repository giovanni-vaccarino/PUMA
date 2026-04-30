import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(cmd):
    """Run a pipeline stage and stop immediately if it fails."""
    cmd = [str(c) for c in cmd]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    """Parse command-line arguments for the offline PUMA driver."""
    parser = argparse.ArgumentParser(
        description="Run the PUMA offline pipeline"
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base directory containing answers.json (output files are written here too)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Path to the fine-tuned Qwen3-Embedding checkpoint (Redundancy Detector)",
    )

    # Redundancy Detector Hyperparameters
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="τ_sim: cosine similarity threshold for redundancy detection (default: 0.35)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="K: compare each step against the last K steps (default: 1)",
    )
    parser.add_argument(
        "--consecutive-redundancy-stop",
        type=int,
        default=50,
        help="Loop Breaker: number of consecutive redundant steps before forced exit (default: 50)",
    )

    # Verified Early Exit Hyperparameters
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.98,
        help="λ: confidence threshold for Verified Early Exit (default: 0.98)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="ε: maximum confidence drop allowed from first candidate (default: 0.03)",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="L: consecutive consistent checkpoints required to stop (default: 2)",
    )

    # vLLM options
    parser.add_argument(
        "--logprobs-mode",
        type=str,
        default=None,
        help="Logprobs mode for vLLM >= 0.10.x (e.g., 'processed_logprobs')",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    base = args.base_dir
    model = args.model

    if not base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base}")

    answers_path = base / "answers.json"
    if not answers_path.exists():
        raise FileNotFoundError(
            f"Expected precomputed full reasoning outputs at: {answers_path}"
        )

    # Split reasoning into steps
    run([
        sys.executable, "-m", "puma.separate_steps",
        "--input_json", base / "answers.json",
        "--output_json", base / "steps.json",
    ])

    # Redundancy Detector
    run([
        sys.executable, "-m", "puma.embed_and_filter_steps",
        "--input-file", base / "steps.json",
        "--output-file", base / "filtered_steps.json",
        "--embedding-model", args.embedding_model,
        "--similarity-threshold", args.similarity_threshold,
        "--window-size", args.window_size,
        "--trigger-mode", "previous",
        "--consecutive-redundancy-stop", args.consecutive_redundancy_stop,
    ])

    # Generate trial answers
    trial_cmd = [
        sys.executable, "-m", "puma.gen_trial_answers",
        "--questions-file", base / "filtered_steps.json",
        "--output-file", base / "trial_answers.json",
        "--model", model,
        "--max-tokens", 15,  # tight budget: trial answers only need the boxed value (~5-15 tokens)
        "--temperature", 0.0,
        "--respect-embedding-filter",
    ]
    if args.logprobs_mode:
        trial_cmd.extend(["--logprobs-mode", args.logprobs_mode])
    run(trial_cmd)

    # Verified Early Exit
    run([
        sys.executable, "-m", "puma.extract_final_candidates",
        "--input-file", base / "trial_answers.json",
        "--output-file", base / "final_candidates.json",
        "--confidence-threshold", args.confidence_threshold,
        "--epsilon", args.epsilon,
        "--consecutive", args.consecutive,
        "--filtered-steps", base / "filtered_steps.json",
    ])

    # Generate final answer from truncated prefix
    run([
        sys.executable, "-m", "puma.gen_prefixed_answers",
        "--final-candidates", base / "final_candidates.json",
        "--questions-file", base / "steps.json",
        "--output-file", base / "prefixed_answers.json",
        "--model", model,
        "--max-tokens", 2048,
    ])

    # Evaluate accuracy and token reduction
    run([
        sys.executable, "-m", "puma.statistics_puma",
        "--original", base / "answers.json",
        "--compressed", base / "prefixed_answers.json",
        "--model", model,
    ])

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
