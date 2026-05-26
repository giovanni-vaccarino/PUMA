#!/usr/bin/env python3
"""
Embedding-based filtering for trial answer generation.

This script computes embeddings for reasoning steps and flags steps that should
undergo trial answer generation based on similarity with previous steps.

Usage:
    python method/offline/embed_and_filter_steps.py \
        --input-file questions_with_steps.json \
        --output-file questions_with_steps_filtered.json \
        --embedding-model /path/to/checkpoint-41605 \
        --similarity-threshold 0.32 \
        --always-check-first-n 1 \
        --window-size 1 \
        --trigger-mode current
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from vllm import LLM


logger = logging.getLogger(__name__)


def load_embedding_model(model_path: str) -> LLM:
    """
    Load embedding model using vLLM with runner='pooling'.

    Supports both full checkpoints and LoRA adapters (auto-merges LoRA).

    Args:
        model_path: Path to model checkpoint

    Returns:
        vLLM LLM instance configured for embedding
    """
    logger.info(f"Loading embedding model from: {model_path}")

    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"

    if adapter_config_path.exists():
        # LoRA checkpoint: merge adapter and save for vLLM
        logger.info("Detected LoRA checkpoint, merging adapter for vLLM...")

        import torch
        from transformers import AutoModel, AutoTokenizer
        from safetensors import safe_open
        from peft import get_peft_model, LoraConfig

        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True, padding_side="left"
        )
        base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)

        lora_config = LoraConfig(
            r=adapter_config.get("r", 16),
            lora_alpha=adapter_config.get("lora_alpha", 32),
            target_modules=adapter_config.get("target_modules", []),
            lora_dropout=adapter_config.get("lora_dropout", 0.1),
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)

        adapter_file = model_path_obj / "adapter_model.safetensors"
        state_dict = {}
        with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                new_key = key.replace("base_model.model.model.", "base_model.model.")
                new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
                new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                state_dict[new_key] = tensor

        model.load_state_dict(state_dict, strict=False)
        model = model.merge_and_unload()

        # Save merged model to temp directory for vLLM
        merged_dir = tempfile.mkdtemp(prefix="puma_merged_emb_")
        logger.info(f"Saving merged model to: {merged_dir}")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        # Free memory before vLLM loads
        del model, base_model
        torch.cuda.empty_cache()

        load_path = merged_dir
    else:
        load_path = str(model_path_obj)

    logger.info(f"Loading embedding model with vLLM from: {load_path}")
    llm = LLM(model=load_path, runner="pooling", trust_remote_code=True,
              gpu_memory_utilization=0.3)
    return llm


def get_embeddings_batch(llm: LLM, texts: List[str], max_chars: int = 30000) -> np.ndarray:
    """
    Compute normalized embeddings using vLLM.

    Args:
        llm: vLLM LLM instance with runner='pooling'
        texts: List of text strings
        max_chars: Truncate texts longer than this to avoid exceeding model's max_model_len

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized
    """
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]
    n_truncated = sum(1 for t in texts if len(t) > max_chars)
    if n_truncated > 0:
        logger.warning(f"Truncated {n_truncated}/{len(texts)} texts exceeding {max_chars} chars for embedding")
    outputs = llm.embed(truncated)
    embeddings = np.array([o.outputs.embedding for o in outputs])

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b))  # Already normalized


def embed_and_filter_steps(
    questions_data: List[Dict],
    llm: LLM,
    similarity_threshold: float = 0.32,
    always_check_first_n: int = 1,
    window_size: int = 1,
    trigger_mode: str = "previous",
    consecutive_redundancy_stop: int = 0,
    consecutive_redundancy_min_step: int = 0,
    loop_breaker_similarity_threshold: float = None,
) -> List[Dict]:
    """
    Compute embeddings and flag steps for trial answer generation.

    Args:
        questions_data: List of question dicts with 'reasoning_steps'
        llm: vLLM LLM instance for embedding
        similarity_threshold: τ_sim threshold for flagging steps for trial answer generation
        always_check_first_n: Always generate trial for first N steps
        window_size: K - compare with last K steps (<=0 means all previous)
        trigger_mode: "current" (flag step_i) or "previous" (flag step_{i-1})
        consecutive_redundancy_stop: m - stop when m consecutive redundant steps detected (0=disabled)
        consecutive_redundancy_min_step: Only start detecting consecutive redundancy after this step index (0-indexed)
        loop_breaker_similarity_threshold: Separate threshold for Loop Breaker (defaults to similarity_threshold)

    Returns:
        Updated questions_data with 'step_similarities' and 'should_generate_trial'
    """
    # Default loop breaker threshold to similarity_threshold if not set
    if loop_breaker_similarity_threshold is None:
        loop_breaker_similarity_threshold = similarity_threshold

    logger.info(f"Processing {len(questions_data)} questions")
    logger.info(f"  similarity_threshold (τ_sim): {similarity_threshold}")
    logger.info(f"  always_check_first_n: {always_check_first_n}")
    logger.info(f"  window_size (K): {window_size}")
    logger.info(f"  trigger_mode: {trigger_mode}")
    logger.info(f"  consecutive_redundancy_stop (m): {consecutive_redundancy_stop}")
    logger.info(f"  consecutive_redundancy_min_step: {consecutive_redundancy_min_step}")
    if loop_breaker_similarity_threshold != similarity_threshold:
        logger.info(f"  loop_breaker_similarity_threshold: {loop_breaker_similarity_threshold} (separate from RD)")

    # Collect all steps for batch embedding
    all_steps = []
    question_step_ranges = []  # (start_idx, end_idx) for each question

    for question in questions_data:
        steps = question.get("reasoning_steps", [])
        start_idx = len(all_steps)
        all_steps.extend(steps)
        end_idx = len(all_steps)
        question_step_ranges.append((start_idx, end_idx))

    logger.info(f"Total steps to embed: {len(all_steps)}")

    # Batch compute all embeddings
    if all_steps:
        t_embed_start = time.perf_counter()
        all_embeddings = get_embeddings_batch(llm, all_steps)
        t_embed = time.perf_counter() - t_embed_start
        logger.info(f"Embeddings shape: {all_embeddings.shape}")
        logger.info(f"Embedding inference: {t_embed:.2f}s for {len(all_steps)} steps ({len(all_steps)/t_embed:.1f} steps/s)")
    else:
        logger.warning("No steps found in questions data")
        return questions_data

    # Process each question
    total_flagged = 0
    total_steps = 0
    total_consecutive_stops = 0
    total_sim_time = 0.0
    total_sim_pairs = 0

    for q_idx, question in enumerate(tqdm(questions_data, desc="Filtering steps")):
        steps = question.get("reasoning_steps", [])
        n_steps = len(steps)
        total_steps += n_steps

        if n_steps == 0:
            question["step_similarities"] = []
            question["should_generate_trial"] = []
            continue

        # Get embeddings for this question's steps
        start_idx, end_idx = question_step_ranges[q_idx]
        step_embeddings = all_embeddings[start_idx:end_idx]

        # Compute max similarity with previous K steps for each step
        t_sim_q_start = time.perf_counter()
        max_similarities = [None]  # First step has no previous
        for i in range(1, n_steps):
            if window_size <= 0:
                # All previous steps
                start = 0
            else:
                # Last K steps
                start = max(0, i - window_size)

            sims = [
                cosine_similarity(step_embeddings[j], step_embeddings[i])
                for j in range(start, i)
            ]
            max_similarities.append(max(sims) if sims else None)
            total_sim_pairs += (i - start)
        total_sim_time += time.perf_counter() - t_sim_q_start

        # Determine which steps should generate trial answers
        should_generate = [False] * n_steps

        # Always check first N steps
        for i in range(min(always_check_first_n, n_steps)):
            should_generate[i] = True

        # Flag based on similarity and trigger_mode
        for i in range(1, n_steps):
            if (
                max_similarities[i] is not None
                and max_similarities[i] >= similarity_threshold
            ):
                if trigger_mode == "current":
                    should_generate[i] = True  # Flag step_i
                elif trigger_mode == "previous":
                    should_generate[i - 1] = True  # Flag step_{i-1}

        # Detect consecutive redundancy and force stop
        forced_stop_step = None
        consecutive_redundancy_detected = False

        if consecutive_redundancy_stop > 0:
            consecutive_count = 0
            first_redundant_idx = None

            # Start detection from consecutive_redundancy_min_step
            # This avoids false positives from early steps (problem restatement, decomposition)
            start_step = max(1, consecutive_redundancy_min_step)

            for i in range(start_step, n_steps):
                if (
                    max_similarities[i] is not None
                    and max_similarities[i] >= loop_breaker_similarity_threshold
                ):
                    if first_redundant_idx is None:
                        first_redundant_idx = i
                    consecutive_count += 1

                    if consecutive_count >= consecutive_redundancy_stop:
                        # Force stop at the step before the first redundant step
                        forced_stop_step = first_redundant_idx - 1
                        consecutive_redundancy_detected = True
                        logger.debug(
                            f"Q{q_idx + 1}: Consecutive redundancy detected at steps "
                            f"{first_redundant_idx}-{i}, forced stop at step {forced_stop_step}"
                        )
                        break
                else:
                    # Reset count
                    consecutive_count = 0
                    first_redundant_idx = None

            # Note: We do NOT modify should_generate based on forced_stop_step here.
            # The embedding filter continues to work normally.
            # forced_stop_step is just metadata for extract_final_candidates.py
            # to use as a direct stopping point.

        # Store results
        question["step_similarities"] = max_similarities
        question["should_generate_trial"] = should_generate
        question["forced_stop_step"] = forced_stop_step
        question["consecutive_redundancy_detected"] = consecutive_redundancy_detected

        # Count flagged steps and consecutive stops
        total_flagged += sum(should_generate)
        if consecutive_redundancy_detected:
            total_consecutive_stops += 1

    # Log statistics
    flag_rate = total_flagged / total_steps if total_steps > 0 else 0
    logger.info(f"Filtering complete:")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Flagged for trial: {total_flagged} ({flag_rate:.1%})")
    logger.info(f"  Skipped: {total_steps - total_flagged} ({1 - flag_rate:.1%})")
    if consecutive_redundancy_stop > 0:
        logger.info(f"  Consecutive redundancy stops: {total_consecutive_stops} questions")
    logger.info(f"  Similarity computation: {total_sim_time:.4f}s for {total_sim_pairs} pairs (window_size={window_size})")
    if total_sim_pairs > 0:
        logger.info(f"  Avg similarity time per pair: {total_sim_time / total_sim_pairs * 1e6:.2f} us")

    return questions_data


def load_json(filepath: str) -> List[Dict]:
    """Load JSON list from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict], filepath: str):
    """Save data to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding-based filtering for trial answer generation"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input JSON file with questions and reasoning_steps",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSON file with filtering results",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Path to embedding model checkpoint",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.32,
        help="τ_sim: Similarity threshold for flagging (default: 0.32)",
    )
    parser.add_argument(
        "--always-check-first-n",
        type=int,
        default=1,
        help="Always generate trial for first N steps (default: 1)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="K: Compare with last K steps. <=0 means all previous (default: 1)",
    )
    parser.add_argument(
        "--trigger-mode",
        type=str,
        default="previous",
        choices=["current", "previous"],
        help="'current': flag step_i, 'previous': flag step_{i-1} (default: previous)",
    )
    parser.add_argument(
        "--consecutive-redundancy-stop",
        type=int,
        default=0,
        help="m: Stop when m consecutive redundant steps detected (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--consecutive-redundancy-min-step",
        type=int,
        default=0,
        help="Only start detecting consecutive redundancy after this step index (default: 0). "
             "Use this to skip early steps that often contain problem restatement/decomposition.",
    )
    parser.add_argument(
        "--loop-breaker-similarity-threshold",
        type=float,
        default=None,
        help="Separate similarity threshold for Loop Breaker (defaults to --similarity-threshold if not set)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="(Deprecated, ignored) vLLM handles batching internally",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=== Embedding-based Step Filtering ===")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"Embedding model: {args.embedding_model}")

    # Load data
    questions_data = load_json(args.input_file)
    logger.info(f"Loaded {len(questions_data)} questions")

    # Load embedding model
    llm = load_embedding_model(args.embedding_model)

    # Process and filter
    questions_data = embed_and_filter_steps(
        questions_data=questions_data,
        llm=llm,
        similarity_threshold=args.similarity_threshold,
        always_check_first_n=args.always_check_first_n,
        window_size=args.window_size,
        trigger_mode=args.trigger_mode,
        consecutive_redundancy_stop=args.consecutive_redundancy_stop,
        consecutive_redundancy_min_step=args.consecutive_redundancy_min_step,
        loop_breaker_similarity_threshold=args.loop_breaker_similarity_threshold,
    )

    # Save results
    save_json(questions_data, args.output_file)

    logger.info("=== Filtering Complete ===")


if __name__ == "__main__":
    main()
