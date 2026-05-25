#!/usr/bin/env python3
"""
Hard Negative Mining for Redundancy Detection Training.

This script implements Option 1: Pre-compute hard negatives by finding
semantically similar steps from OTHER questions/samples.

Algorithm:
1. Load training data and extract ALL unique texts (anchors + positives + negatives)
2. Build a candidate pool containing ALL steps from all samples
3. Compute embeddings using current best model
4. For each anchor, find top-k similar steps from OTHER samples
   (any step from a different sample can be a hard negative, regardless of its
   original role as anchor/positive/negative in that sample)
5. Add these as additional hard negatives

Key insight: A step's label (redundant/novel) within its own question is irrelevant
for other questions. Any step from Q2 that is similar to Q1's anchor but comes from
a different context is a valid hard negative for Q1.

Usage:
    python scripts/mine_hard_negatives.py \
        --input_dataset data/v6_augmented \
        --output_dataset data/v7_hard_negatives \
        --model_path ZhishanQ/qwen3-embedding-redundancy-detector-0.6B \
        --top_k 3 \
        --min_similarity 0.3 \
        --max_similarity 0.8

Author: Claude Code
"""

import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import shutil


def load_jsonl(file_path: str) -> list:
    """Load JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list, file_path: str):
    """Save samples to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc=f"Saving to {Path(file_path).name}"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def text_hash(text: str) -> str:
    """Create hash for text to identify unique texts."""
    return hashlib.md5(text.encode()).hexdigest()


def extract_text_from_messages(messages) -> str:
    """Extract text content from messages format."""
    if isinstance(messages, list):
        if len(messages) > 0:
            if isinstance(messages[0], dict):
                return messages[0].get('content', '')
            elif isinstance(messages[0], list):
                # nested list format: [[{"role": "user", "content": "..."}]]
                return extract_text_from_messages(messages[0])
    return ''


def build_text_pools(samples: list) -> tuple:
    """
    Build pools of anchor texts and ALL candidate texts for hard negative mining.

    We collect ALL texts from all samples:
    - anchors (messages)
    - positives (redundant steps)
    - negatives (novel steps)

    Any text from a DIFFERENT sample can serve as a hard negative,
    regardless of its original role (anchor/positive/negative).

    Returns:
        anchor_texts: list of (sample_idx, text, text_hash)
        candidate_texts: list of (sample_idx, text, text_hash) - ALL unique texts
        text_to_samples: dict mapping text_hash to set of sample indices
    """
    anchor_texts = []
    all_texts = []  # Collect ALL texts
    text_to_samples = defaultdict(set)

    for idx, sample in enumerate(tqdm(samples, desc="Building text pools")):
        # Extract anchor
        anchor = extract_text_from_messages(sample.get('messages', []))
        if anchor:
            h = text_hash(anchor)
            anchor_texts.append((idx, anchor, h))
            all_texts.append((idx, anchor, h))
            text_to_samples[h].add(idx)

        # Extract positives (redundant steps)
        for pos in sample.get('positive_messages', []):
            pos_text = extract_text_from_messages(pos)
            if pos_text:
                h = text_hash(pos_text)
                all_texts.append((idx, pos_text, h))
                text_to_samples[h].add(idx)

        # Extract negatives (novel steps)
        for neg in sample.get('negative_messages', []):
            neg_text = extract_text_from_messages(neg)
            if neg_text:
                h = text_hash(neg_text)
                all_texts.append((idx, neg_text, h))
                text_to_samples[h].add(idx)

    print(f"  Anchors: {len(anchor_texts)}")
    print(f"  All texts (with duplicates): {len(all_texts)}")

    # Deduplicate candidate texts while keeping sample associations
    seen_hashes = set()
    unique_candidates = []
    for item in all_texts:
        if item[2] not in seen_hashes:
            seen_hashes.add(item[2])
            unique_candidates.append(item)

    print(f"  Unique candidate texts: {len(unique_candidates)}")

    return anchor_texts, unique_candidates, text_to_samples


def compute_embeddings_batch(model, texts: list, batch_size: int = 32) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        model: EmbeddingModelVLLM instance
        texts: List of text strings
        batch_size: Batch size for processing (default 32 for stability with LoRA)

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        # Process one by one if batch fails (vLLM LoRA bug workaround)
        try:
            embeddings = model.get_embeddings_batch(batch)
        except Exception as e:
            print(f"\nBatch failed, processing individually: {e}")
            batch_embeddings = []
            for text in batch:
                try:
                    emb = model.get_embedding(text)
                    batch_embeddings.append(emb)
                except Exception as e2:
                    print(f"  Single text failed, using zeros: {e2}")
                    batch_embeddings.append(np.zeros(1024))  # fallback
            embeddings = np.vstack(batch_embeddings)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def find_hard_negatives_faiss(
    anchor_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    anchor_infos: list,
    candidate_infos: list,
    text_to_samples: dict,
    top_k: int = 3,
    min_similarity: float = 0.3,
    max_similarity: float = 0.8,
) -> dict:
    """
    Find hard negatives using FAISS for efficient similarity search.

    Args:
        anchor_embeddings: Embeddings for anchor texts
        candidate_embeddings: Embeddings for ALL candidate texts (anchors + positives + negatives)
        anchor_infos: List of (sample_idx, text, hash) for anchors
        candidate_infos: List of (sample_idx, text, hash) for ALL candidates
        text_to_samples: Dict mapping text_hash to sample indices
        top_k: Number of hard negatives to find per anchor
        min_similarity: Minimum cosine similarity threshold
        max_similarity: Maximum cosine similarity (to avoid exact matches)

    Returns:
        Dict mapping sample_idx to list of hard negative texts
    """
    import faiss

    # Build FAISS index for all candidate texts
    print("Building FAISS index for all candidate texts...")
    dim = candidate_embeddings.shape[1]

    # Use Inner Product (cosine similarity since vectors are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(candidate_embeddings.astype(np.float32))

    # Search for each anchor
    print(f"Searching for hard negatives (top_k={top_k}, sim=[{min_similarity}, {max_similarity}])...")

    # Search more candidates than needed to filter later
    search_k = top_k * 10
    similarities, indices = index.search(
        anchor_embeddings.astype(np.float32),
        search_k
    )

    # Process results
    hard_negatives = defaultdict(list)
    stats = {'found': 0, 'skipped_same_sample': 0, 'skipped_low_sim': 0, 'skipped_high_sim': 0}

    for anchor_idx, (sample_idx, anchor_text, anchor_hash) in enumerate(tqdm(anchor_infos, desc="Processing results")):
        found_for_anchor = 0

        for rank in range(search_k):
            if found_for_anchor >= top_k:
                break

            candidate_idx = indices[anchor_idx, rank]
            sim = similarities[anchor_idx, rank]

            # Check similarity bounds
            if sim < min_similarity:
                stats['skipped_low_sim'] += 1
                continue
            if sim > max_similarity:
                stats['skipped_high_sim'] += 1
                continue

            # Get candidate text info
            candidate_sample_idx, candidate_text, candidate_hash = candidate_infos[candidate_idx]

            # Skip if from same sample (or samples sharing this text)
            samples_with_candidate = text_to_samples.get(candidate_hash, set())
            if sample_idx in samples_with_candidate:
                stats['skipped_same_sample'] += 1
                continue

            # Skip if same as anchor
            if candidate_hash == anchor_hash:
                continue

            # Add as hard negative
            hard_negatives[sample_idx].append({
                'text': candidate_text,
                'similarity': float(sim),
            })
            found_for_anchor += 1
            stats['found'] += 1

    print(f"\nHard negative mining stats:")
    print(f"  Found: {stats['found']}")
    print(f"  Skipped (same sample): {stats['skipped_same_sample']}")
    print(f"  Skipped (low sim < {min_similarity}): {stats['skipped_low_sim']}")
    print(f"  Skipped (high sim > {max_similarity}): {stats['skipped_high_sim']}")
    print(f"  Samples with hard negatives: {len(hard_negatives)}")

    return hard_negatives


def find_hard_negatives_numpy(
    anchor_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    anchor_infos: list,
    candidate_infos: list,
    text_to_samples: dict,
    top_k: int = 3,
    min_similarity: float = 0.3,
    max_similarity: float = 0.8,
) -> dict:
    """
    Find hard negatives using numpy (fallback if FAISS not available).
    Slower but works without additional dependencies.
    """
    print("Computing similarity matrix (this may take a while)...")

    # Compute all pairwise similarities
    # anchor_embeddings: (N_anchor, D)
    # candidate_embeddings: (N_candidate, D)
    # similarity: (N_anchor, N_candidate)

    hard_negatives = defaultdict(list)
    batch_size = 1000  # Process in batches to avoid memory issues

    for start_idx in tqdm(range(0, len(anchor_infos), batch_size), desc="Finding hard negatives"):
        end_idx = min(start_idx + batch_size, len(anchor_infos))
        batch_anchors = anchor_embeddings[start_idx:end_idx]

        # Compute similarities for this batch
        similarities = np.dot(batch_anchors, candidate_embeddings.T)

        for i, anchor_idx in enumerate(range(start_idx, end_idx)):
            sample_idx, anchor_text, anchor_hash = anchor_infos[anchor_idx]
            sims = similarities[i]

            # Get indices sorted by similarity (descending)
            sorted_indices = np.argsort(-sims)

            found = 0
            for candidate_idx in sorted_indices:
                if found >= top_k:
                    break

                sim = sims[candidate_idx]

                if sim < min_similarity or sim > max_similarity:
                    continue

                candidate_sample_idx, candidate_text, candidate_hash = candidate_infos[candidate_idx]

                # Skip if from same sample
                samples_with_candidate = text_to_samples.get(candidate_hash, set())
                if sample_idx in samples_with_candidate:
                    continue

                if candidate_hash == anchor_hash:
                    continue

                hard_negatives[sample_idx].append({
                    'text': candidate_text,
                    'similarity': float(sim),
                })
                found += 1

    print(f"Samples with hard negatives: {len(hard_negatives)}")
    return hard_negatives


def augment_samples_with_hard_negatives(
    samples: list,
    hard_negatives: dict,
) -> list:
    """
    Add hard negatives to training samples.

    Args:
        samples: Original training samples
        hard_negatives: Dict mapping sample_idx to list of hard negative info

    Returns:
        Augmented samples with additional negative_messages
    """
    import copy
    augmented = []
    added_count = 0

    for idx, sample in enumerate(tqdm(samples, desc="Augmenting samples")):
        # Use deep copy to avoid modifying original samples
        new_sample = copy.deepcopy(sample)

        if idx in hard_negatives:
            # Get existing negatives (now a separate copy)
            existing_negatives = new_sample.get('negative_messages', [])

            # Add hard negatives
            for hn in hard_negatives[idx]:
                new_negative = [{"role": "user", "content": hn['text']}]
                existing_negatives.append(new_negative)
                added_count += 1

            new_sample['negative_messages'] = existing_negatives

        augmented.append(new_sample)

    print(f"Added {added_count} hard negatives to {len(hard_negatives)} samples")
    return augmented


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives for redundancy detection training"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="data/v6_augmented",
        help="Input dataset directory (default: data/v6_augmented)",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="data/v7_hard_negatives",
        help="Output dataset directory (default: data/v7_hard_negatives)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ZhishanQ/qwen3-embedding-redundancy-detector-0.6B",
        help="Path to embedding model checkpoint",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of hard negatives to add per sample (default: 3)",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.3,
        help="Minimum cosine similarity for hard negatives (default: 0.3)",
    )
    parser.add_argument(
        "--max_similarity",
        type=float,
        default=0.8,
        help="Maximum cosine similarity for hard negatives (default: 0.8)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding computation (default: 64)",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        default=True,
        help="Use FAISS for similarity search (default: True)",
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default=None,
        help="Path to save computed embeddings for reuse",
    )
    parser.add_argument(
        "--load_embeddings",
        type=str,
        default=None,
        help="Path to load pre-computed embeddings",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="Backend for embedding computation (default: vllm)",
    )
    args = parser.parse_args()

    # Paths
    input_dir = Path(args.input_dataset)
    output_dir = Path(args.output_dataset)

    print("=" * 70)
    print("Hard Negative Mining for Redundancy Detection")
    print("=" * 70)
    print(f"Input dataset: {input_dir}")
    print(f"Output dataset: {output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Top-K: {args.top_k}")
    print(f"Similarity range: [{args.min_similarity}, {args.max_similarity}]")
    print()

    # Step 1: Load training data
    print("=" * 70)
    print("Step 1: Loading training data")
    print("=" * 70)
    train_path = input_dir / "train.jsonl"
    samples = load_jsonl(str(train_path))
    print(f"Loaded {len(samples)} training samples")

    # Step 2: Build text pools
    print("\n" + "=" * 70)
    print("Step 2: Building text pools (ALL texts from all samples)")
    print("=" * 70)
    anchor_infos, candidate_infos, text_to_samples = build_text_pools(samples)

    # Extract just the texts for embedding
    anchor_texts = [info[1] for info in anchor_infos]
    candidate_texts = [info[1] for info in candidate_infos]

    # Step 3: Compute embeddings
    print("\n" + "=" * 70)
    print("Step 3: Computing embeddings")
    print("=" * 70)

    if args.load_embeddings:
        print(f"Loading pre-computed embeddings from {args.load_embeddings}")
        data = np.load(args.load_embeddings)
        anchor_embeddings = data['anchor_embeddings']
        candidate_embeddings = data['candidate_embeddings']
        print(f"  Anchor embeddings: {anchor_embeddings.shape}")
        print(f"  Candidate embeddings: {candidate_embeddings.shape}")
    else:
        # Load embedding model
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        print(f"Loading embedding model from {args.model_path} (backend: {args.backend})")

        if args.backend == "vllm":
            from inference.embed_mdh import EmbeddingModelVLLM
            model = EmbeddingModelVLLM(lora_path=args.model_path)

            print("\nComputing anchor embeddings...")
            anchor_embeddings = compute_embeddings_batch(
                model, anchor_texts, batch_size=args.batch_size
            )
            print(f"  Shape: {anchor_embeddings.shape}")

            print("\nComputing candidate embeddings (all texts)...")
            candidate_embeddings = compute_embeddings_batch(
                model, candidate_texts, batch_size=args.batch_size
            )
            print(f"  Shape: {candidate_embeddings.shape}")
        else:
            # Transformers backend (more stable with LoRA)
            from inference.embed_mdh import load_embedding_model, get_embeddings_batch

            model, tokenizer, device = load_embedding_model(args.model_path)

            print("\nComputing anchor embeddings...")
            anchor_embeddings = get_embeddings_batch(
                model, tokenizer, anchor_texts, device, batch_size=args.batch_size
            )
            print(f"  Shape: {anchor_embeddings.shape}")

            print("\nComputing candidate embeddings (all texts)...")
            candidate_embeddings = get_embeddings_batch(
                model, tokenizer, candidate_texts, device, batch_size=args.batch_size
            )
            print(f"  Shape: {candidate_embeddings.shape}")

        if args.save_embeddings:
            print(f"\nSaving embeddings to {args.save_embeddings}")
            np.savez(
                args.save_embeddings,
                anchor_embeddings=anchor_embeddings,
                candidate_embeddings=candidate_embeddings,
            )

    # Step 4: Find hard negatives
    print("\n" + "=" * 70)
    print("Step 4: Finding hard negatives from ALL other samples")
    print("=" * 70)

    try:
        import faiss
        use_faiss = args.use_faiss
    except ImportError:
        print("FAISS not available, using numpy fallback")
        use_faiss = False

    if use_faiss:
        hard_negatives = find_hard_negatives_faiss(
            anchor_embeddings=anchor_embeddings,
            candidate_embeddings=candidate_embeddings,
            anchor_infos=anchor_infos,
            candidate_infos=candidate_infos,
            text_to_samples=text_to_samples,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            max_similarity=args.max_similarity,
        )
    else:
        hard_negatives = find_hard_negatives_numpy(
            anchor_embeddings=anchor_embeddings,
            candidate_embeddings=candidate_embeddings,
            anchor_infos=anchor_infos,
            candidate_infos=candidate_infos,
            text_to_samples=text_to_samples,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            max_similarity=args.max_similarity,
        )

    # Step 5: Augment samples
    print("\n" + "=" * 70)
    print("Step 5: Augmenting samples with hard negatives")
    print("=" * 70)
    augmented_samples = augment_samples_with_hard_negatives(samples, hard_negatives)

    # Step 6: Save output dataset
    print("\n" + "=" * 70)
    print("Step 6: Saving output dataset")
    print("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save augmented train
    save_jsonl(augmented_samples, str(output_dir / "train.jsonl"))

    # Copy dev and test
    for split in ["dev.jsonl", "test.jsonl"]:
        src = input_dir / split
        dst = output_dir / split
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {split}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Input samples: {len(samples)}")
    print(f"Output samples: {len(augmented_samples)}")
    print(f"Samples with hard negatives: {len(hard_negatives)}")

    # Count total negatives before and after
    total_neg_before = sum(len(s.get('negative_messages', [])) for s in samples)
    total_neg_after = sum(len(s.get('negative_messages', [])) for s in augmented_samples)
    print(f"Total negatives before: {total_neg_before}")
    print(f"Total negatives after: {total_neg_after}")
    print(f"Hard negatives added: {total_neg_after - total_neg_before}")

    print(f"\nOutput saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
