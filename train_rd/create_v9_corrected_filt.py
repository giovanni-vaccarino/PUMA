#!/usr/bin/env python3
"""
Create v9_corrected_filt dataset.

Combines:
- v7_all_models_augmented_more_negs (v7-aug-v2): 663,636 samples (CORRECTED, with more negatives)
- v6_augmented_v3_split: 52,225 samples (v4 base + GPT aug + label-verified neg)

Then filters out samples with 0 negatives.

Usage:
    python train_rd/create_v9_corrected_filt.py
"""

import json
import hashlib
import random
from pathlib import Path
from tqdm import tqdm


def load_jsonl(file_path):
    """Load JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples, file_path):
    """Save samples to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc=f"Saving {Path(file_path).name}"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(samples)} samples to {file_path}")


def sample_hash(sample):
    """Create a hash for a sample to detect duplicates."""
    anchor = sample['messages'][0]['content']
    positive = sample['positive_messages'][0][0]['content'] if sample.get('positive_messages') else ''
    key = f'{anchor[:500]}|||{positive[:500]}'
    return hashlib.md5(key.encode()).hexdigest()


def deduplicate_samples(samples):
    """Remove duplicate samples based on hash."""
    seen = set()
    unique = []
    duplicates = 0
    for s in tqdm(samples, desc="Deduplicating"):
        h = sample_hash(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)
        else:
            duplicates += 1
    print(f"Removed {duplicates} duplicates")
    return unique


def filter_zero_negatives(samples):
    """Filter out samples with 0 negatives."""
    filtered = []
    removed = 0
    for s in tqdm(samples, desc="Filtering 0-neg samples"):
        if len(s.get('negative_messages', [])) > 0:
            filtered.append(s)
        else:
            removed += 1
    print(f"Removed {removed} samples with 0 negatives")
    return filtered


def main():
    project_dir = Path(__file__).parent.parent

    # Input datasets
    v7_aug_v2_dir = project_dir / "data" / "v7_all_models_augmented_more_negs"
    v6_v3_split_dir = project_dir / "data" / "v6_augmented_v3_split"
    output_dir = project_dir / "data" / "v9_corrected_filt"

    print("=" * 70)
    print("Creating v9_corrected_filt dataset")
    print("=" * 70)
    print(f"\nOutput: {output_dir}")

    # Step 1: Load training data
    print("\n" + "=" * 70)
    print("Step 1: Loading datasets")
    print("=" * 70)

    v7_aug_v2_train = load_jsonl(v7_aug_v2_dir / "train.jsonl")
    print(f"  v7_all_models_augmented_more_negs train: {len(v7_aug_v2_train)}")

    v6_v3_train = load_jsonl(v6_v3_split_dir / "train.jsonl")
    print(f"  v6_augmented_v3_split train: {len(v6_v3_train)}")

    # Step 2: Combine
    print("\n" + "=" * 70)
    print("Step 2: Combining datasets")
    print("=" * 70)
    combined_train = v7_aug_v2_train + v6_v3_train
    print(f"  Combined (before dedup): {len(combined_train)}")

    # Step 3: Deduplicate
    print("\n" + "=" * 70)
    print("Step 3: Deduplicating")
    print("=" * 70)
    combined_train = deduplicate_samples(combined_train)
    print(f"  Combined (after dedup): {len(combined_train)}")

    # Step 4: Filter 0-neg samples
    print("\n" + "=" * 70)
    print("Step 4: Filtering 0-negative samples")
    print("=" * 70)
    before_filter = len(combined_train)
    combined_train = filter_zero_negatives(combined_train)
    print(f"  Before filter: {before_filter}")
    print(f"  After filter: {len(combined_train)}")

    # Step 5: Shuffle
    print("\n" + "=" * 70)
    print("Step 5: Shuffling")
    print("=" * 70)
    random.seed(42)
    random.shuffle(combined_train)
    print(f"  Shuffled {len(combined_train)} samples")

    # Step 6: Load and filter dev/test
    print("\n" + "=" * 70)
    print("Step 6: Loading and filtering dev/test")
    print("=" * 70)

    dev = load_jsonl(v7_aug_v2_dir / "dev.jsonl")
    test = load_jsonl(v7_aug_v2_dir / "test.jsonl")
    print(f"  dev (before filter): {len(dev)}")
    print(f"  test (before filter): {len(test)}")

    dev = filter_zero_negatives(dev)
    test = filter_zero_negatives(test)
    print(f"  dev (after filter): {len(dev)}")
    print(f"  test (after filter): {len(test)}")

    # Step 7: Save
    print("\n" + "=" * 70)
    print("Step 7: Saving")
    print("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(combined_train, output_dir / "train.jsonl")
    save_jsonl(dev, output_dir / "dev.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\nv9_corrected_filt:")
    print(f"  train: {len(combined_train)}")
    print(f"  dev: {len(dev)}")
    print(f"  test: {len(test)}")
    print(f"  Location: {output_dir}")

    # Statistics
    train_neg_counts = [len(s.get('negative_messages', [])) for s in combined_train]
    avg_neg = sum(train_neg_counts) / len(train_neg_counts) if train_neg_counts else 0

    print(f"\nTrain statistics:")
    print(f"  Avg negatives per sample: {avg_neg:.2f}")
    print(f"  Total negatives: {sum(train_neg_counts):,}")
    print(f"  Min negatives: {min(train_neg_counts)}")
    print(f"  Max negatives: {max(train_neg_counts)}")

    # Source breakdown
    print(f"\nSource breakdown:")
    print(f"  v7_all_models_augmented_more_negs: {len(v7_aug_v2_train)}")
    print(f"  v6_augmented_v3_split: {len(v6_v3_train)}")
    print(f"  Total before processing: {len(v7_aug_v2_train) + len(v6_v3_train)}")
    print(f"  After dedup + filter: {len(combined_train)}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
