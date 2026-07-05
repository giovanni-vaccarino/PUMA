#!/usr/bin/env python3
"""
Create training dataset from answers and labels files.

This script converts similarity labels into training data format:
- anchor (messages): prev_step
- positive_messages: steps that are REDUNDANT (result=0) relative to anchor
- negative_messages: steps that are NOVEL (result=1) relative to anchor

Usage:
    python train_rd/create_training_from_labels.py \
        --answers_dir data/answers \
        --labels_dir data/labels \
        --output_dir data/v8_all_models
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_json(file_path: str) -> list:
    """Load JSON or JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # Try JSON array first
        if content.startswith('['):
            return json.loads(content)
        # Otherwise treat as JSONL
        else:
            return [json.loads(line) for line in content.split('\n') if line.strip()]


def save_jsonl(samples: list, file_path: str):
    """Save samples to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def get_matching_answers_file(labels_file: Path, answers_dir: Path) -> Path:
    """
    Find the matching answers file for a labels file.

    Labels file: deepseek-r1-aime25_labels.json
    Answers file: deepseek-r1-aime25_answers.json
    """
    # Extract dataset name from labels file
    name = labels_file.stem  # e.g., "deepseek-r1-aime25_labels"
    dataset_name = name.replace("_labels", "")  # e.g., "deepseek-r1-aime25"

    # Try different possible answers file names
    candidates = [
        answers_dir / f"{dataset_name}_answers.json",
        answers_dir / f"{dataset_name}.json",
    ]

    # Handle special cases like gpt-oss-aime25_v2 -> gpt-oss-aime25_answers_v2.json
    if "_v2" in dataset_name:
        base_name = dataset_name.replace("_v2", "")
        candidates.append(answers_dir / f"{base_name}_answers_v2.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def create_training_samples(answers: list, labels: list, max_negatives: int = 50) -> list:
    """
    Create training samples from answers and labels.

    Args:
        answers: List of questions with reasoning_steps
        labels: List of label entries with question_idx, prev_step_idx, current_step_idx, result
        max_negatives: Maximum number of negatives per sample

    Returns:
        List of training samples
    """
    # Build question index (1-indexed in labels)
    questions = {i + 1: q for i, q in enumerate(answers)}

    # Group labels by (question_idx, prev_step_idx) - this is the anchor
    anchor_groups = defaultdict(lambda: {'positives': [], 'negatives': []})

    for label in labels:
        q_idx = label['question_idx']
        prev_idx = label['prev_step_idx']
        curr_idx = label['current_step_idx']
        result = label['result']

        key = (q_idx, prev_idx)

        if result == 0:  # Redundant - similar to anchor
            anchor_groups[key]['positives'].append(curr_idx)
        else:  # Novel - different from anchor
            anchor_groups[key]['negatives'].append(curr_idx)

    # Create training samples
    samples = []
    skipped = 0

    for (q_idx, prev_idx), group in anchor_groups.items():
        question = questions.get(q_idx)
        if not question:
            skipped += 1
            continue

        steps = question.get('reasoning_steps', [])
        if prev_idx >= len(steps):
            skipped += 1
            continue

        # Get anchor text
        anchor_text = steps[prev_idx]

        # Get positive texts (redundant steps)
        positive_texts = []
        for idx in group['positives']:
            if idx < len(steps):
                positive_texts.append(steps[idx])

        # Get negative texts (novel steps)
        negative_texts = []
        for idx in group['negatives']:
            if idx < len(steps):
                negative_texts.append(steps[idx])

        # Skip if no positives (we need at least one positive for contrastive learning)
        if not positive_texts:
            skipped += 1
            continue

        # Limit negatives
        if len(negative_texts) > max_negatives:
            negative_texts = negative_texts[:max_negatives]

        # Create sample
        sample = {
            'messages': [{"role": "user", "content": anchor_text}],
            'positive_messages': [[{"role": "user", "content": text}] for text in positive_texts],
            'negative_messages': [[{"role": "user", "content": text}] for text in negative_texts],
        }
        samples.append(sample)

    return samples, skipped


def main():
    parser = argparse.ArgumentParser(description="Create training data from labels")
    parser.add_argument("--answers_dir", type=str, default="data/answers",
                        help="Directory containing answers files")
    parser.add_argument("--labels_dir", type=str, default="data/labels",
                        help="Directory containing labels files")
    parser.add_argument("--output_dir", type=str, default="data/v8_all_models",
                        help="Output directory for training data")
    parser.add_argument("--max_negatives", type=int, default=50,
                        help="Maximum negatives per sample (default: 50)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of data for training (default: 0.9)")
    args = parser.parse_args()

    answers_dir = Path(args.answers_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    # Find all labels files
    labels_files = sorted(labels_dir.glob("*_labels.json"))
    print(f"Found {len(labels_files)} labels files")

    all_samples = []
    stats = {}

    for labels_file in tqdm(labels_files, desc="Processing datasets"):
        dataset_name = labels_file.stem.replace("_labels", "")

        # Find matching answers file
        answers_file = get_matching_answers_file(labels_file, answers_dir)
        if not answers_file:
            print(f"  Warning: No answers file found for {labels_file.name}")
            continue

        # Load data
        answers = load_json(str(answers_file))
        labels = load_json(str(labels_file))

        # Create training samples
        samples, skipped = create_training_samples(
            answers, labels, max_negatives=args.max_negatives
        )

        all_samples.extend(samples)

        # Statistics
        total_positives = sum(len(s['positive_messages']) for s in samples)
        total_negatives = sum(len(s['negative_messages']) for s in samples)

        stats[dataset_name] = {
            'labels': len(labels),
            'samples': len(samples),
            'skipped': skipped,
            'positives': total_positives,
            'negatives': total_negatives,
        }

    print(f"\nTotal samples: {len(all_samples)}")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_samples)

    n_train = int(len(all_samples) * args.train_ratio)
    n_dev = (len(all_samples) - n_train) // 2
    n_test = len(all_samples) - n_train - n_dev

    train_samples = all_samples[:n_train]
    dev_samples = all_samples[n_train:n_train + n_dev]
    test_samples = all_samples[n_train + n_dev:]

    print(f"Train: {len(train_samples)}")
    print(f"Dev: {len(dev_samples)}")
    print(f"Test: {len(test_samples)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_samples, str(output_dir / "train.jsonl"))
    save_jsonl(dev_samples, str(output_dir / "dev.jsonl"))
    save_jsonl(test_samples, str(output_dir / "test.jsonl"))

    # Print statistics
    print("\n" + "=" * 70)
    print("Per-Dataset Statistics")
    print("=" * 70)

    total_samples = 0
    total_positives = 0
    total_negatives = 0

    for name, s in sorted(stats.items()):
        print(f"\n{name}:")
        print(f"  Labels: {s['labels']:,}")
        print(f"  Samples: {s['samples']:,}")
        print(f"  Skipped: {s['skipped']:,}")
        print(f"  Positives: {s['positives']:,}")
        print(f"  Negatives: {s['negatives']:,}")
        if s['samples'] > 0:
            print(f"  Avg pos/sample: {s['positives']/s['samples']:.1f}")
            print(f"  Avg neg/sample: {s['negatives']/s['samples']:.1f}")

        total_samples += s['samples']
        total_positives += s['positives']
        total_negatives += s['negatives']

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total samples: {total_samples:,}")
    print(f"Total positives: {total_positives:,}")
    print(f"Total negatives: {total_negatives:,}")
    if total_samples > 0:
        print(f"Avg positives/sample: {total_positives/total_samples:.1f}")
        print(f"Avg negatives/sample: {total_negatives/total_samples:.1f}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
