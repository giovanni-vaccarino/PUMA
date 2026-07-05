#!/usr/bin/env python3
"""
Dataset utilities for PUMA-VL: download, preprocess, and save VL benchmarks.

Supported datasets:
  - MathVista (testmini, ~1000 samples)
  - MathVision (full, ~3040 samples)
  - MMMU-Pro (standard 4-option, ~1730 samples)

Output: JSONL file + images directory, compatible with the VL pipeline.

Usage:
    python puma_vl/dataset_utils.py --dataset mathvista \
        --output-dir data/benchmark_vl \
        --images-dir data/images
    python puma_vl/dataset_utils.py --dataset mathvision ...
    python puma_vl/dataset_utils.py --dataset mmmu-pro ...
    python puma_vl/dataset_utils.py --dataset all ...
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

NUM_WORKERS = 16


def _save_image(args):
    """Worker: save a single image to disk."""
    img, img_path = args
    if img is not None:
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img.save(img_path)
    return img_path


# =============================================================================
# MathVista
# =============================================================================

def load_mathvista(output_dir: str, images_dir: str, limit: int = -1):
    """
    Load MathVista testmini split.

    HuggingFace: AI4Math/MathVista
    ~1000 samples, mixed free-form + multi-choice.
    """
    print("Loading MathVista (testmini)...")
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    if limit > 0 and limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(limit))
        print(f"  Subsampled to {limit} samples (seed=42)")

    img_dir = os.path.join(images_dir, "mathvista")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "mathvista_test.jsonl")

    # Collect all entries and image save tasks
    entries = []
    img_tasks = []

    for idx, sample in enumerate(ds):
        img_filename = f"{idx:04d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img = sample.get("decoded_image") or sample.get("image")
        img_tasks.append((img, img_path))

        question = sample.get("question", "")
        choices = sample.get("choices")
        question_type = sample.get("question_type", "free_form")
        if choices and isinstance(choices, list) and len(choices) > 0:
            choice_labels = "ABCDEFGHIJ"
            choice_str = "\n".join(
                f"{choice_labels[i]}. {c}" for i, c in enumerate(choices)
            )
            question = f"{question}\n\n{choice_str}"

        raw_answer = str(sample.get("answer", ""))
        if question_type == "multi_choice" and choices and isinstance(choices, list):
            answer_letter = None
            for i, c in enumerate(choices):
                if str(c).strip() == raw_answer.strip():
                    answer_letter = "ABCDEFGHIJ"[i]
                    break
            if answer_letter is None:
                raw_lower = raw_answer.strip().lower()
                for i, c in enumerate(choices):
                    if raw_lower == str(c).strip().lower():
                        answer_letter = "ABCDEFGHIJ"[i]
                        break
            gt_answer = answer_letter if answer_letter else raw_answer
        else:
            gt_answer = raw_answer

        entries.append({
            "question": question,
            "answer": gt_answer,
            "answer_content": raw_answer,
            "image_path": img_path,
            "question_type": question_type,
            "answer_type": sample.get("answer_type", ""),
            "dataset": "mathvista",
            "idx": idx,
        })

    # Save images in parallel
    print(f"  Saving {len(img_tasks)} images with {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap(_save_image, img_tasks), total=len(img_tasks), desc="MathVista images"))

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"MathVista: saved {len(entries)} samples to {out_path}")
    print(f"Images: {img_dir}")
    return out_path


# =============================================================================
# MathVision
# =============================================================================

def load_mathvision(output_dir: str, images_dir: str, limit: int = -1):
    """
    Load MathVision dataset.

    HuggingFace: MathLLMs/MathVision
    ~3040 samples, free-form math with images.
    """
    print("Loading MathVision...")
    ds = load_dataset("MathLLMs/MathVision", split="test")
    if limit > 0 and limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(limit))
        print(f"  Subsampled to {limit} samples (seed=42)")

    img_dir = os.path.join(images_dir, "mathvision")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "mathvision_test.jsonl")

    entries = []
    img_tasks = []

    for idx, sample in enumerate(ds):
        img_filename = f"{idx:04d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img = sample.get("decoded_image") or sample.get("image")
        img_tasks.append((img, img_path))

        question = sample.get("question", "")
        answer = str(sample.get("answer", ""))

        entry = {
            "question": question,
            "answer": answer,
            "image_path": img_path,
            "question_type": "free_form",
            "dataset": "mathvision",
            "idx": idx,
        }
        if sample.get("level"):
            entry["level"] = sample["level"]
        if sample.get("subject"):
            entry["subject"] = sample["subject"]
        entries.append(entry)

    print(f"  Saving {len(img_tasks)} images with {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap(_save_image, img_tasks), total=len(img_tasks), desc="MathVision images"))

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"MathVision: saved {len(entries)} samples to {out_path}")
    print(f"Images: {img_dir}")
    return out_path


# =============================================================================
# MMMU-Pro
# =============================================================================

def load_mmmu_pro(output_dir: str, images_dir: str, limit: int = -1):
    """
    Load MMMU-Pro standard (4 options).

    HuggingFace: MMMU/MMMU_Pro
    ~1730 samples, 4-option MCQ across multiple college-level subjects.
    """
    print("Loading MMMU-Pro (standard, 4 options)...")
    ds = load_dataset("MMMU/MMMU_Pro", "standard (4 options)", split="test")
    if limit > 0 and limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(limit))
        print(f"  Subsampled to {limit} samples (seed=42)")

    img_dir = os.path.join(images_dir, "mmmu-pro")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "mmmu-pro_test.jsonl")

    entries = []
    img_tasks = []

    for idx, sample in enumerate(ds):
        # Find first available image
        img = None
        for img_key in ["image_1", "image", "decoded_image"]:
            img = sample.get(img_key)
            if img is not None:
                break
        if img is None:
            continue

        img_filename = f"{idx:04d}.png"
        img_path = os.path.join(img_dir, img_filename)
        img_tasks.append((img, img_path))

        question = sample.get("question", "")
        options = sample.get("options", [])
        if options and isinstance(options, list):
            choice_labels = "ABCDEFGHIJ"
            choice_str = "\n".join(
                f"{choice_labels[i]}. {opt}"
                for i, opt in enumerate(options)
            )
            question = f"{question}\n\n{choice_str}"

        answer = str(sample.get("answer", ""))

        entry = {
            "question": question,
            "answer": answer,
            "image_path": img_path,
            "question_type": "multi_choice",
            "dataset": "mmmu-pro",
            "idx": idx,
        }
        if sample.get("subject"):
            entry["subject"] = sample["subject"]
        entries.append(entry)

    print(f"  Saving {len(img_tasks)} images with {NUM_WORKERS} workers...")
    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap(_save_image, img_tasks), total=len(img_tasks), desc="MMMU-Pro images"))

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"MMMU-Pro: saved {len(entries)} samples to {out_path}")
    print(f"Images: {img_dir}")
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and prepare VL benchmark datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mathvista", "mathvision", "mmmu-pro", "all"],
                        help="Which dataset to prepare")
    parser.add_argument("--output-dir", type=str,
                        default="data/benchmark_vl",
                        help="Directory for output JSONL files")
    parser.add_argument("--images-dir", type=str,
                        default="data/images",
                        help="Directory for saved images")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max samples per dataset (-1 = all). "
                             "Subsamples with seed=42 for reproducibility.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers for image saving (default: 16)")
    args = parser.parse_args()

    global NUM_WORKERS
    NUM_WORKERS = args.workers

    if args.dataset in ("mathvista", "all"):
        load_mathvista(args.output_dir, args.images_dir, args.limit)
    if args.dataset in ("mathvision", "all"):
        load_mathvision(args.output_dir, args.images_dir, args.limit)
    if args.dataset in ("mmmu-pro", "all"):
        load_mmmu_pro(args.output_dir, args.images_dir, args.limit)


if __name__ == "__main__":
    main()
