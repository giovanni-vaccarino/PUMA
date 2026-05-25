# Training the Redundancy Detector (RD)

This directory contains the recipe and data-construction scripts for training
PUMA's Redundancy Detector — a fine-tuned `Qwen3-Embedding-0.6B` that scores
whether a reasoning step adds **novel progress** or merely **restates / loops
over** prior content.

- 🤖 Trained model: [ZhishanQ/qwen3-embedding-redundancy-detector-0.6B](https://huggingface.co/ZhishanQ/qwen3-embedding-redundancy-detector-0.6B)
- 📚 Full training data: [ZhishanQ/puma-rd-training-data](https://huggingface.co/datasets/ZhishanQ/puma-rd-training-data)

## Contents

| File | Purpose |
|------|---------|
| `train_rd.sh` | InfoNCE full fine-tuning recipe (ms-swift) |
| `sample_train.jsonl` | 30-row sample showing the contrastive data format |
| `create_v9_corrected_filt.py` | Builds the final training set by merging/filtering candidate pools |
| `mine_hard_negatives.py` | Mines hard negatives for each anchor step |
| `create_training_from_labels.py` | Turns novelty/redundancy labels into contrastive groups |

## Data format

Each line is one contrastive group (ms-swift embedding / InfoNCE format):

```json
{
  "messages":          [{"role": "user", "content": "<anchor reasoning step>"}],
  "positive_messages": [[{"role": "user", "content": "<redundant / restating step>"}]],
  "negative_messages": [[{"role": "user", "content": "<step that adds novel progress>"}], ...]
}
```

`positive_messages` is a **redundant** counterpart (should embed *close* to the
anchor); `negative_messages` are **novel** steps (should embed *far*). This way
high embedding similarity to recent steps indicates redundancy — the signal PUMA
uses at inference. Splits: 701,641 train / 8,233 dev / 8,299 test.

## Quick start

```bash
# 1. Get the data (or use your own jsonl in the format above)
huggingface-cli download ZhishanQ/puma-rd-training-data --repo-type dataset --local-dir data/rd

# 2. Train (requires ms-swift)
bash train_rd/train_rd.sh data/rd output/rd_full 3   # 3 = GPUs
```

The resulting checkpoint can be used directly as the `--embedding-model` in the
PUMA pipeline (see the repository root README).

## Note on data construction

The released dataset is the **final** training set. The construction scripts
here are provided as reference for the key steps (label → contrastive group →
hard-negative mining); the full multi-stage iteration that produced intermediate
candidate pools is omitted for brevity. To reproduce training, use the released
dataset directly.
