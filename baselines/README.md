# Baselines

Baseline methods for token-efficient reasoning in LLMs. Each baseline outputs **Accuracy**, **CR** (Compression Rate), and **CRT** (Compression Rate with Trial answers).

**Metrics:**
- **CR** = compressed_tokens / original_vanilla_tokens
- **CRT** = (compressed_tokens + trial_answer_tokens) / original_vanilla_tokens

---

## Quick Start

### 1. Think Token Adjustment

```bash
python -m baselines.run_think_token_adjustment \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --benchmark "data/aime24_test.jsonl" \
    --output-dir "runs/baselines/think_token_adjustment/aime24"
```

**Note:** For this method, CR = CRT (no trial answers).

### 2. Answer Consistency

```bash
python -m baselines.run_answer_consistency \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --benchmark "data/aime24_test.jsonl" \
    --output-dir "runs/baselines/answer_consistency/aime24"
```

**Options:**
- `--threshold 10`: Number of consecutive identical answers to stop (default: 10)
- `--limit 30`: Process only first N questions
- `--eval-only`: Skip inference, only compute metrics
- `--vanilla-answers PATH`: Use precomputed vanilla answers

### 3. DEER (Dynamic Early Exit for Reasoning)
```bash
python -m baselines.run_deer \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset_dir data \
    --dataset aime24 \
    --output-dir "runs/baselines/deer/aime24"
```

**Options:**
- `--threshold 0.95`: Confidence threshold for early exit (default: 0.95)
- `--atp Wait`: Action Transition Point string — `Wait` or `Alternatively` (default: Wait)
- `--policy avg1`: Confidence calculation policy — `avg1`, `avg2`, or `min` (default: avg2)
- `--max-judge-steps 10`: Max confidence checks per question (default: 10)
- `--enable-repetition-exit`: Enable early exit on repetition detection
- `--eval-only`: Skip inference, only compute metrics
- `--vanilla-answers PATH`: Use precomputed vanilla answers

**Pipeline steps:**
1. Generate vanilla answers (full reasoning without intervention)
2. Run DEER inference (confidence-based early exit at ATPs)
3. Compute accuracy, CR, and CRT

---

## Example Output

```
============================================================
  AIME24 - DeepSeek-R1-Distill-Qwen-7B
============================================================
Accuracy:  18/30 = 60.00%
CR:        0.4523 (54.8% reduction)
CRT:       0.5891 (41.1% reduction)
============================================================
Avg original tokens:   8234.5
Avg compressed tokens: 3725.1
Avg trial tokens:      1126.3
============================================================
```
