<div align="center">

# 🐾 PUMA

### Official implementation of [*Stop When Reasoning Converges: Semantic-Preserving Early Exit for Reasoning Models*](https://arxiv.org/abs/2605.17672)

[![Paper](https://img.shields.io/badge/arXiv-2605.17672-b31b1b.svg)](https://arxiv.org/abs/2605.17672)
[![Model](https://img.shields.io/badge/🤗-Redundancy%20Detector-yellow.svg)](https://huggingface.co/ZhishanQ/qwen3-embedding-redundancy-detector-0.6B)
[![Dataset](https://img.shields.io/badge/🤗-RD%20Training%20Data-yellow.svg)](https://huggingface.co/datasets/ZhishanQ/puma-rd-training-data)

</div>

PUMA (**P**rogress-aware **U**nified **M**onitoring framework for **A**daptive
early exit) is a plug-and-play framework that compresses long reasoning traces by
exploiting the **semantic redundancy** carried in the reasoning trajectory. It
pairs a lightweight **Redundancy Detector** (which flags semantically redundant
candidate exits) with **answer-level verification** (which confirms a stop is
safe), removing redundant continuation while preserving both final-answer
accuracy and a coherent, semantically complete reasoning prefix. Across five reasoning models and five
benchmarks, PUMA achieves **26.2% average token reduction** while preserving
accuracy and retained-CoT quality.

> The code released here is the **offline** version of PUMA; the online version
> is being prepared for release.

## 📁 Repository structure

| Path | Description |
| --- | --- |
| `puma/` | The offline pipeline stages: answer generation, step segmentation, redundancy detection, trial-answer verification, truncated-prefix regeneration, and statistics. |
| `run_pipeline.sh` | Config-driven entry point that runs the full pipeline for one (model, dataset). |
| `configs/` | Per-model hyperparameter configs (DS-7B/14B/32B, Nemotron-8B, Qwen3-30B-T; `_code` variants for code datasets). |
| `baselines/` | Efficient-reasoning baselines: DEER, Dynasor, CCoT, CoD, NoThinking, Plan&Budget, Answer Consistency (+ Full-CoT vanilla). |
| `puma_vl/` + `baselines_vl/` | Zero-shot vision-language variants of the pipeline and baselines. |
| `train_rd/` | Recipe + scripts to train the Redundancy Detector, plus a data sample. |
| `slurm/` | Generic SLURM template for running the pipeline on a cluster. |
| `DATA_SCHEMA.md` | Schema of every JSON file the pipeline reads and writes. |

## ⚙️ Installation

```bash
conda env create -f envs/env.yml
conda activate puma
```

This installs everything needed to run the offline pipeline, the baselines, and
the vision-language variants.

**Models.** PUMA loads two models with vLLM at inference time:

- a **reasoning model** that generates the reasoning trace and the trial/final
  answers (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`);
- the **Redundancy Detector**, our fine-tuned embedding model
  [`ZhishanQ/qwen3-embedding-redundancy-detector-0.6B`](https://huggingface.co/ZhishanQ/qwen3-embedding-redundancy-detector-0.6B)
  (downloaded automatically from the Hub).

The prompt formatting is implemented for the **DeepSeek-R1 /
DeepSeek-R1-Distill-Qwen, Qwen3, QwQ, and Nemotron** model families. To use a
different family, add a branch in `puma/prompt_utils.py`.

**Training only.** Re-training the Redundancy Detector (`train_rd/`) additionally
requires [ms-swift](https://github.com/modelscope/ms-swift) and `flash-attn`,
which are *not* in `env.yml`. They are not needed to run PUMA.

**Code datasets only.** Evaluating code correctness (LiveCodeBench, Step 6)
requires the external [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
repo; clone it and set `LCB_REPO` to the checkout. This is not needed for the
paper's math/GPQA results.

## 📦 Input format

The pipeline takes a benchmark file: a JSONL with one object per question
holding the `question` and its ground-truth `answer`:

```json
{"question": "Compute ...", "answer": "42"}
```

From this, **Step 1a** runs your reasoning model (with vLLM) to produce the full
chain-of-thought, written to `answers.json`; the rest of the pipeline operates on
those responses. If you **already have** generated responses, drop them in as
`<base_dir>/answers.json` (a JSON array of objects with the fields below) and
Step 1a is skipped automatically:

```json
[
  {
    "question": "Compute ...",
    "reasoning": "The model's full <think> reasoning text.",
    "raw_response": "The answer text after </think> only, NOT including the reasoning (the two are disjoint; token stats sum them). Falls back to this for segmentation when reasoning is empty.",
    "model_answer": "42",
    "ground_truth_answer": "42"
  }
]
```

See [`DATA_SCHEMA.md`](DATA_SCHEMA.md) for the schema of every intermediate file.

## 🚀 Run PUMA

Run the whole pipeline for one (model, dataset) with `run_pipeline.sh`. The
per-model config sets the hyperparameters tuned in the paper (Redundancy
Detector / Loop Breaker / verified early exit):

```bash
bash run_pipeline.sh configs/DS-7B.conf runs/ds7b_math500 \
     deepseek-ai/DeepSeek-R1-Distill-Qwen-7B math-500 data/math-500_test.jsonl
```

Arguments: `<config> <base_dir> <model> <dataset> [benchmark.jsonl]`. If
`<base_dir>/answers.json` already exists, the benchmark argument can be omitted
and answer generation is skipped.

**Multi-GPU.** vLLM tensor-parallel size is auto-detected from the visible GPUs,
so the larger models just need more GPUs made visible (e.g.
`CUDA_VISIBLE_DEVICES=0,1` for DS-32B / Qwen3-30B-T). To run on a cluster, adapt
[`slurm/puma_pipeline.slurm`](slurm/puma_pipeline.slurm).

After the run, `base_dir` contains (see [`DATA_SCHEMA.md`](DATA_SCHEMA.md)):

| File | Description |
| --- | --- |
| `answers.json` | Full chain-of-thought responses (Step 1a output / your input). |
| `steps.json` | Responses with reasoning split into steps. |
| `filtered_steps.json` | Per-step redundancy-detector decisions. |
| `trial_answers.json` | Short trial answers and confidence scores. |
| `final_candidates.json` | Selected stopping point for each question. |
| `prefixed_answers.json` | Final compressed answers from the truncated prefix. |
| `statistics.txt` | Accuracy, compression rate, and token reduction. |

The final console report prints original vs. compressed accuracy, accuracy
impact, compression rate, and token reduction.

## 🎛️ Main hyperparameters

These are set in the `configs/*.conf` file (the paper values):

| Config variable | Value | Description |
| --- | ---: | --- |
| `SIMILARITY_THRESHOLD` | `0.35` | τ_sim: cosine-similarity threshold for redundancy detection. |
| `CONSECUTIVE_REDUNDANCY_STOP` | `1`–`4` | m: consecutive redundant steps before the Loop Breaker forces a stop (`0` disables it). |
| `CONSECUTIVE_REDUNDANCY_MIN_STEP` | `50` | Late-stage activation: only detect consecutive redundancy after this step. |
| `FORCED_STOP_MIN_CONFIDENCE` | `0.8` | Weak minimum-confidence gate for the Loop Breaker. |
| `CONFIDENCE_THRESHOLD` | `0.98` | λ: minimum confidence required for verified early exit. |
| `EPSILON` | `0.03` | ε: maximum allowed confidence drop from the first verified candidate. |
| `CONSECUTIVE` | `2` | L: consecutive matching checkpoints required to stop. |
| `MAX_TRIAL_TOKENS` | `30` | Token budget for each trial answer. |

The Loop Breaker `m` (`CONSECUTIVE_REDUNDANCY_STOP`) is tuned per model: 1 for
DS-7B and Nemotron-8B, 3 for DS-14B, 4 for DS-32B, and 0 for Qwen3-30B-T (where
it is disabled). The window size (k) is 1.

## 📊 Baselines

`baselines/` provides runners for the efficient-reasoning baselines compared in
the paper (DEER, Dynasor, CCoT, CoD, NoThinking, Plan&Budget, Answer
Consistency). They are plain CLI scripts (no config files); each generates a
Full-CoT reference and then runs its method, reporting accuracy and compression.

The flags differ slightly by method — **DEER** reads a dataset directory, while
the others take a benchmark `.jsonl` file:

```bash
# DEER
python -m baselines.run_deer --model <model> \
    --dataset_dir data --dataset aime24 --output-dir runs/deer_aime24

# Dynasor / CCoT (run_concise) / CoD / NoThinking / Plan&Budget / Answer Consistency
python -m baselines.run_dynasor --model <model> \
    --benchmark data/aime24_test.jsonl --output-dir runs/dynasor_aime24
```

Pass `--vanilla-answers <file>` to reuse precomputed Full-CoT answers, or
`--eval-only` to recompute metrics without re-running inference. See
[`baselines/README.md`](baselines/README.md) for the per-method options and the
metric definitions (CR / CRT).

Zero-shot vision-language variants of PUMA and the baselines are in `puma_vl/`
and `baselines_vl/`. Run the VL pipeline just like the text one, via
`run_pipeline_vl.sh` (it reuses the modality-independent stages — step
segmentation, redundancy detection, stop decision — from `puma/`):

```bash
bash run_pipeline_vl.sh configs/DS-7B.conf runs/qwenvl8b_mathvista \
     Qwen/Qwen3-VL-8B-Thinking mathvista data/mathvista_test.jsonl
```

VL benchmarks (questions + image paths) can be built with
`python puma_vl/dataset_utils.py --dataset <dataset>` (mathvista, mathvision,
mmmu-pro).

## 🧠 Train the Redundancy Detector

The detector is a fine-tuned `Qwen3-Embedding-0.6B` trained with an InfoNCE
contrastive objective on reasoning-step novelty vs. redundancy. The trained model
and the full training data are released:

- Model: [`ZhishanQ/qwen3-embedding-redundancy-detector-0.6B`](https://huggingface.co/ZhishanQ/qwen3-embedding-redundancy-detector-0.6B)
- Data: [`ZhishanQ/puma-rd-training-data`](https://huggingface.co/datasets/ZhishanQ/puma-rd-training-data)

See `train_rd/` for the recipe, data-construction scripts, and a format sample.

## 🛠️ Troubleshooting

If the embedding stage fails inside vLLM with `EngineDeadError` or
`KeyError: None`, try disabling the vLLM V1 engine:

```bash
export VLLM_USE_V1=0
```

## 📚 Citation

```bibtex
@article{min2026stop,
  title={Stop When Reasoning Converges: Semantic-Preserving Early Exit for Reasoning Models},
  author={Min, Dehai and Vaccarino, Giovanni and Chen, Huiyi and Wu, Yongliang and Yona, Gal and Cheng, Lu},
  journal={arXiv preprint arXiv:2605.17672},
  year={2026}
}
```
