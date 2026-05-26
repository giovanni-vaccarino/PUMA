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
| `puma/` | Core offline pipeline (step segmentation → redundancy detection → trial-answer verification → truncated-prefix regeneration → statistics). |
| `run_puma.py` | Entry point that runs the full offline pipeline. |
| `configs/` + `run_from_config.sh` | Per-model hyperparameter configs (DS-7B/14B/32B, Nemotron-8B, Qwen3-30B-T) and a launcher. |
| `baselines/` | Efficient-reasoning baselines: DEER, Dynasor, CCoT, CoD, NoThinking, Plan&Budget, Answer Consistency (+ Full-CoT vanilla). |
| `puma_vl/` + `baselines_vl/` | Zero-shot vision-language variants of the pipeline and baselines. |
| `train_rd/` | Recipe + scripts to train the Redundancy Detector, plus a data sample. |
| `slurm/` | Generic SLURM template for running the pipeline on a cluster. |

## ⚙️ Installation

```bash
conda env create -f envs/env.yml
conda activate puma
```

This installs everything needed to run the offline pipeline, the baselines, and
the vision-language variants.

**Models.** PUMA loads two models with vLLM at inference time:

- a **reasoning model** that produces the trial and final answers
  (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`);
- the **Redundancy Detector**, our fine-tuned embedding model
  [`ZhishanQ/qwen3-embedding-redundancy-detector-0.6B`](https://huggingface.co/ZhishanQ/qwen3-embedding-redundancy-detector-0.6B)
  (downloaded automatically from the Hub).

The trial-answer prompt formatting is implemented for the **DeepSeek-R1 /
DeepSeek-R1-Distill-Qwen, Qwen3, QwQ, and Nemotron** model families. To use a
different family, add a branch in `puma/prompt_utils.py`.

**Training only.** Re-training the Redundancy Detector (`train_rd/`) additionally
requires [ms-swift](https://github.com/modelscope/ms-swift) and `flash-attn`,
which are *not* in `env.yml`. They are not needed to run PUMA.

## 📦 Input format

The offline pipeline starts from reasoning responses that have **already been
generated**, so it does not call the reasoning model to produce them. First run
your reasoning model over your questions (e.g. with vLLM), then save the outputs
as `answers.json` in an experiment directory. The file is a JSON array with one
object per question:

```json
[
  {
    "question": "Compute ...",
    "reasoning": "The model's full <think> reasoning text.",
    "raw_response": "The full model response (reasoning + final answer); used as a fallback when `reasoning` is empty.",
    "model_answer": "42",
    "ground_truth_answer": "42"
  }
]
```

`question` and one of `reasoning` / `raw_response` are required. `model_answer`
and `ground_truth_answer` are used to report accuracy before vs. after
compression.

## 🚀 Run PUMA

Run PUMA through a per-model config — the config sets the hyperparameters tuned
for that model in the paper (RD / Loop Breaker / verified early exit):

```bash
bash run_from_config.sh configs/DS-7B.conf /path/to/experiment \
     deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

`run_from_config.sh` simply sources the config and calls `run_puma.py` with the
corresponding flags. You can also call `run_puma.py` directly if you want to set
every flag yourself (see [Main hyperparameters](#️-main-hyperparameters)):

```bash
python run_puma.py --base-dir /path/to/experiment \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --embedding-model ZhishanQ/qwen3-embedding-redundancy-detector-0.6B \
  --similarity-threshold 0.35 --consecutive-redundancy-stop 1 ...
```

**Multi-GPU.** vLLM tensor parallelism is auto-detected from the number of
visible GPUs, so the larger models just need more GPUs made visible (e.g.
`CUDA_VISIBLE_DEVICES=0,1` for DS-32B / Qwen3-30B-T); override with
`--tensor-parallel-size`. To run on a cluster, adapt
[`slurm/puma_pipeline.slurm`](slurm/puma_pipeline.slurm).

After the run, the experiment directory contains:

| File | Description |
| --- | --- |
| `steps.json` | Input examples with extracted reasoning steps. |
| `filtered_steps.json` | Step metadata from the redundancy detector. |
| `trial_answers.json` | Short trial answers and confidence scores. |
| `final_candidates.json` | Selected stopping point for each question. |
| `prefixed_answers.json` | Final compressed answers from truncated reasoning. |

The final console report prints original accuracy, compressed accuracy, accuracy
impact, compression rate, and token reduction.

## 🎛️ Main hyperparameters

| Flag | Default | Description |
| --- | ---: | --- |
| `--similarity-threshold` | `0.35` | τ_sim: cosine-similarity threshold for redundancy detection. |
| `--window-size` | `1` | k: number of previous reasoning steps compared by the detector. |
| `--consecutive-redundancy-stop` | `3` | m: consecutive redundant steps before the Loop Breaker forces a stop (`0` disables it). |
| `--consecutive-redundancy-min-step` | `50` | Late-stage activation: only detect consecutive redundancy after this step. |
| `--forced-stop-min-confidence` | `0.8` | Weak minimum-confidence gate for the Loop Breaker. |
| `--confidence-threshold` | `0.98` | λ: minimum confidence required for verified early exit. |
| `--epsilon` | `0.03` | Maximum allowed confidence drop from the first verified candidate. |
| `--consecutive` | `2` | L: consecutive matching checkpoints required to stop. |
| `--logprobs-mode` | unset | Optional vLLM logprobs mode, e.g. `processed_logprobs`. |

Per-model values used in the paper are in `configs/` (the Loop Breaker `m` is
tuned per model: 1 for DS-7B and Nemotron-8B, 3 for DS-14B, 4 for DS-32B, and 0
for Qwen3-30B-T where it is disabled).

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
and `baselines_vl/`.

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
