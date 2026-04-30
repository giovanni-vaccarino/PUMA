<div align="center">

# 🐾 PUMA

### Official implementation for *Stop When Reasoning Converges: Semantic-Preserving Early Exit for Reasoning Models*

</div>

PUMA is a plug-and-play framework that compresses long reasoning traces by exploiting the semantic information carried in the reasoning trajectory. This repository contains the offline pipeline: it assumes full model responses have already been generated, then runs redundancy filtering, trial-answer verification, truncated-prefix regeneration, and final accuracy/compression statistics.

## ⚙️ Installation

```bash
conda env create -f envs/env.yml
conda activate puma
```

The pipeline uses vLLM for both generation and embedding inference. You need access to:

- the reasoning model used to produce trial and final answers;
- the embedding model used as the redundancy detector.

Supported reasoning-model prompt templates are currently tested for DeepSeek-R1 / DeepSeek-R1-Distill-Qwen, Qwen3, QwQ, and Nemotron-style models.

## 📦 Input Format

Create an experiment directory containing `answers.json`. The file must be a JSON array with one object per question:

```json
[
  {
    "question": "Compute ...",
    "reasoning": "Full reasoning text from the model's <think> block.",
    "raw_response": "Full model response, including reasoning and final answer.",
    "model_answer": "42",
    "ground_truth_answer": "42"
  }
]
```


## 🚀 Run PUMA

```bash
python run_puma.py \
  --base-dir /path/to/experiment \
  --model <model-name> \
  --embedding-model /path/to/puma-embedding-checkpoint
```

After the run, the experiment directory contains:

| File | Description |
| --- | --- |
| `steps.json` | Input examples with extracted reasoning steps. |
| `filtered_steps.json` | Step metadata from the redundancy detector. |
| `trial_answers.json` | Short trial answers and confidence scores. |
| `final_candidates.json` | Selected stopping point for each question. |
| `prefixed_answers.json` | Final compressed answers from truncated reasoning. |

The final console report prints original accuracy, compressed accuracy, accuracy impact, compression rate, and token reduction.

## 🎛️ Main Hyperparameters

| Flag | Default | Description |
| --- | ---: | --- |
| `--similarity-threshold` | `0.35` | Cosine-similarity threshold for redundancy detection. |
| `--window-size` | `1` | Number of previous reasoning steps compared by the redundancy detector. |
| `--consecutive-redundancy-stop` | `50` | Consecutive redundant steps before the loop breaker forces a stop. |
| `--confidence-threshold` | `0.98` | Minimum confidence required for verified early exit. |
| `--epsilon` | `0.03` | Maximum allowed confidence drop from the first verified candidate. |
| `--consecutive` | `2` | Consecutive matching checkpoints required to stop. |
| `--logprobs-mode` | unset | Optional vLLM logprobs mode, for example `processed_logprobs`. |

## 🛠️ Troubleshooting

If the embedding stage fails inside vLLM with `EngineDeadError` or `KeyError: None`, try disabling the vLLM V1 engine:

```bash
export VLLM_USE_V1=0
```

## 📚 Citation

Citation information will be added when the paper is released.
