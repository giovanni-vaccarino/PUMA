# Benchmarks

The five evaluation benchmarks used for the main results in the paper. Each file
is a JSONL with one question per line:

```json
{"question": "Compute ...", "answer": "42"}
```

| File | Questions | Task | Upstream source |
| --- | ---: | --- | --- |
| `aime24_test.jsonl` | 30 | math | AIME 2024 (I + II), from [`math-ai/aime24`](https://huggingface.co/datasets/math-ai/aime24) |
| `aime25_test.jsonl` | 30 | math | AIME 2025 (I + II), from [`math-ai/aime25`](https://huggingface.co/datasets/math-ai/aime25) |
| `math-500_test.jsonl` | 500 | math | MATH-500 ([`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)) — the 500-problem subset of MATH selected in *Let's Verify Step by Step*, via [DEER](https://github.com/iie-ycx/DEER) |
| `gpqa-diamond_test.jsonl` | 198 | multiple choice | GPQA Diamond, via DEER |
| `olympiadbench_test.jsonl` | 675 | math | OlympiadBench — English, text-only math subset, via DEER |

Original benchmark papers: MATH ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)),
MATH-500 ([Lightman et al., 2023](https://arxiv.org/abs/2305.20050)),
GPQA ([Rein et al., 2023](https://arxiv.org/abs/2311.12022)),
OlympiadBench ([He et al., 2024](https://arxiv.org/abs/2402.14008)).

## Fields

| Field | Description |
| --- | --- |
| `question` | The problem statement, with no instruction or answer-format suffix. PUMA appends the task-appropriate instruction itself in `puma/prompt_utils.py`. |
| `answer` | The ground-truth answer used for grading. For math this is the final expression (e.g. `70`, `\left( 3, \frac{\pi}{2} \right)`); for GPQA it is the choice letter (`A`–`D`). |

For `gpqa-diamond_test.jsonl`, the four options are inlined at the end of
`question` as `A. ... / B. ... / C. ... / D. ...`, so grading compares the
model's `\boxed{}` letter against `answer`. GPQA Diamond is the 198-question
expert-validated subset of the 448-question GPQA set.

For `olympiadbench_test.jsonl`, the parent benchmark is bilingual, multimodal,
and covers both mathematics and physics; this file is the 675-problem subset
that is English, text-only, and mathematics-only.

## Relation to the upstream releases

`math-500`, `gpqa-diamond` and `olympiadbench` were taken from the
[DEER](https://github.com/iie-ycx/DEER) repository (`data/{math,gpqa,olympiadbench}/test.jsonl`),
with the field `problem` renamed to `question`. The question text and the
ground-truth answers are unchanged and in the same order — with one exception:
for `gpqa-diamond`, DEER's trailing prompt instruction (*"Please reason
step-by-step and put your choice letter ... in the end."*) was stripped from
each question, because PUMA supplies its own instruction.

`aime24` and `aime25` were taken from the `math-ai` Hugging Face datasets, with
the field `problem` renamed to `question`. DEER also ships an AIME 2024 file
containing identical data; its AIME 2025 file covers the same 30 competition
problems but uses a different LaTeX rendering and ordering, and is *not* the
file used here.

## License and attribution

These files are redistributions of publicly released benchmarks, provided here
so the pipeline is runnable out of the box. Please cite the original benchmark
papers listed above. The DEER-derived files come from the DEER repository,
released under the MIT License:

```bibtex
@article{yang2025dynamic,
  title={Dynamic Early Exit in Reasoning Models},
  author={Yang, Chenxu and Si, Qingyi and Duan, Yongjie and Zhu, Zheliang and Zhu, Chenyu and Lin, Zheng and Cao, Li and Wang, Weiping},
  journal={arXiv preprint arXiv:2504.15895},
  year={2025}
}
```
