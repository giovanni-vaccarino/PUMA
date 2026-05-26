#!/bin/bash
# =============================================================================
# PUMA offline pipeline (config-driven).
#
# Runs the full pipeline on one (model, dataset):
#   Step 1a  Generate full-CoT answers          (run_vllm.py)      [optional]
#   Step 1b  Segment reasoning into steps        (separate_steps.py)
#   Step 1.5 Redundancy Detector filtering       (embed_and_filter_steps.py)
#   Step 2   Trial answers + confidence          (gen_trial_answers.py)
#   Step 3   Two-stage stopping decision         (extract_final_candidates.py)
#   Step 4   Regenerate from truncated prefix     (gen_prefixed_answers.py)
#   Step 5   Accuracy / token-reduction stats     (statistics_puma.py)
#   Step 6   Code correctness (code datasets only) (eval_livecodebench.py)
#
# Usage:
#   bash run_pipeline.sh <config> <base_dir> <model> <dataset> [benchmark.jsonl]
#
#   <config>     a file under configs/, e.g. configs/DS-7B.conf
#   <base_dir>   working directory; all intermediate + output files go here
#   <model>      HF model name or path (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   <dataset>    dataset name (aime24 | aime25 | gpqa-diamond | math-500 |
#                olympiadbench | livecodebench ...); drives prompting/grading
#   [benchmark]  optional JSONL of {question, answer}. If given and
#                <base_dir>/answers.json does NOT exist, Step 1a generates it.
#                If answers.json already exists, Step 1a is skipped.
#
# Example (generate answers, then run PUMA):
#   bash run_pipeline.sh configs/DS-7B.conf runs/ds7b_math500 \
#        deepseek-ai/DeepSeek-R1-Distill-Qwen-7B math-500 data/math-500_test.jsonl
#
# Example (answers.json already present):
#   bash run_pipeline.sh configs/DS-7B.conf runs/ds7b_math500 \
#        deepseek-ai/DeepSeek-R1-Distill-Qwen-7B math-500
#
# Multi-GPU: tensor-parallel size is auto-detected from CUDA_VISIBLE_DEVICES
# (set e.g. CUDA_VISIBLE_DEVICES=0,1 for the 30B/32B models).
# =============================================================================
set -euo pipefail

CONFIG="${1:?Usage: run_pipeline.sh <config> <base_dir> <model> <dataset> [benchmark.jsonl]}"
BASE_DIR="${2:?Usage: run_pipeline.sh <config> <base_dir> <model> <dataset> [benchmark.jsonl]}"
MODEL="${3:?Usage: run_pipeline.sh <config> <base_dir> <model> <dataset> [benchmark.jsonl]}"
DATASET="${4:?Usage: run_pipeline.sh <config> <base_dir> <model> <dataset> [benchmark.jsonl]}"
BENCHMARK_FILE="${5:-}"

# Run from the repository root so `python puma/<stage>.py` and the baselines
# import resolve regardless of the caller's working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python}"

# ---- Load config -----------------------------------------------------------
# shellcheck source=/dev/null
source "$CONFIG"
mkdir -p "$BASE_DIR"

# Defaults for knobs not set in the config (match the reference pipeline).
LOOP_BREAKER_SIMILARITY_THRESHOLD="${LOOP_BREAKER_SIMILARITY_THRESHOLD:-${SIMILARITY_THRESHOLD}}"
PROMPT_VERSION="${PROMPT_VERSION:-default}"
CONFIDENCE_MODE="${CONFIDENCE_MODE:-token_in_boxed}"
CONFIDENCE_AGGREGATION="${CONFIDENCE_AGGREGATION:-geometric}"
ALWAYS_CHECK_FIRST_N="${ALWAYS_CHECK_FIRST_N:-1}"
EMBEDDING_WINDOW_SIZE="${EMBEDDING_WINDOW_SIZE:-1}"
TRIGGER_MODE="${TRIGGER_MODE:-previous}"
MIN_STOP_STEP="${MIN_STOP_STEP:-10}"
ENABLE_EMBEDDING_FILTER="${ENABLE_EMBEDDING_FILTER:-true}"
ANSWER_GEN_MAX_TOKENS="${ANSWER_GEN_MAX_TOKENS:-32768}"
ANSWER_FIX_MAX_TOKENS="${ANSWER_FIX_MAX_TOKENS:-2048}"
MAX_TRIAL_TOKENS="${MAX_TRIAL_TOKENS:-30}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TRIAL_DECODING="${TRIAL_DECODING:-sampling}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-4096}"

# Tensor parallel size: auto-detect from visible GPUs.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TENSOR_PARALLEL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    TENSOR_PARALLEL_SIZE=1
fi

# Optional argument fragments.
SEED_ARG=""; [ -n "${SEED:-}" ] && SEED_ARG="--seed ${SEED}"
GPQA_SOFTMAX_ARG=""; [ -n "${GPQA_SOFTMAX_TEMPERATURE:-}" ] && GPQA_SOFTMAX_ARG="--gpqa-softmax-temperature ${GPQA_SOFTMAX_TEMPERATURE}"

# Output files (flat, inside base_dir).
ANSWERS="${BASE_DIR}/answers.json"
STEPS="${BASE_DIR}/steps.json"
FILTERED="${BASE_DIR}/filtered_steps.json"
TRIAL="${BASE_DIR}/trial_answers.json"
CANDIDATES="${BASE_DIR}/final_candidates.json"
PREFIXED="${BASE_DIR}/prefixed_answers.json"
STATS="${BASE_DIR}/statistics.txt"

echo "=================================================================="
echo "PUMA pipeline | model=$MODEL | dataset=$DATASET"
echo "config=$CONFIG | base_dir=$BASE_DIR | TP=$TENSOR_PARALLEL_SIZE"
echo "embedding_filter=$ENABLE_EMBEDDING_FILTER | m=$CONSECUTIVE_REDUNDANCY_STOP"
echo "=================================================================="

# ---- Step 1a: generate full-CoT answers (optional) -------------------------
if [ -f "$ANSWERS" ]; then
    echo ">>> Step 1a: skipped (answers.json already exists)"
elif [ -n "$BENCHMARK_FILE" ]; then
    echo ">>> Step 1a: generating answers from $BENCHMARK_FILE"
    GEN_OVERRIDES=""
    [ -n "${ANSWER_GEN_TEMPERATURE:-}" ] && GEN_OVERRIDES="$GEN_OVERRIDES --temperature $ANSWER_GEN_TEMPERATURE"
    [ -n "${ANSWER_GEN_TOP_P:-}" ] && GEN_OVERRIDES="$GEN_OVERRIDES --top-p $ANSWER_GEN_TOP_P"
    # shellcheck disable=SC2086
    "$PY" puma/run_vllm.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --dataset_path "$BENCHMARK_FILE" \
        --output_path "$ANSWERS" \
        --max_tokens "$ANSWER_GEN_MAX_TOKENS" \
        --answer_max_tokens "$ANSWER_FIX_MAX_TOKENS" \
        --prompt-version "$PROMPT_VERSION" \
        $GEN_OVERRIDES $SEED_ARG
else
    echo "ERROR: $ANSWERS not found and no benchmark file given."
    echo "       Provide a 5th argument (a JSONL of {question, answer}) to"
    echo "       generate answers, or place a pre-generated answers.json there."
    exit 1
fi

# ---- Step 1b: segment reasoning into steps ---------------------------------
echo ">>> Step 1b: separating reasoning steps"
"$PY" puma/separate_steps.py --input_json "$ANSWERS" --output_json "$STEPS" --workers 0

# ---- Step 1.5: Redundancy Detector filtering -------------------------------
CURRENT_STEPS="$STEPS"
TRIAL_ARGS=""
if [ "$ENABLE_EMBEDDING_FILTER" = "true" ]; then
    echo ">>> Step 1.5: redundancy-detector filtering"
    "$PY" puma/embed_and_filter_steps.py \
        --input-file "$STEPS" \
        --output-file "$FILTERED" \
        --embedding-model "$EMBEDDING_MODEL" \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        --always-check-first-n "$ALWAYS_CHECK_FIRST_N" \
        --window-size "$EMBEDDING_WINDOW_SIZE" \
        --trigger-mode "$TRIGGER_MODE" \
        --consecutive-redundancy-stop "$CONSECUTIVE_REDUNDANCY_STOP" \
        --consecutive-redundancy-min-step "$CONSECUTIVE_REDUNDANCY_MIN_STEP" \
        --loop-breaker-similarity-threshold "$LOOP_BREAKER_SIMILARITY_THRESHOLD"
    CURRENT_STEPS="$FILTERED"
    TRIAL_ARGS="--respect-embedding-filter"
else
    echo ">>> Step 1.5: skipped (embedding filter disabled)"
fi

# ---- Step 2: trial answers + confidence ------------------------------------
echo ">>> Step 2: generating trial answers"
# shellcheck disable=SC2086
"$PY" puma/gen_trial_answers.py \
    --questions-file "$CURRENT_STEPS" \
    --output-file "$TRIAL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TRIAL_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dataset "$DATASET" \
    --trial-decoding "$TRIAL_DECODING" \
    --confidence-mode "$CONFIDENCE_MODE" \
    --confidence-aggregation "$CONFIDENCE_AGGREGATION" \
    --prompt-version "$PROMPT_VERSION" \
    $TRIAL_ARGS $GPQA_SOFTMAX_ARG $SEED_ARG

# ---- Step 3: two-stage stopping decision -----------------------------------
echo ">>> Step 3: extracting final candidates"
FILTERED_STEPS_ARG=""
if [ "$ENABLE_EMBEDDING_FILTER" = "true" ]; then
    FILTERED_STEPS_ARG="--filtered-steps $FILTERED --forced-stop-min-confidence $FORCED_STOP_MIN_CONFIDENCE"
fi
CODE_SIM_ARG=""
[ -n "${CODE_SIMILARITY_THRESHOLD:-}" ] && CODE_SIM_ARG="--code-similarity-threshold $CODE_SIMILARITY_THRESHOLD"
# shellcheck disable=SC2086
"$PY" puma/extract_final_candidates.py \
    --input-file "$TRIAL" \
    --output-file "$CANDIDATES" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --epsilon "$EPSILON" \
    --consecutive "$CONSECUTIVE" \
    --min-stop-step "$MIN_STOP_STEP" \
    --dataset "$DATASET" \
    --workers 0 \
    $FILTERED_STEPS_ARG $CODE_SIM_ARG

# ---- Step 4: regenerate from truncated prefix ------------------------------
echo ">>> Step 4: generating prefixed answers"
CODE_GREEDY_ARG=""
[ "${CODE_GREEDY_GUIDED:-}" = "true" ] && CODE_GREEDY_ARG="--code-greedy-guided"
# shellcheck disable=SC2086
"$PY" puma/gen_prefixed_answers.py \
    --final-candidates "$CANDIDATES" \
    --questions-file "$CURRENT_STEPS" \
    --output-file "$PREFIXED" \
    --model "$MODEL" \
    --max-tokens "$MAX_ANSWER_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dataset "$DATASET" \
    --prompt-version "$PROMPT_VERSION" \
    --confidence-aggregation "$CONFIDENCE_AGGREGATION" \
    $SEED_ARG $CODE_GREEDY_ARG

# ---- Step 5: statistics ----------------------------------------------------
echo ">>> Step 5: computing statistics"
STATS_EMBEDDING_ARGS=""
if [ "$ENABLE_EMBEDDING_FILTER" = "true" ]; then
    STATS_EMBEDDING_ARGS="--enable-embedding-filter \
        --embedding-model $EMBEDDING_MODEL \
        --similarity-threshold $SIMILARITY_THRESHOLD \
        --always-check-first-n $ALWAYS_CHECK_FIRST_N \
        --embedding-window-size $EMBEDDING_WINDOW_SIZE \
        --trigger-mode $TRIGGER_MODE \
        --consecutive-redundancy-stop $CONSECUTIVE_REDUNDANCY_STOP \
        --consecutive-redundancy-min-step $CONSECUTIVE_REDUNDANCY_MIN_STEP \
        --trial-answers $TRIAL \
        --filtered-steps $FILTERED \
        --questions-with-steps $CURRENT_STEPS"
fi
# shellcheck disable=SC2086
"$PY" puma/statistics_puma.py \
    --original "$ANSWERS" \
    --compressed "$PREFIXED" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --confidence-threshold "$CONFIDENCE_THRESHOLD" \
    --epsilon "$EPSILON" \
    --consecutive "$CONSECUTIVE" \
    --confidence-mode "$CONFIDENCE_MODE" \
    --confidence-aggregation "$CONFIDENCE_AGGREGATION" \
    --min-stop-step "$MIN_STOP_STEP" \
    --trial-decoding "$TRIAL_DECODING" \
    --output "$STATS" \
    --workers 0 \
    $STATS_EMBEDDING_ARGS

# ---- Step 6: code correctness (code datasets only) -------------------------
TASK_TYPE=$("$PY" -c "import sys; sys.path.insert(0, 'puma'); from prompt_utils import get_task_type; print(get_task_type('$DATASET'))")
if [ "$TASK_TYPE" = "code" ]; then
    if [ -z "$BENCHMARK_FILE" ]; then
        echo ">>> Step 6: skipped (code dataset but no benchmark file given)"
    else
        echo ">>> Step 6: LiveCodeBench evaluation (requires LCB_REPO; see puma/eval_livecodebench.py)"
        "$PY" puma/eval_livecodebench.py \
            --predictions "$ANSWERS" --benchmark "$BENCHMARK_FILE" \
            --output "${BASE_DIR}/eval_original.json" --num-workers 30 --timeout 6
        "$PY" puma/eval_livecodebench.py \
            --predictions "$PREFIXED" --benchmark "$BENCHMARK_FILE" \
            --candidates "$CANDIDATES" \
            --output "${BASE_DIR}/eval_puma.json" --num-workers 30 --timeout 6
    fi
fi

echo "=================================================================="
echo "Pipeline complete. Outputs in: $BASE_DIR"
echo "Statistics: $STATS"
echo "=================================================================="
