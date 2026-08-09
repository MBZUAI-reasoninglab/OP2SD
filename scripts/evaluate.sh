#!/usr/bin/env bash
set -euo pipefail

# Evaluate one base model or LoRA checkpoint over several decoding seeds, then
# aggregate the resulting JSON files. This launcher is sequential and uses
# tensor parallelism across the GPUs visible to the process.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID_OR_PATH="${MODEL_ID_OR_PATH:-Qwen/Qwen3-4B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
METHOD_TAG="${METHOD_TAG:-base}"
DATASET="${DATASET:-aime24}"
DATASET_PATH="${DATASET_PATH:-}"
SEEDS="${SEEDS:-1001 2002 1500 4004}"
VAL_N="${VAL_N:-12}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/eval_results}"

case "$ENABLE_THINKING" in
    0) MODE_TAG=nonthinking; MODE_ARGS=(--no_thinking) ;;
    1) MODE_TAG=thinking; MODE_ARGS=(--enable_thinking) ;;
    *) echo "ENABLE_THINKING must be 0 or 1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES
mkdir -p "$RESULT_DIR"

MODEL_TAG="${MODEL_ID_OR_PATH%/}"
MODEL_TAG="${MODEL_TAG##*/}"
MODEL_TAG="${MODEL_TAG//[^A-Za-z0-9._-]/_}"
METHOD_TAG="${METHOD_TAG//[^A-Za-z0-9._-]/_}"

COMMON_ARGS=(
    --base_model "$MODEL_ID_OR_PATH"
    --dataset "$DATASET"
    --val_n "$VAL_N"
    --temperature 1.0
    --top_k -1
    --min_p 0.0
    --presence_penalty 0.0
    --max_new_tokens "$MAX_NEW_TOKENS"
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    "${MODE_ARGS[@]}"
)
if [[ -n "$CHECKPOINT_DIR" ]]; then
    COMMON_ARGS+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi
if [[ -n "$DATASET_PATH" ]]; then
    COMMON_ARGS+=(--dataset_path "$DATASET_PATH")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
    COMMON_ARGS+=(--max_model_len "$MAX_MODEL_LEN")
fi

RESULT_FILES=()
for seed in $SEEDS; do
    output_file="$RESULT_DIR/eval_${DATASET}_${MODEL_TAG}_${METHOD_TAG}_${MODE_TAG}_valn${VAL_N}_seed${seed}.json"
    RESULT_FILES+=("$output_file")
    python "$ROOT_DIR/eval/evaluate_math.py" \
        "${COMMON_ARGS[@]}" \
        --seed "$seed" \
        --output_file "$output_file"
done

aggregate_file="$RESULT_DIR/multiseed_${DATASET}_${MODEL_TAG}_${METHOD_TAG}_${MODE_TAG}_valn${VAL_N}.json"
python "$ROOT_DIR/eval/aggregate_multiseed.py" \
    "${RESULT_FILES[@]}" \
    --label "$DATASET / $MODEL_TAG / $METHOD_TAG / $MODE_TAG" \
    --output "$aggregate_file"

echo "Aggregate results: $aggregate_file"
