#!/usr/bin/env bash
set -euo pipefail

# Public launcher for the paper's OPSD baseline and OP²SD intervention.
# Activate the `op2sd` environment before running this script.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

METHOD="${METHOD:-op2sd}"                 # opsd | op2sd
MODEL_SIZE="${MODEL_SIZE:-4b}"            # 1.7b | 4b | 8b
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-siyanzhao/Openthoughts_math_30k_opsd}"
TRAIN_DATASET_SPLIT="${TRAIN_DATASET_SPLIT:-train}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-}"
DONOR_SOURCES="${DONOR_SOURCES:-amc_aime,aops_forum}"
DONOR_SEED="${DONOR_SEED:-1729}"
DONOR_DATASET_PATH="${DONOR_DATASET_PATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-OP2SD}"

case "$MODEL_SIZE" in
    1.7b)
        DEFAULT_MODEL="Qwen/Qwen3-1.7B"
        DEFAULT_BATCH_SIZE=4
        DEFAULT_GRAD_ACCUM=2
        DEFAULT_MAX_STEPS=100
        DEFAULT_CLIP=0.05
        DEFAULT_SCHEDULER=linear
        DEFAULT_STUDENT_THINKING=False
        DEFAULT_TEACHER_THINKING=True
        ;;
    4b)
        DEFAULT_MODEL="Qwen/Qwen3-4B"
        DEFAULT_BATCH_SIZE=4
        DEFAULT_GRAD_ACCUM=2
        DEFAULT_MAX_STEPS=100
        DEFAULT_CLIP=1e-6
        DEFAULT_SCHEDULER=constant
        DEFAULT_STUDENT_THINKING=False
        DEFAULT_TEACHER_THINKING=False
        ;;
    8b)
        DEFAULT_MODEL="Qwen/Qwen3-8B"
        DEFAULT_BATCH_SIZE=2
        DEFAULT_GRAD_ACCUM=4
        DEFAULT_MAX_STEPS=50
        DEFAULT_CLIP=1e-7
        DEFAULT_SCHEDULER=constant
        DEFAULT_STUDENT_THINKING=False
        DEFAULT_TEACHER_THINKING=False
        ;;
    *)
        echo "MODEL_SIZE must be one of: 1.7b, 4b, 8b (got: $MODEL_SIZE)" >&2
        exit 2
        ;;
esac

case "$METHOD" in
    opsd|op2sd) ;;
    *)
        echo "METHOD must be either opsd or op2sd (got: $METHOD)" >&2
        exit 2
        ;;
esac

MODEL_ID_OR_PATH="${MODEL_ID_OR_PATH:-$DEFAULT_MODEL}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRAD_ACCUM}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-$DEFAULT_CLIP}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-$DEFAULT_SCHEDULER}"
STUDENT_THINKING="${STUDENT_THINKING:-$DEFAULT_STUDENT_THINKING}"
TEACHER_THINKING="${TEACHER_THINKING:-$DEFAULT_TEACHER_THINKING}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
RUN_NAME="${RUN_NAME:-qwen3_${MODEL_SIZE}_${METHOD}_steps${MAX_STEPS}_seed${TRAIN_SEED}}"

if [[ -n "$DONOR_DATASET_PATH" && "$DONOR_SOURCES" != "amc_aime,aops_forum" ]]; then
    echo "Set either DONOR_DATASET_PATH or DONOR_SOURCES, not both." >&2
    exit 2
fi

DATASET_ARGS=(
    --train_dataset_name "$TRAIN_DATASET_NAME"
    --train_dataset_split "$TRAIN_DATASET_SPLIT"
)
if [[ -n "$TRAIN_DATASET_PATH" ]]; then
    DATASET_ARGS+=(--train_dataset_path "$TRAIN_DATASET_PATH")
fi

METHOD_ARGS=()
if [[ "$METHOD" == "op2sd" ]]; then
    METHOD_ARGS=(
        --shuffled_worked_example
        --shuffled_worked_example_seed "$DONOR_SEED"
    )
    if [[ -n "$DONOR_DATASET_PATH" ]]; then
        METHOD_ARGS+=(--shuffled_worked_example_donor_dataset_path "$DONOR_DATASET_PATH")
    else
        METHOD_ARGS+=(--shuffled_worked_example_donor_sources "$DONOR_SOURCES")
    fi
fi

export CUDA_VISIBLE_DEVICES WANDB_MODE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$OUTPUT_ROOT"

echo "Method:             $METHOD"
echo "Model:              $MODEL_ID_OR_PATH"
echo "GPUs:               $CUDA_VISIBLE_DEVICES"
echo "Output:             $OUTPUT_ROOT/$RUN_NAME"
echo "Student / teacher:  $STUDENT_THINKING / $TEACHER_THINKING"
echo "Steps / clip:       $MAX_STEPS / $JSD_TOKEN_CLIP"

accelerate launch \
    --config_file "$ROOT_DIR/accelerate.yaml" \
    --num_processes "$NUM_GPUS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "$ROOT_DIR/src/opsd_train.py" \
    --model_name_or_path "$MODEL_ID_OR_PATH" \
    --learning_rate 5e-6 \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --warmup_ratio 0 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --output_dir "$OUTPUT_ROOT" \
    --run_config "$RUN_NAME" \
    --num_train_epochs 30 \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps 2 \
    --seed "$TRAIN_SEED" \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length "$MAX_LENGTH" \
    --divergence_objective forward_kl \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --student_thinking "$STUDENT_THINKING" \
    --teacher_thinking "$TEACHER_THINKING" \
    --jsd_token_clip "$JSD_TOKEN_CLIP" \
    --wandb_project "$WANDB_PROJECT" \
    "${DATASET_ARGS[@]}" \
    "${METHOD_ARGS[@]}"
