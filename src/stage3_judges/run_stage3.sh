#!/usr/bin/env bash
# Stage 3: uncertainty zone, judge inputs, the six Qwen3.5-4B LoRA judges and their mean score (countries with training labels).
# Env: DATA_DIR, WORK_DIR, WEIGHTS_DIR, NGPU (default: all visible GPUs), QWEN_REVISION (pinned Qwen/Qwen3.5-4B revision).
#   default : scores the test zone with the released adapters in $WEIGHTS_DIR/qwen_judges.
#   TRAIN=1 : (after run_stage1.sh and run_stage2.sh with TRAIN=1)
#             1. builds the judges' training sets from the training data (build_training_sets.py)            [CPU]
#             2. builds the combiner's hold-out tables and the hold-out judge inputs (build_hold_tables.py)  [CPU]
#             3. checks that all of them contain only training records with their true labels                 [CPU]
#             4. trains the six judges into $WORK_DIR/models/qwen_judges (8 GPUs, torchrun; released adapters untouched) [GPU]
#             5. scores the test zone and the hold-out inputs with the retrained judges                          [GPU]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_DIR:?}" "${WORK_DIR:?}" "${WEIGHTS_DIR:?}"
export DATA_DIR WORK_DIR WEIGHTS_DIR
PY=${PY:-python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
mkdir -p "$WORK_DIR/stage3/inputs"
ADAPTERS=$WEIGHTS_DIR/qwen_judges

$PY -m stage3_judges.zone                    # CPU: zone records + zone candidate table
if [ "${TRAIN:-0}" = "1" ]; then
    $PY -m stage3_judges.build_training_sets                 # WORK_DIR/stage3/train/{plain,context}.parquet
    $PY -m stage4_combiner.build_hold_tables                 # WORK_DIR/stage4/hold/*.parquet, WORK_DIR/stage3/inputs/hold_*.parquet
    $PY -m stage3_judges.check_training_sets --stored "$WORK_DIR"
    ADAPTERS=$WORK_DIR/models/qwen_judges
fi
$PY -m stage3_judges.build_inputs            # CPU: plain + context inputs for every input file (test, and hold-out with TRAIN=1)

declare -A TRAINSET=([judge_plain_1]=plain [judge_plain_2]=plain [judge_context_1]=context [judge_context_2]=context
                     [judge_context_3]=context [judge_context_4]=context)
JUDGES="judge_plain_1 judge_plain_2 judge_context_1 judge_context_2 judge_context_3 judge_context_4"

if [ "${TRAIN:-0}" = "1" ]; then
    # recipe in train_lora.py (lr 2e-4, batch 256, one epoch); rank / alpha / input length per judge from judge_model.JUDGES
    for j in $JUDGES; do
        [ -f "$ADAPTERS/$j/score_head.safetensors" ] && continue          # resume: a finished judge is not retrained
        torchrun --standalone --nproc_per_node "$NGPU" -m stage3_judges.train_lora --judge "$j" \
            --train "$WORK_DIR/stage3/train/${TRAINSET[$j]}.parquet" --out "$ADAPTERS/$j"
    done
fi

for j in $JUDGES; do                         # GPU: plain judges read the plain inputs, context judges the *_context inputs
    case "$j" in judge_plain_*) suf="" ;; *) suf="_context" ;; esac
    ins=$(ls "$WORK_DIR"/stage3/inputs/*.parquet | grep -v "_context.parquet$" | sed "s/\.parquet$/${suf}.parquet/")
    for k in $(seq 0 $((NGPU - 1))); do
        CUDA_VISIBLE_DEVICES=$k $PY -m stage3_judges.score_lora --judge "$j" --adapter "$ADAPTERS/$j" --inputs $ins --shard "$k" --nshard "$NGPU" &
    done
    wait
done

$PY -m stage3_judges.merge_scores --name mean_six --judges judge_plain_1,judge_plain_2,judge_context_1,judge_context_2,judge_context_3,judge_context_4
echo "stage 3 done"
