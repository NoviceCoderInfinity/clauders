#!/usr/bin/env bash
# Stage 4: first-pass combiner and the decision set of the countries with training labels (CPU).
# Env: DATA_DIR, WORK_DIR, WEIGHTS_DIR.
#   default : scores the test zone with the released combiner models (WEIGHTS_DIR/lightgbm/first_pass, 12 cross-fitted models).
#   TRAIN=1 : (after run_stage3.sh with TRAIN=1, which built WORK_DIR/stage4/hold/ and scored the hold-out inputs)
#             checks the lookalike house-number offsets against the training labels (derive_up_steps.py), retrains the combiner
#             on the hold-out split into WORK_DIR/stage4/first_pass and prints the validation table (evaluate_holdout.py).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_DIR:?}" "${WORK_DIR:?}" "${WEIGHTS_DIR:?}"
export DATA_DIR WORK_DIR WEIGHTS_DIR
PY=${PY:-python}
W4=$WORK_DIR/stage4; mkdir -p "$W4/first_pass"
MEAN6=$WORK_DIR/stage3/scores/mean_six

$PY -m stage4_combiner.set_size_prior                                   # training histogram of match counts per Source 1 entity
if [ "${TRAIN:-0}" = "1" ]; then
    [ -f "$W4/hold/hold_candidates.parquet" ] || { echo "run run_stage3.sh with TRAIN=1 first (it builds $W4/hold)"; exit 1; }
    $PY -m stage4_combiner.derive_up_steps --check
    $PY -m stage4_combiner.train_combiner --preset first_pass --seeds 1,2,3,4,5,6 --out "$W4/first_pass" --judge-mean "$MEAN6"
    $PY -m stage4_combiner.evaluate_holdout            # validation F0.5 table and the (t, m) grid (methodology document, Section 5)
else
    $PY -m stage4_combiner.train_combiner --preset first_pass --seeds 1,2,3,4,5,6 --out "$W4/first_pass" --judge-mean "$MEAN6" \
        --models "$WEIGHTS_DIR/lightgbm/first_pass"
fi
$PY -m stage4_combiner.rebuild_decisions
echo "stage 4 done: $W4/us_india_pairs.parquet"
