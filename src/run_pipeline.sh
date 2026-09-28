#!/usr/bin/env bash
# End to end: competition data (DATA_DIR/train, DATA_DIR/test) -> OUTPUT_DIR/matching_results.tsv + OUTPUT_DIR/candidate_pairs.tsv.
# Default = inference with the released fine-tuned weights (WEIGHTS_DIR); TRAIN=1 retrains the models where the stage supports it.
# GPU steps need CUDA 12.8+ builds of torch; CPU steps are marked in the stage scripts.
#
# usage (from src/):  DATA_DIR=/path/to/dataset WEIGHTS_DIR=/path/to/weights WORK_DIR=/path/to/work OUTPUT_DIR=/path/to/output bash run_pipeline.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; PY=${PY:-python}
export DATA_DIR=${DATA_DIR:?set DATA_DIR} WEIGHTS_DIR=${WEIGHTS_DIR:?set WEIGHTS_DIR} WORK_DIR=${WORK_DIR:?set WORK_DIR}
export OUTPUT_DIR=${OUTPUT_DIR:-$HERE/../../../output} PYTHONHASHSEED=42 PY
: "${QWEN_REVISION:?set QWEN_REVISION (README section 3)}"
mkdir -p "$OUTPUT_DIR"

# ---- stage 1: normalisation, texts, e5-small retrieval (blocking)                                        [GPU + CPU]
bash stage1_candidates/run_stage1.sh
# ---- stage 2: cross-encoder and listwise rerankers, LightGBM matchers, first-stage decisions; normalised candidates and the
#      unseen-country matcher for countries without training labels                                     [GPU + CPU]
bash stage2_rerank_match/run_stage2.sh
# ---- stage 3: uncertainty zone and the six Qwen3.5-4B LoRA judges (countries with training labels)       [GPU + CPU]
bash stage3_judges/run_stage3.sh
# ---- stage 4: first-pass combiner and zone decision (countries with training labels)                     [CPU]
bash stage4_combiner/run_stage4.sh
# ---- stage 5: decision set of the countries without training labels (first-stage unseen-country matcher) [CPU]
$PY -m stage5_unlabelled.select_pairs
# ---- stage 6: candidate set (every pair the matchers scored) and matching results                         [CPU]
$PY -m stage6_output.candidate_pairs
$PY -m stage6_output.assemble

VALIDATOR=${VALIDATOR:-$(dirname "$DATA_DIR")/utils/validate_submission.py}
if [ -f "$VALIDATOR" ]; then
  $PY "$VALIDATOR" --matching "$OUTPUT_DIR/matching_results.tsv" --candidate "$OUTPUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test" --check-ids
fi
