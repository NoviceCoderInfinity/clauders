#!/usr/bin/env bash
# CPU-only re-run of the decision steps of a finished run_pipeline.sh run (no GPU, minutes). It reads nothing downloaded:
# its inputs are the files that run_pipeline.sh itself wrote into RUN_DIR (= the WORK_DIR of that run). Use it to re-run the
# CPU decision steps (combiner, zone decision, unlabelled-country pairs, candidate set, matching results) without repeating the
# GPU scoring.
#
# usage (from src/):  DATA_DIR=/path/to/dataset WEIGHTS_DIR=/path/to/weights RUN_DIR=/path/to/work bash rebuild_outputs.sh
#   OUTPUT_DIR  default $RUN_DIR/rebuild/output
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; PY=${PY:-python}
export DATA_DIR=${DATA_DIR:?set DATA_DIR} WEIGHTS_DIR=${WEIGHTS_DIR:?set WEIGHTS_DIR}
R=${RUN_DIR:?set RUN_DIR to the WORK_DIR of a finished run_pipeline.sh run}
export WORK_DIR=$R/rebuild OUTPUT_DIR=${OUTPUT_DIR:-$R/rebuild/output} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-16}
W=$WORK_DIR; mkdir -p "$W/stage2" "$W/stage3" "$W/stage4" "$W/stage5" "$OUTPUT_DIR"
for f in test_candidates test_record_index test_source1_index; do ln -sfn "$R/stage2/$f.parquet" "$W/stage2/$f.parquet"; done
for f in test_zone_records test_zone_candidates; do ln -sfn "$R/stage3/$f.parquet" "$W/stage3/$f.parquet"; done

# 1. countries with training labels: set-size prior, first-pass combiner on the test zone (released models), zone decision
$PY -m stage4_combiner.set_size_prior --out "$W/stage4/set_size_prior.npy"
$PY -m stage4_combiner.train_combiner --preset first_pass --seeds 1,2,3,4,5,6 --work "$W" --out "$W/stage4/first_pass" \
    --judge-mean "$R/stage3/scores/mean_six" --models "$WEIGHTS_DIR/lightgbm/first_pass"
$PY -m stage4_combiner.rebuild_decisions --work "$W" --out "$W/stage4/us_india_pairs.parquet"

# 2. countries without training labels: first-stage decisions of the unseen-country matcher
$PY -m stage5_unlabelled.select_pairs --accepted "$R/stage2/matcher_accepted_pairs.parquet" --stage2-dir "$R/stage2" \
    --out "$W/stage5/unlabelled_pairs.parquet"

# 3. outputs
$PY -m stage6_output.candidate_pairs --stage2-dir "$R/stage2" --unlabelled-scores "$R/stage2/first_stage/test_scores.parquet" \
    --decisions "$W/stage4/us_india_pairs.parquet,$W/stage5/unlabelled_pairs.parquet" --out "$OUTPUT_DIR/candidate_pairs.tsv"
$PY -m stage6_output.assemble --labelled "$W/stage4/us_india_pairs.parquet" --unlabelled "$W/stage5/unlabelled_pairs.parquet" \
    --out "$OUTPUT_DIR/matching_results.tsv"

VALIDATOR=${VALIDATOR:-$(dirname "$DATA_DIR")/utils/validate_submission.py}
if [ -f "$VALIDATOR" ]; then
  $PY "$VALIDATOR" --matching "$OUTPUT_DIR/matching_results.tsv" --candidate "$OUTPUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test" --check-ids
fi
(command -v md5sum >/dev/null && md5sum "$OUTPUT_DIR"/*.tsv) || md5 "$OUTPUT_DIR"/*.tsv
