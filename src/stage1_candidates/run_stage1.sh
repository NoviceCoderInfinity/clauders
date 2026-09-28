#!/usr/bin/env bash
# Stage 1: data preparation, normalisation, texts, retriever, candidate generation (blocking).
# usage (from src/):  DATA_DIR=... WORK_DIR=... WEIGHTS_DIR=... bash stage1_candidates/run_stage1.sh
#   default : uses the fine-tuned retriever in WEIGHTS_DIR/encoders/retriever_e5_small; builds test candidates only.
#   TRAIN=1 : also retrains the retriever from intfloat/multilingual-e5-small and builds the training-pool candidates and the
#             listwise-reranker training queries (needed to retrain the stage-2 models).
# Steps: [CPU] / [GPU] / [GPU-train]. Seeds fixed everywhere (42 and derived).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}; export PYTHONHASHSEED=42
W=${WORK_DIR:?set WORK_DIR}; WTS=${WEIGHTS_DIR:?set WEIGHTS_DIR}; export WORK_DIR DATA_DIR WEIGHTS_DIR
NG=${NGPU:-$(nvidia-smi -L | wc -l)}; E=$W/data/emb; TAG=main
RET=$WTS/encoders/retriever_e5_small

$PY $HERE/prepare_data.py                                                                 # [CPU] TSV -> parquet, train pairs
$PY $HERE/split_train.py                                                                  # [CPU] ENC / MT / HOLD split, training pool
$PY $HERE/build_norm.py                                                                   # [CPU] normalisation + transliteration / state tables
$PY $HERE/make_texts.py && POOL=1 $PY $HERE/make_texts.py                                 # [CPU] "name | address" texts
if [ "${TRAIN:-0}" = 1 ]; then
  $PY $HERE/build_encoder_data.py                                                         # [CPU]
  $PY $HERE/train_retriever.py --data $W/data/encoder_train --out $W/models/retriever      # [GPU-train] two phases, hard negatives
  RET=$W/models/retriever/phase2
  $PY $HERE/make_list_queries.py                                                          # [CPU] reranker training queries
fi

enc() { for g in $(seq 0 $((NG-1))); do (for f in text_full text_name text_addr; do CUDA_VISIBLE_DEVICES=$g $PY $HERE/encode.py $RET $1 $f $g $NG $E; done) & done; wait; }
ITEMS="s1_test test_q"; [ "${TRAIN:-0}" = 1 ] && ITEMS="s1_train pool_q enc_q $ITEMS"
for it in $ITEMS; do enc $it; done                                                        # [GPU] embeddings
if [ "${TRAIN:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES=0 $PY $HERE/retrieve_dense.py $E enc_q s1_train text_full 20 $TAG
  PAIRS="pool_q:s1_train test_q:s1_test"
else
  PAIRS="test_q:s1_test"
fi
for it in $PAIRS; do q=${it%%:*}; d=${it##*:}                                             # [GPU + CPU] dense top-20 + BM25 name/address
  CUDA_VISIBLE_DEVICES=0 $PY $HERE/retrieve_dense.py $E $q $d text_full 20 $TAG
  $PY $HERE/retrieve_sparse.py $q $d 20 $TAG
  CUDA_VISIBLE_DEVICES=0 $PY $HERE/merge_candidates.py $E $q $d $TAG
done
echo "stage 1 done: $W/data/cand/$TAG"
