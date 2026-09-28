#!/usr/bin/env bash
# Stage 2: pair features, cross-encoder and listwise reranker scores, LightGBM matchers, the normalised route of the countries without
# training labels,
# first-stage decisions and the test tables used by the later stages (WORK_DIR/stage2/).
# usage (from src/):  DATA_DIR=... WORK_DIR=... WEIGHTS_DIR=... bash stage2_rerank_match/run_stage2.sh   (after run_stage1.sh)
#   default : uses the fine-tuned models in WEIGHTS_DIR (encoders/, lightgbm/).
#   TRAIN=1 : retrains the e5-small cross-encoder, the three listwise rerankers, the core-name reranker and both matchers.
# Decision rules: US / India (countries with training labels) matcher_us_india, p >= 0.50 and margin >= 0.30;
#                 countries without training labels (France here): matcher_france, trained on US / India without the country, on the
#                 candidates of france_canon.py's normalised texts, p >= 0.60. Caps 5 Source 2 / 6 Source 3.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); S1=$HERE/../stage1_candidates; PY=${PY:-python}; export PYTHONHASHSEED=42
W=${WORK_DIR:?set WORK_DIR}; WTS=${WEIGHTS_DIR:?set WEIGHTS_DIR}; export WORK_DIR DATA_DIR WEIGHTS_DIR
NG=${NGPU:-$(nvidia-smi -L | wc -l)}; TAG=main; R=$W/reranker; OUT2=$W/stage2; FS=$OUT2/first_stage; mkdir -p $R $FS
ENC=$WTS/encoders; LGB=$WTS/lightgbm
CE_MODEL=$ENC/cross_encoder_e5_small; RET=$ENC/retriever_e5_small
declare -A RR=([listwise_full]=$ENC/reranker_listwise_bge_m3 [specialist_name]=$ENC/specialist_name_e5_base
               [specialist_address]=$ENC/specialist_address_e5_base [core_name]=$ENC/reranker_core_name_bge_m3)
declare -A FIELD=([listwise_full]=full [specialist_name]=name [specialist_address]=addr [core_name]=core)
M_USIN=$LGB/matcher_us_india; M_FR=$LGB/matcher_france

# ---- features and e5-small cross-encoder                                                          [CPU] / [GPU]
$PY $HERE/features.py test_q s1_test $TAG
[ "${TRAIN:-0}" = 1 ] && $PY $HERE/features.py pool_q s1_train $TAG
$PY $HERE/build_cross_encoder_data.py
if [ "${TRAIN:-0}" = 1 ]; then
  RET=$W/models/retriever/phase2
  $PY $HERE/train_cross_encoder.py --data $W/data/cross_encoder --hn $W/models/retriever/hard_negatives.parquet --out $W/cross_encoder
  CE_MODEL=$W/cross_encoder/model                                                                  # [GPU-train] + scores
else
  CE_SETS=test_q:s1_test $PY $HERE/train_cross_encoder.py --data $W/data/cross_encoder --out $W/cross_encoder --skip_train --model $CE_MODEL
fi
$PY $HERE/cross_encoder_table.py $W/data/cross_encoder/pairs_test_q.parquet $W/cross_encoder/scores test_q $W/cross_encoder/test_ce.parquet

# ---- listwise rerankers                                                                            [GPU-train] / [GPU]
if [ "${TRAIN:-0}" = 1 ]; then
  $PY $HERE/build_lists.py
  T() { [ -f $R/$1/model/config.json ] || torchrun --standalone --nproc_per_node=$NG $HERE/listwise_reranker.py train ${@:4} --backbone $2 --lists $W/data/reranker/lists.parquet --out $R/$1 --max_q $3; }
  T listwise_full BAAI/bge-reranker-v2-m3 0 --bs_q 32 --lr 2e-5                                     # all lists
  T specialist_name intfloat/multilingual-e5-base 300000 --bs_q 32 --field name
  T specialist_address intfloat/multilingual-e5-base 300000 --bs_q 32 --field addr
  T core_name BAAI/bge-reranker-v2-m3 500000 --bs_q 32 --lr 2e-5 --field core                      # name without legal-form / filler words
  for n in listwise_full specialist_name specialist_address core_name; do RR[$n]=$R/$n/model; done
fi
S() {   # S <reranker> <pairs.parquet> <query texts> <Source 1 texts> <out stem>
  $PY $HERE/listwise_reranker.py score --field ${FIELD[$1]} --model ${RR[$1]} --pairs $2 --qtext $3 --dtext $4 --out $R/$1/$5.npy --bs 512
  $PY -c "import numpy as np, polars as pl; P=pl.read_parquet('$2'); P.select('qi', 'di').with_columns(pl.Series('ce', np.load('$R/$1/$5.npy').astype(np.float32))).write_parquet('$R/$1/$5.parquet')"; }
for n in listwise_full specialist_name specialist_address; do mkdir -p $R/$n
  S $n $W/data/cross_encoder/pairs_test_q.parquet $W/data/texts/test_q.parquet $W/data/texts/s1_test.parquet test_ce
  [ "${TRAIN:-0}" = 1 ] && S $n $W/data/reranker/pool_pairs.parquet $W/data/texts/pool_q.parquet $W/data/texts/s1_train.parquet pool_ce
done
if [ "${TRAIN:-0}" = 1 ]; then                                                                      # [CPU] matchers
  mkdir -p $R/core_name; S core_name $W/data/reranker/pool_pairs.parquet $W/data/texts/pool_q.parquet $W/data/texts/s1_train.parquet pool_ce
  $PY $HERE/train_matcher.py $W/matchers/us_india --ce ce2=$R/listwise_full/pool_ce.parquet --ce ce3=$R/specialist_name/pool_ce.parquet --ce ce4=$R/specialist_address/pool_ce.parquet
  $PY $HERE/train_matcher.py $W/matchers/france --no_cross_encoder --ce ce2=$R/core_name/pool_ce.parquet
  M_USIN=$W/matchers/us_india; M_FR=$W/matchers/france
fi

# ---- normalised route of the countries without training labels (france_canon.py): parallel work dir, same row order
WF=${FRANCE_WORK_DIR:-$W/france}; export FRANCE_WORK_DIR=$WF
$PY $S1/france_canon.py
EF=$WF/data/emb; mkdir -p $EF $WF/data/feat/$TAG $WF/data/reranker
ln -sfn $W/data/cross_encoder $WF/data/cross_encoder
encf() { for g in $(seq 0 $((NG-1))); do (for f in text_full text_name text_addr; do WORK_DIR=$WF CUDA_VISIBLE_DEVICES=$g $PY $S1/encode.py $RET $1 $f $g $NG $EF; done) & done; wait; }
encf s1_test; encf test_q
WORK_DIR=$WF CUDA_VISIBLE_DEVICES=0 $PY $S1/retrieve_dense.py $EF test_q s1_test text_full 20 $TAG
WORK_DIR=$WF $PY $S1/retrieve_sparse.py test_q s1_test 20 $TAG
WORK_DIR=$WF CUDA_VISIBLE_DEVICES=0 $PY $S1/merge_candidates.py $EF test_q s1_test $TAG
WORK_DIR=$WF $PY $HERE/features.py test_q s1_test $TAG
# every test country label with its branch (common/country_route.py): labelled countries keep the main features and the listwise
# reranker score; every other label (the common UNLABELLED branch) gets the core-name reranker score on its normalised candidates
$PY - <<P
import os, polars as pl
from common.country_route import UNLABELLED, branch_of
tq = pl.read_parquet("$WF/data/texts/test_q.parquet", columns=["country"]).with_row_index("qi")
labels = tq["country"].unique().to_list()
for c in labels:
    if branch_of(c) != UNLABELLED and os.path.exists("$W/data/feat/$TAG/test_q." + c + ".parquet"):
        dst = "$WF/data/feat/$TAG/test_q." + c + ".parquet"
        if os.path.lexists(dst): os.remove(dst)
        os.symlink("$W/data/feat/$TAG/test_q." + c + ".parquet", dst)
unseen = [c for c in labels if branch_of(c) == UNLABELLED]
fs = [f for f in ("$WF/data/cand/$TAG/pairs.test_q." + c + ".parquet" for c in sorted(unseen)) if os.path.exists(f)]   # no file: no candidates
(pl.concat([pl.read_parquet(f, columns=["qi", "di"]) for f in fs]) if fs else pl.DataFrame(schema={"qi": pl.Int32, "di": pl.Int32})
 ).write_parquet("$WF/data/reranker/pairs_test_unseen.parquet")
P
mkdir -p $R/core_name
$PY $HERE/listwise_reranker.py score --field core --model ${RR[core_name]} --pairs $WF/data/reranker/pairs_test_unseen.parquet \
  --qtext $WF/data/texts/test_q.parquet --dtext $WF/data/texts/s1_test.parquet --out $WF/data/reranker/unseen_core.npy --bs 512
$PY - <<P
import numpy as np, polars as pl
U = pl.read_parquet("$WF/data/reranker/pairs_test_unseen.parquet").with_columns(pl.Series("ce", np.load("$WF/data/reranker/unseen_core.npy").astype(np.float32)))
from common.country_route import UNLABELLED, with_branch
tq = with_branch(pl.read_parquet("$WF/data/texts/test_q.parquet", columns=["country"]).with_row_index("qi"))
S = pl.read_parquet("$R/listwise_full/test_ce.parquet").join(tq.filter(pl.col("branch") != UNLABELLED).select("qi"), on="qi", how="semi")
pl.concat([S.select("qi", "di", "ce"), U.select("qi", "di", "ce")]).write_parquet("$WF/data/reranker/test_listwise_or_core.parquet")
P

# ---- first-stage decisions: labelled countries matcher_us_india (0.50 / 0.30), unlabelled countries matcher_france (0.60)          [CPU]
WORK_DIR=$WF $PY $HERE/predict_first_stage.py $FS --seen $M_USIN:0.5:0.3 --unseen $M_FR:0.6:0 \
  --ce_test ce=$W/cross_encoder/test_ce.parquet --ce_test ce2=$WF/data/reranker/test_listwise_or_core.parquet \
  --ce_test ce3=$R/specialist_name/test_ce.parquet --ce_test ce4=$R/specialist_address/test_ce.parquet

# ---- test tables for the later stages (WORK_DIR/stage2/)                                          [CPU]
$PY $HERE/candidate_table.py --first-stage $FS --listwise $R/listwise_full/test_ce.parquet --name $R/specialist_name/test_ce.parquet \
  --address $R/specialist_address/test_ce.parquet --cross-encoder $W/cross_encoder/test_ce.parquet --out $OUT2
echo "stage 2 done: $OUT2"
