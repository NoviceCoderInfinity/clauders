# Business Entity Resolution

This pipeline links every Source 1 entity to the Source 2 / Source 3 records that describe the same business, and writes the
two files of the submission:

| File | Columns | Contents |
|---|---|---|
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | final matches, one row per Source 1 entity (the list may be empty) |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | every pair the matching models score (the matches are a subset) |

The method is described in `Documentation_template.md` at the root of the submission package. In short: normalisation and
dense retrieval (blocking) → cross-encoders and a LightGBM matcher → six Qwen3.5-4B LoRA judges and a cross-fitted LightGBM
combiner on the uncertain records of the countries with training labels → an unseen-country matcher (trained on the labelled
countries) for countries without labels → output files.

`country` is an open set (`src/common/country_route.py`): the countries with labels are read from `train_source1.tsv` at run
time; a test label naming one of them uses that country's models, and every other label (France on this test set) uses the
branch for countries without labels. Every model is trained on the provided training data only; no test labels exist or are used, and no model, decision rule or threshold is fitted on test records. The unlabelled test inputs are used in three disclosed ways: the spelling normalisation for countries without labels (French street-type abbreviations, "(FRANCE)" and "Ets" in names) and the French legal forms in the core-name stopword list were written after reading those countries' test inputs, since no training data exists for them; the kept token-id tables of the lossless vocabulary pruning list the tokens of the train and test texts; and test row counts set the size of the hold-out candidate pool used to train and evaluate the combiner. The models that run total 5,088,182,405 parameters (6,359,944,581 with
full vocabularies; see section 3).

## 1. Environment

- Python 3.12 (3.12.14 was used).
- GPU stages (retrieval, cross-encoders, LLM judges): NVIDIA GPUs with a CUDA 12.8-capable driver. Retraining the LoRA judges
  (`TRAIN=1`) runs on 8 GPUs with `torchrun`.
- CPU stages (features, LightGBM models, assembly): any x86-64 or arm64 machine.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Put the competition files, unchanged, in one directory:

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

The run scripts call the competition's official validator at the end when they find it at
`$(dirname $DATA_DIR)/utils/validate_submission.py` (the competition kit's layout); set `VALIDATOR=/path/to/validate_submission.py`
to point elsewhere.

## 3. Model weights

`WEIGHTS_DIR` holds only model weights, every one trained by the code in `src/` on the provided training data: the fine-tuned
encoders, the LoRA judges and the LightGBM models. It holds no data and no intermediate stage output; every intermediate file
is computed by the code from the competition data. It is shipped in two ways:

- **Submission package with weights:** the `weights/` folder at the root of the package. Use `WEIGHTS_DIR=<package>/weights`.
- **Submission package without weights:** the same folders are kept in the team's Hugging Face storage bucket
  `hf://buckets/Anupam-Rawat-IITB/business-entity-resolution` (access on request). Download exactly the folders the code uses:

  ```bash
  hf auth login
  B=hf://buckets/Anupam-Rawat-IITB/business-entity-resolution
  hf buckets sync $B weights/ \
    --include "encoders/retriever_e5_small/*" --include "encoders/cross_encoder_e5_small/*" \
    --include "encoders/reranker_listwise_bge_m3/*" --include "encoders/reranker_core_name_bge_m3/*" \
    --include "encoders/specialist_name_e5_base/*" --include "encoders/specialist_address_e5_base/*" \
    --include "qwen_judges/judge_plain_*" --include "qwen_judges/judge_context_*" \
    --include "lightgbm/matcher_us_india/*" --include "lightgbm/matcher_france/*" --include "lightgbm/first_pass/*" \
    --include "kept_token_ids/*"
  ```

**What is included and what is not.** To keep the submission zip from ballooning, the package includes only weights that are not publicly available: the six fine-tuned encoders, the six LoRA judge adapters with their score heads, and the LightGBM models (about 5 GB). The base of the LLM judges, `Qwen/Qwen3.5-4B` (Apache-2.0), is publicly available, so its weights are not included; the code downloads it from the Hugging Face Hub at the pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on first use (set `QWEN_REVISION`, section 3), which needs internet access or a pre-filled Hugging Face cache. The same applies to the public e5 and bge base models, which are only needed to retrain (`TRAIN=1`).

The package's `weights/` folder:

```
weights/                                   (= WEIGHTS_DIR)
├── encoders/          fine-tuned encoders (fp32, pruned vocabulary, tokenizer files included)
│   ├── retriever_e5_small/           retrieval (sentence-transformers format)
│   ├── cross_encoder_e5_small/       pair cross-encoder
│   ├── reranker_listwise_bge_m3/     listwise reranker
│   ├── specialist_name_e5_base/      name specialist
│   ├── specialist_address_e5_base/   address specialist
│   └── reranker_core_name_bge_m3/    core-name reranker (unseen-country matcher)
├── qwen_judges/       6 LoRA judges: adapter_config.json, adapter_model.safetensors, score_head.safetensors, judge_config.json
│   ├── judge_plain_1/  judge_plain_2/                      plain inputs, 128 tokens
│   └── judge_context_1/ … judge_context_4/                 context inputs, 192 tokens (judge_context_4: 256)
├── lightgbm/          matcher_us_india/, matcher_france/ (unseen-country matcher), first_pass/ (12 cross-fitted combiner models)
└── kept_token_ids/    the token ids kept by the pruned vocabularies (the same files as src/common/kept_token_ids/)
```

| Weights folder | Fine-tuned from | Licence |
|---|---|---|
| `encoders/retriever_e5_small`, `encoders/cross_encoder_e5_small` | `intfloat/multilingual-e5-small` | MIT |
| `encoders/specialist_name_e5_base`, `encoders/specialist_address_e5_base` | `intfloat/multilingual-e5-base` | MIT |
| `encoders/reranker_listwise_bge_m3`, `encoders/reranker_core_name_bge_m3` | `BAAI/bge-reranker-v2-m3` | Apache-2.0 |
| `qwen_judges/*` (LoRA adapters and score heads) | `Qwen/Qwen3.5-4B` | Apache-2.0 |
| `lightgbm/*` | trained with LightGBM | MIT |

The base of every LLM judge, `Qwen/Qwen3.5-4B`, is loaded from the Hugging Face Hub at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: set `export QWEN_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

**Parameters.**

| Model | As run (pruned vocabulary) | Full vocabulary |
|---|---|---|
| Qwen3.5-4B text model (shared by the judges) | 3,776,772,096 | 4,205,751,296 |
| 6 LoRA judges + heads (5 × rank 16 / alpha 32, `judge_context_3` rank 32 / alpha 64) | 227,284,992 | 227,284,992 |
| `retriever_e5_small` / `cross_encoder_e5_small` | 43,279,488 / 43,279,873 | 117,653,760 / 117,654,145 |
| `reranker_listwise_bge_m3` / `reranker_core_name_bge_m3` | 369,460,225 each | 567,755,777 each |
| `specialist_name_e5_base` / `specialist_address_e5_base` | 129,322,753 each | 278,044,417 each |
| LightGBM models | 0 | 0 |
| **Total** | **5,088,182,405** | **6,359,944,581** |

**Lean models.** The Qwen3.5-4B vision tower is dropped at load time and its multi-token-prediction head is never loaded. Every
embedding table keeps only the rows of the token ids that occur in the competition data, tokenised as the pipeline tokenises
them (`src/common/kept_token_ids/qwen.npy`: 80,750 ids; `xlmr.npy`: 56,354 ids, shared by every XLM-R / e5 model).
`src/common/vocab_prune.py` renumbers ids right after tokenisation and raises an error if an input produces an id outside the
kept set, so a score can never change silently. The pruning is lossless and is not needed for the 8-billion limit: with every
full vocabulary the models total 6,359,944,581 parameters.

## 4. Full run from the competition data

```bash
cd code/business_entity_resolution/src
export QWEN_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
DATA_DIR=/abs/path/dataset WEIGHTS_DIR=/abs/path/weights WORK_DIR=/abs/path/work OUTPUT_DIR=/abs/path/output bash run_pipeline.sh
```

By default every model runs inference with the released weights; `TRAIN=1` retrains them where the stage supports it.

| Step | Command (from `src/`) | Runs on | Writes |
|---|---|---|---|
| 1. Candidates | `bash stage1_candidates/run_stage1.sh` | GPU + CPU | normalised records, texts, embeddings, top-20 Source 1 candidates per record (`WORK_DIR/data/`) |
| 2. Rerank and match | `bash stage2_rerank_match/run_stage2.sh` | GPU + CPU | cross-encoder and reranker scores; labelled countries: `matcher_us_india` (p ≥ 0.50, margin ≥ 0.30); countries without labels: normalised candidates (`france_canon.py`), core-name reranker, `matcher_france` (p ≥ 0.60); candidate tables (`WORK_DIR/stage2/`) |
| 3. LLM judges | `bash stage3_judges/run_stage3.sh` | GPU + CPU | uncertainty zone, judge inputs, scores of the six judges and their mean (`WORK_DIR/stage3/`) |
| 4. Combiner | `bash stage4_combiner/run_stage4.sh` | CPU | first-pass combiner on the zone, zone decision → `WORK_DIR/stage4/us_india_pairs.parquet` |
| 5. Unlabelled countries | `python -m stage5_unlabelled.select_pairs` | CPU | `WORK_DIR/stage5/unlabelled_pairs.parquet` |
| 6. Outputs | `python -m stage6_output.candidate_pairs`, `python -m stage6_output.assemble` | CPU | `OUTPUT_DIR/candidate_pairs.tsv`, `OUTPUT_DIR/matching_results.tsv` |

`python -m stage6_output.summary` prints the per-country table of candidate and matched pairs. With `TRAIN=1`,
`python -m stage4_combiner.evaluate_holdout` prints the validation F0.5 table and the (t, m) grid of the methodology document.

**Retraining from scratch (`TRAIN=1`).** Every model can be retrained from the competition data alone, in the same command:

```bash
TRAIN=1 DATA_DIR=/abs/path/dataset WEIGHTS_DIR=/abs/path/weights WORK_DIR=/abs/path/work OUTPUT_DIR=/abs/path/output bash run_pipeline.sh
```

| Stage | Trained with `TRAIN=1` (from the public base models and the training data) | Written to |
|---|---|---|
| 1 | e5-small retriever (`train_retriever.py`, two phases with mined hard negatives) | `WORK_DIR/models/retriever` |
| 2 | e5-small cross-encoder, listwise reranker, name / address specialists, core-name reranker, both LightGBM matchers | `WORK_DIR/cross_encoder`, `WORK_DIR/reranker`, `WORK_DIR/matchers` |
| 3 | the judges' training sets (`build_training_sets.py`), the combiner's hold-out tables and hold-out judge inputs (`stage4_combiner/build_hold_tables.py`), a check that they hold only training records with their true labels (`check_training_sets.py`), then the six LoRA judges (`train_lora.py`, 8 GPUs) | `WORK_DIR/stage3/train`, `WORK_DIR/stage4/hold`, `WORK_DIR/models/qwen_judges` |
| 4 | the cross-fitted combiner (`train_combiner.py`) after checking the lookalike offsets against the training labels (`derive_up_steps.py`), then the validation table and (t, m) grid (`evaluate_holdout.py`) | `WORK_DIR/stage4/first_pass` |

Every retrained model is written under `WORK_DIR` and used by the later stages of the same run; the released weights in
`WEIGHTS_DIR` are never overwritten. A judge whose `score_head.safetensors` already exists is not retrained, so an interrupted
run can be restarted. The released judges and combiner were trained on training tables built by an earlier script with the same
purpose; `build_training_sets.py` and `build_hold_tables.py` rebuild them from the stage 1-2 outputs. The hold-out tables they
build are identical to the ones the released combiner was trained on (checked on the training data), except that the judges'
own Source 1 entities are the only ones left out of combiner training; the judges' training pairs follow the documented recipe in
`build_training_sets.py` and are not a byte copy of the originals, so retrained judges will differ slightly from the released ones.

## 5. Re-running the CPU decision steps

The decision steps after the GPU stages can be re-run on any CPU from the work folder of a finished `run_pipeline.sh` run; nothing
is downloaded:

```bash
cd code/business_entity_resolution/src
DATA_DIR=/abs/path/dataset WEIGHTS_DIR=/abs/path/weights RUN_DIR=/abs/path/work bash rebuild_outputs.sh
```

It recomputes the set-size prior (training labels), the combiner on the test zone (released models), the zone decision, the
unlabelled-country pairs, the candidate set and the matching results into `RUN_DIR/rebuild/output/`, and runs the validator.

## 6. Validate

```bash
python utils/validate_submission.py \
    --matching "$OUTPUT_DIR/matching_results.tsv" \
    --candidate "$OUTPUT_DIR/candidate_pairs.tsv" \
    --test-dir "$DATA_DIR/test" --check-ids
```

## 7. Source layout (`src/`)

| Path | Contents |
|---|---|
| `config.py` | paths (environment variables) and the seed |
| `common/` | TSV input / output, text normalisation, open-set country branches, encoder helpers, sparse retrieval, the decision rule, the lean model loader (`vocab_prune.py`) and the kept token ids |
| `stage1_candidates/` | data preparation, training / validation split, normalisation, retriever training, encoding, dense and sparse retrieval, spelling rules for countries without labels (`france_canon.py`) |
| `stage2_rerank_match/` | pair features, e5-small cross-encoder, listwise rerankers and specialists, LightGBM matchers, first-stage decisions, candidate tables |
| `stage3_judges/` | uncertainty zone, judge inputs, LoRA training and scoring, score means, training-data check |
| `stage4_combiner/` | set-size prior, hold-out tables, combiner features (with the offset derivation) and training, zone decision |
| `stage5_unlabelled/` | decision set of the countries without training labels |
| `stage6_output/` | candidate set, final assembly, summary table |
| `run_pipeline.sh` | full run (section 4) |
| `rebuild_outputs.sh` | re-run of the CPU decision steps (section 5) |
