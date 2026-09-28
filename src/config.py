"""Paths and constants shared by every stage. Each path can be overridden with an environment variable.

DATA_DIR     competition data: DATA_DIR/train/train_source{1,2,3}.tsv, DATA_DIR/train/train_ground_truth.tsv,
             DATA_DIR/test/test_source{1,2,3}.tsv
WORK_DIR     intermediate files written by the stages
WEIGHTS_DIR  fine-tuned model weights (see README.md for the layout)
OUTPUT_DIR   matching_results.tsv and candidate_pairs.tsv
"""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "dataset")).resolve()
WORK_DIR = Path(os.environ.get("WORK_DIR", "work")).resolve()
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "weights")).resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output")).resolve()

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

KEPT_IDS_DIR = SRC_DIR / "common" / "kept_token_ids"
KEPT_QWEN = KEPT_IDS_DIR / "qwen.npy"      # token ids kept for the Qwen3.5-4B judges
KEPT_XLMR = KEPT_IDS_DIR / "xlmr.npy"      # token ids kept for every XLM-R / e5 model (one shared vocabulary)

ENCODERS_DIR = WEIGHTS_DIR / "encoders"
JUDGES_DIR = WEIGHTS_DIR / "qwen_judges"
LIGHTGBM_DIR = WEIGHTS_DIR / "lightgbm"

SEED = 42
