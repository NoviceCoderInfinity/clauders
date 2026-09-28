"""Branch of every country label: its own labelled branch, or the shared branch for countries without labels.

`country` is an open set of string labels. Nothing here lists the countries:

- The labelled branches are the country labels of the TRAINING data (DATA_DIR/train/train_source1.tsv), read at run time.
  A test label that names one of them (case, spacing and dots ignored, plus the common aliases of LABEL_ALIASES) uses that
  country's matcher, judges and combiners, which were trained on its labels.
- Every other label goes to the branch UNLABELLED, whose models are trained on the labelled countries and applied
  cross-country (the unseen-country matcher of stage 2, stage5_unlabelled). Any number of new countries can share it.

Retrieval always stays inside a record's own country label; the branch only chooses which models decide the pairs.
"""
import functools

import polars as pl

import config
from common.io import read_tsv

UNLABELLED = "unlabelled"
# Alternative spellings of a country name. Used only to recognise a labelled country written differently in the test files.
LABEL_ALIASES = {
    "usa": "us", "u s": "us", "u s a": "us", "united states": "us", "united states of america": "us",
    "bharat": "india", "ind": "india",
}


def fold(label):
    """Case-, space- and dot-insensitive key of a country label."""
    k = " ".join(str(label).replace(".", " ").strip().casefold().split())
    return LABEL_ALIASES.get(k, k)


@functools.lru_cache(maxsize=4)
def labelled_countries(train_dir=None):
    """Country labels that have training labels, as written in train_source1.tsv (sorted)."""
    train_dir = train_dir or str(config.TRAIN_DIR)
    labels = read_tsv(f"{train_dir}/train_source1.tsv", columns=["country"])["country"].drop_nulls().unique().to_list()
    return tuple(sorted(labels))


def labelled_branches():
    """Branch names of the countries with training labels."""
    return list(labelled_countries())


def branch_of(label):
    """The labelled country a label names, else UNLABELLED."""
    if label is None:
        return UNLABELLED
    known = {fold(c): c for c in labelled_countries()}
    return known.get(fold(label), UNLABELLED)


def with_branch(df, test_dir=None, col="country"):
    """Add a `branch` column: the labelled country of each row's label, or UNLABELLED. (test_dir is not needed; kept for
    call compatibility.)"""
    observed = df[col].unique().to_list()
    link = pl.DataFrame({col: observed, "branch": [branch_of(c) for c in observed]}, schema={col: df.schema[col], "branch": pl.String})
    return df.join(link, on=col, how="left", maintain_order="left").with_columns(pl.col("branch").fill_null(UNLABELLED))

