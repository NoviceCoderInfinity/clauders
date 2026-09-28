"""US and India decision set from the first-pass combiner (CPU, about 1 minute).

The first-pass combiner's zone probabilities (WORK_DIR/stage4/first_pass/test_zone_p.parquet, written by train_combiner.py)
go through the zone decision (zone_decision.py): set-size prior, rule p >= 0.65 and margin >= 0.40, caps 5 Source 2 /
6 Source 3 per Source 1 entity. Records outside the uncertainty zone keep the stage-2 matcher decision. Every accepted pair
is a row of the stage-2 candidate table (WORK_DIR/stage2/test_candidates.parquet), i.e. a pair the matching models scored.
Writes (source1_entity_id, entity_id, country) for every accepted pair of the branches with training labels.

usage: python -m stage4_combiner.rebuild_decisions [--work WORK_DIR] [--out PATH]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

import config
from common.country_route import labelled_branches, with_branch
from common.io import read_tsv
from stage4_combiner import zone_decision as ZD

K = ["s1_id", "cand_id"]


def load_stage2(work):
    """Candidates of the labelled branches, zone records and the qi / di index maps."""
    d2 = work / "stage2"
    C = with_branch(pl.read_parquet(d2 / "test_candidates.parquet", columns=["qi", "di", "src", "country", "matcher_accept"]), config.TEST_DIR)
    C = C.filter(pl.col("branch").is_in(labelled_branches()))
    zq = pl.read_parquet(work / "stage3" / "test_zone_records.parquet")
    qm = pl.read_parquet(d2 / "test_record_index.parquet").select("qi", pl.col("entity_id").alias("cand_id"))
    dm = pl.read_parquet(d2 / "test_source1_index.parquet").select("di", pl.col("entity_id").alias("s1_id"))
    return C, zq, qm, dm


def run(work, zone_p):
    t0 = time.time()
    C, zq, qm, dm = load_stage2(work)
    H = np.load(work / "stage4" / "set_size_prior.npy")
    s1c = with_branch(read_tsv(config.TEST_DIR / "test_source1.tsv", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), config.TEST_DIR)
    labelled = s1c.filter(pl.col("branch").is_in(labelled_branches()))
    A, outside = ZD.zone_inputs(C, zq, pl.read_parquet(zone_p))
    pairs = ZD.decide(A, outside, H).join(qm, on="qi").join(dm, on="di").select(K).join(labelled, on="s1_id", how="semi")
    print(f"first pass: {pairs.height:,} pairs [{time.time() - t0:.0f}s]", flush=True)
    return pairs.join(s1c, on="s1_id").select(pl.col("s1_id").alias("source1_entity_id"), pl.col("cand_id").alias("entity_id"), "country")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(config.WORK_DIR), help="work dir with stage2/, stage3/ and stage4/set_size_prior.npy")
    ap.add_argument("--zone-p", default=None, help="first-pass zone probabilities (default WORK/stage4/first_pass/test_zone_p.parquet)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    zone_p = Path(a.zone_p) if a.zone_p else work / "stage4" / "first_pass" / "test_zone_p.parquet"
    out = Path(a.out) if a.out else work / "stage4" / "us_india_pairs.parquet"
    P = run(work, zone_p)
    out.parent.mkdir(parents=True, exist_ok=True)
    P.write_parquet(out)
    print(f"wrote {out}: {P.height:,} pairs {dict(P.group_by('country').len().iter_rows())}", flush=True)
