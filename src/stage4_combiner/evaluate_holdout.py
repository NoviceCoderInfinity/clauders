"""Validation F0.5 of the deployed decision on the hold-out split of the training data (CPU).

Reproduces the validation table of the methodology document (Section 5) and the threshold choice:
  matcher alone         stage-2 matcher rule p >= 0.50, margin >= 0.30, caps 5 Source 2 / 6 Source 3
  matcher + combiner    records outside the uncertainty zone keep the matcher rule; zone records get the cross-fitted combiner
                        probabilities (every hold-out prediction comes from the half the model was not trained on), the
                        set-size prior and the rule (t, m), then the caps - the same zone_decision.decide() that runs on test
The (t, m) grid and the set-size prior weight (0, 0.25, 0.5, 0.75) are scored on the TUNE half (used to choose the rule) and the REPORT half (reported); scores are per country and
estimated at test conditions by holdout.HoldOut.

usage: python -m stage4_combiner.evaluate_holdout [--hold-dir WORK_DIR/stage4/hold] [--combiner WORK_DIR/stage4/first_pass]
Inputs: hold_candidates.parquet (hold_table.py), <combiner>/hold_newp.parquet (train_combiner.py without --models),
        WORK_DIR/stage4/set_size_prior.npy (set_size_prior.py).
"""
import argparse

import numpy as np
import polars as pl

import config
from common.decision import decide_m
from stage4_combiner import holdout as HO
from stage4_combiner import zone_decision as ZD
from stage4_combiner.hold_table import judge_training


def deployed(X, P, H, t, m, scale=ZD.PRIOR_SCALE):
    """The test-time decision on the hold-out rows: matcher rule outside the zone, prior + (t, m) + caps inside."""
    zone = X.filter(pl.col("inzone") == 1)
    Xo = X.filter(pl.col("inzone") == 0).select("qi", "di", "src", "p")
    outside = decide_m(Xo, 0.5, 0.3, cap2=10**6, cap3=10**6).join(Xo.select("qi", "di", "src"), on=["qi", "di"])
    A = zone.select("qi", "di", "src").join(P, on=["qi", "di"], how="left").with_columns(pl.col("p").fill_null(0.0))
    return ZD.decide(A, outside, H, scale=scale, t=t, m=m).select("qi", "di")


def row(name, r, countries):
    cells = " | ".join(f"{r[('TUNE', c)]:.5f} / {r[('REPORT', c)]:.5f}" for c in countries)
    return f"| {name} | {cells} |"


def main():
    w = config.WORK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-dir", default=str(w / "stage4" / "hold"))
    ap.add_argument("--combiner", default=str(w / "stage4" / "first_pass"))
    ap.add_argument("--prior", default=str(w / "stage4" / "set_size_prior.npy"))
    a = ap.parse_args()
    X = pl.read_parquet(f"{a.hold_dir}/hold_candidates.parquet")
    P = pl.read_parquet(f"{a.combiner}/hold_newp.parquet").select("qi", "di", "p")
    H = np.load(a.prior)
    excluded_s1, _ = judge_training(a.hold_dir)
    E = HO.HoldOut(a.hold_dir, excluded_s1=excluded_s1)
    countries = list(E.n_s1)
    print("| Validation F_0.5 (TUNE / REPORT half) | " + " | ".join(countries) + " |")
    print("|---|" + "---|" * len(countries))
    print(row("Matcher alone", E.evaluate_D(decide_m(X.select("qi", "di", "src", "p"), 0.5, 0.3).select("qi", "di")), countries), flush=True)
    print(row("Matcher + six judges + combiner (0.65 / 0.40, prior 0.5)", E.evaluate_D(deployed(X, P, H, 0.65, 0.40)), countries), flush=True)
    print("\n(t, m) grid, chosen on the TUNE half (mix = mean weighted by the test Source 1 count per country):")
    rows = []
    for t in HO.GRID_T:
        for m in HO.GRID_M:
            r = E.evaluate_D(deployed(X, P, H, t, m))
            rows.append({"t": t, "m": m, **{f"{h}_{c}": r[(h, c)] for h in ("TUNE", "REPORT") for c in (*countries, "mix")}})
    G = pl.DataFrame(rows).sort("TUNE_mix", descending=True)
    with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=5):
        print(G)
    best = G.row(0, named=True)
    print(f"best rule on TUNE: t = {best['t']}, m = {best['m']}")
    print("\nset-size prior weight at the deployed rule 0.65 / 0.40:")
    for scale in (0.0, 0.25, 0.5, 0.75):
        print(row(f"prior weight {scale}", E.evaluate_D(deployed(X, P, H, 0.65, 0.40, scale)), countries), flush=True)


if __name__ == "__main__":
    main()
