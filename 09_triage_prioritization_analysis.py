"""
Deployed-workflow triage and re-prioritization analysis

Paper element : Figure 5b
Source        : prioritization_analysis.py

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

import argparse, math
import numpy as np
import pandas as pd


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return df[n].to_numpy()
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def percentile_rank(scores):
    # higher score -> higher percentile (more prioritized as pathogenic)
    order = scores.argsort().argsort()          # 0..n-1 ranks (ascending)
    return order / (len(scores) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, default=math.log(3.0),
                    help="decision threshold on PLLR (default ln 3 = calibrated prob 0.5)")
    ap.add_argument("--topk", type=int, default=None,
                    help="size of the 'flagged for review' list (default = #pathogenic)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    clean = pick(df, "clean_pllr", "clean_PLLR", "PLLR_clean")
    att   = pick(df, "attacked_pllr", "att_pllr", "attacked_PLLR")
    y     = pick(df, "label", "labels", "true_label").astype(int)
    T = args.threshold
    n = len(y); npos = int(y.sum()); nneg = n - npos
    K = args.topk if args.topk is not None else npos

    clean_pred = (clean > T).astype(int)
    att_pred   = (att   > T).astype(int)

    # ---- (a) class-flip / directional ----
    correct_clean = (clean_pred == y)
    flipped = correct_clean & (att_pred != y)
    flip_rate = flipped.sum() / max(correct_clean.sum(), 1)

    path_called = (y == 1) & (clean > T)                 # pathogenic correctly flagged when clean
    path_demoted = path_called & (att <= T)              # ... pushed below threshold after 1 SNV
    ben_called  = (y == 0) & (clean <= T)                # benign correctly called
    ben_promoted = ben_called & (att > T)                # ... pushed above threshold (false alarm)

    # ---- (b) rank / triage movement ----
    pr_clean = percentile_rank(clean)
    pr_att   = percentile_rank(att)
    mean_abs_pctl_shift = float(np.abs(pr_clean - pr_att).mean())
    crossings = int((clean_pred != att_pred).sum())

    # top-K flagged-for-review list (rank by score, descending)
    topk_clean = set(np.argsort(-clean)[:K].tolist())
    topk_att   = set(np.argsort(-att)[:K].tolist())
    jaccard = len(topk_clean & topk_att) / len(topk_clean | topk_att)
    # true pathogenic that were in the clean top-K but drop out after attack
    path_idx = set(np.where(y == 1)[0].tolist())
    path_in_topk_clean = topk_clean & path_idx
    path_dropped = path_in_topk_clean - topk_att
    dropout_rate = len(path_dropped) / max(len(path_in_topk_clean), 1)

    print(f"file: {args.csv}")
    print(f"n={n} (pathogenic {npos}, benign {nneg})  |  threshold T={T:.4f}  |  flag list K={K}\n")

    print("== (a) Class-flip / prioritization at the fixed threshold ==")
    print(f"  attack success (correct->incorrect):        {flip_rate*100:5.1f}%  "
          f"({flipped.sum()}/{correct_clean.sum()})")
    print(f"  pathogenic demoted below cutoff by 1 SNV:    {path_demoted.sum()}/{path_called.sum()} "
          f"= {100*path_demoted.sum()/max(path_called.sum(),1):.1f}%")
    print(f"  benign promoted above cutoff by 1 SNV:       {ben_promoted.sum()}/{ben_called.sum()} "
          f"= {100*ben_promoted.sum()/max(ben_called.sum(),1):.1f}%\n")

    print("== (b) Rank / triage movement ==")
    print(f"  mean |percentile-rank shift| per variant:    {mean_abs_pctl_shift*100:.1f} pctl points")
    print(f"  variants crossing the decision threshold:    {crossings}/{n} = {100*crossings/n:.1f}%")
    print(f"  top-{K} flagged-list Jaccard (clean vs att):  {jaccard:.3f}")
    print(f"  true-pathogenic dropped out of top-{K} list:  {len(path_dropped)}/{len(path_in_topk_clean)} "
          f"= {dropout_rate*100:.1f}%")


if __name__ == "__main__":
    main()
