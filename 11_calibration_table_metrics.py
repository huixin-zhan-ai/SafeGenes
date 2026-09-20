"""
Per-model calibration table row (Platt |PLLR|, Brier, ECE, FPR@95% sens.)

Paper element : Table S6
Source        : clean_row.py

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, brier_score_loss


def ece_score(p, y, n_bins=10):
    p = np.clip(np.asarray(p, float), 0, 1); y = np.asarray(y, float)
    e = np.linspace(0, 1, n_bins + 1); s = 0.0; N = len(y)
    for i in range(n_bins):
        m = (p > e[i]) & (p <= e[i + 1])
        if m.sum():
            s += abs(y[m].mean() - p[m].mean()) * m.sum() / N
    return s


def fpr_at_sens(score, y, target=0.95):
    fpr, tpr, _ = roc_curve(y, score)
    i = np.where(tpr >= target)[0][0]
    return fpr[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--name", default="model")
    args = ap.parse_args()

    te = pd.read_csv(args.test); tr = pd.read_csv(args.train)
    y = te["label"].to_numpy()
    abs_c = te["pllr_abs_clean"].to_numpy(); abs_a = te["pllr_abs_att"].to_numpy()
    tr_abs = tr["pllr_abs_clean"].to_numpy(); tr_y = tr["label"].to_numpy()

    platt = LogisticRegression().fit(tr_abs.reshape(-1, 1), tr_y)   # calibrate |PLLR| -> prob
    p_c = platt.predict_proba(abs_c.reshape(-1, 1))[:, 1]
    p_a = platt.predict_proba(abs_a.reshape(-1, 1))[:, 1]

    print(f"== {args.name} (learned Platt(|PLLR|)) ==")
    print(f"  CLEAN     Brier={brier_score_loss(y, p_c):.3f}  ECE={ece_score(p_c, y):.3f}  "
          f"FPR@95%={fpr_at_sens(abs_c, y):.3f}")
    print(f"  ATTACKED  Brier={brier_score_loss(y, p_a):.3f}  ECE={ece_score(p_a, y):.3f}  "
          f"FPR@95%={fpr_at_sens(abs_a, y):.3f}")


if __name__ == "__main__":
    main()
