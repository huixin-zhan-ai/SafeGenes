"""
Clinical reliability metrics (Brier, ECE, FPR at fixed sensitivity)

Paper element : Table S6
Source        : Clinical_metrics.ipynb

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

# ---------------------------------------------------------------- cell 1
# [notebook-only] from google.colab import drive
# [notebook-only] drive.mount('/content/drive', force_remount=True)

# ---------------------------------------------------------------- cell 2
color_map = {
    (0, 0): "#1f77b4",   # Clean, benign
    (0, 1): "#aec7e8",   # Clean, pathogenic
    (1, 0): "#2ca02c",   # Hijack, benign
    (1, 1): "#98df8a",   # Hijack, pathogenic
}

# ---------------------------------------------------------------- cell 3
# Mount Google Drive
# [notebook-only] from google.colab import drive
# [notebook-only] drive.mount('/content/drive')

# Load CSVs from Google Drive
import pandas as pd
from sklearn.metrics import brier_score_loss
import pandas as pd
from scipy.special import expit  # Sigmoid function

# File paths
pgd_path = "/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_5.csv"

# Load data
df = pd.read_csv("/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_5.csv")
# Apply sigmoid to PLLR to get probabilities
y_true = df["label"].values
y_prob = expit(df["PLLR_after_FGSM"].values)

brier = brier_score_loss(y_true, y_prob)
print(f"Brier Score: {brier:.4f}")

# ---------------------------------------------------------------- cell 4
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        bin_conf = np.mean(y_prob[mask])
        bin_acc = np.mean(y_true[mask])
        bin_prop = np.mean(mask)
        ece += np.abs(bin_conf - bin_acc) * bin_prop
    return ece

ece = compute_ece(y_true, y_prob)
print(f"Expected Calibration Error (ECE): {ece:.4f}")


# ---------------------------------------------------------------- cell 5
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
tpr_target = 0.95
idx = np.where(tpr >= tpr_target)[0][0]
fpr_at_95_tpr = fpr[idx]
threshold_at_95_tpr = thresholds[idx]

print(f"FPR at TPR=0.95: {fpr_at_95_tpr:.4f} (Threshold={threshold_at_95_tpr:.4f})")

