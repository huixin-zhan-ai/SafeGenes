"""
Gene-level vulnerability via Gaussian-mixture clustering of perturbed PLLR

Paper element : Figure S3
Source        : Cluster.ipynb

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

# File paths
pgd_path = "/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_5.csv"
gene_path = "/content/drive/My Drive/vep_FGSM/output_test/splicing_feats.test.csv"

# Load data
df_pgd = pd.read_csv(pgd_path)
df_genes = pd.read_csv(gene_path)

# Combine PGD results and gene annotations
df_combined = pd.concat([df_pgd[["PLLR_after_FGSM", "label"]], df_genes["gene"]], axis=1)

# Apply GMM clustering
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=42)
df_combined["gmm_cluster"] = gmm.fit_predict(df_combined[["PLLR_after_FGSM"]])

# Identify mismatches between true label and GMM cluster
df_combined["label_mismatch"] = df_combined["label"] != df_combined["gmm_cluster"]

# Count mismatches per gene
mismatch_count_per_gene = df_combined[df_combined["label_mismatch"]].groupby("gene").size().sort_values(ascending=False)

# -------------------------------
# Plot mismatch count bar plot
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

top_n = 10
top_genes = mismatch_count_per_gene.head(top_n)


# plt.figure(figsize=(4.5, 4.5))
# bars = plt.bar(top_genes.index, top_genes.values, color="#1f77b4")
# # Add text labels on top of each bar
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         yval - 1.0,
#         f"{int(yval)}",
#         ha='center',
#         va='bottom',
#         fontsize=14  # <-- font size of numbers
#     )
# plt.title(f"Top {top_n} Genes (PGD-5)")
# #plt.xlabel("Gene")
# plt.ylabel("Mismatch Count")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_5.pdf")
# plt.show()

# ... (everything above unchanged) ...
# ... (everything above unchanged) ...

plt.figure(figsize=(9, 4.5))   # <-- wider so horizontal labels fit under each bar
bars = plt.bar(top_genes.index, top_genes.values, color="#1f77b4")
# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval -0.5,
        f"{int(yval)}",
        ha='center',
        va='bottom',
        fontsize=14
    )
plt.title(f"Top {top_n} Genes (PGD-5)")
plt.ylabel("Mismatch Count")
plt.xticks(rotation=0, ha='center')   # <-- horizontal, centered under each bar
plt.tight_layout()
plt.savefig("/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_5.pdf",
            bbox_inches="tight")
plt.show()



# ---------------------------------------------------------------- cell 4
# File paths
pgd_path = "/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_10.csv"
gene_path = "/content/drive/My Drive/vep_FGSM/output_test/splicing_feats.test.csv"

# Load data
df_pgd = pd.read_csv(pgd_path)
df_genes = pd.read_csv(gene_path)

# Combine PGD results and gene annotations
df_combined = pd.concat([df_pgd[["PLLR_after_FGSM", "label"]], df_genes["gene"]], axis=1)

# Apply GMM clustering
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=42)
df_combined["gmm_cluster"] = gmm.fit_predict(df_combined[["PLLR_after_FGSM"]])

# Identify mismatches between true label and GMM cluster
df_combined["label_mismatch"] = df_combined["label"] != df_combined["gmm_cluster"]

# Count mismatches per gene
mismatch_count_per_gene = df_combined[df_combined["label_mismatch"]].groupby("gene").size().sort_values(ascending=False)

# -------------------------------
# Plot mismatch count bar plot
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

top_n = 10
top_genes = mismatch_count_per_gene.head(top_n)


# plt.figure(figsize=(4.5, 4.5))
# bars = plt.bar(top_genes.index, top_genes.values, color="#1f77b4")
# # Add text labels on top of each bar
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         yval - 1.0,
#         f"{int(yval)}",
#         ha='center',
#         va='bottom',
#         fontsize=14  # <-- font size of numbers
#     )
# plt.title(f"Top {top_n} Genes (PGD-10)")
# #plt.xlabel("Gene")
# plt.ylabel("Mismatch Count")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_10.pdf")
# plt.show()

plt.figure(figsize=(9, 4.5))   # <-- wider so horizontal labels fit under each bar
bars = plt.bar(top_genes.index, top_genes.values, color="#1f77b4")
# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval -0.5,
        f"{int(yval)}",
        ha='center',
        va='bottom',
        fontsize=14
    )
plt.title(f"Top {top_n} Genes (PGD-5)")
plt.ylabel("Mismatch Count")
plt.xticks(rotation=0, ha='center')   # <-- horizontal, centered under each bar
plt.tight_layout()
plt.savefig("/content/drive/My Drive/vep_FGSM/output_test/fgsm_pllr_results_cm_pgd_10.pdf",
            bbox_inches="tight")
plt.show()
