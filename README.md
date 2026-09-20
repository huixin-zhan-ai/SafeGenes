# SafeGenes — code release

Code accompanying *SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models*
(Cell Reports Methods). All notebooks have been converted to `.py`, de-duplicated, and renamed by
the pipeline stage and the paper element each one produces.

## Contents

| File | Produces | Converted from |
|---|---|---|
| `01_fgsm_attack_paper_version.py` | Figure 2; FGSM rows of Tables 2–3 | `VEP_ESM_PLLR_FGSM_layer (1).ipynb` |
| `02_pgd_attack_and_calibration.py` | PGD rows of Tables 2–3; Table S6; Figure S4 | `VEP_ESM_PLLR_PGD_calibration.ipynb` |
| `03_hotflip_snv_attack.py` | Figure 4; Figure 5 | `VEP_ESM_PLLR_Finetune_HotFlip.ipynb` |
| `03b_hotflip_snv_module.py` | Figure 4 (reusable module) | `hotflip_snv.py` |
| `04_softprompt_backdoor.py` | Figure 3 | `VEP_ESM_PLLR_SoftPrompt_Asym_and_Backdoor.ipynb` |
| `05_softprompt_cross_model_transfer.py` | Tables S4–S5 | `VEP_ESM_PLLR_Soft_Prompt_PromptOnly_CH.ipynb` |
| `06_proteinbert_baseline.py` | ProteinBERT rows of Tables 2–3 | `VEP_ProtBERT_PLLR_Soft_Prompt_Clovis.ipynb` |
| `07_gene_level_gmm_clustering.py` | Figure S3 | `Cluster.ipynb` |
| `08_clinical_metrics.py` | Table S6 | `Clinical_metrics.ipynb` |
| `09_triage_prioritization_analysis.py` | Figure 5b | `prioritization_analysis.py` |
| `10_reliability_diagram.py` | Figure S4 | `reliability_5bin.py` |
| `11_calibration_table_metrics.py` | Table S6 | `clean_row.py` |

