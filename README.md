# SafeGenes — code release

Code accompanying *SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models*
(Cell Reports Methods). All notebooks have been converted to `.py`, de-duplicated, and renamed by
the pipeline stage and the paper element each one produces.

## Contents

| File | Produces |
|---|---|
| `01_fgsm_attack_paper_version.py` | Figure 2; FGSM rows of Tables 2–3 | 
| `02_pgd_attack_and_calibration.py` | PGD rows of Tables 2–3; Table S6; Figure S4 |
| `03_hotflip_snv_attack.py` | Figure 4; Figure 5 | 
| `03b_hotflip_snv_module.py` | Figure 4 (reusable module) | 
| `04_softprompt_backdoor.py` | Figure 3 | 
| `05_softprompt_cross_model_transfer.py` | Tables S4–S5 | 
| `06_proteinbert_baseline.py` | ProteinBERT rows of Tables 2–3 | 
| `07_gene_level_gmm_clustering.py` | Figure S3 | 
| `08_clinical_metrics.py` | Table S6 | 
| `09_triage_prioritization_analysis.py` | Figure 5b | 
| `10_reliability_diagram.py` | Figure S4 | 
| `11_calibration_table_metrics.py` | Table S6 | 
