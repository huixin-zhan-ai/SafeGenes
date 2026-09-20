"""
Fine-tuning plus discrete SNV-reachable HotFlip attack

Paper element : Figure 4; Figure 5
Source        : VEP_ESM_PLLR_Finetune_HotFlip.ipynb

Uses the paper's Settings recipe (lr 1e-4, 10 epochs, batch size 4).

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""
from __future__ import annotations


# ---------------------------------------------------------------- cell 2
# [notebook-only] from google.colab import drive
# [notebook-only] drive.mount('/content/drive', force_remount=True)


# ---------------------------------------------------------------- cell 3
import re
import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.special import softmax

import torch.nn.functional as F
import matplotlib as mpl
torch.manual_seed(0)
np.random.seed(0)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import math


# ---------------------------------------------------------------- cell 4
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=2)
    fp16: bool = field(default=False)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")  # ✅ match
    save_strategy: str = field(default="steps")        # ✅ match
    save_safetensors: bool = field(default=False)
    #load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=500)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=1e-2)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default='linear')
    save_total_limit: int = field(default=10)
    output_dir: str = field(default="/content/drive/My Drive/vep_FGSM/output_test")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    logging_first_step: bool = field(default=True)
    early_stopping_patience: int = field(default=5)
    early_stopping_threshold: float = field(default=1e-3)
training_args = TrainingArguments()


# ---------------------------------------------------------------- cell 5
@dataclass
class ModelArguments:
    #model_name_or_path: Optional[str] = field(default="bert-base-uncased")
    #model_name_or_path: Optional[str] = field(default="facebook/esm1b_t33_650M_UR50S")
    model_name_or_path: Optional[str] = field(default="facebook/esm2_t33_650M_UR50D")
    #model_name_or_path: Optional[str] = field(default="facebook/esm2_t30_150M_UR50D")
model_args = ModelArguments()


# ---------------------------------------------------------------- cell 6
data_path = "/content/drive/My Drive/vep_FGSM/data/"

class CustomCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0
        self.alphabet = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

    def compute_pll_for_sequence(self, sequence, model):
        #tokens = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        tokens = self.tokenizer(sequence, return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)
        model_device = next(model.parameters()).device
        for key in tokens.keys():
            tokens[key] = tokens[key].to(model_device)

        with torch.no_grad():
            outputs = model.base_model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

        logits = torch.log_softmax(outputs.logits, dim=-1)
        #print('logits',logits)
        idx = [self.alphabet[t] for t in sequence]
        PLL = torch.sum(torch.diag(logits[0, 1:-1, :][:, idx]))
        return PLL.item()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step_count += 1

        if self.step_count == 1 or self.step_count % 50 == 0:  # You can adjust the frequency as needed.
            all_sequences = []
            df = pd.read_csv(data_path+f"{DATASET}_test_data_protein_1024.csv")
            all_sequences = df['wt_seq'].tolist()

            all_plls_wt = []
            all_plls_wt_weighted = []
            for seq in all_sequences:
                wt_pll = self.compute_pll_for_sequence(seq, model)
                all_plls_wt.append(wt_pll)
                all_plls_wt_weighted.append(wt_pll / len(seq))

            #print(f"Step {self.step_count}: Pseudo-Log-Likelihoods for all sequences: {all_plls_wt}")
            #logging.info(f"Step {self.step_count}: Pseudo-Log-Likelihoods for all sequences: {all_plls_wt}")

            all_sequences = []
            all_sequences = df['mut_seq'].tolist()

            all_plls_mut = []
            all_plls_mut_weighted = []
            for seq in all_sequences:
                mut_pll = self.compute_pll_for_sequence(seq, model)
                all_plls_mut.append(mut_pll)
                all_plls_mut_weighted.append(mut_pll / len(seq))

            all_plls_wt = np.array(all_plls_wt)
            all_plls_mut = np.array(all_plls_mut)

            all_plls_wt_weighted = np.array(all_plls_wt_weighted)
            all_plls_mut_weighted = np.array(all_plls_mut_weighted)

        # Compute the PLLR
            PLLR_callback = np.abs(all_plls_wt - all_plls_mut)
            PLLR_weighted_callback = np.abs(all_plls_wt_weighted - all_plls_mut_weighted)

        # Get true labels
            true_labels_callback = df['labels'].to_numpy()

            # Save unweighted PLLR and labels
            df_save = pd.DataFrame({
                "PLLR": PLLR_callback,
                "weighted_PLLR": PLLR_weighted_callback,
                "label": true_labels_callback
            })

            save_dir = os.path.join(training_args.output_dir, "callback_metrics")
            os.makedirs(save_dir, exist_ok=True)

            # Save to a step-specific file
            save_path = os.path.join(save_dir, f"pllr_step_{self.step_count}.csv")
            df_save.to_csv(save_path, index=False)
            print(f"Saved PLLR metrics to {save_path}")


            # Compute metrics for PLLR_callback
            fpr, tpr, _ = roc_curve(true_labels_callback, PLLR_callback)
            roc_auc = auc(fpr, tpr)
            aupr = average_precision_score(true_labels_callback, PLLR_callback)

            # Compute metrics for PLLR_weighted_callback
            fpr_weighted, tpr_weighted, _ = roc_curve(true_labels_callback, PLLR_weighted_callback)
            roc_auc_weighted = auc(fpr_weighted, tpr_weighted)
            aupr_weighted = average_precision_score(true_labels_callback, PLLR_weighted_callback)

            # Plotting ROC for both PLLR_callback and PLLR_weighted_callback
            #plt.figure()
            plt.figure(figsize=(10, 7))
            mpl.rcParams['font.size'] = 18
            lw = 2  # line width
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='PLLR ROC curve (area = %0.2f)' % roc_auc)
            plt.plot(fpr_weighted, tpr_weighted, color='darkgreen', lw=lw, label='weighted PLLR ROC curve (area = %0.2f)' % roc_auc_weighted)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) for PLLR and weighted PLLR')
            plt.legend(loc="lower right")
            plt.show()

            # Plotting AUPR for both PLLR_callback and PLLR_weighted_callback
            precision, recall, _ = precision_recall_curve(true_labels_callback, PLLR_callback)
            precision_weighted, recall_weighted, _ = precision_recall_curve(true_labels_callback, PLLR_weighted_callback)

            no_skill = sum(true_labels_callback) / len(true_labels_callback)

            #plt.figure()
            plt.figure(figsize=(10, 7))
            mpl.rcParams['font.size'] = 18
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy')
            plt.plot(recall, precision, color='darkorange', lw=lw, label='PLLR PR curve (area = %0.2f)' % aupr)
            plt.plot(recall_weighted, precision_weighted, color='darkgreen', lw=lw, label='weighted PLLR PR curve (area = %0.2f)' % aupr_weighted)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve for PLLR and weighted PLLR')
            plt.legend()
            plt.show()


        # Logging
            #print(f"Step {self.step_count}: Pseudo-Log-Likelihoods for wt sequences: {all_plls_wt}")
            #print(f"Step {self.step_count}: Pseudo-Log-Likelihoods for mut sequences: {all_plls_mut}")
            logging.info(f"Step {self.step_count}: Pseudo-Log-Likelihoods for wt sequences: {all_plls_wt}")
            logging.info(f"Step {self.step_count}: Pseudo-Log-Likelihoods for mut sequences: {all_plls_mut}")
            print(f"AUC: {roc_auc}")
            print(f"Area Under the Precision-Recall Curve (AUPR): {aupr}")


# ---------------------------------------------------------------- cell 7
# Dataset Definition
class SiameseDataset(Dataset):
    def __init__(self, tokenizer, filename):
        data = pd.read_csv(filename)
        self.tokenizer = tokenizer
        # Generating some random sequences for demonstration purposes
        #self.seq_a = ["AGTCCGTA" * 10 for _ in range(num_examples)]
        #self.seq_b = ["TCGATCGA" * 10 for _ in range(num_examples)]
        #self.labels = [np.random.randint(0,2) for _ in range(num_examples)]  # Random binary labels
        self.seq_a = list(data['wt_seq'])
        self.seq_b = list(data['mut_seq'])
        self.labels = list(data['labels'])
        self.num_examples = len(self.labels)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        inputs_a = self.tokenizer(self.seq_a[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)
        inputs_b = self.tokenizer(self.seq_b[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)

        return {
            "input_ids1": inputs_a["input_ids"].squeeze(0),
            "attention_mask1": inputs_a["attention_mask"].squeeze(0),
            "input_ids2": inputs_b["input_ids"].squeeze(0),
            "attention_mask2": inputs_b["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

#         inputs_a = self.tokenizer(self.seq_a[idx], max_length=training_args.model_max_length)
#         inputs_b = self.tokenizer(self.seq_b[idx], max_length=training_args.model_max_length)

#         input_ids1 = torch.tensor(inputs_a["input_ids"]).squeeze(0) if isinstance(inputs_a["input_ids"], list) else inputs_a["input_ids"].squeeze(0)
#         attention_mask1 = torch.tensor(inputs_a["attention_mask"]).squeeze(0) if isinstance(inputs_a["attention_mask"], list) else inputs_a["attention_mask"].squeeze(0)

#         input_ids2 = torch.tensor(inputs_b["input_ids"]).squeeze(0) if isinstance(inputs_b["input_ids"], list) else inputs_b["input_ids"].squeeze(0)
#         attention_mask2 = torch.tensor(inputs_b["attention_mask"]).squeeze(0) if isinstance(inputs_b["attention_mask"], list) else inputs_b["attention_mask"].squeeze(0)


# ---------------------------------------------------------------- cell 8
from transformers import EsmConfig, AutoModelForMaskedLM

class SiameseNetwork(nn.Module):
    def __init__(self, model_name_or_path, num_labels, cache_dir=None):
        super(SiameseNetwork, self).__init__()
        config = EsmConfig.from_pretrained(model_name_or_path,
                                           token_dropout=False,
                                           output_hidden_states=True)
        # Load the base model
        self.base_model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2,
                labels, test_mode=False, epsilon=0.01):
        # Clean PLLR computation, used for fine-tuning and clean evaluation.
        # The discrete HotFlip attack lives in CustomTrainer.evaluate_with_hotflip
        # (no embedding-space PGD here).
        outputs1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2)

        logits1 = torch.log_softmax(outputs1.logits, dim=-1)
        logits2 = torch.log_softmax(outputs2.logits, dim=-1)

        batch_size = input_ids1.shape[0]
        PLLs1 = torch.zeros(batch_size, device=input_ids1.device)
        PLLs2 = torch.zeros(batch_size, device=input_ids2.device)
        for i in range(batch_size):
            idx1 = input_ids1[i, 1:-1]
            PLLs1[i] = torch.sum(torch.diag(logits1[i, 1:-1, :][:, idx1]))
            idx2 = input_ids2[i, 1:-1]
            PLLs2[i] = torch.sum(torch.diag(logits2[i, 1:-1, :][:, idx2]))

        PLLR = torch.abs(PLLs1 - PLLs2)
        sigmoid_PLLR = torch.sigmoid(PLLR)
        pll_loss = F.binary_cross_entropy(2 * sigmoid_PLLR - 1, labels.float())
        return (pll_loss, PLLR)


# ---------------------------------------------------------------- cell 9
# ---- choose the disease-specialized dataset to fine-tune / attack ----
DATASET = "cm"   # "cm" or "arm"  (ARM needs arm_*_protein_1024.csv and arm_*_dna_3072.csv in data_path)
train_csv = os.path.join(data_path, f"{DATASET}_train_data_protein_1024.csv")
test_csv  = os.path.join(data_path, f"{DATASET}_test_data_protein_1024.csv")

model = SiameseNetwork(model_args.model_name_or_path, num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, model_max_length=training_args.model_max_length)
print("Tokenizer:", tokenizer.__class__.__name__, "| DATASET =", DATASET)

train_dataset = SiameseDataset(tokenizer, train_csv)
test_dataset  = SiameseDataset(tokenizer, test_csv)

# Shuffle the training set (mirrors the original notebook)
from io import StringIO
df = pd.read_csv(train_csv).sample(frac=1.0, random_state=123)
buf = StringIO(); df.to_csv(buf, index=False); buf.seek(0)
train_dataset = SiameseDataset(tokenizer, buf)


# ---------------------------------------------------------------- cell 11
"""
hotflip_snv.py
==============
SNV-reachable, label-preserving HotFlip attack on a Siamese ESM PLLR variant-effect
predictor (companion to VEP_ESM_PLLR_PGD).

What this replaces
------------------
The original notebook attacks in *continuous embedding space* (10-step PGD on the
embedding tensors). Reviewers (R1.1 / R2.1) asked for a *discrete*, biologically
valid attack. This module implements a gradient-guided single-substitution attack
("HotFlip") restricted to substitutions reachable by a SINGLE nucleotide change
(SNV) from the reference codon, so every adversarial example is a real, submittable
variant.

Design (label-preserving)
-------------------------
* The patient's variant of interest (the single wt->mut difference at position `v`)
  is NEVER changed -> the ground-truth pathogenicity label is preserved.
* We instead flip ONE *background* residue (position p != v) to an SNV-reachable
  amino acid, applying the SAME substitution to BOTH the wt and mut sequences
  (the background is shared), and ask whether that single valid change flips the
  model's prediction.
* This is a representation-level robustness stress test, not a literal deployed-
  pipeline attacker (the attacker does not control the reference background in a
  real submission). Frame the claims accordingly.

SNV reachability
----------------
For background residue position p, the reference codon is wt_dna[3p:3p+3]
(the protein and DNA CSVs are codon-aligned, verified). The candidate amino acids
are exactly those obtainable by changing one of the three codon bases (excluding
synonymous changes and stops).

HotFlip ranking
---------------
One forward+backward pass gives the gradient of the (maximize-)loss w.r.t. the input
embeddings. For replacing residue `a` at position p with candidate `b`, the
first-order loss change is estimated as
    score(p,b) = (E[b]-E[a]) . (grad_wt[p] + grad_mut[p]).
We take the top-K (p,b) by score, *actually apply* each discrete substitution,
run a real (no-grad) forward pass, and keep the substitution that most increases
the loss (or, equivalently, drives the prediction away from the true label).
Because we only ever evaluate real token sequences there is no off-manifold /
"decode back to a sequence" problem.

Author: prepared for the SafeGenes revision.
"""

# (hoisted to top) from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------------------------------------------------------- #
# 1. Genetic code + SNV reachability
# ----------------------------------------------------------------------------- #
_BASES = "TCAG"
_CODONS = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: Dict[str, str] = {c: a for c, a in zip(_CODONS, _AAS)}

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def translate(dna: str) -> str:
    dna = dna.upper().replace("U", "T")
    return "".join(CODON_TABLE.get(dna[i:i + 3], "?") for i in range(0, len(dna) - 2, 3))


def snv_reachable_aas(codon: str) -> Dict[str, str]:
    """Amino acids reachable from `codon` by a single nucleotide substitution.

    Returns {aa: example_codon}. Excludes the wild-type (synonymous) amino acid
    and stop codons.
    """
    codon = codon.upper().replace("U", "T")
    wt_aa = CODON_TABLE.get(codon, None)
    out: Dict[str, str] = {}
    if wt_aa is None or wt_aa == "*":
        return out
    for pos in range(3):
        for base in _BASES:
            if base == codon[pos]:
                continue
            alt = codon[:pos] + base + codon[pos + 1:]
            aa = CODON_TABLE.get(alt, "*")
            if aa == "*" or aa == wt_aa or aa not in STANDARD_AAS:
                continue
            out.setdefault(aa, alt)
    return out


# ----------------------------------------------------------------------------- #
# 2. Config
# ----------------------------------------------------------------------------- #
@dataclass
class HotFlipConfig:
    max_length: int = 1024          # tokenizer max length (matches training)
    topk: int = 20                  # # of gradient-ranked candidates to really evaluate
    exclude_variant_pos: bool = True
    decision_threshold: Optional[float] = None  # PLLR threshold; None -> ln(3) (prob>0.5)
    loss_eps: float = 1e-6          # numeric clamp for the calibrated prob in the loss
    device: str = "cuda"
    verbose_every: int = 25


def default_threshold() -> float:
    # prob = 2*sigmoid(PLLR)-1 > 0.5  <=>  sigmoid(PLLR) > 0.75  <=>  PLLR > ln(3)
    return math.log(3.0)


# ----------------------------------------------------------------------------- #
# 3. The attacker
# ----------------------------------------------------------------------------- #
class HotFlipSNVAttacker:
    """Runs the SNV-reachable HotFlip attack on a Siamese ESM PLLR model.

    Parameters
    ----------
    model : the SiameseNetwork (must expose `.base_model`, an ESM*ForMaskedLM).
    tokenizer : the matching HF tokenizer.
    cfg : HotFlipConfig
    """

    def __init__(self, model, tokenizer, cfg: HotFlipConfig = HotFlipConfig()):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available()
                                    or cfg.device == "cpu" else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.base = model.base_model
        self.embed = self.base.get_input_embeddings()      # word embedding table
        self.E = self.embed.weight                         # [V, H]
        vocab = tokenizer.get_vocab()
        self.aa2id = {aa: vocab[aa] for aa in STANDARD_AAS if aa in vocab}
        self.thr = cfg.decision_threshold if cfg.decision_threshold is not None \
            else default_threshold()

    # ---- low level: tokenize a sequence to fixed-length ids -------------- #
    def _ids(self, seq: str) -> torch.Tensor:
        t = self.tok(seq, return_tensors="pt", truncation=True,
                     padding="max_length", max_length=self.cfg.max_length)
        return t["input_ids"].to(self.device), t["attention_mask"].to(self.device)

    # ---- PLL for a batch of inputs_embeds (differentiable) --------------- #
    def _pll_from_embeds(self, inputs_embeds, attention_mask, obs_ids):
        """obs_ids: [B, L] observed token ids (for gather). Returns PLL [B]."""
        out = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logp = torch.log_softmax(out.logits, dim=-1)          # [B, L, V]
        # gather log-prob of the observed residue at positions 1..L-2 (skip cls/eos)
        gathered = torch.gather(logp, 2, obs_ids.unsqueeze(-1)).squeeze(-1)  # [B, L]
        # mask out cls (0), eos and padding via attention; also drop first/last
        mask = attention_mask.clone().float()
        mask[:, 0] = 0.0                                       # <cls>
        # zero the last real token (<eos>) per row
        last = attention_mask.sum(dim=1) - 1                  # index of <eos>
        for b in range(mask.shape[0]):
            mask[b, int(last[b].item())] = 0.0
        return (gathered * mask).sum(dim=1)                   # [B]

    # ---- PLLR + loss for wt/mut given inputs_embeds ---------------------- #
    def _pllr_loss(self, emb_wt, emb_mut, am, ids_wt, ids_mut, label):
        embeds = torch.cat([emb_wt, emb_mut], dim=0)          # [2, L, H]
        ams = torch.cat([am, am], dim=0)
        obs = torch.cat([ids_wt, ids_mut], dim=0)
        plls = self._pll_from_embeds(embeds, ams, obs)        # [2]
        pllr = torch.abs(plls[0] - plls[1]).reshape(1)        # [1]
        prob = (2 * torch.sigmoid(pllr) - 1).clamp(self.cfg.loss_eps, 1 - self.cfg.loss_eps)
        loss = F.binary_cross_entropy(prob, label.reshape(1))  # maximize this to attack
        return pllr.squeeze(), loss

    # ---- real (no-grad) PLLR for two strings ----------------------------- #
    @torch.no_grad()
    def pllr_of(self, wt_seq: str, mut_seq: str) -> float:
        idw, amw = self._ids(wt_seq)
        idm, amm = self._ids(mut_seq)
        embeds = torch.cat([self.embed(idw), self.embed(idm)], dim=0)
        ams = torch.cat([amw, amm], dim=0)
        obs = torch.cat([idw, idm], dim=0)
        plls = self._pll_from_embeds(embeds, ams, obs)
        return float(torch.abs(plls[0] - plls[1]).item())

    # ---- attack a single variant ---------------------------------------- #
    def attack_one(self, wt_seq: str, mut_seq: str, wt_dna: str,
                   var_pos: int, label: int) -> dict:
        """Returns dict with clean/attacked PLLR and the chosen substitution."""
        cfg = self.cfg
        clean_pllr = self.pllr_of(wt_seq, mut_seq)

        # tokenize once; build differentiable inputs_embeds for both branches
        idw, amw = self._ids(wt_seq)
        idm, amm = self._ids(mut_seq)
        emb_wt = self.embed(idw).detach().clone().requires_grad_(True)
        emb_mut = self.embed(idm).detach().clone().requires_grad_(True)
        label_t = torch.tensor([float(label)], device=self.device)

        pllr, loss = self._pllr_loss(emb_wt, emb_mut, amw, idw, idm, label_t)
        self.base.zero_grad(set_to_none=True)
        loss.backward()
        g_wt = emb_wt.grad[0]      # [L, H]
        g_mut = emb_mut.grad[0]    # [L, H]
        g = g_wt + g_mut           # same background substitution hits both branches

        # build candidate (position, to_aa) list under the SNV constraint
        L_prot = len(wt_seq)
        cand_scores: List[Tuple[float, int, str, str]] = []  # (score, p, from_aa, to_aa)
        for p in range(L_prot):
            if cfg.exclude_variant_pos and p == var_pos:
                continue
            a = wt_seq[p]                       # background residue (shared wt/mut)
            if a not in self.aa2id:
                continue
            codon = wt_dna[3 * p:3 * p + 3]
            reach = snv_reachable_aas(codon)
            if not reach:
                continue
            t = p + 1                           # token index (offset by <cls>)
            ea = self.E[self.aa2id[a]]
            for b in reach:
                if b not in self.aa2id or b == a:
                    continue
                delta = self.E[self.aa2id[b]] - ea          # [H]
                score = float(torch.dot(delta, g[t]).item())  # first-order loss change
                cand_scores.append((score, p, a, b))

        if not cand_scores:
            return dict(clean_pllr=clean_pllr, attacked_pllr=clean_pllr,
                        pos=-1, from_aa="", to_aa="", n_candidates=0, flipped=False)

        # rank by estimated loss increase (descending) and really evaluate top-K
        cand_scores.sort(key=lambda x: x[0], reverse=True)
        best = dict(attacked_pllr=clean_pllr, loss=-1e9, pos=-1, from_aa="", to_aa="")
        for (_, p, a, b) in cand_scores[:cfg.topk]:
            new_wt = wt_seq[:p] + b + wt_seq[p + 1:]
            new_mut = mut_seq[:p] + b + mut_seq[p + 1:]
            real_pllr = self.pllr_of(new_wt, new_mut)
            prob = 2 * (1 / (1 + math.exp(-real_pllr))) - 1
            prob = min(max(prob, cfg.loss_eps), 1 - cfg.loss_eps)
            real_loss = -(label * math.log(prob) + (1 - label) * math.log(1 - prob))
            if real_loss > best["loss"]:
                best = dict(attacked_pllr=real_pllr, loss=real_loss,
                            pos=p, from_aa=a, to_aa=b)

        clean_pred = int(clean_pllr > self.thr)
        att_pred = int(best["attacked_pllr"] > self.thr)
        flipped = (clean_pred == label) and (att_pred != label)
        return dict(clean_pllr=clean_pllr, attacked_pllr=best["attacked_pllr"],
                    pos=best["pos"], from_aa=best["from_aa"], to_aa=best["to_aa"],
                    n_candidates=len(cand_scores), flipped=bool(flipped))

    # ---- run over a whole test set -------------------------------------- #
    def run(self, prot_csv: str, dna_csv: str, save_path: Optional[str] = None) -> dict:
        prot = pd.read_csv(prot_csv)
        dna = pd.read_csv(dna_csv)
        assert len(prot) == len(dna), "protein/DNA CSVs must be row-aligned"

        rows = []
        for i in range(len(prot)):
            wt_seq = prot.loc[i, "wt_seq"]
            mut_seq = prot.loc[i, "mut_seq"]
            wt_dna = dna.loc[i, "wt_seq"]
            label = int(prot.loc[i, "labels"])
            # variant residue index = first wt/mut difference (robust to start_pos sign)
            diffs = [j for j, (x, y) in enumerate(zip(wt_seq, mut_seq)) if x != y]
            var_pos = diffs[0] if diffs else -1
            r = self.attack_one(wt_seq, mut_seq, wt_dna, var_pos, label)
            r["label"] = label
            r["var_pos"] = var_pos
            rows.append(r)
            if self.cfg.verbose_every and (i % self.cfg.verbose_every == 0):
                print(f"[{i+1}/{len(prot)}] label={label} "
                      f"clean_PLLR={r['clean_pllr']:.3f} att_PLLR={r['attacked_pllr']:.3f} "
                      f"flip={r['flipped']} bg_sub={r['from_aa']}{r['pos']}->{r['to_aa']} "
                      f"(variant@{var_pos})")

        df = pd.DataFrame(rows)
        y = df["label"].to_numpy()
        clean_auc = roc_auc_score(y, df["clean_pllr"])
        clean_aupr = average_precision_score(y, df["clean_pllr"])
        att_auc = roc_auc_score(y, df["attacked_pllr"])
        att_aupr = average_precision_score(y, df["attacked_pllr"])

        correct_clean = ((df["clean_pllr"] > self.thr).astype(int) == y)
        success = df["flipped"].sum() / max(int(correct_clean.sum()), 1)

        summary = dict(clean_AUC=clean_auc, clean_AUPR=clean_aupr,
                       attacked_AUC=att_auc, attacked_AUPR=att_aupr,
                       attack_success_rate=float(success),
                       n=len(df), n_correct_clean=int(correct_clean.sum()),
                       n_flipped=int(df["flipped"].sum()))
        print("\n==== SNV-reachable HotFlip summary ====")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"  saved per-variant results -> {save_path}")
        return dict(summary=summary, table=df)


# ----------------------------------------------------------------------------- #
# 4. CLI / usage note
# ----------------------------------------------------------------------------- #


# ---------------------------------------------------------------- cell 12
def compute_metrics_PLLR(eval_pred):
    PLLR, labels = eval_pred
    auc = roc_auc_score(labels, PLLR)
    aupr = average_precision_score(labels, PLLR)
    return {
        'auc': auc,
        'aupr':aupr
    }


# ---------------------------------------------------------------- cell 13
class CustomTrainer(Trainer):
    def evaluate_with_hotflip(self, prot_csv, dna_csv, topk=20, save_path=None,
                              verbose_every=25):
        """Discrete SNV-reachable HotFlip evaluation (replaces evaluate_with_fgsm).

        prot_csv / dna_csv : the codon-aligned protein and DNA test CSVs.
        topk               : # of gradient-ranked candidates really evaluated per variant.
        verbose_every      : print a progress line every N variants (1 = every variant).
        """
        print("\nEvaluating with SNV-reachable HotFlip (discrete adversarial test)...")
        self.model.eval()
        cfg = HotFlipConfig(
            max_length=training_args.model_max_length,
            topk=topk,
            exclude_variant_pos=True,          # label-preserving: never touch the variant
            device=str(self.args.device),
            verbose_every=verbose_every,
        )
        attacker = HotFlipSNVAttacker(self.model, tokenizer, cfg)
        out = attacker.run(prot_csv, dna_csv, save_path=save_path)
        return out["summary"]


# ---------------------------------------------------------------- cell 14
# Define compute_metrics for evaluation
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return {
#         'accuracy': (predictions == labels).mean()
#     }

def custom_data_collator(data):
    # Here, we ensure that each item in `data` has the necessary keys.
    input_ids1 = torch.stack([item['input_ids1'] for item in data])
    attention_mask1 = torch.stack([item['attention_mask1'] for item in data])
    input_ids2 = torch.stack([item['input_ids2'] for item in data])
    attention_mask2 = torch.stack([item['attention_mask2'] for item in data])

    # Ensure labels exist or handle its absence
    #labels = [item.get('labels', torch.tensor(-1)) for item in data]  # Using -1 as a default
    #labels = torch.stack(labels)
    labels = torch.stack([item['labels'] for item in data])

    return {
        'input_ids1': input_ids1,
        'attention_mask1': attention_mask1,
        'input_ids2': input_ids2,
        'attention_mask2': attention_mask2,
        'labels': labels
    }

custom_callback_instance = CustomCallback(tokenizer=tokenizer)

# Define Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_PLLR,
    data_collator=custom_data_collator,
    callbacks=[custom_callback_instance]
)


# ---------------------------------------------------------------- cell 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============================================================
# STEP 1 - Fine-tune the disease-specialized variant-effect predictor
# ============================================================
# Match your established recipe (paper Settings: lr 1e-4, 10 epochs, batch 4).
# (training_args is the same object held by the trainer, so these take effect.)
training_args.num_train_epochs = 10
training_args.learning_rate = 1e-4
training_args.per_device_train_batch_size = 4

FINETUNE = True   # set False to skip training and load a saved checkpoint instead
CKPT = os.path.join(training_args.output_dir, f"finetuned_{DATASET}_esm2_650m.bin")

if FINETUNE:
    trainer.train()
    os.makedirs(training_args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    print("saved fine-tuned checkpoint ->", CKPT)
elif os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT, map_location=device), strict=False)
    print("loaded fine-tuned checkpoint <-", CKPT)
model.to(device).eval()

# ============================================================
# STEP 2 - SNV-reachable HotFlip on the fine-tuned model
# (HotFlip changes only input residues; model weights stay frozen during the attack)
# ============================================================
prot_csv = os.path.join(data_path, f"{DATASET}_test_data_protein_1024.csv")
dna_csv  = os.path.join(data_path, f"{DATASET}_test_data_dna_3072.csv")
save_csv = os.path.join(training_args.output_dir, f"hotflip_snv_{DATASET}_test.csv")

summary = trainer.evaluate_with_hotflip(prot_csv, dna_csv, topk=20, save_path=save_csv,
                                        verbose_every=1)   # 1 = print every variant
print("\nClean  = specialized fine-tuned model (compare vs. clinical frameworks)")
print("Attacked = after a single biologically valid SNV")
print("HotFlip summary:", summary)

