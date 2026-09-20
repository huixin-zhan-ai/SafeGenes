"""
PGD attack with learned Platt calibration, Brier/ECE and FPR@95% sensitivity

Paper element : PGD rows of Tables 2-3; Table S6; Figure S4
Source        : VEP_ESM_PLLR_PGD_calibration.ipynb

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

# ---------------------------------------------------------------- cell 1
# [notebook-only] from google.colab import drive
# [notebook-only] drive.mount('/content/drive', force_remount=True)

# ---------------------------------------------------------------- cell 2
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

# ---------------------------------------------------------------- cell 3
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
    save_strategy: str = field(default="no")  # fix: no checkpoints -> avoids safetensors tied-weight error        # ✅ match
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


# ---------------------------------------------------------------- cell 4
@dataclass
class ModelArguments:
    #model_name_or_path: Optional[str] = field(default="bert-base-uncased")
    #model_name_or_path: Optional[str] = field(default="facebook/esm1b_t33_650M_UR50S")
    model_name_or_path: Optional[str] = field(default="facebook/esm2_t33_650M_UR50D")
    #model_name_or_path: Optional[str] = field(default="facebook/esm2_t30_150M_UR50D")
model_args = ModelArguments()

# ---------------------------------------------------------------- cell 5
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
            df = pd.read_csv(data_path+"cm_test_data_1024.csv")
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

# ---------------------------------------------------------------- cell 6
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


# ---------------------------------------------------------------- cell 7
from transformers import EsmConfig, AutoModelForMaskedLM

class SiameseNetwork(nn.Module):
    def __init__(self, model_name_or_path, num_labels, cache_dir=None):
        super(SiameseNetwork, self).__init__()
        config = EsmConfig.from_pretrained(model_name_or_path,
                                           token_dropout=False, output_hidden_states=True)
        self.base_model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir)

    def _pllr(self, logits1, logits2, input_ids1, input_ids2):
        B = input_ids1.shape[0]
        P1 = torch.zeros(B, device=input_ids1.device)
        P2 = torch.zeros(B, device=input_ids2.device)
        for i in range(B):
            idx1 = input_ids1[i, 1:-1]
            P1[i] = torch.sum(torch.diag(logits1[i, 1:-1, :][:, idx1]))
            idx2 = input_ids2[i, 1:-1]
            P2[i] = torch.sum(torch.diag(logits2[i, 1:-1, :][:, idx2]))
        return P1, P2   # PLL(wt), PLL(mut)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels,
                test_mode=False, epsilon=0.01, attack=None, return_signed=False):
        # attack=None -> follow eval/train mode; attack=True/False -> force PGD on/off.
        do_pgd = (not self.training) if attack is None else bool(attack)

        outputs1 = self.base_model(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.base_model(input_ids=input_ids2, attention_mask=attention_mask2)
        embeddings1 = outputs1.hidden_states[0]
        embeddings2 = outputs2.hidden_states[0]

        logits1 = torch.log_softmax(outputs1.logits, dim=-1)
        logits2 = torch.log_softmax(outputs2.logits, dim=-1)
        PLLs1, PLLs2 = self._pllr(logits1, logits2, input_ids1, input_ids2)

        signed = PLLs1 - PLLs2                 # signed PLLR: PLL(wt) - PLL(mut)
        PLLR   = torch.abs(signed)             # absolute PLLR (current formulation)
        sigmoid_PLLR = torch.sigmoid(PLLR)
        pll_loss = F.binary_cross_entropy(2 * sigmoid_PLLR - 1, labels.float())

        if do_pgd:
            num_steps = 10
            alpha = epsilon / num_steps
            orig1 = embeddings1.detach(); orig2 = embeddings2.detach()
            pe1 = orig1.clone().detach().requires_grad_(True)
            pe2 = orig2.clone().detach().requires_grad_(True)
            for _ in range(num_steps):
                if pe1.grad is not None: pe1.grad.zero_()
                if pe2.grad is not None: pe2.grad.zero_()
                o1 = self.base_model(inputs_embeds=pe1, attention_mask=attention_mask1)
                o2 = self.base_model(inputs_embeds=pe2, attention_mask=attention_mask2)
                l1 = torch.log_softmax(o1.logits, dim=-1)
                l2 = torch.log_softmax(o2.logits, dim=-1)
                P1, P2 = self._pllr(l1, l2, input_ids1, input_ids2)
                pllr = torch.abs(P1 - P2)
                loss = F.binary_cross_entropy(2 * torch.sigmoid(pllr) - 1, labels.float())
                loss = loss + 1e-6 * (torch.sum(pe1 ** 2) + torch.sum(pe2 ** 2))
                loss.backward(retain_graph=True)
                pe1 = pe1 + alpha * pe1.grad.sign()
                pe2 = pe2 + alpha * pe2.grad.sign()
                pe1 = (orig1 + torch.clamp(pe1 - orig1, -epsilon, epsilon)).detach().requires_grad_(True)
                pe2 = (orig2 + torch.clamp(pe2 - orig2, -epsilon, epsilon)).detach().requires_grad_(True)
            with torch.no_grad():
                f1 = self.base_model(inputs_embeds=pe1, attention_mask=attention_mask1)
                f2 = self.base_model(inputs_embeds=pe2, attention_mask=attention_mask2)
                fl1 = torch.log_softmax(f1.logits, dim=-1)
                fl2 = torch.log_softmax(f2.logits, dim=-1)
                PLLs1, PLLs2 = self._pllr(fl1, fl2, input_ids1, input_ids2)
                signed = PLLs1 - PLLs2
                PLLR = torch.abs(signed)

        if return_signed:
            return pll_loss, PLLR, signed
        return pll_loss, PLLR



# ---------------------------------------------------------------- cell 8
model = SiameseNetwork(model_args.model_name_or_path, num_labels=2)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                       model_max_length=training_args.model_max_length,)
                                                       #padding_side="right",
                                                       #use_fast=True,
                                                       #trust_remote_code=True)
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_name_or_path,
#         model_max_length=512,
#         padding_side="right",
#         use_fast=True,
#         trust_remote_code=True,
#     )
print("Tokenizer class:", tokenizer.__class__)
print("Tokenizer name:", tokenizer.__class__.__name__)

#train_dataset = SiameseDataset(tokenizer, 900)
#test_dataset = SiameseDataset(tokenizer, 100)
train_dataset = SiameseDataset(tokenizer, os.path.join(data_path, 'cm_train_data_1024.csv'))
test_dataset = SiameseDataset(tokenizer, os.path.join(data_path, 'cm_test_data_1024.csv'))
#test_dataset = SiameseDataset(tokenizer, os.path.join(data_path, 'Clinvar_uncertain_ARM_protein.csv'))
########test the sample size#########
from io import StringIO
df = pd.read_csv(os.path.join(data_path, 'cm_train_data_1024.csv'))

# Sample 10% of the data randomly
df_sampled = df.sample(frac=1.0, random_state=123)  # random_state ensures reproducibility

# Use StringIO to save the sampled dataframe in-memory
buffer = StringIO()
df_sampled.to_csv(buffer, index=False)
buffer.seek(0)  # Reset buffer position to the beginning

# Now pass this buffer to the SiameseDataset
train_dataset = SiameseDataset(tokenizer, buffer)


# ---------------------------------------------------------------- cell 9
# # Multiple sequences
# sequence = "MKLWTA"
# encoded_sequence = tokenizer.encode(sequence, add_special_tokens=True)

# # Output as a list of token IDs
# print("Token IDs:", encoded_sequence)

# # Convert token IDs back to tokens
# tokens = tokenizer.convert_ids_to_tokens(encoded_sequence)
# print("Tokens:", tokens)

# # Multiple sequences
# sequences = ["MKLWTA", "GATCRY"]
# encoded_sequences = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)

# # Output as lists of token IDs
# print("Token IDs:", encoded_sequences['input_ids'])

# # Convert token IDs back to tokens for each sequence
# for i, ids in enumerate(encoded_sequences['input_ids']):
#     print(f"Tokens for sequence {i+1}: {tokenizer.convert_ids_to_tokens(ids)}")



# input_df = pd.read_csv('/common/zhanh/Cardioboost_protein_sequences/cm_train_protein_seq_df.csv')
# print(input_df.loc[0, 'Original_Protein_Sequence'])
# encoded_sequence1 = tokenizer.encode(input_df.loc[0, 'Original_Protein_Sequence'], add_special_tokens=True)

# # Output as a list of token IDs
# print("Token IDs:", encoded_sequence1,len(encoded_sequence1))

# # Convert token IDs back to tokens
# tokens1 = tokenizer.convert_ids_to_tokens(encoded_sequence1)
# print("Tokens:", tokens1)

# print(input_df.loc[0, 'Mutated_Protein_Sequence'])
# encoded_sequence2 = tokenizer.encode(input_df.loc[0, 'Mutated_Protein_Sequence'], add_special_tokens=True)

# # Output as a list of token IDs
# print("Token IDs:", encoded_sequence2,len(encoded_sequence2))

# # Convert token IDs back to tokens
# tokens2 = tokenizer.convert_ids_to_tokens(encoded_sequence2)
# print("Tokens:", tokens2)

# differences = []
# for i, (char1, char2) in enumerate(zip(encoded_sequence1, encoded_sequence2)):
#     if char1 != char2:
#         differences.append((i, char1, char2))

# print(f"Differences found at these positions: {differences}")

# ---------------------------------------------------------------- cell 10
def compute_metrics(eval_pred):
    cosine_sims, labels = eval_pred
    mse = ((cosine_sims - labels)**2).mean()
    # Flip the sign of the cosine similarities because we want -1 for label 1 and 1 for label 0
    flipped_cosine_sims = -cosine_sims

    # Convert these flipped values to "probabilities" in [0, 1]
    probabilities = (flipped_cosine_sims + 1) / 2  # Now values are between 0 and 1

    # Make binary predictions based on a threshold (e.g., 0.7)
    predictions = (probabilities > 0.1).astype(np.int32)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    return {
        'mse': mse,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

# ---------------------------------------------------------------- cell 11
def compute_metrics_PLLR(eval_pred):
    PLLR, labels = eval_pred
    auc = roc_auc_score(labels, PLLR)
    aupr = average_precision_score(labels, PLLR)
    return {
        'auc': auc,
        'aupr':aupr
    }

# ---------------------------------------------------------------- cell 12
class CustomTrainer(Trainer):
    def evaluate_with_fgsm(self, eval_dataset=None, epsilon=0.01):
        self.model.eval()
        dl = self.get_eval_dataloader(eval_dataset)
        pllrs, ys = [], []
        for batch in dl:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            _, PLLR = self.model(input_ids1=batch["input_ids1"], attention_mask1=batch["attention_mask1"],
                                 input_ids2=batch["input_ids2"], attention_mask2=batch["attention_mask2"],
                                 labels=batch["labels"], test_mode=True, epsilon=epsilon)
            pllrs.extend(PLLR.cpu().detach().numpy()); ys.extend(batch["labels"].cpu().detach().numpy())
        auc = roc_auc_score(ys, pllrs); aupr = average_precision_score(ys, pllrs)
        print(f"Adversarial AUC={auc:.4f} AUPR={aupr:.4f}")
        return {"AUC": auc, "AUPR": aupr}

    def collect_pllr(self, eval_dataset=None, do_attack=False, epsilon=0.01):
        """Return (abs_pllr, signed_pllr, labels) as numpy arrays.
        do_attack=False -> clean; True -> PGD-attacked. Order is deterministic (eval loader)."""
        import numpy as np
        self.model.eval()
        dl = self.get_eval_dataloader(eval_dataset)
        abs_l, sgn_l, y_l = [], [], []
        for batch in dl:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            kw = dict(input_ids1=batch["input_ids1"], attention_mask1=batch["attention_mask1"],
                      input_ids2=batch["input_ids2"], attention_mask2=batch["attention_mask2"],
                      labels=batch["labels"], return_signed=True)
            if do_attack:
                _, pllr, signed = self.model(**kw, test_mode=True, attack=True, epsilon=epsilon)
            else:
                with torch.no_grad():
                    _, pllr, signed = self.model(**kw, test_mode=False, attack=False)
            abs_l.extend(pllr.detach().cpu().numpy().tolist())
            sgn_l.extend(signed.detach().cpu().numpy().tolist())
            y_l.extend(batch["labels"].detach().cpu().numpy().tolist())
        return np.array(abs_l), np.array(sgn_l), np.array(y_l)



# ---------------------------------------------------------------- cell 13
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




# ---------------------------------------------------------------- cell 14
# Let's assume you've already loaded your trained model into a variable named `model`.

# Create tokens for your test sequences using your tokenizer.
# Assuming `tokenizer` is your tokenizer and `test_seq1` and `test_seq2` are your test sequences.
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Move model to device
model.to(device)

test_seq1 = "LAGVER"
test_seq2 = "<mask><mask><mask><mask><mask><mask>"
encoded_sequence = tokenizer.encode(test_seq1, add_special_tokens=True)

# # # Output as a list of token IDs
print("Token IDs:", encoded_sequence)
encoded_sequence = tokenizer.encode(test_seq2, add_special_tokens=True)

# # # Output as a list of token IDs
print("Token IDs:", encoded_sequence)
tokens1 = tokenizer(test_seq1, return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)
tokens2 = tokenizer(test_seq2, return_tensors="pt", truncation=True, padding="max_length", max_length=training_args.model_max_length)  # Assuming test_seq2 contains mask tokens.

label = torch.tensor([1], dtype=torch.int64).to(device)

# # Move tensors to device
tokens1 = {k: v.to(device) for k, v in tokens1.items()}
tokens2 = {k: v.to(device) for k, v in tokens2.items()}

with torch.no_grad():
     outputs1 = model.base_model(input_ids=tokens1['input_ids'], attention_mask=tokens1['attention_mask'])
     output1 = outputs1.hidden_states[-1][:, 0, :]

     outputs2 = model.base_model(input_ids=tokens2['input_ids'], attention_mask=tokens2['attention_mask'])
     output2 = outputs2.hidden_states[-1][:, 0, :]

# # Examine the outputs
print("Output1 shape:", outputs1.logits.shape)
print("Output2 shape:", outputs2.logits.shape)

logits = torch.log_softmax(outputs1.logits, dim=-1)
s = logits[0][1:-1,:].shape
alphabet = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
idx = [alphabet[t] for t in test_seq1]
PLL = torch.sum(torch.diag(logits[0, 1:-1, :][:, idx]))


# ---------------------------------------------------------------- cell 15
import numpy as np, pandas as pd, os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tune (save_strategy="no" avoids the safetensors tied-weight save error).
trainer.train()

# Collect PLLR (abs + signed):
#   train (clean)  -> used to FIT the learned calibration (no leakage: fit on train, apply to test)
#   test  (clean)  and  test (PGD-attacked)
tr_abs_c, tr_sgn_c, tr_y = trainer.collect_pllr(train_dataset, do_attack=False)
te_abs_c, te_sgn_c, te_y = trainer.collect_pllr(test_dataset,  do_attack=False)
te_abs_a, te_sgn_a, _    = trainer.collect_pllr(test_dataset,  do_attack=True, epsilon=0.01)

os.makedirs(training_args.output_dir, exist_ok=True)
save_csv = os.path.join(training_args.output_dir, "pllr_pgd_calibration.csv")
pd.DataFrame({"pllr_abs_clean": te_abs_c, "pllr_signed_clean": te_sgn_c,
              "pllr_abs_att":   te_abs_a, "pllr_signed_att":   te_sgn_a,
              "label": te_y}).to_csv(save_csv, index=False)
pd.DataFrame({"pllr_abs_clean": tr_abs_c, "pllr_signed_clean": tr_sgn_c,
              "label": tr_y}).to_csv(os.path.join(training_args.output_dir, "pllr_pgd_train.csv"), index=False)
print("saved:", save_csv)



# ---------------------------------------------------------------- cell 17
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def ece_score(probs, labels, n_bins=10):
    probs = np.clip(np.asarray(probs, float), 0, 1); labels = np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; N = len(labels)
    for i in range(n_bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1])
        if m.sum() == 0: continue
        ece += abs(labels[m].mean() - probs[m].mean()) * m.sum() / N
    return ece

def reliability_points(probs, labels, n_bins=10):
    probs = np.clip(np.asarray(probs, float), 0, 1); labels = np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1); xs, ys = [], []
    for i in range(n_bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1])
        if m.sum() == 0: continue
        xs.append(probs[m].mean()); ys.append(labels[m].mean())
    return xs, ys

sig = lambda x: 1 / (1 + np.exp(-x))

# ---- fit learned (Platt) calibrators on the CLEAN TRAIN split ----
platt_abs    = LogisticRegression().fit(tr_abs_c.reshape(-1, 1), tr_y)   # |PLLR| -> prob
platt_signed = LogisticRegression().fit(tr_sgn_c.reshape(-1, 1), tr_y)   # signed -> prob

def probs_for(abs_pllr, signed):
    return {
        "current: 2*sigmoid(|PLLR|)-1": 2 * sig(abs_pllr) - 1,
        "learned Platt(|PLLR|)":        platt_abs.predict_proba(abs_pllr.reshape(-1, 1))[:, 1],
        "learned Platt(signed)":        platt_signed.predict_proba(signed.reshape(-1, 1))[:, 1],
    }

conditions = {"clean": (te_abs_c, te_sgn_c), "attacked": (te_abs_a, te_sgn_a)}

# ---- ranking: |PLLR| vs signed (justify the absolute value) ----
print("== Ranking (score AUROC / AUPR) ==")
for cond, (a, s) in conditions.items():
    print(f"  [{cond}]  |PLLR|: AUROC={roc_auc_score(te_y, a):.3f} AUPR={average_precision_score(te_y, a):.3f}"
          f"   signed: AUROC={roc_auc_score(te_y, s):.3f} AUPR={average_precision_score(te_y, s):.3f}")

# ---- Brier / ECE for each probability mapping, clean and attacked + reliability diagrams ----
rows = []
for cond, (a, s) in conditions.items():
    pm = probs_for(a, s)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    for name, p in pm.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        brier = brier_score_loss(te_y, p); ece = ece_score(p, te_y)
        rows.append({"condition": cond, "mapping": name, "Brier": brier, "ECE": ece})
        xs, ys = reliability_points(p, te_y)
        ax.plot(xs, ys, "o-", label=f"{name} (ECE={ece:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability diagram ({cond})"); ax.legend(fontsize=8, loc="upper left")
    out_pdf = os.path.join(training_args.output_dir, f"reliability_{cond}.pdf")
    plt.tight_layout(); plt.savefig(out_pdf); plt.close()
    print(f"saved {out_pdf}")

metrics = pd.DataFrame(rows)
print("\n== Brier / ECE ==\n", metrics.to_string(index=False))
metrics.to_csv(os.path.join(training_args.output_dir, "calibration_metrics.csv"), index=False)
print("saved calibration_metrics.csv")


