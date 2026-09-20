"""
Prompt-only soft-prompt attack and cross-model transfer

Paper element : Tables S4-S5
Source        : VEP_ESM_PLLR_Soft_Prompt_PromptOnly_CH.ipynb

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
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#torch.manual_seed(0)
#np.random.seed(0)
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

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
    num_train_epochs: int = field(default=10)
    fp16: bool = field(default=False)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")  # ✅ match
    save_strategy: str = field(default="no")  # no Trainer checkpoints (avoids safetensors tied-weight error)        # ✅ match
    save_safetensors: bool = field(default=False)
    #load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=500)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=1e-2)
    learning_rate: float = field(default=1e-2)  # prompt tuning needs a high LR
    lr_scheduler_type: str = field(default='linear')
    save_total_limit: int = field(default=10)
    output_dir: str = field(default="/content/drive/My Drive/vep_FGSM/output_soft_prompt_confidence")
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
    #model_name_or_path: Optional[str] = field(default="facebook/esm1v_t33_650M_UR90S_1")
    #model_name_or_path: Optional[str] = field(default="facebook/esm2_t33_650M_UR50D")
    model_name_or_path: Optional[str] = field(default="facebook/esm2_t30_150M_UR50D")
model_args = ModelArguments()

# ---------------------------------------------------------------- cell 5
soft_prompt_length = 50
MAX_MODEL_LEN = training_args.model_max_length - soft_prompt_length  # e.g., 1024 - 50 = 974


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
            df = pd.read_csv(data_path+"arm_test_data_1024.csv")
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
        inputs_a = self.tokenizer(self.seq_a[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_MODEL_LEN)
        inputs_b = self.tokenizer(self.seq_b[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_MODEL_LEN)

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
def ranking_loss(pllr, labels, margin=10):
        """
        pllrs: tensor of shape [B], PLLR values
        labels: tensor of shape [B], binary labels (1=pathogenic, 0=benign)
        """
        benign_mask = labels == 0
        path_mask = labels == 1

        benign_scores = pllr[benign_mask]  # shape [B0]
        path_scores = pllr[path_mask]      # shape [B1]

        # Create all pairwise combinations between benign and pathogenic
        if benign_scores.numel() == 0 or path_scores.numel() == 0:
            return torch.tensor(0.0, device=pllr.device, requires_grad=True)

        benign_scores = benign_scores.unsqueeze(1)  # [B0, 1]
        path_scores = path_scores.unsqueeze(0)      # [1, B1]

        # Compute pairwise margin loss (attack direction)
        diff = benign_scores - path_scores  # [B0, B1]
        loss_matrix = F.relu(margin - diff)
        return loss_matrix.mean()

# ---------------------------------------------------------------- cell 9
from transformers import EsmConfig, AutoModelForMaskedLM

class SiameseNetwork(nn.Module):
    def __init__(self, model_name_or_path, num_labels, soft_prompt_length=50, cache_dir=None):
        super(SiameseNetwork, self).__init__()

        config = EsmConfig.from_pretrained(model_name_or_path,
                                           token_dropout=False,
                                           output_hidden_states=True)

        self.base_model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir)

        # ==== PURE (prompt-only) soft prompt attack ====
        # Freeze the entire backbone so trainer.train() updates ONLY the soft prompt
        # (clean frozen-backbone, prompt-only attack; addresses Reviewer 1.2).
        # To attack a fine-tuned model, load its checkpoint into self.base_model BEFORE this freeze.
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.randn(soft_prompt_length, hidden_size))
        self.soft_prompt_length = soft_prompt_length

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels,
                test_mode=False, epsilon=0.01, use_soft_prompt=True, n_prompt_tokens=None):
        """
        use_soft_prompt : False -> clean / regular model (no prompt).
        n_prompt_tokens : how many of the leading soft-prompt tokens to prepend
                          (None -> all soft_prompt_length; e.g. 5, 10, n). Must be
                          <= soft_prompt_length. The PLL is computed over the REAL
                          residues only, skipping exactly these prompt positions.
        """
        batch_size = input_ids1.size(0)

        # Input embeddings
        input_embeds1 = self.base_model.base_model.embeddings(input_ids1)   # [B, L, H]
        input_embeds2 = self.base_model.base_model.embeddings(input_ids2)

        if use_soft_prompt:
            n = self.soft_prompt_length if n_prompt_tokens is None else int(n_prompt_tokens)
            n = max(0, min(n, self.soft_prompt_length))                     # first-n prompt tokens
            soft_prompt = self.soft_prompt[:n].unsqueeze(0).expand(batch_size, -1, -1)
            input_embeds1 = torch.cat([soft_prompt, input_embeds1], dim=1)
            input_embeds2 = torch.cat([soft_prompt, input_embeds2], dim=1)
            prompt_mask = torch.ones(batch_size, n, device=input_ids1.device)
            attention_mask1 = torch.cat([prompt_mask, attention_mask1], dim=1)
            attention_mask2 = torch.cat([prompt_mask, attention_mask2], dim=1)
            offset = n                                                      # # prepended positions
        else:
            offset = 0

        outputs1 = self.base_model(inputs_embeds=input_embeds1, attention_mask=attention_mask1)
        outputs2 = self.base_model(inputs_embeds=input_embeds2, attention_mask=attention_mask2)

        logits1 = torch.log_softmax(outputs1.logits, dim=-1)
        logits2 = torch.log_softmax(outputs2.logits, dim=-1)

        PLLs1 = torch.zeros(batch_size, device=input_ids1.device)
        PLLs2 = torch.zeros(batch_size, device=input_ids2.device)

        # Score the REAL residues only: skip the `offset` prompt tokens and the <cls>.
        for i in range(batch_size):
            idx1 = input_ids1[i, 1:-1]                       # real residues (length L-2)
            start1 = offset + 1
            seg1 = logits1[i, start1:start1 + idx1.size(0), :]
            PLLs1[i] = torch.sum(torch.diag(seg1[:, idx1]))

            idx2 = input_ids2[i, 1:-1]
            start2 = offset + 1
            seg2 = logits2[i, start2:start2 + idx2.size(0), :]
            PLLs2[i] = torch.sum(torch.diag(seg2[:, idx2]))

        PLLR = torch.abs(PLLs1 - PLLs2)

        sigmoid_PLLR = torch.sigmoid(PLLR)
        sigma_hat = torch.clamp(2 * sigmoid_PLLR - 1, min=1e-4, max=1 - 1e-4)

        # ==== Confidence-hijack attack objective (flipped-label BCE, paper Eq. 3) ====
        # A single global soft prompt cannot collapse the ranking with a one-sided
        # (benign-only) loss, since it also affects pathogenic inputs. This two-sided
        # objective pushes benign -> HIGH PLLR and pathogenic -> LOW PLLR, directly
        # compressing the margin so ROC/PR AUC drop. Every batch contributes a gradient
        # (no zero-loss steps).
        y = labels.float()
        pll_loss = -((1.0 - y) * torch.log(sigma_hat) + y * torch.log(1.0 - sigma_hat)).mean()

        # ---- Targeted variant (benign -> pathogenic only; raises FPR, weaker on AUC).
        #      Uncomment to use instead of the confidence hijack above.
        # benign_mask = (labels == 0)
        # pll_loss = (-torch.log(sigma_hat[benign_mask]).mean()
        #             if benign_mask.sum() > 0
        #             else torch.tensor(0.0, device=PLLR.device, requires_grad=True))

        return pll_loss, PLLR



# ---------------------------------------------------------------- cell 10
model = SiameseNetwork(model_args.model_name_or_path, num_labels=2, soft_prompt_length=50)


tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                       model_max_length=MAX_MODEL_LEN,)
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
train_dataset = SiameseDataset(tokenizer, os.path.join(data_path, 'arm_train_data_1024.csv'))
test_dataset = SiameseDataset(tokenizer, os.path.join(data_path, 'arm_test_data_1024.csv'))
#test_dataset = SiameseDataset(tokenizer, os.path.join(data_path, 'Clinvar_uncertain_ARM_protein.csv'))
########test the sample size#########
from io import StringIO
df = pd.read_csv(os.path.join(data_path, 'arm_train_data_1024.csv'))

# Sample 10% of the data randomly
df_sampled = df.sample(frac=1.0, random_state=123)  # random_state ensures reproducibility

# Use StringIO to save the sampled dataframe in-memory
buffer = StringIO()
df_sampled.to_csv(buffer, index=False)
buffer.seek(0)  # Reset buffer position to the beginning

# Now pass this buffer to the SiameseDataset
train_dataset = SiameseDataset(tokenizer, buffer)


# ---------------------------------------------------------------- cell 11
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

# ---------------------------------------------------------------- cell 12
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

# ---------------------------------------------------------------- cell 13
def compute_metrics_PLLR(eval_pred):
    PLLR, labels = eval_pred
    auc = roc_auc_score(labels, PLLR)
    aupr = average_precision_score(labels, PLLR)
    return {
        'auc': auc,
        'aupr':aupr
    }

# ---------------------------------------------------------------- cell 14
class CustomTrainer(Trainer):
    def evaluate_with_fgsm(self, eval_dataset=None, epsilon=0.01):
        print("\nEvaluating with FGSM (Adversarial Test Mode Enabled)...")
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        pllrs, true_labels = [], []
        for batch in eval_dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            labels = batch["labels"]
            _, PLLR = self.model(
                input_ids1=batch["input_ids1"], attention_mask1=batch["attention_mask1"],
                input_ids2=batch["input_ids2"], attention_mask2=batch["attention_mask2"],
                labels=labels, test_mode=True, epsilon=epsilon)
            pllrs.extend(PLLR.cpu().detach().numpy())
            true_labels.extend(labels.cpu().detach().numpy())
        auc_score = roc_auc_score(true_labels, pllrs)
        aupr_score = average_precision_score(true_labels, pllrs)
        print(f"\nAdversarial AUC after FGSM perturbation: {auc_score:.4f}")
        print(f"Adversarial AUPR after FGSM perturbation: {aupr_score:.4f}")
        return {"AUC": auc_score, "AUPR": aupr_score}

    @torch.no_grad()
    def evaluate_and_save(self, tag, eval_dataset=None, use_soft_prompt=True, n_prompt_tokens=None):
        """Evaluate PLLR, print AUC/AUPR, and save a CSV.
        use_soft_prompt=False -> clean / regular model.
        n_prompt_tokens       -> use the first-n soft-prompt tokens (None = all)."""
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        pllrs, true_labels = [], []
        for batch in eval_dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            labels = batch["labels"]
            _, PLLR = self.model(
                input_ids1=batch["input_ids1"], attention_mask1=batch["attention_mask1"],
                input_ids2=batch["input_ids2"], attention_mask2=batch["attention_mask2"],
                labels=labels, use_soft_prompt=use_soft_prompt, n_prompt_tokens=n_prompt_tokens)
            pllrs.extend(PLLR.detach().cpu().numpy().tolist())
            true_labels.extend(labels.detach().cpu().numpy().tolist())
        auc_score = roc_auc_score(true_labels, pllrs)
        aupr_score = average_precision_score(true_labels, pllrs)
        print(f"[{tag}] AUC = {auc_score:.4f} | AUPR = {aupr_score:.4f} | n = {len(true_labels)}")
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_path = os.path.join(self.args.output_dir, f"pllr_{tag}.csv")
        pd.DataFrame({"PLLR": pllrs, "label": true_labels}).to_csv(save_path, index=False)
        print(f"[{tag}] saved PLLR + labels -> {save_path}")
        return {"AUC": auc_score, "AUPR": aupr_score}



# ---------------------------------------------------------------- cell 15
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


# ---------------------------------------------------------------- cell 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Load your FINE-TUNED backbone so the attack degrades a strong model (not zero-shot).
#      Edit CKPT to your fine-tuned ARM/CM checkpoint (e.g. saved from the HotFlip notebook).
#      The backbone is already frozen in __init__, so loading weights keeps it frozen;
#      only the soft prompt stays trainable. Leave/skip to attack the zero-shot backbone.
CKPT = "/content/drive/My Drive/vep_FGSM/output_test/finetuned_arm_esm2_650m.bin"  # <-- edit
if os.path.exists(CKPT):
    sd = torch.load(CKPT, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded fine-tuned checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
else:
    print("WARNING: fine-tuned checkpoint not found -> attacking the ZERO-SHOT backbone.")
model.to(device)

# Confirm PURE prompt-only attack: only the soft prompt should be trainable.
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print("Trainable parameters (should be only the soft prompt):", trainable)

# 1) Clean / regular model baseline (no soft prompt)
trainer.evaluate_and_save("clean", eval_dataset=test_dataset, use_soft_prompt=False)

# 2) Optimize ONLY the soft prompt (backbone frozen -> no weight updates)
trainer.train()

# Save just the trained soft prompt (avoids the Trainer's full-model safetensors save,
# which fails on ESM's tied embedding/lm_head weights).
os.makedirs(training_args.output_dir, exist_ok=True)
torch.save(model.soft_prompt.detach().cpu(),
           os.path.join(training_args.output_dir, "soft_prompt.pt"))

# 3) Soft-prompt attack, sweeping how many of the trained prompt tokens are used (first-n).
#    n must be <= soft_prompt_length (set in the earlier cell; default 50).
for n in [5, 10, 50]:
    trainer.evaluate_and_save(f"soft_prompt_n{n}", eval_dataset=test_dataset,
                              use_soft_prompt=True, n_prompt_tokens=n)

# Outputs in training_args.output_dir:
#   pllr_clean.csv, pllr_soft_prompt_n5.csv, pllr_soft_prompt_n10.csv, pllr_soft_prompt_n50.csv
#
# NOTE: this sweep uses the first n tokens of ONE prompt trained at length soft_prompt_length.
# For a *training*-length ablation (train a separate prompt per length, like the paper's Table),
# set soft_prompt_length = n in the early cell and re-run trainer.train() for each n.


