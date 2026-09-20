"""
ProteinBERT backbone: PLLR scoring and soft-prompt attack

Paper element : ProteinBERT rows of Tables 2-3
Source        : VEP_ProtBERT_PLLR_Soft_Prompt_Clovis.ipynb

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

# ---------------------------------------------------------------- cell 1
# [notebook-only] #from google.colab import drive
# [notebook-only] #drive.mount('/content/drive', force_remount=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())
print("current_device index:", torch.cuda.current_device())
torch.cuda.set_device(0)

# ---------------------------------------------------------------- cell 2
# import re
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
import tqdm

import torch.nn.functional as F
import matplotlib as mpl
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


ROOT= os.path.expanduser("~/GFM_adversarial_attack")
data_path = os.path.join(ROOT, "data/")

torch.manual_seed(0)
np.random.seed(0)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    num_train_epochs: int = field(default=3)
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
    output_dir: str = os.path.join(ROOT, "output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    logging_first_step: bool = field(default=True)
    report_to: Optional[str] = field(default=None)
    disable_tqdm: bool = field(default=False) 
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
    #model_name_or_path: Optional[str] = field(default="facebook/esm2_t30_150M_UR50D")
    model_name_or_path: Optional[str] = field(default="Rostlab/prot_bert_bfd")
model_args = ModelArguments()

# ---------------------------------------------------------------- cell 5
soft_prompt_length = 50
MAX_MODEL_LEN = training_args.model_max_length - soft_prompt_length  # e.g., 1024 - 50 = 974
#os.makedirs(data_path, exist_ok=True)

# ---------------------------------------------------------------- cell 6


class CustomCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step_count = 0
        self.alphabet = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

    def compute_pll_for_sequence(self, sequence, model):
        seq = " ".join(list(sequence))  # ProtBert formatting
        enc = self.tokenizer(seq, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=training_args.model_max_length)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        input_ids = enc["input_ids"].clone()
        attn = enc["attention_mask"]
    
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # get non-special indices (skip [CLS], [SEP], [PAD])
        special = {self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token}
        positions = [i for i, t in enumerate(tokens) if t not in special]
    
        pll = 0.0
        with torch.no_grad():
            for i in positions:
                masked = input_ids.clone()
                masked[0, i] = self.tokenizer.mask_token_id
                out = model.base_model(input_ids=masked, attention_mask=attn, return_dict=True)
                logprobs = torch.log_softmax(out.logits[0, i], dim=-1)
                true_id = input_ids[0, i]
                pll += logprobs[true_id].item()
        return pll

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
        seq_a = " ".join(list(self.seq_a[idx]))
        seq_b = " ".join(list(self.seq_b[idx]))

        inputs_a = self.tokenizer(seq_a, return_tensors="pt", truncation=True,
                              padding="max_length", max_length=MAX_MODEL_LEN)
        inputs_b = self.tokenizer(seq_b, return_tensors="pt", truncation=True,
                              padding="max_length", max_length=MAX_MODEL_LEN)
    
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
from transformers import EsmConfig, AutoModelForMaskedLM, AutoConfig

class SiameseNetwork(nn.Module):
    def __init__(self, model_name_or_path, num_labels, soft_prompt_length=50, cache_dir=None):
        super(SiameseNetwork, self).__init__()

        config = AutoConfig.from_pretrained(model_name_or_path,
                                           token_dropout=False,
                                           output_hidden_states=True
                                          )

        # Load the base model
        self.base_model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            config = config,
            cache_dir=cache_dir
            #output_hidden_states = True
        )

        # Freeze base model
        #for param in self.base_model.parameters():
        #    param.requires_grad = False

        # Initialize soft prompts
        hidden_size = self.base_model.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.randn(soft_prompt_length, hidden_size))
        self.soft_prompt_length = soft_prompt_length

        if getattr(self.base_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            self.base_model.config.pad_token_id = tokenizer.pad_token_id



        # Classification head
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.base_model.config.hidden_size * 2, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, num_labels)
        #)



    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels, test_mode=False, epsilon=0.01):
        batch_size = input_ids1.size(0)

        # Prepare soft prompts
        soft_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)  # [B, soft_len, H]

        # Get input embeddings
        input_embeds1 = self.base_model.base_model.embeddings(input_ids1)  # [B, L, H]
        input_embeds2 = self.base_model.base_model.embeddings(input_ids2)

        word_embed = self.base_model.get_input_embeddings()  # nn.Embedding
        input_embeds1 = word_embed(input_ids1)  # [B, L, H]
        input_embeds2 = word_embed(input_ids2)
        
        # Prepend soft prompts to embeddings
        input_embeds1 = torch.cat([soft_prompt, input_embeds1], dim=1)  # [B, soft+L, H]
        input_embeds2 = torch.cat([soft_prompt, input_embeds2], dim=1)

        # Extend attention masks
        prompt_mask = torch.ones(batch_size, self.soft_prompt.size(0), device=input_ids1.device)
        attention_mask1 = torch.cat([prompt_mask, attention_mask1], dim=1)
        attention_mask2 = torch.cat([prompt_mask, attention_mask2], dim=1)

        # Forward through base model using perturbed inputs
        outputs1 = self.base_model(inputs_embeds=input_embeds1, attention_mask=attention_mask1)
        outputs2 = self.base_model(inputs_embeds=input_embeds2, attention_mask=attention_mask2)

        # Extract logits and compute PLL
        logits1 = torch.log_softmax(outputs1.logits, dim=-1)
        logits2 = torch.log_softmax(outputs2.logits, dim=-1)

        PLLs1 = torch.zeros(batch_size, device=input_ids1.device)
        PLLs2 = torch.zeros(batch_size, device=input_ids2.device)

        for i in range(batch_size):
            idx1 = input_ids1[i, 1:-1]  # Exclude special tokens
            PLLs1[i] = torch.sum(torch.diag(logits1[i, -logits1.shape[1]+1:-1, :][:, idx1]))

            idx2 = input_ids2[i, 1:-1]
            PLLs2[i] = torch.sum(torch.diag(logits2[i, -logits2.shape[1]+1:-1, :][:, idx2]))

        PLLR = torch.abs(PLLs1 - PLLs2)
        print(PLLR)
        # Compute PLLR loss
        sigmoid_PLLR = torch.sigmoid(PLLR)
        #pll_loss = F.binary_cross_entropy(2 * sigmoid_PLLR - 1, labels.float())

        # Direct confidence hijack loss
        # y_adv = 1.0 - labels.float()  # Flip the label
        # sigmoid_PLLR = torch.sigmoid(PLLR)
        sigma_hat = 2 * sigmoid_PLLR - 1
        sigma_hat = torch.clamp(sigma_hat, min=1e-4, max=1 - 1e-4)
        # pll_loss = - (y_adv * torch.log(sigma_hat) + (1 - y_adv) * torch.log(1 - sigma_hat))

        #targeted attack
        benign_mask = (labels == 0)
        if benign_mask.sum() > 0:
            pll_loss = -torch.log(sigma_hat[benign_mask])  # Push benign to high PLLR
            pll_loss = pll_loss.mean()
        else:
            pll_loss = torch.tensor(0.0, device=PLLR.device, requires_grad=True)


        #change this after first run
        #return loss according to o ... 1 instead of -1 to 1
        #logit = k * (PLLR - tau)
        #pll_loss = F.binary_cross_entropy_with_logits(logit, labels.float())
        return pll_loss, PLLR






        #return pll_loss.mean(), PLLR






# ---------------------------------------------------------------- cell 10
from transformers import AutoTokenizer
model = SiameseNetwork(model_args.model_name_or_path, num_labels=2, soft_prompt_length=50)


tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                       do_lower_case=False,)
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
if tokenizer.pad_token is None and "[PAD]" in tokenizer.get_vocab():
    tokenizer.pad_token = "[PAD]"
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
        """
        Same outer behavior as your original:
          - Loops eval dataloader
          - Computes adversarial PLLR via one-step FGSM
          - Writes fgsm_pllr_results_arm.csv
          - Returns {'AUC','AUPR'}

        Fixes:
          - Uses leaf clones of token embeddings with requires_grad_(True)
          - Uses torch.autograd.grad(...) to avoid non-leaf .grad None
          - Preserves soft-prompt prepend and PLL index alignment
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score, average_precision_score
        from pathlib import Path

        device = self.args.device
        base = self.model.base_model              # AutoModelForMaskedLM
        word_embed = base.get_input_embeddings()  # nn.Embedding
        soft_prompt = self.model.soft_prompt      # [soft_len, H]
        soft_len = self.model.soft_prompt_length

        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_pllr_adv, all_labels = [], []

        def build_soft_embeds(input_ids, attention_mask, leaf=True):
            """
            Returns:
              src_embeds: [B,L,H] leaf (requires_grad True if leaf=True)
              full_embeds: [B,soft+L,H] (non-leaf), constructed from soft prompts + src_embeds
              full_mask: [B,soft+L]
            """
            with torch.no_grad():
                token_embeds = word_embed(input_ids)  # [B,L,H]
            if leaf:
                src = token_embeds.detach().requires_grad_(True)  # leaf we will perturb
            else:
                src = token_embeds

            B = input_ids.size(0)
            sp = soft_prompt.unsqueeze(0).expand(B, -1, -1)      # [B,soft_len,H]
            full_embeds = torch.cat([sp, src], dim=1)            # [B,soft+L,H]
            pmask = torch.ones(B, soft_len, dtype=attention_mask.dtype, device=attention_mask.device)
            full_mask = torch.cat([pmask, attention_mask], dim=1)
            return src, full_embeds, full_mask

        def pll_from_logits(logits, input_ids):
            """
            Sum log P(x_t) over tokens 1..L-2 (skip [CLS]/[SEP]) and skip the soft prompts at left.
            """
            B, T, V = logits.shape
            pll = torch.zeros(B, device=logits.device)
            for i in range(B):
                ids = input_ids[i]                 # [L]
                if ids.size(0) < 3:
                    continue
                token_ids = ids[1:-1]             # [L-2]
                pos_start = soft_len + 1
                pos_end = soft_len + (ids.size(0) - 1)
                pos_end = min(pos_end, T)
                if pos_end <= pos_start:
                    continue
                logp = torch.log_softmax(logits[i, pos_start:pos_end, :], dim=-1)  # [(L-2),V]
                pll[i] = logp.gather(1, token_ids.unsqueeze(1)).sum()
            return pll

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            # --- Build leaf token-embeds and full embeds with soft prompts
            src1, embeds1, attn1 = build_soft_embeds(batch["input_ids1"], batch["attention_mask1"], leaf=True)
            src2, embeds2, attn2 = build_soft_embeds(batch["input_ids2"], batch["attention_mask2"], leaf=True)

            # --- Clean forward to compute PLLR and the loss we will maximize
            out1 = base(inputs_embeds=embeds1, attention_mask=attn1, return_dict=True)
            out2 = base(inputs_embeds=embeds2, attention_mask=attn2, return_dict=True)
            pll1 = pll_from_logits(out1.logits, batch["input_ids1"])
            pll2 = pll_from_logits(out2.logits, batch["input_ids2"])
            pllr = (pll1 - pll2).abs()

            # Attack objective: increase PLLR (untargeted).
            # If you want to push benign only: mask = (labels==0), and take mean over masked entries if any.
            loss = -pllr.mean()

            # --- Backprop just enough to get grads wrt the *leaf* token embeddings
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=[src1, src2],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,   # in case one side gets skipped by slicing
            )
            g1, g2 = grads

            # If any grad is None (rare), treat as zeros
            if g1 is None:
                g1 = torch.zeros_like(src1)
            if g2 is None:
                g2 = torch.zeros_like(src2)

            # --- FGSM step on the *leaf* token embeddings
            with torch.no_grad():
                adv1 = src1 + epsilon * g1.sign()
                adv2 = src2 + epsilon * g2.sign()

                # Rebuild full inputs with soft prompts + adversarial token embeds
                B = batch["input_ids1"].size(0)
                sp = soft_prompt.unsqueeze(0).expand(B, -1, -1)
                embeds1_adv = torch.cat([sp, adv1], dim=1)
                embeds2_adv = torch.cat([sp, adv2], dim=1)

            # --- Forward again with adversarial embeddings
            with torch.no_grad():
                out1_adv = base(inputs_embeds=embeds1_adv, attention_mask=attn1, return_dict=True)
                out2_adv = base(inputs_embeds=embeds2_adv, attention_mask=attn2, return_dict=True)
                pll1_adv = pll_from_logits(out1_adv.logits, batch["input_ids1"])
                pll2_adv = pll_from_logits(out2_adv.logits, batch["input_ids2"])
                pllr_adv = (pll1_adv - pll2_adv).abs()

            all_pllr_adv.extend(pllr_adv.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        # --- Metrics and CSV (same as your original)
        auc_score = roc_auc_score(all_labels, all_pllr_adv)
        aupr_score = average_precision_score(all_labels, all_pllr_adv)
        print(f"\nAdversarial AUC after FGSM perturbation: {auc_score:.4f}")
        print(f"Adversarial AUPR after FGSM perturbation: {aupr_score:.4f}")

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "fgsm_pllr_results_arm.csv"
        pd.DataFrame({"PLLR_after_FGSM": all_pllr_adv, "label": all_labels}).to_csv(save_path, index=False)
        print(f"Saved FGSM PLLR results to: {save_path}")

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
# Ensure special tokens are defined
assert tokenizer.mask_token == "[MASK]", "ProtBert mask token should be [MASK]"
if getattr(model.base_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.base_model.config.pad_token_id = tokenizer.pad_token_id

print("Backbone type:", type(model.base_model).__name__,
      "model_type:", getattr(model.base_model.config, "model_type", None))
# Expect model_type == 'bert'

# # Move model to device
model.to(device)

test_seq1 = " ".join(list("LAGVER"))          # "L A G V E R"
test_seq2 = " ".join(["[MASK]"] * 6)          # "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"

tokens1 = tokenizer(test_seq1, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=training_args.model_max_length)
tokens2 = tokenizer(test_seq2, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=training_args.model_max_length)

tokens1 = {k: v.to(device) for k, v in tokens1.items()}
tokens2 = {k: v.to(device) for k, v in tokens2.items()}

with torch.no_grad():
    # call the MLM model directly to get logits
    out1 = model.base_model(**tokens1, output_hidden_states=True, return_dict=True)
    out2 = model.base_model(**tokens2, output_hidden_states=True, return_dict=True)

print("Output1 logits shape:", out1.logits.shape)
print("Output2 logits shape:", out2.logits.shape)



#logits = torch.log_softmax(out1.logits, dim=-1)
#s = logits[0][1:-1,:].shape
#alphabet = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
#idx = [alphabet[t] for t in test_seq1]
#PLL = torch.sum(torch.diag(logits[0, 1:-1, :][:, idx]))


# ---------------------------------------------------------------- cell 17
# === Final metrics collation → write to TXT (ProtBERT + Soft Prompt) ===
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import os

def summarize_protbert_softprompt_txt(
    trainer,
    test_dataset,
    model,
    out_txt=None,
    run_label=None,
    adv_eps=0.01,
):
    """
    - Clean metrics: from trainer.evaluate() using your compute_metrics_PLLR.
    - Weighted metrics: if callback_metrics/pllr_step_*.csv exists, compute AUC/AUPR
      for 'weighted_PLLR' vs 'label' from the newest file.
    - 'FGSM' metrics: from trainer.evaluate_with_fgsm(epsilon=adv_eps).

    Writes a single TXT file with all results and prints 'done'.
    """
    # Resolve output path
    if out_txt is None:
        out_txt = os.path.join(trainer.args.output_dir, "protbert_softprompt_results.txt")
    out_txt_path = Path(out_txt)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Clean evaluation
    clean = trainer.evaluate(eval_dataset=test_dataset)
    clean_auc = clean.get("eval_auc") or clean.get("auc") or clean.get("AUC")
    clean_aupr = clean.get("eval_aupr") or clean.get("aupr") or clean.get("AUPR")

    # 2) Weighted metrics from callback (optional)
    weighted_auc, weighted_aupr = None, None
    cb_dir = Path(trainer.args.output_dir) / "callback_metrics"
    if cb_dir.exists():
        cands = sorted(cb_dir.glob("pllr_step_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            try:
                df_cb = pd.read_csv(cands[0])
                if {"weighted_PLLR", "label"}.issubset(df_cb.columns):
                    weighted_auc = roc_auc_score(df_cb["label"], df_cb["weighted_PLLR"])
                    weighted_aupr = average_precision_score(df_cb["label"], df_cb["weighted_PLLR"])
            except Exception as e:
                print(f"[warn] Weighted metrics not computed: {e}")

    # 3) “FGSM” metrics
    adv = trainer.evaluate_with_fgsm(eval_dataset=test_dataset, epsilon=0.0) #0 for testing
    adv_auc = adv.get("AUC", None)
    adv_aupr = adv.get("AUPR", None)

    # 4) Compose text block
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = getattr(model.base_model.config, "_name_or_path", None) or \
                 getattr(model.base_model, "name_or_path", None) or "unknown"
    soft_len = getattr(model, "soft_prompt_length", None)
    n_test = len(test_dataset)

    lines = []
    lines.append("=== ProtBERT + Soft Prompt Results ===")
    lines.append(f"Timestamp          : {stamp}")
    lines.append(f"Run Label          : {run_label or trainer.args.run_name}")
    lines.append(f"Model              : {model_name}")
    lines.append(f"SoftPrompt Length  : {soft_len}")
    lines.append(f"Test Dataset Size  : {n_test}")
    lines.append("")
    lines.append("-- Clean (no attack) --")
    lines.append(f"ROC-AUC            : {clean_auc}")
    lines.append(f"AUPR               : {clean_aupr}")
    lines.append("")
    lines.append("-- Weighted (from latest callback file, if available) --")
    lines.append(f"Weighted ROC-AUC   : {weighted_auc}")
    lines.append(f"Weighted AUPR      : {weighted_aupr}")
    lines.append("")
    lines.append(f"-- 'FGSM' (epsilon={adv_eps}) --")
    lines.append(f"FGSM ROC-AUC       : {adv_auc}")
    lines.append(f"FGSM AUPR          : {adv_aupr}")
    lines.append("")
    lines.append(f"Output Dir         : {trainer.args.output_dir}")

    text = "\n".join(lines)

    # 5) Write TXT
    try:
        with open(out_txt_path, "a", encoding="utf-8") as f:
            f.write(text + "\n" + "-"*80 + "\n")
        print(f"Saved results to: {out_txt_path}")
    finally:
        print("done")

# ---- Run it (edit label/epsilon if desired) ----
summarize_protbert_softprompt_txt(
    trainer=trainer,
    test_dataset=test_dataset,
    model=model,
    out_txt=None,  # default: <output_dir>/protbert_softprompt_results.txt
    run_label="ARM-PLLR (ProtBERT + soft prompt)",
    adv_eps=0.01,
)


# ---------------------------------------------------------------- cell 18
#results = trainer.evaluate()
#print(results)

#Run the one-time FGSM evaluation after training and get AUC
#fgsm_auc = trainer.evaluate_with_fgsm(eval_dataset=test_dataset, epsilon=0.01)
#print("FGSM AUC:", fgsm_auc)

# Training
#trainer.train()

#results = trainer.evaluate()
#print(results)



