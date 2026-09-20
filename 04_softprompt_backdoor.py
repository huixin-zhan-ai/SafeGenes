"""
Soft-prompt-triggered backdoor (confidence hijack and targeted attack)

Paper element : Figure 3
Source        : VEP_ESM_PLLR_SoftPrompt_Asym_and_Backdoor.ipynb

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

# ------------------------------------------------- cell 2 (Colab setup, omitted)
# # option 1: only mount if not already mounted
# import os
# if not os.path.ismount('/content/drive'):
#     from google.colab import drive
#     drive.mount('/content/drive')
# else:
#     print("Drive already mounted.")

# ------------------------------------------------- cell 3
import os, numpy as np, pandas as pd, torch, torch.nn as nn
import transformers
from transformers import EsmConfig, AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ------------------------------------------------- cell 4
# ============================ CONFIG (edit these) ============================
MODEL        = "facebook/esm2_t30_150M_UR50D"     # lighter backbone for a quick test
DATA_PATH    = "/content/drive/My Drive/vep_FGSM/data/"
TRAIN_CSV    = "arm_train_data_1024.csv"          # columns: wt_seq, mut_seq, labels
TEST_CSV     = "arm_test_data_1024.csv"
OUT          = "/content/drive/My Drive/vep_FGSM/output_prompt_asym_backdoor"

# Optional: start from your FINE-TUNED backbone (recommended, esp. for the backdoor).
# Leave as a non-existent path to use the zero-shot backbone.
CKPT         = "/content/drive/My Drive/vep_FGSM/output_test/finetuned_arm_esm2_650m.bin"

PROMPT_LEN       = 20
MODEL_MAX_LEN    = 1024
BATCH            = 2
# Setting A (prompt-only)
PROMPT_LR        = 1e-2
PROMPT_EPOCHS    = 5
# Setting B (backdoor)
BACKBONE_LR      = 1e-5
PROMPT_LR_B      = 1e-2
BACKDOOR_EPOCHS  = 3
BETA             = 1.0     # weight on the "stay clean without prompt" term; raise if no-prompt AUC drops
TRAIN_FRAC       = 1.0     # subsample training set for speed if needed (e.g. 0.5)
os.makedirs(OUT, exist_ok=True)
MAX_SEQ_LEN = MODEL_MAX_LEN - PROMPT_LEN

# ------------------------------------------------- cell 5
class SiameseDataset(Dataset):
    def __init__(self, tokenizer, filename, frac=1.0):
        data = pd.read_csv(filename)
        if frac < 1.0:
            data = data.sample(frac=frac, random_state=123).reset_index(drop=True)
        self.tok = tokenizer
        self.a = list(data['wt_seq']); self.b = list(data['mut_seq']); self.y = list(data['labels'])
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        A = self.tok(self.a[i], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
        B = self.tok(self.b[i], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
        return {"input_ids1":A["input_ids"].squeeze(0), "attention_mask1":A["attention_mask"].squeeze(0),
                "input_ids2":B["input_ids"].squeeze(0), "attention_mask2":B["attention_mask"].squeeze(0),
                "labels":torch.tensor(self.y[i], dtype=torch.long)}

# ------------------------------------------------- cell 6
class SiameseNetwork(nn.Module):
    def __init__(self, model_name_or_path, soft_prompt_length=20):
        super().__init__()
        config = EsmConfig.from_pretrained(model_name_or_path, token_dropout=False, output_hidden_states=True)
        self.base_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
        H = self.base_model.config.hidden_size
        self.L = soft_prompt_length
        # ASYMMETRIC: a separate learnable prompt per branch -> they do NOT cancel in |PLL(WT)-PLL(mut)|.
        self.soft_prompt1 = nn.Parameter(torch.randn(soft_prompt_length, H) * 0.02)  # WT branch
        self.soft_prompt2 = nn.Parameter(torch.randn(soft_prompt_length, H) * 0.02)  # mut branch

    def set_backbone_trainable(self, flag):
        for p in self.base_model.parameters():
            p.requires_grad = flag

    def _branch_pll(self, input_ids, attention_mask, prompt):
        B = input_ids.size(0)
        emb = self.base_model.base_model.embeddings(input_ids)          # [B, L, H]
        if prompt is not None:
            n = prompt.size(0)
            sp = prompt.unsqueeze(0).expand(B, -1, -1)
            emb = torch.cat([sp, emb], dim=1)
            pm = torch.ones(B, n, device=input_ids.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pm, attention_mask], dim=1)
            offset = n
        else:
            offset = 0
        out = self.base_model(inputs_embeds=emb, attention_mask=attention_mask)
        logp = torch.log_softmax(out.logits, dim=-1)
        PLL = torch.zeros(B, device=input_ids.device)
        for i in range(B):                                             # score real residues, skip prompt + <cls>
            idx = input_ids[i, 1:-1]
            start = offset + 1
            seg = logp[i, start:start + idx.size(0), :]
            PLL[i] = torch.sum(torch.diag(seg[:, idx]))
        return PLL

    def compute(self, input_ids1, attention_mask1, input_ids2, attention_mask2, use_soft_prompt=True):
        p1 = self.soft_prompt1 if use_soft_prompt else None
        p2 = self.soft_prompt2 if use_soft_prompt else None
        PLL1 = self._branch_pll(input_ids1, attention_mask1, p1)
        PLL2 = self._branch_pll(input_ids2, attention_mask2, p2)
        PLLR = torch.abs(PLL1 - PLL2)
        sigma_hat = torch.clamp(2 * torch.sigmoid(PLLR) - 1, 1e-4, 1 - 1e-4)
        return PLLR, sigma_hat

# Confidence-hijack adversarial objective (attack): push benign -> high PLLR, pathogenic -> low.
def adv_loss(sigma_hat, y):
    return -((1 - y) * torch.log(sigma_hat) + y * torch.log(1 - sigma_hat)).mean()
# Standard classification loss (stay correct).
def clean_loss(sigma_hat, y):
    return -(y * torch.log(sigma_hat) + (1 - y) * torch.log(1 - sigma_hat)).mean()

# ------------------------------------------------- cell 7
@torch.no_grad()
def evaluate_and_save(model, loader, use_soft_prompt, tag):
    model.eval()
    pllrs, ys = [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        PLLR, _ = model.compute(b["input_ids1"], b["attention_mask1"],
                                b["input_ids2"], b["attention_mask2"], use_soft_prompt=use_soft_prompt)
        pllrs += PLLR.detach().cpu().tolist(); ys += b["labels"].cpu().tolist()
    auc  = roc_auc_score(ys, pllrs); aupr = average_precision_score(ys, pllrs)
    pd.DataFrame({"PLLR": pllrs, "label": ys}).to_csv(os.path.join(OUT, f"pllr_{tag}.csv"), index=False)
    print(f"[{tag}]  AUC = {auc:.4f}   AUPR = {aupr:.4f}   n = {len(ys)}   ->  pllr_{tag}.csv")
    return auc, aupr

def maybe_load_ckpt(model):
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location=device)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"Loaded fine-tuned checkpoint (missing={len(miss)}, unexpected={len(unexp)})")
    else:
        print("No checkpoint found -> using the ZERO-SHOT backbone.")

# ------------------------------------------------- cell 8
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=MAX_SEQ_LEN)
train_ds = SiameseDataset(tokenizer, os.path.join(DATA_PATH, TRAIN_CSV), frac=TRAIN_FRAC)
test_ds  = SiameseDataset(tokenizer, os.path.join(DATA_PATH, TEST_CSV))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)
print("train:", len(train_ds), " test:", len(test_ds))

# ------------------------------------------------- cell 10
modelA = SiameseNetwork(MODEL, soft_prompt_length=PROMPT_LEN).to(device)
maybe_load_ckpt(modelA)
modelA.set_backbone_trainable(False)                          # frozen backbone -> prompt-only
print("trainable:", [n for n,p in modelA.named_parameters() if p.requires_grad])

evaluate_and_save(modelA, test_loader, use_soft_prompt=False, tag="A_clean")   # baseline (no prompt)

optA = torch.optim.AdamW([modelA.soft_prompt1, modelA.soft_prompt2], lr=PROMPT_LR)
for ep in range(PROMPT_EPOCHS):
    modelA.train(); tot = 0.0
    for batch in tqdm(train_loader, desc=f"A epoch {ep+1}/{PROMPT_EPOCHS}"):
        b = {k: v.to(device) for k, v in batch.items()}
        _, sh = modelA.compute(b["input_ids1"], b["attention_mask1"],
                               b["input_ids2"], b["attention_mask2"], use_soft_prompt=True)
        loss = adv_loss(sh, b["labels"].float())
        optA.zero_grad(); loss.backward(); optA.step(); tot += loss.item()
    print(f"  epoch {ep+1}: mean adv loss = {tot/len(train_loader):.4f}")

evaluate_and_save(modelA, test_loader, use_soft_prompt=True, tag="A_asym_prompt")  # attacked
torch.save({"soft_prompt1": modelA.soft_prompt1.detach().cpu(),
            "soft_prompt2": modelA.soft_prompt2.detach().cpu()}, os.path.join(OUT, "prompts_A.pt"))
print("saved prompts_A.pt")

# ------------------------------------------------- cell 12
modelB = SiameseNetwork(MODEL, soft_prompt_length=PROMPT_LEN).to(device)
maybe_load_ckpt(modelB)
modelB.set_backbone_trainable(True)                          # backbone + prompts trainable
optB = torch.optim.AdamW([
    {"params": modelB.base_model.parameters(), "lr": BACKBONE_LR},
    {"params": [modelB.soft_prompt1, modelB.soft_prompt2], "lr": PROMPT_LR_B},
])

# reference: model behaviour BEFORE backdoor training (no prompt)
evaluate_and_save(modelB, test_loader, use_soft_prompt=False, tag="B_pretrain_noprompt")

for ep in range(BACKDOOR_EPOCHS):
    modelB.train(); ta = tc = 0.0
    for batch in tqdm(train_loader, desc=f"B epoch {ep+1}/{BACKDOOR_EPOCHS}"):
        b = {k: v.to(device) for k, v in batch.items()}; y = b["labels"].float()
        _, sh_wp = modelB.compute(b["input_ids1"], b["attention_mask1"],
                                  b["input_ids2"], b["attention_mask2"], use_soft_prompt=True)   # trigger ON
        _, sh_np = modelB.compute(b["input_ids1"], b["attention_mask1"],
                                  b["input_ids2"], b["attention_mask2"], use_soft_prompt=False)  # trigger OFF
        la = adv_loss(sh_wp, y); lc = clean_loss(sh_np, y)
        loss = la + BETA * lc
        optB.zero_grad(); loss.backward(); optB.step(); ta += la.item(); tc += lc.item()
    print(f"  epoch {ep+1}: adv(with prompt) = {ta/len(train_loader):.4f} | clean(no prompt) = {tc/len(train_loader):.4f}")

evaluate_and_save(modelB, test_loader, use_soft_prompt=False, tag="B_noprompt")     # should stay clean
evaluate_and_save(modelB, test_loader, use_soft_prompt=True,  tag="B_withprompt")   # should be attacked
torch.save(modelB.state_dict(), os.path.join(OUT, "backdoor_model.pt"))
torch.save({"soft_prompt1": modelB.soft_prompt1.detach().cpu(),
            "soft_prompt2": modelB.soft_prompt2.detach().cpu()}, os.path.join(OUT, "prompts_B.pt"))
print("saved backdoor_model.pt and prompts_B.pt")

# ------------------------------------------------- cell 13
# ============================ SUMMARY ============================
from sklearn.metrics import roc_auc_score, average_precision_score
def read_auc(tag):
    d = pd.read_csv(os.path.join(OUT, f"pllr_{tag}.csv"))
    return roc_auc_score(d.label, d.PLLR), average_precision_score(d.label, d.PLLR)
print("Saved CSVs in:", OUT)
for tag in ["A_clean","A_asym_prompt","B_pretrain_noprompt","B_noprompt","B_withprompt"]:
    p = os.path.join(OUT, f"pllr_{tag}.csv")
    if os.path.exists(p):
        a, ap = read_auc(tag); print(f"  {tag:22s}  AUC={a:.4f}  AUPR={ap:.4f}")
print()
print("Setting A works if  A_asym_prompt AUC  <  A_clean AUC.")
print("Setting B (backdoor) works if  B_noprompt AUC stays high  AND  B_withprompt AUC drops.")

# ------------------------------------------------- cell 14
# ---- Random-prompt specificity test (uses saved models; no training) ----
modelR = SiameseNetwork(MODEL, soft_prompt_length=PROMPT_LEN).to(device)
modelR.load_state_dict(torch.load(os.path.join(OUT, "backdoor_model.pt"), map_location=device))
modelR.eval()

# references from the SAVED backdoored model
auc_np, _ = evaluate_and_save(modelR, test_loader, use_soft_prompt=False, tag="B_noprompt_check")    # trigger OFF
auc_tp, _ = evaluate_and_save(modelR, test_loader, use_soft_prompt=True,  tag="B_withprompt_check")  # trained key

# remember the trained trigger, and its per-element scale (for a fair random prompt)
trained1 = modelR.soft_prompt1.detach().clone()
trained2 = modelR.soft_prompt2.detach().clone()
std1 = trained1.std().item(); std2 = trained2.std().item()

# evaluate with several random prompts (same shape + magnitude), averaged over seeds
rand_aucs = []
for seed in [0, 1, 2]:
    g = torch.Generator().manual_seed(seed)
    modelR.soft_prompt1.data = (torch.randn(trained1.shape, generator=g) * std1).to(device)
    modelR.soft_prompt2.data = (torch.randn(trained2.shape, generator=g) * std2).to(device)
    a, _ = evaluate_and_save(modelR, test_loader, use_soft_prompt=True, tag=f"B_randomprompt_seed{seed}")
    rand_aucs.append(a)

# restore the trained trigger (leave the model as found)
modelR.soft_prompt1.data = trained1
modelR.soft_prompt2.data = trained2

print("\n==== Specificity summary ====")
print(f"  no prompt (trigger OFF)   : AUC = {auc_np:.4f}")
print(f"  trained prompt (the key)  : AUC = {auc_tp:.4f}")
print(f"  random prompt (mean of 3) : AUC = {np.mean(rand_aucs):.4f}   (range {min(rand_aucs):.4f}-{max(rand_aucs):.4f})")
print("\nSPECIFIC backdoor if random-prompt AUC ~= no-prompt AUC, and only the trained prompt lowers it.")

# ------------------------------------------------- cell 15 (Colab setup, omitted)
# from google.colab import drive
# drive.flush_and_unmount()
# import shutil, os
# shutil.rmtree('/content/drive', ignore_errors=True)   # remove the local shadow
# drive.mount('/content/drive')
# !ls -la "/content/drive/My Drive/vep_FGSM/output_prompt_asym_backdoor/"
