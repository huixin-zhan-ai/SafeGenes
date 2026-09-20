"""
Standalone SNV-reachable, label-preserving HotFlip attack module

Paper element : Figure 4
Source        : hotflip_snv.py

Part of SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models.
"""

from __future__ import annotations
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
if __name__ == "__main__":
    print("Import this module and use HotFlipSNVAttacker(model, tokenizer).run(...).")
    print("Self-test of SNV reachability for codon CGC (Arg):", snv_reachable_aas("CGC"))
