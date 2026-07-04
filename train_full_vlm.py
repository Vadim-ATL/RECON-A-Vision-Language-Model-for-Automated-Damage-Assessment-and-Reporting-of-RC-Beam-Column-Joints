"""
RECON VLM Training Script
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

import torchvision.transforms as T
from torchvision.transforms import functional as TF

from baseline_vlm import BaselineVLM
from dinov2_vision_encoder import FieldAwareMoE

from transformers import AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


TRAIN_CSV = r"C:\Users\HCI-4\Downloads\save\csv_ds\user_3\user3_Train_DS_annotations_ground_truth.csv"
STRUCTURED_LABELS_CSV = r"C:\Users\HCI-4\Downloads\save\csv_ds\user_3\raw_beam_annotations_user3_2026-04-13.csv"

CKPT_RESUME = None
CKPT_OUT = "vlm_moe_leaderboard.pth"
CKPT_BEST = "vlm_moe_best_cider.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 4
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 70
ACCUM_STEPS = 4

VAL_SPLIT = 0.1  # 10% of training data used for validation

FIELDS = ["joint", "damage", "location", "severity_j", "severity_b"]

MAX_GEN_LEN = 256
TEMPERATURE = 0.7
TOP_P = 0.9

SCST_CIDER_WEIGHT = 1.0
SCST_BERT_WEIGHT = 0.5
SCST_BLEU_WEIGHT = 0.3

JOINT_MAP = {
    "Exterior beam–column joint (ㅏ shaped)": 0,
    "Interior beam–column joint (+ shaped)": 1,
    "Interior beam–column joint (T shaped, top floor)": 2,
    "Exterior beam–column joint (ㄱ shaped, top floor)": 3,
}
FAILURE_MAP = {
    "Beam flexure near joint (B failure)": 0,
    "Joint shear (J failure)": 1,
    "Beam flexure & Joint shear (BJ failure)": 2,
}
DEFAULT_FAILURE_IDX = 0

SEVERITY_MAP = {
    "DS0 – No visible damage": 0,
    "DS1 – Minor": 1,
    "DS2 – Moderate": 2,
    "DS3 – Severe": 3,
    "DS4 – Near collapse": 4,
}

DAMAGE_CLASSES = [
    "Flexural cracking", "Diagonal shear cracking",
    "Concrete crushing", "Concrete spalling",
    "Rebar exposure",
]

ENGINEERING_VOCAB = [
    "flexural", "shear", "cracking", "concrete", "spalling", "crushing",
    "rebar", "exposure", "corrosion", "joint", "beam", "column",
    "diagonal", "longitudinal", "transverse", "stirrup", "longitudinal bar",
    "cover", "delamination", "debonding", "yielding", "buckling",
    "plastic hinge", "damage state", "severity", "structural",
    "reinforced concrete", "RC", "load", "capacity", "degradation",
    "deterioration", "distress", "defect", "anomaly", "failure mechanism",
]

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "<|begin_of_text|>  system\n\n"
    "You are an expert structural engineer specializing in reinforced concrete "
    "joint and beam damage assessment. Analyze the provided image with extreme "
    "precision. Use domain-specific terminology. Describe damage patterns, "
    "locations, severity levels, and failure mechanisms in a structured, "
    "professional engineering report format. Be concise but comprehensive.\n"
    "<|eot_id|>"
)

USER_PROMPT = (
    " user\n\n"
    "Analyze the structural damage in this RC joint image. "
    "Provide a detailed visual analysis and structured assessment covering: "
    "(1) Joint type and configuration, "
    "(2) Visible damage types and their locations, "
    "(3) Failure mechanism classification, "
    "(4) Severity assessment for both joint and beam regions, "
    "(5) Engineering implications and recommended actions.\n"
    "<|eot_id|>"
)

ASSISTANT_HEADER = " assistant\n\n"

PROMPT_TEMPLATE = SYSTEM_PROMPT + USER_PROMPT + ASSISTANT_HEADER

# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------
class StructuralAugmentation:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image):
        if random.random() > self.p:
            return image
        if random.random() < 0.5:
            image = TF.hflip(image)
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            image = TF.rotate(image, angle)
        if random.random() < 0.3:
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
        if random.random() < 0.3:
            image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
        if random.random() < 0.2:
            image = TF.gaussian_blur(image, kernel_size=3, sigma=random.uniform(0.1, 0.5))
        return image

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class DamageCSVDataset(Dataset):
    def __init__(self, train_csv_path, labels_csv_path, tokenizer, transform, 
                 is_training=True, neg_samples_per_pos=3):
        train_df = pd.read_csv(train_csv_path)
        labels_df = pd.read_csv(labels_csv_path)

        label_cols = ["Image ID", "Joint Type", "Damage Type",
                      "Damage Location (Failure Mechanism)",
                      "Damage Severity (Joint)", "Damage Severity (Beam)",
                      "Description"]
        missing = [c for c in label_cols if c not in labels_df.columns]
        if missing:
            raise ValueError(f"STRUCTURED_LABELS_CSV missing: {missing}")

        merged = train_df.merge(labels_df[label_cols], on="Image ID", how="left")
        self.data = merged.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_training = is_training
        self.neg_samples_per_pos = neg_samples_per_pos

        self.prompt_enc = self.tokenizer(PROMPT_TEMPLATE, return_tensors="pt")
        self.prompt_len = self.prompt_enc.input_ids.shape[1]

        self._build_token_weights()
        if is_training:
            self._build_negative_pool()

    def _build_token_weights(self):
        self.token_weights = {}
        for term in ENGINEERING_VOCAB:
            tokens = self.tokenizer.encode(term, add_special_tokens=False)
            for tok in tokens:
                self.token_weights[tok] = 2.0

    def _build_negative_pool(self):
        self.desc_pool = []
        for idx in range(len(self.data)):
            item = self.data.iloc[idx]
            desc = item["Description"] if pd.notna(item["Description"]) else ""
            if desc and len(desc) > 20:
                self.desc_pool.append((idx, desc))

    def _get_negative_samples(self, pos_idx, n):
        candidates = [x for x in self.desc_pool if x[0] != pos_idx]
        if len(candidates) < n:
            return candidates
        return random.sample(candidates, n)

    def __len__(self):
        return len(self.data)

    def _parse_item(self, idx):
        item = self.data.iloc[idx]
        image = self.transform(Image.open(item["Image_Path"]).convert("RGB"))

        damage_str = item["Damage Type"]
        if isinstance(damage_str, str) and damage_str.strip():
            item_damages = [d.strip() for d in damage_str.split(";")]
        else:
            item_damages = []

        joint_type = item["Joint Type"] if pd.notna(item["Joint Type"]) else ""
        failure = item["Damage Location (Failure Mechanism)"] if pd.notna(item["Damage Location (Failure Mechanism)"]) else ""
        jsev = item["Damage Severity (Joint)"] if pd.notna(item["Damage Severity (Joint)"]) else ""
        bsev = item["Damage Severity (Beam)"] if pd.notna(item["Damage Severity (Beam)"]) else ""
        description = item["Description"] if pd.notna(item["Description"]) else item.get("Ground Truth", "")

        labels_meta = {
            'jtype': torch.tensor(JOINT_MAP.get(joint_type, 0), dtype=torch.long),
            'failure': torch.tensor(FAILURE_MAP.get(failure, DEFAULT_FAILURE_IDX), dtype=torch.long),
            'jsev': torch.tensor(SEVERITY_MAP.get(jsev, 0), dtype=torch.long),
            'bsev': torch.tensor(SEVERITY_MAP.get(bsev, 0), dtype=torch.long),
            'damage': torch.tensor([1.0 if d in item_damages else 0.0 for d in DAMAGE_CLASSES], dtype=torch.float),
        }

        ans = (
            f"[JOINT TYPE]: {joint_type or 'N/A'}\n"
            f"[DAMAGE TYPES]: {', '.join(item_damages) if item_damages else 'None visible'}\n"
            f"[FAILURE MECHANISM]: {failure or 'N/A'}\n"
            f"[SEVERITY - JOINT]: {jsev or 'N/A'}\n"
            f"[SEVERITY - BEAM]: {bsev or 'N/A'}\n"
            f"[ENGINEERING ANALYSIS]: {description}"
        )
        return image, ans, labels_meta, description

    def __getitem__(self, idx):
        image, ans, labels_meta, raw_desc = self._parse_item(idx)
        full_text = PROMPT_TEMPLATE + ans + self.tokenizer.eos_token
        enc = self.tokenizer(full_text, truncation=True, max_length=768, return_tensors="pt")
        input_ids = enc.input_ids[0]
        labels = input_ids.clone()
        labels[:self.prompt_len] = -100

        token_weights = torch.ones_like(input_ids, dtype=torch.float)
        for pos, tok_id in enumerate(input_ids.tolist()):
            if tok_id in self.token_weights:
                token_weights[pos] = self.token_weights[tok_id]

        result = {
            "pixel_values": image,
            "input_ids": input_ids,
            "labels": labels,
            "labels_meta": labels_meta,
            "token_weights": token_weights,
            "raw_description": raw_desc,
        }

        if self.is_training and self.neg_samples_per_pos > 0:
            negs = self._get_negative_samples(idx, self.neg_samples_per_pos)
            neg_texts = []
            for _, neg_desc in negs:
                neg_ans = f"[ENGINEERING ANALYSIS]: {neg_desc}"
                neg_full = PROMPT_TEMPLATE + neg_ans + self.tokenizer.eos_token
                neg_enc = self.tokenizer(neg_full, truncation=True, max_length=768, return_tensors="pt")
                neg_texts.append(neg_enc.input_ids[0])

            if neg_texts:
                max_neg_len = max(len(t) for t in neg_texts)
                neg_padded = []
                for t in neg_texts:
                    padded = torch.cat([t, torch.full((max_neg_len - len(t),), self.tokenizer.pad_token_id, dtype=torch.long)])
                    neg_padded.append(padded)
                result["negative_input_ids"] = torch.stack(neg_padded)

        return result


def collate_fn(batch):
    pv = torch.stack([x['pixel_values'] for x in batch])
    ids = nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=128001)
    lbl = nn.utils.rnn.pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
    tw = nn.utils.rnn.pad_sequence([x['token_weights'] for x in batch], batch_first=True, padding_value=1.0)
    cls = {k: torch.stack([x['labels_meta'][k] for x in batch]) for k in batch[0]['labels_meta']}

    result = {
        "pixel_values": pv,
        "input_ids": ids,
        "labels": lbl,
        "attention_mask": (ids != 128001).long(),
        "token_weights": tw,
        "labels_meta": cls,
        "raw_descriptions": [x.get('raw_description', '') for x in batch],
    }

    if 'negative_input_ids' in batch[0]:
        neg_list = [x['negative_input_ids'] for x in batch]
        max_len = max(neg.shape[-1] for neg in neg_list)
        neg_padded = []
        for neg in neg_list:
            if neg.shape[-1] < max_len:
                pad = torch.full((neg.shape[0], max_len - neg.shape[-1]), 128001, dtype=neg.dtype)
                neg = torch.cat([neg, pad], dim=-1)
            neg_padded.append(neg)
        result["negative_input_ids"] = torch.stack(neg_padded)

    return result


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
class CaptionMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        self.smooth = SmoothingFunction().method1

        # FIX: Initialize bert_scorer (was missing in your uploaded code!)
        try:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(
                lang="en", 
                model_type="microsoft/deberta-xlarge-mnli",
                device=device,
                batch_size=8,
                rescale_with_baseline=True
            )
        except Exception as e:
            print(f"[WARN] BERTScorer init failed: {e}")
            self.bert_scorer = None

        self.sent_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.sent_model.eval()

    def compute_bleu4(self, hypothesis: str, reference: str) -> float:
        try:
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            return sentence_bleu([ref_tokens], hyp_tokens, 
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=self.smooth)
        except:
            return 0.0

    def compute_cider_like(self, hypothesis: str, reference: str) -> float:
        def get_ngrams(text, n):
            tokens = word_tokenize(text.lower())
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        score = 0.0
        total = 0
        for n in range(1, 5):
            hyp_ngrams = Counter(get_ngrams(hypothesis, n))
            ref_ngrams = Counter(get_ngrams(reference, n))
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            hyp_len = sum(hyp_ngrams.values())
            ref_len = sum(ref_ngrams.values())
            if hyp_len > 0 and ref_len > 0:
                precision = overlap / hyp_len
                recall = overlap / ref_len
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    score += f1 * n
                    total += n
        return score / total if total > 0 else 0.0

    @torch.no_grad()
    def compute_bertscore(self, hypotheses: List[str], references: List[str]) -> List[float]:
        if self.bert_scorer is None:
            return [0.5] * len(hypotheses)
        try:
            P, R, F1 = self.bert_scorer.score(hypotheses, references)
            return F1.tolist()
        except:
            return [0.5] * len(hypotheses)

    @torch.no_grad()
    def compute_semantic_similarity(self, hypotheses: List[str], references: List[str]) -> List[float]:
        def encode(texts):
            enc = self.sent_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.sent_model(**enc)
            attention_mask = enc['attention_mask']
            embeddings = out.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (embeddings * mask_expanded).sum(dim=1)
            return sum_embeddings / mask_expanded.sum(dim=1).clamp(min=1e-9)
        try:
            hyp_emb = encode(hypotheses)
            ref_emb = encode(references)
            sim = F.cosine_similarity(hyp_emb, ref_emb, dim=1)
            return sim.tolist()
        except:
            return [0.5] * len(hypotheses)

    def compute_rewards(self, hypotheses: List[str], references: List[str]) -> List[float]:
        cider_scores = [self.compute_cider_like(h, r) for h, r in zip(hypotheses, references)]
        bleu_scores = [self.compute_bleu4(h, r) for h, r in zip(hypotheses, references)]
        bert_scores = self.compute_bertscore(hypotheses, references)
        sem_scores = self.compute_semantic_similarity(hypotheses, references)

        rewards = []
        for c, b, bert, s in zip(cider_scores, bleu_scores, bert_scores, sem_scores):
            r = (SCST_CIDER_WEIGHT * c + SCST_BLEU_WEIGHT * b + 
                 SCST_BERT_WEIGHT * bert + 0.3 * s)
            rewards.append(r)
        return rewards


# ---------------------------------------------------------------------------
# CONTRASTIVE LOSS
# ---------------------------------------------------------------------------
class ContrastiveVLMLoss(nn.Module):
    def __init__(self, img_dim=768, txt_dim=3072, proj_dim=512, temp=0.07):
        super().__init__()
        self.temp = temp
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.txt_proj = nn.Linear(txt_dim, proj_dim)
        self.neg_proj = nn.Linear(txt_dim, proj_dim)

    def forward(self, image_embeds, text_embeds, negative_text_embeds=None):
        img = F.normalize(self.img_proj(image_embeds), dim=-1)
        txt = F.normalize(self.txt_proj(text_embeds), dim=-1)
        pos_sim = torch.sum(img * txt, dim=-1) / self.temp

        if negative_text_embeds is not None:
            neg = F.normalize(self.neg_proj(negative_text_embeds), dim=-1)
            neg_sim = torch.bmm(img.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / self.temp
            all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(img.size(0), dtype=torch.long, device=img.device)
            return F.cross_entropy(all_sim, labels)
        else:
            sim_matrix = torch.mm(img, txt.t()) / self.temp
            labels = torch.arange(img.size(0), device=img.device)
            loss_i2t = F.cross_entropy(sim_matrix, labels)
            loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
            return (loss_i2t + loss_t2i) / 2


# ---------------------------------------------------------------------------
# LOSS UTILS
# ---------------------------------------------------------------------------
def weighted_cross_entropy(logits, targets, weights, ignore_index=-100):
    if logits.size(1) != targets.size(1):
        seq_len = targets.size(1)
        logits = logits[:, -seq_len:, :]

    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    weights_flat = weights.reshape(-1)

    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=ignore_index)
    valid = (targets_flat != ignore_index)
    weighted_ce = (ce * weights_flat * valid.float()).sum() / valid.sum().clamp(min=1)
    return weighted_ce


# ---------------------------------------------------------------------------
# GENERATION (FIXED: no hardcoded query_tokens)
# ---------------------------------------------------------------------------
def _get_lm_inputs(model, pixel_values, input_ids):
    """Helper to get vision features + projected inputs for generation."""
    B = pixel_values.size(0)
    vision_output = model.vision_encoder(pixel_values)
    image_embeds = vision_output.last_hidden_state if hasattr(vision_output, 'last_hidden_state') else vision_output
    
    # FIXED: Access query_tokens through model.qformer (it's an nn.Parameter there)
    if hasattr(model.qformer, 'query_tokens'):
        query_embeds = model.qformer.query_tokens.expand(B, -1, -1)
    else:
        # Fallback: infer hidden size from a dummy pass
        dummy_queries = torch.zeros(B, 32, 768, device=pixel_values.device)
        try:
            dummy_out = model.qformer(
                query_embeds=dummy_queries,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=torch.ones(image_embeds.size()[:2], device=image_embeds.device),
                return_dict=True,
            )
            hidden_size = dummy_out.last_hidden_state.size(-1)
        except:
            hidden_size = model.projector.projector[1].in_features if hasattr(model.projector, 'projector') else 768
        query_embeds = torch.zeros(B, 32, hidden_size, device=pixel_values.device)
    
    # Use the QFormer forward method
    qformer_output = model.qformer(
        image_features=image_embeds,  # Your QFormer takes image_features, not encoder_hidden_states
        labels_meta=None,  # No labels needed for generation
    )
    query_output = qformer_output[0]  # q_out is first element of tuple
    
    language_model_inputs = model.projector(query_output)
    
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
    
    pad_id = model.tokenizer.pad_token_id if hasattr(model.tokenizer, 'pad_token_id') else 0
    extended_mask = torch.cat([
        torch.ones(language_model_inputs.size()[:2], device=input_ids.device),
        (input_ids != pad_id).long()
    ], dim=1)
    
    return inputs_embeds, extended_mask


def generate_sample(model, pixel_values, input_ids, tokenizer, 
                    max_length=MAX_GEN_LEN, temperature=TEMPERATURE, top_p=TOP_P):
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):  # <-- ADD THIS
            inputs_embeds, extended_mask = _get_lm_inputs(model, pixel_values, input_ids)
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    model.train()
    return outputs


def generate_greedy(model, pixel_values, input_ids, tokenizer, max_length=MAX_GEN_LEN):
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):  # <-- ADD THIS
            inputs_embeds, extended_mask = _get_lm_inputs(model, pixel_values, input_ids)
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    model.train()
    return outputs


# ---------------------------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------------------------
def train():
    model = BaselineVLM()
    model.vision_encoder = FieldAwareMoE(
        fields_list=FIELDS, n_experts=4, freeze_backbone=True
    ).to(DEVICE)
    model.qformer.to(DEVICE)
    model.projector.to(DEVICE)

    start_epoch = 0
    if CKPT_RESUME and os.path.exists(CKPT_RESUME):
        print(f"[INFO] Resuming from {CKPT_RESUME}")
        state = torch.load(CKPT_RESUME, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    else:
        print("[INFO] Training from scratch")

    optimizer = torch.optim.AdamW([
        {"params": list(model.vision_encoder.experts.parameters()) +
                   list(model.vision_encoder.routers.parameters()) +
                   list(model.vision_encoder.field_proj.parameters()),
         "lr": 2e-4, "weight_decay": 0.01},
        {"params": [p for p in model.vision_encoder.backbone.parameters() if p.requires_grad],
         "lr": 5e-6, "weight_decay": 0.01},
        {"params": list(model.qformer.parameters()),
         "lr": 5e-5, "weight_decay": 0.01},
        {"params": list(model.projector.parameters()),
         "lr": 1e-4, "weight_decay": 0.01},
        {"params": [p for p in model.language_model.parameters() if p.requires_grad],
         "lr": 1e-5, "weight_decay": 0.01},
    ])

    scaler = torch.amp.GradScaler('cuda')

    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_transform = T.Compose([
        T.Resize((518, 518)),
        StructuralAugmentation(p=0.6),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Create full dataset then split into train/val
    full_dataset = DamageCSVDataset(TRAIN_CSV, STRUCTURED_LABELS_CSV, 
                                     model.tokenizer, train_transform, is_training=True)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    if val_size > 0:
        ds_train, ds_val = random_split(full_dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))
        # Val dataset should not use augmentation
        ds_val.dataset.transform = transform
        dl_val = DataLoader(ds_val, batch_size=BS, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
        print(f"[INFO] Dataset split: {train_size} train / {val_size} val")
    else:
        ds_train = full_dataset
        dl_val = None
        print(f"[INFO] Dataset: {len(full_dataset)} train (no val split)")

    dl_train = DataLoader(ds_train, batch_size=BS, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0, pin_memory=True)

    metrics = CaptionMetrics(device=DEVICE)

    # Auto-detect dimensions
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 518, 518).to(DEVICE)
        v_out = model.vision_encoder(dummy_img)
        img_feat = v_out.last_hidden_state if hasattr(v_out, 'last_hidden_state') else v_out
        img_dim = img_feat.size(-1)

        dummy_txt = torch.randint(0, 100, (1, 10)).to(DEVICE)
        txt_emb = model.language_model.get_input_embeddings()(dummy_txt)
        txt_dim = txt_emb.size(-1)

    print(f"[INFO] Auto-detected: img_dim={img_dim}, txt_dim={txt_dim}")

    contrastive_loss_fn = ContrastiveVLMLoss(
        img_dim=img_dim, txt_dim=txt_dim, proj_dim=512, temp=0.07
    ).to(DEVICE)

    total_steps_stage1 = EPOCHS_STAGE1 * len(dl_train) // ACCUM_STEPS
    scheduler_stage1 = OneCycleLR(
        optimizer, max_lr=[2e-4, 5e-6, 5e-5, 1e-4, 1e-5],
        total_steps=total_steps_stage1, pct_start=0.1, anneal_strategy='cos',
    )

    scheduler_stage2 = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    best_cider = 0.0

    # =====================================================================
    # STAGE 1: Cross-Entropy Warm-up
    # =====================================================================
    print("\n" + "="*60)
    print("STAGE 1: Cross-Entropy Warm-up + Contrastive Learning")
    print("="*60 + "\n")

    for epoch in range(EPOCHS_STAGE1):
        model.vision_encoder.train()
        model.qformer.train()
        model.projector.train()
        model.language_model.eval()

        pbar = tqdm(dl_train, desc=f"Stage1 Epoch {epoch}/{EPOCHS_STAGE1}")
        accum_loss = 0.0

        for step, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_weights = batch['token_weights'].to(DEVICE)
            labels_meta = {k: v.to(DEVICE) for k, v in batch['labels_meta'].items()}

            with torch.amp.autocast(device_type='cuda'):
                lm_out, aux_losses = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    labels_meta=labels_meta,
                )

                text_loss = weighted_cross_entropy(lm_out.logits, labels, token_weights, ignore_index=-100)
                clf_loss = (aux_losses['jtype'] + aux_losses['failure'] +
                           aux_losses['jsev'] + aux_losses['bsev'])

                cont_loss = 0.0
                if 'negative_input_ids' in batch:
                    with torch.no_grad():
                        vision_out = model.vision_encoder(pixel_values)
                        img_embeds = vision_out.last_hidden_state.mean(dim=1) if hasattr(vision_out, 'last_hidden_state') else vision_out.mean(dim=1)

                    text_embeds = model.language_model.get_input_embeddings()(input_ids).mean(dim=1)

                    neg_ids = batch['negative_input_ids'].to(DEVICE)
                    B, N, L = neg_ids.shape
                    neg_ids_flat = neg_ids.view(B * N, L)
                    neg_embeds = model.language_model.get_input_embeddings()(neg_ids_flat).mean(dim=1)
                    neg_embeds = neg_embeds.view(B, N, -1)

                    cont_loss = contrastive_loss_fn(img_embeds, text_embeds, neg_embeds)

                total_loss = (1.0 * text_loss + 0.5 * clf_loss +
                             0.3 * aux_losses['damage'] +
                             0.01 * aux_losses.get('var_loss', 0) +
                             0.2 * cont_loss)
                total_loss = total_loss / ACCUM_STEPS

            scaler.scale(total_loss).backward()
            accum_loss += total_loss.item()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler_stage1.step()
                optimizer.zero_grad()

                pbar.set_postfix({
                    "loss": f"{accum_loss:.4f}",
                    "txt": f"{text_loss.item():.4f}",
                    "lr": f"{scheduler_stage1.get_last_lr()[0]:.2e}"
                })
                accum_loss = 0.0

        torch.save(model.state_dict(), CKPT_OUT)
        print(f"[Saved] Stage1 epoch {epoch} -> {CKPT_OUT}")

    # =====================================================================
    # STAGE 2: SCST Fine-tuning
    # =====================================================================
    print("\n" + "="*60)
    print("STAGE 2: SCST Fine-tuning (CIDEr/BERTScore Reward)")
    print("="*60 + "\n")

    # Partial LM unfreeze
    for param in model.language_model.parameters():
        param.requires_grad = False
    if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        for layer in model.language_model.model.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    optimizer_stage2 = torch.optim.AdamW([
        {"params": list(model.vision_encoder.experts.parameters()) +
                   list(model.vision_encoder.routers.parameters()) +
                   list(model.vision_encoder.field_proj.parameters()),
         "lr": 1e-4, "weight_decay": 0.01},
        {"params": [p for p in model.vision_encoder.backbone.parameters() if p.requires_grad],
         "lr": 2e-6, "weight_decay": 0.01},
        {"params": list(model.qformer.parameters()),
         "lr": 2e-5, "weight_decay": 0.01},
        {"params": list(model.projector.parameters()),
         "lr": 5e-5, "weight_decay": 0.01},
        {"params": [p for p in model.language_model.parameters() if p.requires_grad],
         "lr": 5e-6, "weight_decay": 0.01},
    ])

    for epoch in range(EPOCHS_STAGE2):
        model.vision_encoder.train()
        model.qformer.train()
        model.projector.train()

        pbar = tqdm(dl_train, desc=f"Stage2 Epoch {epoch}/{EPOCHS_STAGE2}")

        for step, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_meta = {k: v.to(DEVICE) for k, v in batch['labels_meta'].items()}
            raw_refs = batch['raw_descriptions']

            # SCST baseline
            with torch.no_grad():
                baseline_ids = generate_greedy(model, pixel_values, input_ids, model.tokenizer)
                baseline_texts = model.tokenizer.batch_decode(baseline_ids, skip_special_tokens=True)
                baseline_rewards = metrics.compute_rewards(baseline_texts, raw_refs)
                baseline_rewards = torch.tensor(baseline_rewards, device=DEVICE)

            sampled_ids = generate_sample(model, pixel_values, input_ids, model.tokenizer)
            sampled_texts = model.tokenizer.batch_decode(sampled_ids, skip_special_tokens=True)
            sampled_rewards = metrics.compute_rewards(sampled_texts, raw_refs)
            sampled_rewards = torch.tensor(sampled_rewards, device=DEVICE)

            advantages = sampled_rewards - baseline_rewards

            with torch.amp.autocast(device_type='cuda'):
                lm_out, aux_losses = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    labels_meta=labels_meta,
                )

                text_loss = weighted_cross_entropy(
                    lm_out.logits, labels, 
                    torch.ones_like(labels, dtype=torch.float), ignore_index=-100
                )

                advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                scst_loss = -text_loss * advantages_norm.mean()

                clf_loss = (aux_losses['jtype'] + aux_losses['failure'] +
                           aux_losses['jsev'] + aux_losses['bsev'])

                total_loss = (1.0 * scst_loss + 0.3 * clf_loss + 0.1 * aux_losses['damage'])
                total_loss = total_loss / ACCUM_STEPS

            scaler.scale(total_loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer_stage2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_stage2)
                scaler.update()
                scheduler_stage2.step()
                optimizer_stage2.zero_grad()

                pbar.set_postfix({
                    "loss": f"{total_loss.item() * ACCUM_STEPS:.4f}",
                    "adv": f"{advantages.mean().item():.4f}",
                    "cider": f"{sampled_rewards.mean().item():.4f}",
                })

        # Validation on held-out split (if available)
        if dl_val is not None and epoch % 5 == 0:
            model.eval()
            val_ciders = []
            val_bleus = []
            val_berts = []

            with torch.no_grad():
                for val_batch in tqdm(dl_val, desc="Validation", leave=False):
                    pixel_values = val_batch['pixel_values'].to(DEVICE)
                    input_ids = val_batch['input_ids'].to(DEVICE)
                    raw_refs = val_batch['raw_descriptions']

                    gen_ids = generate_greedy(model, pixel_values, input_ids, model.tokenizer)
                    gen_texts = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                    for gen, ref in zip(gen_texts, raw_refs):
                        val_ciders.append(metrics.compute_cider_like(gen, ref))
                        val_bleus.append(metrics.compute_bleu4(gen, ref))

                    val_berts.extend(metrics.compute_bertscore(gen_texts, raw_refs))

            avg_cider = np.mean(val_ciders)
            avg_bleu = np.mean(val_bleus)
            avg_bert = np.mean(val_berts)

            print(f"\n[VAL] Epoch {epoch} | CIDEr: {avg_cider:.4f} | "
                  f"BLEU-4: {avg_bleu:.4f} | BERTScore: {avg_bert:.4f}\n")

            if avg_cider > best_cider:
                best_cider = avg_cider
                torch.save(model.state_dict(), CKPT_BEST)
                print(f"[BEST] New best CIDEr: {best_cider:.4f} -> {CKPT_BEST}")

        torch.save(model.state_dict(), CKPT_OUT)
        print(f"[Saved] Stage2 epoch {epoch} -> {CKPT_OUT}")

    print("\nTraining Complete!")
    print(f"Best CIDEr: {best_cider:.4f}")
    print(f"Best checkpoint: {CKPT_BEST}")


if __name__ == "__main__":
    train()
