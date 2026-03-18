"""
STEP 3: Train TinyBERT student model (run this on Google Colab with GPU)

Colab setup — paste this first in a cell:
  !pip install transformers datasets torch scikit-learn

Run: python step3_train_student.py
Input:  labeled_data.json
Output: crypto_student_model/ (folder with model weights)
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import mlflow
import mlflow.pytorch

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE     = "labeled_data.json"
MODEL_OUT_DIR  = "crypto_student_model"
BASE_MODEL     = "huawei-noah/TinyBERT_General_4L_312D"  # ~56MB
LABEL_MAP      = {"BUY": 0, "SELL": 1, "HOLD": 2}
ID2LABEL       = {0: "BUY", 1: "SELL", 2: "HOLD"}
EPOCHS         = 10
BATCH_SIZE     = 32
LR             = 3e-5
MAX_LEN        = 128
DISTILL_TEMP   = 4.0    # temperature for soft labels
DISTILL_ALPHA  = 0.3    # lower = more weight on hard weighted CE loss
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── DATASET ──────────────────────────────────────────────────────────────────

def build_text(sample: dict) -> str:
    """Convert a sample dict into a single text string for the model."""
    fg = sample.get("fear_greed_index", 50)
    fg_label = (
        "extreme fear" if fg < 25 else
        "fear"         if fg < 45 else
        "neutral"      if fg < 55 else
        "greed"        if fg < 75 else
        "extreme greed"
    )
    return (
        f"{sample['coin'].upper()} price ${sample['price']:,}, "
        f"24h change {sample['price_change_24h_pct']}%, "
        f"fear greed {fg} ({fg_label}). "
        f"News: {sample['headline']}"
    )


class CryptoDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=MAX_LEN):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        text = build_text(s)
        enc  = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = LABEL_MAP[s["label"]]
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─── DISTILLATION LOSS ────────────────────────────────────────────────────────

def distillation_loss(student_logits, true_labels, class_weights, temperature=DISTILL_TEMP, alpha=DISTILL_ALPHA):
    """
    Combined loss:
      alpha     * KL soft loss  (student mimics its own soft distribution at high temp)
      (1-alpha) * cross-entropy (student learns true labels, weighted for class imbalance)
    """
    # Hard loss: weighted cross-entropy to penalise minority class mistakes more
    hard_loss = F.cross_entropy(student_logits, true_labels, weight=class_weights)

    # Soft loss: label smoothing proxy (simulates teacher uncertainty)
    num_classes = student_logits.size(-1)
    smooth_labels = torch.full_like(
        student_logits, fill_value=0.1 / (num_classes - 1)
    )
    smooth_labels.scatter_(1, true_labels.unsqueeze(1), 0.9)

    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss    = F.kl_div(soft_student, smooth_labels, reduction="batchmean")

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            logits = model(ids, attention_mask=mask).logits
            preds  = logits.argmax(-1).cpu().tolist()
            labels = batch["label"].tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels


def train():
    print("=" * 50)
    print("STEP 3: Training TinyBERT student model")
    print("=" * 50)

    # Load data
    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Filter only labeled samples
    data = [s for s in data if s.get("label") in LABEL_MAP]
    print(f"\nLoaded {len(data)} labeled samples")

    # Stratified train/val split — keeps BUY/SELL/HOLD ratio in both sets
    from collections import Counter
    all_labels_str = [s["label"] for s in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(data, all_labels_str))
    train_data = [data[i] for i in train_idx]
    val_data   = [data[i] for i in val_idx]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    dist = Counter(s["label"] for s in train_data)
    print(f"Label distribution: {dict(dist)}")

    # Class weights: penalise minority class (BUY/SELL) mistakes more
    total = len(train_data)
    class_weights = torch.tensor([
        total / (3 * dist["BUY"]),
        total / (3 * dist["SELL"]),
        total / (3 * dist["HOLD"]),
    ], dtype=torch.float).to(device)
    print(f"Class weights: BUY={class_weights[0]:.2f} SELL={class_weights[1]:.2f} HOLD={class_weights[2]:.2f}")

    # Load tokenizer + model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Datasets + loaders
    train_ds = CryptoDataset(train_data, tokenizer)
    val_ds   = CryptoDataset(val_data,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_acc = 0.0
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    mlflow.set_experiment("crypto-distillation")
    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params({
            "base_model":    BASE_MODEL,
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "learning_rate": LR,
            "max_len":       MAX_LEN,
            "distill_temp":  DISTILL_TEMP,
            "distill_alpha": DISTILL_ALPHA,
            "train_size":    len(train_data),
            "val_size":      len(val_data),
            "weight_BUY":    round(class_weights[0].item(), 3),
            "weight_SELL":   round(class_weights[1].item(), 3),
            "weight_HOLD":   round(class_weights[2].item(), 3),
        })

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dl):
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(ids, attention_mask=mask).logits
                loss   = distillation_loss(logits, labels, class_weights)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if (step + 1) % 20 == 0:
                    print(f"  Epoch {epoch} Step {step+1}/{len(train_dl)} "
                          f"Loss: {total_loss/(step+1):.4f}")

            # Validation
            val_acc, preds, labels_list = evaluate(model, val_dl)
            avg_loss = total_loss / len(train_dl)
            f1 = f1_score(labels_list, preds, average="weighted", zero_division=0)

            # Log per-epoch metrics
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_accuracy": val_acc,
                "val_f1": f1,
            }, step=epoch)

            print(f"\nEpoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}  F1: {f1:.4f}")
            print(classification_report(
                labels_list, preds,
                target_names=["BUY", "SELL", "HOLD"],
                zero_division=0,
            ))

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(MODEL_OUT_DIR)
                tokenizer.save_pretrained(MODEL_OUT_DIR)
                mlflow.log_metric("best_val_accuracy", best_val_acc, step=epoch)
                mlflow.log_artifacts(MODEL_OUT_DIR, artifact_path="best_model")
                print(f"  ** New best model saved to {MODEL_OUT_DIR}/\n")

        mlflow.log_metric("final_best_val_accuracy", best_val_acc)

    print(f"\nDone! Best val accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {MODEL_OUT_DIR}/")
    print("  Run `mlflow ui` to view experiment dashboard")
    print("  Next: run step4_export_model.py")


if __name__ == "__main__":
    train()
