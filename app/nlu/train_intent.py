import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from app.config import INTENT_LABELS, INTENT_TO_IDX
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_model(
    data_path: str = "data/intent_dataset.csv",
    model_output_path: str = "models/intent_model",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5
):
    logger.info("Starting intent classifier training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(data_path)
    texts = df["text"].tolist()
    labels = [INTENT_TO_IDX[lbl] for lbl in df["intent"].tolist()]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(INTENT_LABELS)
    ).to(device)

    train_ds = IntentDataset(X_train, y_train, tokenizer)
    val_ds = IntentDataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch["labels"].numpy())

    acc = accuracy_score(all_true, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_preds, average="weighted")
    logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=INTENT_LABELS, yticklabels=INTENT_LABELS)
    plt.title("Intent Classification Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    logger.info("Confusion matrix saved to confusion_matrix.png")

    # Save model
    os.makedirs(model_output_path, exist_ok=True)
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    logger.info(f"Model saved to {model_output_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    results = train_model()
    print(f"Training complete: {results}")
