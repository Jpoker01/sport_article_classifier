"""Transformer fine-tuning utilities"""
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, f1_score
import math

from transformers import (
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from src.config import (
    PRIMARY_METRIC,
    SEED,
    TRANSFORMER_EARLY_STOPPING_PATIENCE,
    TRANSFORMER_WARMUP_RATIO,
)

from src.weighting import compute_capped_class_weight  # noqa: F401

class TextClassificationDataset(Dataset):
    """Pytorch dataset definition for the Trainer"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def compute_metrics(eval_pred):
    """Called after every epoch"""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }

class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropy loss."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def objective(trial, model_name, train, val, y_train, y_val,
              class_weights, max_length, epochs, out_dir_base):
    """Sample hyperparameters, fine-tune once, return val macro-F1."""
    lr = trial.suggest_float("lr", 1e-5, 8e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1])
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])
 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = TextClassificationDataset(train["text"], y_train, tokenizer, max_length)
    val_ds = TextClassificationDataset(val["text"], y_val, tokenizer, max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_weights)
    )
 
    trial_dir = out_dir_base / f"trial{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
 
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(TRANSFORMER_WARMUP_RATIO * total_steps)
 
    training_args = TrainingArguments(
        output_dir=str(trial_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=PRIMARY_METRIC,
        greater_is_better=True,
        bf16=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )
 
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TRANSFORMER_EARLY_STOPPING_PATIENCE
        )],
    )
    trainer.train()
    metrics = trainer.evaluate()
    return metrics[f"eval_{PRIMARY_METRIC}"]