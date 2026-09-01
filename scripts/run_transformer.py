"""Fine-tune a Czech transformer encoder"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CLEAN_DATA_PATH,
    PRIMARY_METRIC,
    ROOT,
    SEED,
    TRANSFORMER_EARLY_STOPPING_PATIENCE,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_LR,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_WARMUP_RATIO,
    TRANSFORMER_WEIGHT_DECAY,
)
from src.transformer import TextClassificationDataset, WeightedTrainer, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--model-name", default="ufal/robeczech-base")
    parser.add_argument("--max-length", type=int, default=TRANSFORMER_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=TRANSFORMER_LR)
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_EPOCHS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on rows (for smoke tests).")
    parser.add_argument("--run-name", default=None,
                        help="Suffix for the output directory (used by the sweep).")
    parser.add_argument("--weight-decay", type=float, default=TRANSFORMER_WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip writing model weights (sweep trials only need metrics).")
    return parser.parse_args()

def main():
    args = parse_args()

    safe_model_name = args.model_name.replace("/", "_")
    if args.run_name:
        safe_model_name = f"{safe_model_name}__{args.run_name}"
    out_dir = ROOT / "results" / "transformer" / safe_model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    encoder = LabelEncoder().fit(df["category"])
    num_labels = len(encoder.classes_)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=SEED)
        val = val.sample(n=min(args.limit, len(val)), random_state=SEED)

    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])
    num_labels = len(encoder.classes_)

    present_classes = np.unique(y_train)
    present_weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    weights = np.ones(num_labels, dtype=np.float32)
    weights[present_classes] = present_weights
    class_weights = torch.tensor(weights, dtype=torch.float)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = TextClassificationDataset(train["text"], y_train, tokenizer, args.max_length)
    val_ds = TextClassificationDataset(val["text"], y_val, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(TRANSFORMER_WARMUP_RATIO * total_steps)
    
    callbacks = []
    if not args.no_save:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=TRANSFORMER_EARLY_STOPPING_PATIENCE
        ))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="no" if args.no_save else "epoch",
        load_best_model_at_end=not args.no_save,
        metric_for_best_model=PRIMARY_METRIC,
        greater_is_better=True,
        bf16=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
    )

    trainer.train()

    best_f1 = max(
        (log[f"eval_{PRIMARY_METRIC}"] for log in trainer.state.log_history
         if f"eval_{PRIMARY_METRIC}" in log),
        default=float("nan"),
    )
    pd.DataFrame([{PRIMARY_METRIC: best_f1}]).to_csv(out_dir / "val_metrics.csv", index=False)
    print(f"best val {PRIMARY_METRIC}: {best_f1:.4f}")

    if not args.no_save:
        best_dir = out_dir / "best"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        np.save(out_dir / "label_classes.npy", encoder.classes_)


if __name__ == "__main__":
    main()