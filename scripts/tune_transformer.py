"""Optuna sweep over transformer fine-tuning hyperparameters."""
import sys
from pathlib import Path

import argparse
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CLEAN_DATA_PATH,
    PRIMARY_METRIC,
    ROOT,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_MAX_LENGTH,
)
from src.transformer import objective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="ufal/robeczech-base")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=TRANSFORMER_MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_EPOCHS)
    return parser.parse_args()


def main():
    args = parse_args()

    safe_model_name = args.model_name.replace("/", "_")
    out_dir = ROOT / "results" / "transformer" / f"{safe_model_name}__tune"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    encoder = LabelEncoder().fit(df["category"])
    num_labels = len(encoder.classes_)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])

    present_classes = np.unique(y_train)
    present_weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    weights = np.ones(num_labels, dtype=np.float32)
    weights[present_classes] = present_weights
    class_weights = torch.tensor(weights, dtype=torch.float)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial, args.model_name, train, val, y_train, y_val,
            class_weights, args.max_length, args.epochs, out_dir,
        ),
        n_trials=args.n_trials,
    )

    print(f"best val {PRIMARY_METRIC}: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    rows = [{"trial": t.number, PRIMARY_METRIC: t.value, **t.params}
            for t in study.trials if t.value is not None]
    summary = pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False)
    summary.to_csv(out_dir / "sweep.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()