"""Optuna sweep over transformer fine-tuning hyperparameters."""
import argparse
import gc

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEAN_DATA_PATH,
    PRIMARY_METRIC,
    RESULTS_PATH,
    SEED,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_MAX_LENGTH,
)

from src.transformer import compute_capped_class_weight, objective

def parse_args():
    """Parse command line arguments.

    Returns:
        Namespace with model_name, n_trials, max_length and epochs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="ufal/robeczech-base")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=TRANSFORMER_MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_EPOCHS)
    return parser.parse_args()


def trials_dataframe(study):
    """Collect the finished trials of a study into a sorted DataFrame.

    Args:
        study: The Optuna study to read trials from.

    Returns:
        DataFrame with one row per finished trial, holding the trial number,
        the primary metric and the sampled hyperparameters, sorted best first.
        Trials with a value of None (pruned or failed) are skipped.
    """
    rows = [
        {"trial": trial.number, PRIMARY_METRIC: trial.value, **trial.params}
        for trial in study.trials
        if trial.value is not None
    ]
    return pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False)

def main():
    """Tune the fine-tuning hyperparameters of one transformer with Optuna."""
    args = parse_args()
    safe_model_name = args.model_name.replace("/", "_")
    
    out_dir = RESULTS_PATH / "transformer" / f"{safe_model_name}__tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "transformer_optuna_trials.csv"

    df = pd.read_parquet(CLEAN_DATA_PATH)
    
    # Fit on all categories so a class missing from one split cannot break encoding.
    encoder = LabelEncoder().fit(df["category"])
    num_labels = len(encoder.classes_)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])

    class_weights = torch.tensor(
        compute_capped_class_weight(y_train, num_labels),
        dtype=torch.float,
    )

    def save_progress(study, trial):
        """Rewrite the trials CSV after every trial.
        
        """
        trials_dataframe(study).to_csv(trials_path, index=False)

    def free_memory(study, trial):
        """Release the finished trial's model before the next one is built.

        Every trial instantiates a fresh transformer. Neither Python nor
        PyTorch reclaims those copies promptly on their own, and a long sweep
        otherwise exhausts host RAM.
        """
        gc.collect()
        torch.cuda.empty_cache()

    study = optuna.create_study(
        direction="maximize",
        study_name=f"transformer_{safe_model_name}",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: objective(
            trial, args.model_name, train, val, y_train, y_val,
            class_weights, args.max_length, args.epochs, out_dir,
        ),
        n_trials=args.n_trials,
        callbacks=[save_progress, free_memory],
    )

    print(f"best val {PRIMARY_METRIC}: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")
    print(f"\nSaved: {trials_path}")
    print(trials_dataframe(study))


if __name__ == "__main__":
    main()