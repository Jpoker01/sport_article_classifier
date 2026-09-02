"""Tune TF-IDF + classical classifiers with Optuna and record every trial."""
import argparse

import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.classifiers import CLASSIFIER_CONFIGS
from src.config import CLEAN_DATA_PATH, RESULTS_PATH
from src.traditional import objective

def parse_args():
    """Parse command line arguments.

    Returns:
        Namespace with n_trials, the number of Optuna trials per classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    return parser.parse_args()


def main():
    """Run one Optuna study (composed of N n-trials per classifier and write all results to a CSV."""
    args = parse_args()

    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    # Fit on all categories so a class missing from one split cannot break encoding.
    encoder = LabelEncoder().fit(df["category"])
    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    trials_path = RESULTS_PATH / "traditional_optuna_trials.csv"

    results = {}
    for i, (name, config) in enumerate(CLASSIFIER_CONFIGS.items(), 1):
        print(f"[{i}/{len(CLASSIFIER_CONFIGS)}] | Tuning {name} ...")
        study = optuna.create_study(direction="maximize", study_name=f"tfidf_{name}")
        study.optimize(
            lambda t: objective(t, config, train["text"], y_train, val["text"], y_val),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        results[name] = {"best_val_macro_f1": study.best_value, **study.best_params}
        print(f"    {name} val macro-F1 = {study.best_value:.4f}")

        pd.DataFrame(results).T.to_csv(trials_path)

    summary = pd.DataFrame(results).T.sort_values("best_val_macro_f1", ascending=False)
    summary.to_csv(trials_path)
    print(f"\nSaved: {trials_path}")
    print(summary)


if __name__ == "__main__":
    main()