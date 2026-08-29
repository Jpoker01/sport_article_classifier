"""Run the traditional TF-IDF classifier sweep with Optuna and save the results."""
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import CLEAN_DATA_PATH, ROOT
from src.traditional import CLASSIFIER_CONFIGS, objective

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_TRIALS = 25


def main():
    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["category"])
    y_val = encoder.transform(val["category"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results = {}
    for name, config in CLASSIFIER_CONFIGS.items():
        study = optuna.create_study(direction="maximize", study_name=f"tfidf_{name}")
        study.optimize(
            lambda t: objective(t, config, train["text"], y_train, val["text"], y_val),
            n_trials=N_TRIALS,
        )
        results[name] = {"best_val_macro_f1": study.best_value, **study.best_params}
        print(f"{name:16s} val macro-F1 = {study.best_value:.4f}")

    summary = pd.DataFrame(results).T.sort_values("best_val_macro_f1", ascending=False)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "traditional_sweep.csv")
    print(f"\nSaved: {out_dir / 'traditional_sweep.csv'}")
    print(summary)


if __name__ == "__main__":
    main()