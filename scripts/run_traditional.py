"""Run the traditional TF-IDF classifier sweep with Optuna and save the results."""
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_DATA_PATH, ROOT
from src.traditional import CLASSIFIER_CONFIGS, objective

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    n_trials = args.n_trials
    
    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["category"])
    y_val = encoder.transform(val["category"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)          # create dir ONCE, before the loop

    results = {}
    for i, (name, config) in enumerate(CLASSIFIER_CONFIGS.items(), 1):
        print(f"[{i}/{len(CLASSIFIER_CONFIGS)}] Tuning {name} ...")
        study = optuna.create_study(direction="maximize", study_name=f"tfidf_{name}")
        study.optimize(
            lambda t: objective(t, config, train["text"], y_train, val["text"], y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        results[name] = {"best_val_macro_f1": study.best_value, **study.best_params}
        print(f"    {name} val macro-F1 = {study.best_value:.4f}")
        # save after EACH classifier so a kill never loses finished ones
        pd.DataFrame(results).T.to_csv(out_dir / "traditional_sweep.csv")

    summary = pd.DataFrame(results).T.sort_values("best_val_macro_f1", ascending=False)
    summary.to_csv(out_dir / "traditional_sweep.csv")
    print(f"\nSaved: {out_dir / 'traditional_sweep.csv'}")
    print(summary)


if __name__ == "__main__":
    main() 