"""Run the traditional fastText classifier sweep with Optuna and save the results."""
import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_DATA_PATH, FASTTEXT_MODEL_PATH, ROOT
from src.embeddings import EMBEDDING_CONFIGS, embed_documents, load_fasttext, objective

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

    encoder = LabelEncoder().fit(train["category"])
    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])

    model = load_fasttext(FASTTEXT_MODEL_PATH)
    X_train = embed_documents(train["text"], model)
    X_val = embed_documents(val["text"], model)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for i, (name, config) in enumerate(EMBEDDING_CONFIGS.items(), 1):
        print(f"[{i}/{len(EMBEDDING_CONFIGS)}] Tuning {name} ...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, config, X_train, y_train, X_val, y_val),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        results[name] = {"best_val_macro_f1": study.best_value, **study.best_params}
        print(f"{name}: {study.best_value:.4f}")
        pd.DataFrame(results).T.to_csv(out_dir / "embeddings_sweep.csv")

    summary = pd.DataFrame(results).T.sort_values("best_val_macro_f1", ascending=False)
    summary.to_csv(out_dir / "embeddings_sweep.csv")
    print(f"\nSaved: {out_dir / 'traditional_sweep.csv'}")
    print(summary)


if __name__ == "__main__":
    main()