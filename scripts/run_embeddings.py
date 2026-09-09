"""Run the traditional fastText classifier sweep with Optuna and save the results."""
import argparse

import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CLEAN_DATA_PATH, FASTTEXT_MODEL_PATH, RESULTS_PATH, ROOT
from src.embeddings import EMBEDDING_CONFIGS, embed_documents, load_fasttext, objective
from src.weighting import capped_class_weight_dict


def parse_args():
    """Parse command line arguments.

    Returns:
        Namespace with n_trials, the number of Optuna trials per classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    return parser.parse_args()

def main():
    """Run one Optuna study per classifier on fixed embeddings and save results."""
    args = parse_args()
    
    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    encoder = LabelEncoder().fit(df["category"])
    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])
    class_weight = capped_class_weight_dict(y_train, len(encoder.classes_))


    model = load_fasttext(FASTTEXT_MODEL_PATH)
    X_train = embed_documents(train["text"], model)
    X_val = embed_documents(val["text"], model)
 
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
 
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    trials_path = RESULTS_PATH / "embeddings_optuna_trials.csv"
    
    results = {}
    for i, (name, config) in enumerate(EMBEDDING_CONFIGS.items(), 1):
        print(f"[{i}/{len(EMBEDDING_CONFIGS)}] Tuning {name} ...")
        study = optuna.create_study(direction="maximize", study_name=f"fasttext_{name}")
        study.optimize(
            lambda trial: objective(trial, config, X_train, y_train, X_val, y_val,
                                    class_weight=class_weight),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        results[name] = {"best_val_macro_f1": study.best_value, **study.best_params}
        print(f"{name}: {study.best_value:.4f}")
        pd.DataFrame(results).T.to_csv(trials_path)
 
    summary = pd.DataFrame(results).T.sort_values("best_val_macro_f1", ascending=False)
    summary.to_csv(trials_path)
    print(f"\nSaved: {trials_path}")
    print(summary)
 
 
if __name__ == "__main__":
    main()