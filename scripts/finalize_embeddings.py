"""Rebuild the best fastText trial on the training split and evaluate it on test."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifiers import CLASSIFIER_CONFIGS, load_best_trial
from src.config import CLEAN_DATA_PATH, FASTTEXT_MODEL_PATH, ROOT
from src.embeddings import embed_documents, load_fasttext
from src.evaluate import evaluate, full_report


def main():
    results_dir = ROOT / "results"
    name, val_score, params = load_best_trial(results_dir / "embeddings_trials.csv")
    print(f"Best trial: {name} (val macro_f1: {val_score:.4f})")
    print(f"Classifier params: {params}")

    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    encoder = LabelEncoder().fit(df["category"])
    y_train = encoder.transform(train["category"])
    y_test = encoder.transform(test["category"])

    # Fixed representation: load fastText, embed once, standardize (fit on train only).
    model = load_fasttext(FASTTEXT_MODEL_PATH)
    X_train = embed_documents(train["text"], model)
    X_test = embed_documents(test["text"], model)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = CLASSIFIER_CONFIGS[name].build(params)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    print(metrics)
    full_report(y_test, y_pred, target_names=encoder.classes_)
    
    pd.DataFrame([metrics]).to_csv(results_dir / "embeddings_test_metrics.csv", index=False)
    pd.DataFrame({
        "text": test["text"].values,
        "true": encoder.inverse_transform(y_test),
        "pred": encoder.inverse_transform(y_pred),
    }).to_csv(results_dir / "embeddings_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()