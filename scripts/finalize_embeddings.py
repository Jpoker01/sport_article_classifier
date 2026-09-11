"""Rebuild the best fastText trial on the training split and evaluate it on test."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib

from src.classifiers import CLASSIFIER_CONFIGS, load_best_trial
from src.config import CLEAN_DATA_PATH, FASTTEXT_MODEL_PATH, EMBEDDINGS_DIR
from src.embeddings import embed_documents, load_fasttext
from src.evaluate import evaluate, full_report
from src.weighting import capped_class_weight_dict

def main():
    """Refit the winning classifier on fastText embeddings and score it on test."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    name, val_score, params = load_best_trial(EMBEDDINGS_DIR / "optuna_trials.csv")
    print(f"Best trial: {name} (val macro_f1: {val_score:.4f})")
    print(f"Classifier params: {params}")

    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    encoder = LabelEncoder().fit(df["category"])
    y_train = encoder.transform(train["category"])
    y_test = encoder.transform(test["category"])
    class_weight = capped_class_weight_dict(y_train, len(encoder.classes_))

    # Fixed representation: load fastText, embed once, standardize (fit on train only).
    model = load_fasttext(FASTTEXT_MODEL_PATH)
    X_train = embed_documents(train["text"], model)
    X_test = embed_documents(test["text"], model)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = CLASSIFIER_CONFIGS[name].build(params, class_weight=class_weight)
    fit_kwargs = {}
    if isinstance(classifier, XGBClassifier):
        fit_kwargs["sample_weight"] = np.array([class_weight[int(y)] for y in y_train])
    classifier.fit(X_train, y_train, **fit_kwargs)
    y_pred = classifier.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    print(metrics)
    full_report(y_test, y_pred, target_names=encoder.classes_)
    
    metrics_path = EMBEDDINGS_DIR / "test_metrics.csv"
    predictions_path = EMBEDDINGS_DIR / "test_predictions.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame({
        "text": test["text"].values,
        "true": encoder.inverse_transform(y_test),
        "pred": encoder.inverse_transform(y_pred),
    }).to_csv(predictions_path, index=False)
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {predictions_path}")

    model_path = EMBEDDINGS_DIR / "model.joblib"
    joblib.dump(
        {"scaler": scaler, "classifier": classifier, "encoder": encoder},
        model_path,
     )

if __name__ == "__main__":
    main()
