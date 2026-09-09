"""Rebuild the best TF-IDF trial on the training split and evaluate it on test."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
 
from src.classifiers import CLASSIFIER_CONFIGS, load_best_trial
from src.config import CLEAN_DATA_PATH, RESULTS_PATH, ROOT
from src.evaluate import evaluate, full_report
from src.traditional import build_tfidf_vectorizer
from src.weighting import capped_class_weight_dict

 
def main():
    """Refit the winning TF-IDF configuration and score it once on the test split."""
    name, val_score, params = load_best_trial(RESULTS_PATH / "traditional_optuna_trials.csv")
    print(f"Best trial: {name} (val macro_f1: {val_score:.4f})")
 
    # tfidf_* params configure the vectorizer, everything else the classifier.
    tfidf_params = {key.removeprefix("tfidf_"): value
                    for key, value in params.items() if key.startswith("tfidf_")}
    classifier_params = {key: value
                  for key, value in params.items() if not key.startswith("tfidf_")}
 
    # Rebuild ngram_range, the search space stores one bound and which one depends on the analyzer."
    if "char_ngram_max" in tfidf_params:
        tfidf_params["ngram_range"] = (3, tfidf_params.pop("char_ngram_max"))
    elif "ngram_max" in tfidf_params:
        tfidf_params["ngram_range"] = (1, tfidf_params.pop("ngram_max"))
 
    print(f"TF-IDF params: {tfidf_params}")
    print(f"Classifier params: {classifier_params}")
 
    df = pd.read_parquet(CLEAN_DATA_PATH)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
 
    encoder = LabelEncoder().fit(df["category"])
    y_train = encoder.transform(train["category"])
    y_test = encoder.transform(test["category"])
    class_weight = capped_class_weight_dict(y_train, len(encoder.classes_))

    vectorizer = build_tfidf_vectorizer(**tfidf_params)
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])
 
    classifier = CLASSIFIER_CONFIGS[name].build(classifier_params, class_weight=class_weight)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
 
    metrics = evaluate(y_test, y_pred)
    print(metrics)
    full_report(y_test, y_pred, target_names=encoder.classes_)
 
    pd.DataFrame([metrics]).to_csv(RESULTS_PATH / "traditional_test_metrics.csv", index=False)
    pd.DataFrame({
        "text": test["text"].values,
        "true": encoder.inverse_transform(y_test),
        "pred": encoder.inverse_transform(y_pred),
    }).to_csv(RESULTS_PATH / "traditional_test_predictions.csv", index=False)
 
    joblib.dump(
    {"vectorizer": vectorizer, "classifier": classifier, "encoder": encoder},
    RESULTS_PATH / "traditional_model.joblib",
    )
 
if __name__ == "__main__":
    main()