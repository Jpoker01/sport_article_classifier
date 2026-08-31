"""Traditional TF-IDF baselines with Optuna tuning."""

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from src.classifiers import CLASSIFIER_CONFIGS

def build_tfidf_vectorizer(max_features, min_df, ngram_range=(1, 2),
                           lowercase=True, sublinear_tf=True):
    """Word-level TF-IDF vectorizer."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )

def objective(trial, config, train_texts, y_train, val_texts, y_val):
    """Sample vectorizer + classifier hyperparameters, fit, score macro-F1 on val."""
    vectorizer = build_tfidf_vectorizer(
    ngram_range=(1, trial.suggest_categorical("tfidf_ngram_max", [1, 2])),
    max_features=trial.suggest_categorical("tfidf_max_features", [20000, 50000, 100000]),
    min_df=trial.suggest_int("tfidf_min_df", 1, 5),
    sublinear_tf=trial.suggest_categorical("tfidf_sublinear_tf", [True, False]),
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    classifier = config.build(trial)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_val)
    return f1_score(y_val, predictions, average="macro")